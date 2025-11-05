import numpy as np
import matplotlib.pyplot as plt
from fixedpoint import q, FX_NARROW
from prbs_qpsk import prbs_qpsk_iq
from channel import channel
from mu_switch import make_mu_switch
from pbfdaf_lms import ffe_lms_freq_iq
from metrics import ber as _ber
from plots import (
    plot_weights_labeled, plot_filter_profile, plot_channel_profile, plot_combined_response,
    plot_time_iq_complex, plot_constellation, plot_freq_responses_db, sym_ylim
)
from math import erfc as _erfc_scalar  

# teoria QPSK: Q(x) y BER vs Es/N0 
def _erfc_vec(x):
    x = np.asarray(x, dtype=float)
    return np.vectorize(_erfc_scalar)(x)

def _Q(x):
    # Q(x) = 0.5 * erfc(x / sqrt(2))
    return 0.5 * _erfc_vec(x / np.sqrt(2.0))

def ber_qpsk_theory_esn0(EsN0_dB):
    # convertir Es/N0(dB) A Eb/N0(dB) (QPSK: 2 bits/simbolo)
    EbN0_dB  = np.asarray(EsN0_dB) - 10*np.log10(2.0)  # -3.0103 dB
    EbN0_lin = 10**(EbN0_dB/10.0)
    return _Q(np.sqrt(2.0 * EbN0_lin))  # BER QPSK coherente + Gray

def suggest_center_tap(h, L_eq, pre_ratio=0.15):
        k_peak = int(np.argmax(np.abs(h)))
        off = max(1, int(pre_ratio * L_eq))
        return min(k_peak + off, L_eq - 1)

# clase del simulador 
class equalizerSimulator:
    def __init__(self,
                 N_SYM, N_PLOT, N_SKIP,
                 CHAN_MODE, H_TAPS, SNR_DB, SEED_NOISE,
                 L_EQ, PART_N, CENTER_TAP,
                 MU, MU_SWITCH_ENABLE, MU_FINAL, N_SWITCH,
                 USE_STABLE, STABLE_WIN, STABLE_TOL, STABLE_PATIENCE,
                 seedI, seedQ,
                 # flags de canal
                 NORM_H_POWER=False, SNR_REF="post"):
        # sim
        self.N_SYM        = N_SYM
        self.N_PLOT       = N_PLOT
        self.N_SKIP       = N_SKIP
        # canal
        self.CHAN_MODE    = CHAN_MODE
        self.SNR_DB       = SNR_DB      # Es/N0(dB)
        self.SEED_NOISE   = SEED_NOISE
        self.H_TAPS = (
            np.array([
                0.72+0.00j, 0.20+0.05j, -0.15+0.04j, 0.12-0.03j,
               -0.10+0.02j, 0.08-0.02j, 0.06+0.01j, -0.05-0.015j,
                0.04+0.01j, -0.03-0.008j, 0.02+0.005j, 0.015+0.003j
            ], dtype=np.complex128)
            if H_TAPS is None else np.asarray(H_TAPS, np.complex128)
        )
        
        self.NORM_H_POWER = bool(NORM_H_POWER)
        self.SNR_REF      = str(SNR_REF)

        # eq
        self.L_EQ         = L_EQ
        self.PART_N       = PART_N
        self.CENTER_TAP = (suggest_center_tap(self.H_TAPS, L_EQ) 
                           if CENTER_TAP is None else CENTER_TAP)
        # mu / switch
        self.MU               = MU
        self.MU_SWITCH_ENABLE = MU_SWITCH_ENABLE
        self.MU_FINAL         = MU_FINAL
        self.N_SWITCH         = N_SWITCH
        self.USE_STABLE       = USE_STABLE
        self.STABLE_WIN       = STABLE_WIN
        self.STABLE_TOL       = STABLE_TOL
        self.STABLE_PATIENCE  = STABLE_PATIENCE
        # PRBS
        self.seedI        = seedI
        self.seedQ        = seedQ
        # buffers
        self.sI = self.sQ = None
        self.x_n = self.x_det = None
        self.yI = self.yQ = self.yhatI = self.yhatQ = None
        self.eI = self.eQ = None
        self.W_histI = self.W_histQ = None
        self.w_finI = self.w_finQ = None
        self.bI_src = self.bQ_src = None
        self.k0 = None


    def _build_mu_step(self):
        return make_mu_switch(
            enable=self.MU_SWITCH_ENABLE, mu_init=self.MU, mu_final=self.MU_FINAL,
            n_switch=self.N_SWITCH, use_stable=self.USE_STABLE,
            win=self.STABLE_WIN, tol=self.STABLE_TOL, patience=self.STABLE_PATIENCE
        )

    def set_source(self, sI, sQ):
        # fija fuente externa
        self.sI = np.asarray(sI, dtype=np.float64)
        self.sQ = np.asarray(sQ, dtype=np.float64)
        self.bI_src = (self.sI < 0).astype(np.uint8)
        self.bQ_src = (self.sQ < 0).astype(np.uint8)

    def gen_source(self):
        # PRBS -> QPSK
        self.sI, self.sQ = prbs_qpsk_iq(self.N_SYM, self.seedI, self.seedQ)
        self.bI_src = (self.sI < 0).astype(np.uint8)
        self.bQ_src = (self.sQ < 0).astype(np.uint8)
        return self.sI + 1j*self.sQ

    def pass_channel(self, s):
        # canal FIR + AWGN
        Hq = np.array([q(h.real, FX_NARROW) + 1j*q(h.imag, FX_NARROW) for h in self.H_TAPS],
                      dtype=np.complex128)
        self.x_n, self.x_det = channel(
            s, mode=self.CHAN_MODE, h=Hq, SNR_dB=self.SNR_DB,
            seed=self.SEED_NOISE, return_det=True,
            
            norm_h_power=self.NORM_H_POWER, snr_ref=self.SNR_REF
        )
        if self.SNR_DB is not None:
            noise = self.x_n - self.x_det
            snr_meas = 10*np.log10(
                (np.mean(np.abs(self.x_det)**2)+1e-12) /
                (np.mean(np.abs(noise)**2)+1e-12)
            )
            print(f"SNR pedida: {self.SNR_DB:.2f} dB | SNR medida: {snr_meas:.2f} dB")
        return self.x_n

    def equalize(self):
        # FFE PBFDAF (LMS freq)
        mu_step = self._build_mu_step()
        (self.yI, self.yQ, self.yhatI, self.yhatQ, self.eI, self.eQ,
         self.W_histI, self.W_histQ, self.w_finI, self.w_finQ, k0) = ffe_lms_freq_iq(
            self.x_n.real.copy(), self.x_n.imag.copy(),
            L_eq=self.L_EQ, part_N=self.PART_N, mu=self.MU,
            center_index=self.CENTER_TAP, mu_step=mu_step
        )
        self.k0 = int(k0)
        return k0

    def plot(self,
         weights=True, profile=True, conv=True,
         time_in=True, time_out=True,
         const_in=True, const_out=True, const_dec=True,
         chan_profile=True, freq=True):
        # graficos
        W_hist = self.W_histI + 1j*self.W_histQ
        w_fin  = self.w_finI  + 1j*self.w_finQ
        y_n    = self.yI + 1j*self.yQ
        yhat_n = self.yhatI + 1j*self.yhatQ
    
        # misma escala Re/Im filtro y canal
        ylims = sym_ylim(np.real(w_fin), np.imag(w_fin), np.real(self.H_TAPS), np.imag(self.H_TAPS))
    
        if weights:
            plot_weights_labeled(W_hist, center_index=self.CENTER_TAP)
        if profile:
            plot_filter_profile(w_fin, center_index=self.CENTER_TAP,
                                ylims_re=ylims, ylims_im=ylims)
        if chan_profile:
            plot_channel_profile(self.H_TAPS, center_index=None, ylims=ylims)
        if conv:
            plot_combined_response(w_fin, self.H_TAPS)
        if time_in:
            plot_time_iq_complex(self.x_n, self.N_PLOT, "Salida del canal: x_n")
        if time_out:
            tail = y_n[max(0, len(y_n) - max(self.N_PLOT, len(y_n) - self.N_SKIP)):]
            plot_time_iq_complex(tail, self.N_PLOT, "Salida del ecualizador: y_n")
        if const_in:
            x_tail = self.x_n[-min(self.N_PLOT, len(self.x_n)):]
            plot_constellation(x_tail, title="Constelacion x_n")
        if const_out:
            tail = y_n[-min(self.N_PLOT, len(y_n)):]
            plot_constellation(tail, title="Constelacion y_n")
        if const_dec:
            tail_hat = yhat_n[-min(self.N_PLOT, len(yhat_n)):]
            plot_constellation(tail_hat, title="Constelacion y_hat")
        if freq:
            plot_freq_responses_db( w_fin,np.asarray(self.H_TAPS, np.complex128), pad_n=8192)
    
    def run(self):
        # pipeline
        s = self.gen_source()
        self.pass_channel(s)
        self.equalize()
        return self

    def ber(self, *, skip=0, lag=None, win=None, mN=None, min_overlap=None):
     if lag is None:
         lag_eff = 0 if (self.CHAN_MODE == "ideal") else int(self.k0)
     else:
         lag_eff = int(lag)

     if win is None: win = 4 * self.PART_N
     if mN  is None: mN  = 8
     if min_overlap is None: min_overlap = max(4 * self.PART_N, 256)

     return _ber(self.bI_src, self.bQ_src, self.yhatI, self.yhatQ,
                 lag=lag_eff, N=int(self.PART_N), win=int(win),
                 mN=int(mN), skip=int(skip), min_overlap=int(min_overlap))


# BER vs SNR (Es/N0)
def ber_at_snr(snr_db, *, N_SYM=30000, skip=2000, runs=1, eq_params=None, reuse_src=None):
    if eq_params is None: eq_params = {}
    ber_sum = 0.0; berI_sum = 0.0; berQ_sum = 0.0; nbits_sum = 0
    for r in range(runs):
        print(f"[SNR={snr_db} dB] run {r+1}/{runs}…", flush=True)
        sim = equalizerSimulator(
            N_SYM             =  N_SYM,
            N_PLOT            =  0,
            N_SKIP            =  0,
            CHAN_MODE         =  eq_params.get("CHAN_MODE", "fir"),
            H_TAPS            =  eq_params.get("H_TAPS",[
                                 0.72+0.00j, 0.20+0.05j, -0.15+0.04j, 0.12-0.03j,
                                -0.10+0.02j, 0.08-0.02j, 0.06+0.01j, -0.05-0.015j,
                                 0.04+0.01j, -0.03-0.008j, 0.02+0.005j, 0.015+0.003j
                                                        ]),
            SNR_DB            =  snr_db,              # Es/N0(dB)
            SEED_NOISE        =  eq_params.get("SEED_NOISE", 5678 + r),
            L_EQ              =  eq_params.get("L_EQ", 31),
            PART_N            =  eq_params.get("PART_N", 16),
            CENTER_TAP        =  eq_params.get("CENTER_TAP", None),
            MU                =  eq_params.get("MU", 0.015),
            MU_SWITCH_ENABLE  =  eq_params.get("MU_SWITCH_ENABLE", True),
            MU_FINAL          =  eq_params.get("MU_FINAL", 0.006),
            N_SWITCH          =  eq_params.get("N_SWITCH", 200),
            USE_STABLE        =  eq_params.get("USE_STABLE", False),
            STABLE_WIN        =  eq_params.get("STABLE_WIN", 300),
            STABLE_TOL        =  eq_params.get("STABLE_TOL", 1.0),
            STABLE_PATIENCE   =  eq_params.get("STABLE_PATIENCE", 80),
            seedI             =  eq_params.get("seedI", 0x17F + r),
            seedQ             =  eq_params.get("seedQ", 0x11D + r),
           
            NORM_H_POWER      =  eq_params.get("NORM_H_POWER", False),
            SNR_REF           =  eq_params.get("SNR_REF", "post"),
        )
        if reuse_src is not None:
            sI, sQ = reuse_src
            sim.set_source(sI, sQ); s = sim.sI + 1j*sim.sQ
        else:
            s = sim.gen_source()
        sim.pass_channel(s)
        sim.equalize()

        
        force_lag = 0 if (sim.CHAN_MODE == "ideal") else None
        res = sim.ber(skip=skip, win=4*sim.PART_N, mN=6, lag=force_lag)
        

        print(f"[SNR={snr_db} dB] run {r+1}/{runs} -> "
              f"BER={res['BER']:.3e}, Nbits={res['Nbits']}, "
              f"lag={res.get('lag_used')}, xform={res.get('xform_used')}", flush=True)
        if res["Nbits"] > 0:
            ber_sum  += res["BER"]   * res["Nbits"]
            berI_sum += res["BER_I"] * res["Nbits"] / 2
            berQ_sum += res["BER_Q"] * res["Nbits"] / 2
            nbits_sum += res["Nbits"]
    if nbits_sum == 0:
        print(f"[SNR={snr_db} dB] sin datos", flush=True)
        return {"SNR": snr_db, "BER": np.nan, "BER_I": np.nan, "BER_Q": np.nan, "Nbits": 0}
    ber_avg  = ber_sum  / nbits_sum
    beri_avg = (2 * berI_sum) / nbits_sum
    berq_avg = (2 * berQ_sum) / nbits_sum
    print(f"[SNR={snr_db} dB] DONE -> BER={ber_avg:.3e}  (Nbits={nbits_sum})", flush=True)
    return {"SNR": snr_db, "BER": ber_avg, "BER_I": beri_avg, "BER_Q": berq_avg, "Nbits": nbits_sum}

def sweep_snr(snr_list, *, N_SYM=30000, skip=2000, runs=1, eq_params=None, reuse_src=True):
    # barrido de SNR (Es/N0)
    src = None
    if reuse_src:
        sI, sQ = prbs_qpsk_iq(N_SYM, 0x17F, 0x11D)
        src = (sI, sQ)
    results = []
    for i, snr in enumerate(snr_list, 1):
        print(f"\n=== {i}/{len(snr_list)} : SNR={snr} dB ===", flush=True)
        results.append(ber_at_snr(snr, N_SYM=N_SYM, skip=skip, runs=runs,
                                  eq_params=eq_params, reuse_src=src))
    return results

def plot_ber_curve(results, title="BER vs SNR"):
    # curva BER: teorica (Es/N0) + simulada
    snr_esn0 = np.array([r["SNR"] for r in results], dtype=float)  # Es/N0(dB)
    ber_sim  = np.array([max(r["BER"], 1e-12) for r in results], dtype=float)

    # teoria sobre el mismo rango observado
    xmin = float(np.nanmin(snr_esn0)) if snr_esn0.size else 0.0
    xmax = float(np.nanmax(snr_esn0)) if snr_esn0.size else 12.0
    x_dense_esn0 = np.linspace(xmin, xmax, 500)
    ber_th = ber_qpsk_theory_esn0(x_dense_esn0)

    plt.figure(figsize=(6.5, 4.2))
    plt.semilogy(x_dense_esn0, ber_th, '-', linewidth=2, label='QPSK teoria (BER vs Es/N0)')
    plt.semilogy(snr_esn0, ber_sim, 'o-', linewidth=1, label='QPSK simulada')
    plt.grid(True, which="both", linestyle=":")
    plt.xlabel("Es/N0 [dB]")   
    plt.ylabel("BER")
    plt.xlim(xmin, xmax)
    plt.ylim(1e-7, 1)
    plt.title(title)
    plt.legend()
    plt.tight_layout()


# main 
if __name__ == "__main__":
    sim = equalizerSimulator(
        N_SYM            =  1000000,
        N_PLOT           =  1000000,
        N_SKIP           =  0,
        CHAN_MODE        =  "fir",
        #H_TAPS          =  [0,0,0,0,0,0,1,0,0,0,0,0,0],  # impulso
        H_TAPS = [ 0.9091025 +0.0000000j, 
                   0.2525285 +0.0631321j, 
                  -0.1893963 +0.0505057j, 
                   0.1515171 -0.0378793j, 
                  -0.1262642 +0.0252529j, 
                   0.1010114 -0.0252529j, 
                   0.0757585 +0.0126264j, 
                  -0.0631321 -0.0189396j, 
                   0.0505057 +0.0126264j, 
                  -0.0378793 -0.0101011j, 
                   0.0252529 +0.0063132j, 
                   0.0189396 +0.0037879j ],
        
        SNR_DB           =  20,    
        SEED_NOISE       =  5678,
        L_EQ             =  31,
        PART_N           =  16,
        CENTER_TAP       =  None,
        MU               =  0.006,
        MU_SWITCH_ENABLE =  True,
        MU_FINAL         =  0.0005,
        N_SWITCH         =  500,
        USE_STABLE       =  False,
        STABLE_WIN       =  300,
        STABLE_TOL       =  1.0,
        STABLE_PATIENCE  =  80,
        seedI            =  0x17F,
        seedQ            =  0x11D,
        # canal unitario o referenciar SNR al TX
        NORM_H_POWER     =  False,
        SNR_REF          =  "post"
    ).run()

    # graficos 
    sim.plot(
        weights      =  True,
        profile      =  True,
        chan_profile =  True,
        freq         =  True,
        conv         =  True,
        time_in      =  True,
        const_in     =  True,
        time_out     =  True,
        const_out    =  True,
        const_dec    =  True
    )

    print("CENTER_TAP =", sim.CENTER_TAP, "k0 =", sim.k0)
    res = sim.ber(skip=2000, win=4*sim.PART_N, mN=8)
    print("BER:", res)

    # sweep 
    RUN_BER_SWEEP = True
    if RUN_BER_SWEEP:
        SNRS = list(range(0, 13, 1))  # inicio, fin, paso
        EQ_PARAMS = dict(
            CHAN_MODE        =  "fir",
            L_EQ             =  31,
            PART_N           =  16,
            CENTER_TAP       =  None,
            MU               =  0.005,
            MU_SWITCH_ENABLE =  True,
            MU_FINAL         =  0.0002,
            N_SWITCH         =  1000,
            USE_STABLE       =  False,
            STABLE_WIN       =  300,
            STABLE_TOL       =  1.0,
            STABLE_PATIENCE  =  80,
            #H_TAPS           =  [0,0,0,0,0,0,1,0,0,0,0,0,0],  # impulso
            H_TAPS = [ 0.9091025 +0.0000000j, 
                       0.2525285 +0.0631321j, 
                      -0.1893963 +0.0505057j, 
                       0.1515171 -0.0378793j, 
                      -0.1262642 +0.0252529j, 
                       0.1010114 -0.0252529j, 
                       0.0757585 +0.0126264j, 
                      -0.0631321 -0.0189396j, 
                       0.0505057 +0.0126264j, 
                      -0.0378793 -0.0101011j, 
                       0.0252529 +0.0063132j, 
                       0.0189396 +0.0037879j ],

            # canal:
            NORM_H_POWER     =  False,
            SNR_REF          =  "post",  # para calcular sigma
        )
        SWEEP_N_SYM     = 1000000
        SWEEP_SKIP      = 10000
        SWEEP_RUNS      = 1
        SWEEP_REUSE_SRC = True

        results = sweep_snr(
            SNRS,
            N_SYM     = SWEEP_N_SYM,
            skip      = SWEEP_SKIP,
            runs      = SWEEP_RUNS,
            eq_params = EQ_PARAMS,
            reuse_src = SWEEP_REUSE_SRC
        )
        for r in results:
            print(f"SNR(Es/N0)={r['SNR']:>2} dB | BER={r['BER']:.3e} | Nbits={r['Nbits']}")
        plot_ber_curve(results, title="QPSK: BER teorica vs simulada")
        plt.show()


"""

- Graficar el perfil del canal OK
- Graficar la resp en frecuencia del canal y del ecualizador 
- Hacer las graficas de convolucion canal-filtro en varios puntos de ber 
- Validar el canal con la forma de onda de ber teorica 

"""