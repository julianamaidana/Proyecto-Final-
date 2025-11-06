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


# ============================ Utilidades teóricas ============================

def _erfc_vec(x):
    x = np.asarray(x, dtype=float)
    return np.vectorize(_erfc_scalar)(x)

def _Q(x):
    # Q(x) = 0.5 * erfc(x / sqrt(2))
    return 0.5 * _erfc_vec(x / np.sqrt(2.0))

def ber_qpsk_theory_esn0(EsN0_dB):
    # convertir Es/N0(dB) a Eb/N0(dB) (QPSK: 2 bits/símbolo)
    EbN0_dB  = np.asarray(EsN0_dB) - 10*np.log10(2.0)  # -3.0103 dB
    EbN0_lin = 10**(EbN0_dB/10.0)
    return _Q(np.sqrt(2.0 * EbN0_lin))  # BER QPSK coherente + Gray

def suggest_center_tap(h, L_eq, pre_ratio=0.15):
<<<<<<< HEAD
    k_peak = int(np.argmax(np.abs(h)))
    off = max(1, int(pre_ratio * L_eq))
    return min(k_peak + off, L_eq - 1)


# ============================ Pulso RC (original) ============================
=======
        k_peak = int(np.argmax(np.abs(h)))
        off = max(1, int(pre_ratio * L_eq))
        return min(k_peak + off, L_eq - 1)
        
def taps_rc_narrow(beta=0.15, span_sym=13, frac=0.12, sps=64, cascade=2):
    """Canal T-spaced derivado de RC muy angosto + opcionalmente en cascada."""
    # RC sobremuestreado
    T = 1.0
    Nhalf = span_sym//2 + 1
    t = np.arange(-Nhalf*T, Nhalf*T, T/sps)
    t[t == 0] = 1e-8
    denom0 = 1.0 - (2.0*beta*t/T)**2
    denom0[np.abs(denom0) < 1e-8] += 1e-8
    h_os = np.sinc(t/T) * (np.cos(np.pi*beta*t/T) / denom0)

    # centrar y aplicar retardo fraccional
    c = len(h_os)//2 + int(round(frac*sps))
    idx = c + np.arange(-span_sym//2, span_sym//2+1)*sps
    h = h_os[idx.astype(int)].astype(complex)

    # cascada para “cerrar” más la banda
    for _ in range(cascade-1):
        h = np.convolve(h, h)

    # normalizar energía
    h = h / np.sqrt((np.abs(h)**2).sum() + 1e-15)
    return h

>>>>>>> main

def rcosine(beta, sps, num_symbols_half):
    """
    Raised Cosine (pulso de Nyquist) sobremuestreado.
    beta in [0,1], sps: oversampling, num_symbols_half: cola a cada lado (en símbolos).
    Normaliza energía total a 1.
    """
    Tbaud = 1.0
    Nbauds = int(num_symbols_half) * 2
    t = np.arange(-0.5*Nbauds*Tbaud, 0.5*Nbauds*Tbaud, Tbaud/float(sps))
    eps = 1e-12

    # Evitar divisiones por cero en t=0 y en los ceros del denominador
    t_safe = np.where(np.abs(t) < eps, eps, t)
    den = 1.0 - (2.0*beta*t_safe/Tbaud)**2
    den = np.where(np.abs(den) < eps, eps, den)

    y = np.sinc(t_safe/Tbaud) * (np.cos(np.pi*beta*t_safe/Tbaud) / den)
    h = y.astype(np.complex128)
    h /= np.sqrt(np.sum(np.abs(h)**2) + eps)
    return h


# ======= Canal agresivo: RC + desfase fraccional + eco complejo (T-spaced) =======

def rc_aggressive_channel(beta=0.40, span_sym=13, sps=64, frac=0.18,
                          echo_a=0.35, echo_d=1, echo_phi=np.pi*0.6,
                          normalize=True):
    """
    Construye taps T-spaced "agresivos":
    - RC sobremuestreado con roll-off beta
    - muestreo a tasa de símbolo con desfase fraccional 'frac' (rompe Nyquist)
    - suma un eco complejo (amplitud 'echo_a', retardo 'echo_d', fase 'echo_phi')
    - normaliza energía total
    """
    assert span_sym % 2 == 1, "span_sym debe ser impar (tap central)."
    # RC muy fino
    h_os = rcosine(beta=beta, sps=sps, num_symbols_half=span_sym//2 + 1)

    # Centro + desfase fraccional
    c    = len(h_os)//2 + int(round(frac * sps))
    half = span_sym//2
    idx  = c + np.arange(-half, half+1) * sps
    idx  = np.clip(idx, 0, len(h_os)-1)

    # T-spaced con timing fraccional
    h0 = h_os[idx.astype(int)]

    # Eco complejo desplazado d símbolos
    he = echo_a * np.exp(1j*echo_phi) * np.roll(h0, +int(echo_d))

    h = h0 + he
    if normalize:
        h /= np.sqrt(np.sum(np.abs(h)**2) + 1e-15)
    return h.astype(np.complex128)


# =============================== Simulador FFE ===============================

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
        self.N_SYM        = int(N_SYM)
        self.N_PLOT       = int(N_PLOT)
        self.N_SKIP       = int(N_SKIP)

        # canal
        self.CHAN_MODE    = str(CHAN_MODE)
        self.SNR_DB       = None if SNR_DB is None else float(SNR_DB)  # Es/N0(dB)
        self.SEED_NOISE   = int(SEED_NOISE)

        # Si no pasan H_TAPS, construyo un canal "agresivo"
        if H_TAPS is None:
<<<<<<< HEAD
            self.H_TAPS = rc_aggressive_channel(
                beta=0.20, span_sym=13, sps=64, frac=0.18,
                echo_a=0.35, echo_d=1, echo_phi=np.pi*0.6, normalize=True
            )
=======
            # Parámetros del canal RC "angosto"
            sps       = 64        # sobremuestreo para construir el RC
            beta      = 0.15      # achicar banda => más ISI
            span_sym  = 13        # longitud impar del canal T-spaced
            frac      = 0.12      # retardo fraccional opcional (0..1)

            print(f"Generando canal RC: sps={sps}, beta={beta}, span={span_sym}, frac={frac}")

            h_os = rcosine(beta=beta, sps=sps, num_symbols_half=span_sym//2 + 1)

            # centro + fracción
            c    = len(h_os)//2 + int(round(frac * sps))
            half = span_sym//2
            idx  = c + np.arange(-half, half+1) * sps
            h_diezmado = h_os[idx.astype(int)]

            # normalización de energía del canal T-spaced
            h_diezmado = h_diezmado / np.sqrt(np.sum(np.abs(h_diezmado)**2) + 1e-15)

            self.H_TAPS = h_diezmado.astype(np.complex128)
>>>>>>> main
        else:
            self.H_TAPS = np.asarray(H_TAPS, np.complex128)

        self.NORM_H_POWER = bool(NORM_H_POWER)
        self.SNR_REF      = str(SNR_REF)

        # eq
<<<<<<< HEAD
        self.L_EQ         = int(L_EQ)
        self.PART_N       = int(PART_N)
        self.CENTER_TAP   = (suggest_center_tap(self.H_TAPS, self.L_EQ)
                             if CENTER_TAP is None else int(CENTER_TAP))

=======
        self.L_EQ         = L_EQ
        self.PART_N       = PART_N
        #self.CENTER_TAP = (suggest_center_tap(self.H_TAPS, L_EQ) 
        #                   if CENTER_TAP is None else CENTER_TAP)
        self.CENTER_TAP = CENTER_TAP if CENTER_TAP is not None else L_EQ // 2
>>>>>>> main
        # mu / switch
        self.MU               = float(MU)
        self.MU_SWITCH_ENABLE = bool(MU_SWITCH_ENABLE)
        self.MU_FINAL         = float(MU_FINAL)
        self.N_SWITCH         = int(N_SWITCH)
        self.USE_STABLE       = bool(USE_STABLE)
        self.STABLE_WIN       = int(STABLE_WIN)
        self.STABLE_TOL       = float(STABLE_TOL)
        self.STABLE_PATIENCE  = int(STABLE_PATIENCE)

        # PRBS
        self.seedI = int(seedI)
        self.seedQ = int(seedQ)

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
        # FFE PBFDAF (LMS en frecuencia)
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
        # Gráficos
        W_hist = self.W_histI + 1j*self.W_histQ
        w_fin  = self.w_finI  + 1j*self.w_finQ
        y_n    = self.yI + 1j*self.yQ
        yhat_n = self.yhatI + 1j*self.yhatQ

        # misma escala Re/Im filtro y canal
        ylims = sym_ylim(np.real(w_fin), np.imag(w_fin),
                         np.real(self.H_TAPS), np.imag(self.H_TAPS))

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
            plot_constellation(x_tail, title="Constelación x_n")
        if const_out:
            tail = y_n[-min(self.N_PLOT, len(y_n)):]
            plot_constellation(tail, title="Constelación y_n")
        if const_dec:
            tail_hat = yhat_n[-min(self.N_PLOT, len(yhat_n)):]
            plot_constellation(tail_hat, title="Constelación y_hat")
        if freq:
            plot_freq_responses_db(w_fin, np.asarray(self.H_TAPS, np.complex128), pad_n=8192)

    def run(self):
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


# ============================== Runners auxiliares ==============================

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
            H_TAPS            =  eq_params.get("H_TAPS", None),
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
    snr_esn0 = np.array([r["SNR"] for r in results], dtype=float)  # Es/N0(dB)
    ber_sim  = np.array([max(r["BER"], 1e-12) for r in results], dtype=float)

    xmin = float(np.nanmin(snr_esn0)) if snr_esn0.size else 0.0
    xmax = float(np.nanmax(snr_esn0)) if snr_esn0.size else 12.0
    x_dense_esn0 = np.linspace(xmin, xmax, 500)
    ber_th = ber_qpsk_theory_esn0(x_dense_esn0)

    plt.figure(figsize=(6.5, 4.2))
    plt.semilogy(x_dense_esn0, ber_th, '-', linewidth=2, label='QPSK teoría (BER vs Es/N0)')
    plt.semilogy(snr_esn0, ber_sim, 'o-', linewidth=1, label='QPSK simulada')
    plt.grid(True, which="both", linestyle=":")
    plt.xlabel("Es/N0 [dB]")
    plt.ylabel("BER")
    plt.xlim(xmin, xmax)
    plt.ylim(1e-7, 1)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

def measure_ber_vs_mu(SNR_DB=8.0, mus=None, repeats=3,
                      NSYM=10000, skip=2000, min_overlap=256,
                      eq_mu_params=None):
    if mus is None:
        mus = np.logspace(-4, -1, 16)
    if eq_mu_params is None:
        eq_mu_params = {}

    ber_means, ber_stds = [], []

    for mu in mus:
        vals = []
        for r in range(repeats):
            sim = equalizerSimulator(
                N_SYM=NSYM, N_PLOT=0, N_SKIP=0,
                CHAN_MODE=eq_mu_params.get("CHAN_MODE", "fir"),
                H_TAPS=eq_mu_params.get("H_TAPS", None),
                SNR_DB=SNR_DB, SEED_NOISE=5678 + r,
                L_EQ=eq_mu_params.get("L_EQ", 31),
                PART_N=eq_mu_params.get("PART_N", 16),
                CENTER_TAP=eq_mu_params.get("CENTER_TAP", None),
                MU=mu,
                MU_SWITCH_ENABLE=eq_mu_params.get("MU_SWITCH_ENABLE", False),
                MU_FINAL=eq_mu_params.get("MU_FINAL", 0.0005),
                N_SWITCH=eq_mu_params.get("N_SWITCH", 500),
                USE_STABLE=eq_mu_params.get("USE_STABLE", False),
                STABLE_WIN=eq_mu_params.get("STABLE_WIN", 300),
                STABLE_TOL=eq_mu_params.get("STABLE_TOL", 1.0),
                STABLE_PATIENCE=eq_mu_params.get("STABLE_PATIENCE", 80),
                seedI=eq_mu_params.get("seedI", 0x17F),
                seedQ=eq_mu_params.get("seedQ", 0x11D),
                NORM_H_POWER=eq_mu_params.get("NORM_H_POWER", False),
                SNR_REF=eq_mu_params.get("SNR_REF", "post"),
            )
            sim.run()

            res = sim.ber(skip=skip, win=4*sim.PART_N, mN=8, min_overlap=min_overlap)
            vals.append(res["BER"])

        ber_means.append(np.mean(vals))
        ber_stds.append(np.std(vals, ddof=1) if repeats > 1 else 0.0)

    return np.array(mus), np.array(ber_means), np.array(ber_stds)


# =================================== Main ===================================

if __name__ == "__main__":
    sim = equalizerSimulator(
        N_SYM            =  10000,
        N_PLOT           =  10000,
        N_SKIP           =  0,
        CHAN_MODE        =  "fir",
<<<<<<< HEAD
        H_TAPS           =  None,   # usa el canal agresivo por defecto
        SNR_DB           =  15,
=======
        H_TAPS           =  taps_rc_narrow(beta=0.15, span_sym=13, frac=0.12, cascade=2),  # <-- ¡MODIFICADO!
        SNR_DB           =  20,    
>>>>>>> main
        SEED_NOISE       =  5678,
        L_EQ             =  31,
        PART_N           =  16,
        CENTER_TAP       =  None,
<<<<<<< HEAD
        MU               =  0.009, #si lo subis se rompe 
        MU_SWITCH_ENABLE =  True, 
        MU_FINAL         =  0.0003,
=======
        MU               =  0.004,
        MU_SWITCH_ENABLE =  False,
        MU_FINAL         =  0.0004,
>>>>>>> main
        N_SWITCH         =  500,
        USE_STABLE       =  False,
        STABLE_WIN       =  300,
        STABLE_TOL       =  1.0,
        STABLE_PATIENCE  =  80,
        seedI            =  0x17F,
        seedQ            =  0x11D,
        NORM_H_POWER     =  False,
        SNR_REF          =  "post"
    ).run()

    # Gráficos principales
    sim.plot(
        weights      =  True,
        profile      =  False,
        chan_profile =  True,
        freq         =  True,
        conv         =  True,
        time_in      =  False,
        const_in     =  True,
        time_out     =  False,
        const_out    =  True,
        const_dec    =  True
    )

    print("CENTER_TAP =", sim.CENTER_TAP, "k0 =", sim.k0)
    res = sim.ber(skip=5000, win=4*sim.PART_N, mN=8)
    print("BER:", res)

    # Si querés un sweep de SNR:
    RUN_BER_SWEEP = False
    if RUN_BER_SWEEP:
        SNRS = list(range(0, 13, 1))
        EQ_PARAMS = dict(
            CHAN_MODE        =  "fir",
            L_EQ             =  31,
            PART_N           =  16,
            CENTER_TAP       =  None,
            MU               =  0.008,
            MU_SWITCH_ENABLE =  True,
            MU_FINAL         =  0.0006,
            N_SWITCH         =  1000,
            USE_STABLE       =  False,
            STABLE_WIN       =  300,
            STABLE_TOL       =  1.0,
            STABLE_PATIENCE  =  80,
            H_TAPS           =  None,
            NORM_H_POWER     =  False,
            SNR_REF          =  "post",
        )
        SWEEP_N_SYM     = 10000
        SWEEP_SKIP      = 5000
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
        plot_ber_curve(results, title="QPSK: BER teórica vs simulada")

    plt.show()
