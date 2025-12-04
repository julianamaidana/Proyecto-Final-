# simulator.py
# Simulador con FFE por bloques y barridos de BER

import numpy as np
import matplotlib.pyplot as plt
from math import erfc as _erfc_scalar

from fixedpoint import q, FX_NARROW
from prbs_qpsk import prbs_qpsk_iq
from channel import channel
from mu_switch import make_mu_switch
from pbfdaf_lms import FFE_Engine
from metrics import ber as _ber
from plots import (
    plot_weights_labeled, plot_filter_profile, plot_channel_profile, plot_combined_response,
    plot_time_iq_complex, plot_constellation, plot_freq_responses_db, sym_ylim
)

# ============================ Utilidades teóricas ============================

def _erfc_vec(x):
    x = np.asarray(x, dtype=float)
    return np.vectorize(_erfc_scalar)(x)

def _Q(x):
    # Q(x) = 0.5·erfc(x / sqrt(2))
    return 0.5 * _erfc_vec(x / np.sqrt(2.0))

def _ber_qpsk_awgn_from_EsN0_dB(esn0_dB):
    """BER teórico QPSK en AWGN a partir de Es/N0 [dB] (mapeo Gray)."""
    EbN0_dB  = np.asarray(esn0_dB) - 10*np.log10(2.0)  # QPSK: Es=2·Eb
    EbN0_lin = 10**(EbN0_dB/10.0)
    return float(_Q(np.sqrt(2.0 * EbN0_lin)))


# ============================ Utilidades de canal ============================

def suggest_center_tap(h, L_eq, pre_ratio=0.15):
    k_peak = int(np.argmax(np.abs(h)))
    off = max(1, int(pre_ratio * L_eq))
    return min(k_peak + off, L_eq - 1)

def rcosine(beta, sps, num_symbols_half):
    """
    Raised-Cosine sobremuestreado. Normaliza energía total a 1.
    """
    Tbaud = 1.0
    Nbauds = int(num_symbols_half) * 2
    t = np.arange(-0.5*Nbauds*Tbaud, 0.5*Nbauds*Tbaud, Tbaud/float(sps))
    eps = 1e-12
    t_safe = np.where(np.abs(t) < eps, eps, t)
    den = 1.0 - (2.0*beta*t_safe/Tbaud)**2
    den = np.where(np.abs(den) < eps, eps, den)
    y = np.sinc(t_safe/Tbaud) * (np.cos(np.pi*beta*t_safe/Tbaud) / den)
    h = y.astype(np.complex128)
    h /= np.sqrt(np.sum(np.abs(h)**2) + eps)
    return h

def rc_aggressive_channel(beta=0.20, span_sym=13, sps=64, frac=0.18,
                          echo_a=0.35, echo_d=1, echo_phi=np.pi*0.6,
                          normalize=True):
    """
    Taps T-spaced agresivos: RC sobremuestreado + desfase fraccional + eco complejo.
    """
    assert span_sym % 2 == 1, "span_sym debe ser impar (tap central)."
    h_os = rcosine(beta=beta, sps=sps, num_symbols_half=span_sym//2 + 1)

    c    = len(h_os)//2 + int(round(frac * sps))
    half = span_sym//2
    idx  = c + np.arange(-half, half+1) * sps
    idx  = np.clip(idx, 0, len(h_os)-1)

    h0 = h_os[idx.astype(int)]
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
                 NORM_H_POWER=False, SNR_REF="post"):

        # sim
        self.N_SYM  = int(N_SYM)
        self.N_PLOT = int(N_PLOT)
        self.N_SKIP = int(N_SKIP)

        # canal
        self.CHAN_MODE  = str(CHAN_MODE)
        self.SNR_DB     = None if SNR_DB is None else float(SNR_DB)
        self.SEED_NOISE = int(SEED_NOISE)

        if H_TAPS is None:
            self.H_TAPS = rc_aggressive_channel(
                beta=0.20, span_sym=13, sps=64, frac=0.18,
                echo_a=0.35, echo_d=1, echo_phi=np.pi*0.6, normalize=True
            )
        else:
            self.H_TAPS = np.asarray(H_TAPS, np.complex128)

        self.NORM_H_POWER = bool(NORM_H_POWER)
        self.SNR_REF      = str(SNR_REF)

        # equalizador
        self.L_EQ       = int(L_EQ)
        self.PART_N     = int(PART_N)
        self.CENTER_TAP = (suggest_center_tap(self.H_TAPS, self.L_EQ)
                           if CENTER_TAP is None else int(CENTER_TAP))

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

        # --- Motor FFE por bloques ---
        self.mu_step_func = self._build_mu_step()
        try:
            self.ffe_motor = FFE_Engine(
                L_eq=self.L_EQ, part_N=self.PART_N,
                mu_init=self.MU, center_index=self.CENTER_TAP,
                mu_step=self.mu_step_func
            )
        except TypeError:
            self.ffe_motor = FFE_Engine(
                L_eq=self.L_EQ, part_N=self.PART_N,
                mu=self.MU, center_index=self.CENTER_TAP,
                mu_step=self.mu_step_func
            )

        # buffers de salida/historial
        self.num_blocks = (self.N_SYM - self.PART_N) // self.PART_N + 1 if self.N_SYM >= self.PART_N else 0
        if self.num_blocks <= 0:
            raise ValueError("N_SYM demasiado chico para el PART_N elegido.")

        self.yI = np.zeros(self.num_blocks * self.PART_N, dtype=np.float64)
        self.yQ = np.zeros(self.num_blocks * self.PART_N, dtype=np.float64)
        self.yhatI = np.zeros(self.num_blocks * self.PART_N, dtype=np.float64)
        self.yhatQ = np.zeros(self.num_blocks * self.PART_N, dtype=np.float64)
        self.eI = np.zeros(self.num_blocks * self.PART_N, dtype=np.float64)
        self.eQ = np.zeros(self.num_blocks * self.PART_N, dtype=np.float64)
        self.W_histI = np.zeros((self.num_blocks, self.L_EQ), dtype=np.float64)
        self.W_histQ = np.zeros((self.num_blocks, self.L_EQ), dtype=np.float64)

        self.sI = self.sQ = None
        self.x_n = self.x_det = None
        self.bI_src = self.bQ_src = None
        self.k0 = getattr(self.ffe_motor, "center_index",
                   getattr(self.ffe_motor, "center", self.CENTER_TAP))

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
        self.sI, self.sQ = prbs_qpsk_iq(self.N_SYM, self.seedI, self.seedQ)
        self.bI_src = (self.sI < 0).astype(np.uint8)
        self.bQ_src = (self.sQ < 0).astype(np.uint8)
        return self.sI + 1j*self.sQ

    def pass_channel(self, s):
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
        """Procesa x_n bloque a bloque con el motor FFE."""
        xI_full = self.x_n.real.copy()
        xQ_full = self.x_n.imag.copy()
        out_idx = 0

        for b in range(self.num_blocks):
            start, stop = b * self.PART_N, b * self.PART_N + self.PART_N
            x_newI = xI_full[start:stop]
            x_newQ = xQ_full[start:stop]

            (y_blkI, y_blkQ,
             d_blkI, d_blkQ,
             e_blkI, e_blkQ) = self.ffe_motor.process_block(x_newI, x_newQ)

            self.yI[out_idx:out_idx+self.PART_N]    = y_blkI
            self.yQ[out_idx:out_idx+self.PART_N]    = y_blkQ
            self.yhatI[out_idx:out_idx+self.PART_N] = d_blkI
            self.yhatQ[out_idx:out_idx+self.PART_N] = d_blkQ
            self.eI[out_idx:out_idx+self.PART_N]    = e_blkI
            self.eQ[out_idx:out_idx+self.PART_N]    = e_blkQ

            wI, wQ, _ = self.ffe_motor.get_final_taps()
            self.W_histI[b, :] = wI
            self.W_histQ[b, :] = wQ

            out_idx += self.PART_N

        self.w_finI, self.w_finQ, _ = self.ffe_motor.get_final_taps()
        return self.k0

    def plot(self,
             weights=True, profile=True, conv=True,
             time_in=True, time_out=True,
             const_in=True, const_out=True, const_dec=True,
             chan_profile=True, freq=True,
             weights_smoothing_window=None):
        W_hist = self.W_histI + 1j*self.W_histQ
        w_fin  = self.w_finI  + 1j*self.w_finQ
        y_n    = self.yI + 1j*self.yQ
        yhat_n = self.yhatI + 1j*self.yhatQ

        ylims = sym_ylim(np.real(w_fin), np.imag(w_fin),
                         np.real(self.H_TAPS), np.imag(self.H_TAPS))

        if weights:
            plot_weights_labeled(W_hist, center_index=self.CENTER_TAP,
                                 smoothing_window=weights_smoothing_window)
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
        lag_eff = 0 if (lag is None and self.CHAN_MODE == "ideal") else (self.k0 if lag is None else int(lag))
        if win is None: win = 4 * self.PART_N
        if mN  is None: mN  = 8
        if min_overlap is None: min_overlap = max(4 * self.PART_N, 256)
        return _ber(self.bI_src, self.bQ_src, self.yhatI, self.yhatQ,
                    lag=lag_eff, N=int(self.PART_N), win=int(win),
                    mN=int(mN), skip=int(skip), min_overlap=int(min_overlap))


# ================== Barrido BER acelerado (dos tiros) ==================

def ber_at_snr_until_errors(snr_db, *, E_TARGET=100, N_SYM_MAX=1_000_000, skip=2000,
                            eq_params=None, safety=1.25):
    """
    1) Estima N_SYM para ~E_TARGET errores (QPSK-AWGN).
    2) Corre una simulación con N_SYM_try. Si alcanza E_TARGET, termina.
    3) Si no, corre una segunda simulación a N_SYM_MAX.
    """
    if eq_params is None:
        eq_params = {}

    p_theory   = max(_ber_qpsk_awgn_from_EsN0_dB(snr_db), 1e-12)
    bits_need  = E_TARGET / p_theory
    sym_need   = int(np.ceil(bits_need / 2.0))
    N_SYM_try  = int(min(N_SYM_MAX, max(2*skip, int(safety * sym_need))))

    def _one_shot(N_SYM):
        sim = equalizerSimulator(
            N_SYM=N_SYM, N_PLOT=0, N_SKIP=0,
            CHAN_MODE=eq_params.get("CHAN_MODE", "fir"),
            H_TAPS=eq_params.get("H_TAPS", None),
            SNR_DB=snr_db, SEED_NOISE=eq_params.get("SEED_NOISE", 5678),
            L_EQ=eq_params.get("L_EQ", 31),
            PART_N=eq_params.get("PART_N", 16),
            CENTER_TAP=eq_params.get("CENTER_TAP", None),
            MU=eq_params.get("MU", 0.01),
            MU_SWITCH_ENABLE=eq_params.get("MU_SWITCH_ENABLE", True),
            MU_FINAL=eq_params.get("MU_FINAL", 0.0005),
            N_SWITCH=eq_params.get("N_SWITCH", 800),
            USE_STABLE=eq_params.get("USE_STABLE", False),
            STABLE_WIN=eq_params.get("STABLE_WIN", 300),
            STABLE_TOL=eq_params.get("STABLE_TOL", 1.0),
            STABLE_PATIENCE=eq_params.get("STABLE_PATIENCE", 1),
            seedI=eq_params.get("seedI", 0x17F),
            seedQ=eq_params.get("seedQ", 0x11D),
            NORM_H_POWER=eq_params.get("NORM_H_POWER", False),
            SNR_REF=eq_params.get("SNR_REF", "post")
        )
        sim.run()
        return sim.ber(skip=skip)

    print(f"  [SNR={snr_db} dB] 1er tiro: {N_SYM_try} símbolos (~{E_TARGET} errores)")
    res = _one_shot(N_SYM_try)
    if (res.get("E_total", 0) >= E_TARGET) or (N_SYM_try >= N_SYM_MAX):
        print(f"  [SNR={snr_db} dB] DONE -> BER={res['BER']:.3e} (Nbits={res['Nbits']}, E={res.get('E_total', -1)})")
        return {"SNR": snr_db, "BER": res["BER"], "BER_I": res["BER_I"], "BER_Q": res["BER_Q"],
                "Nbits": res["Nbits"], "E_total": res.get("E_total", -1)}

    print(f"  [SNR={snr_db} dB] 2do tiro: {N_SYM_MAX} símbolos (faltaron errores)")
    res = _one_shot(N_SYM_MAX)
    print(f"  [SNR={snr_db} dB] DONE -> BER={res['BER']:.3e} (Nbits={res['Nbits']}, E={res.get('E_total', -1)})")
    return {"SNR": snr_db, "BER": res["BER"], "BER_I": res["BER_I"], "BER_Q": res["BER_Q"],
            "Nbits": res["Nbits"], "E_total": res.get("E_total", -1)}


def sweep_snr_until_errors(snr_list, *, E_TARGET=100, N_SYM_MAX=1_000_000, skip=2000,
                           eq_params=None, safety=1.25):
    results = []
    for i, snr in enumerate(snr_list, 1):
        print(f"\n=== {i}/{len(snr_list)} : SNR={snr} dB ===")
        results.append(ber_at_snr_until_errors(
            snr, E_TARGET=E_TARGET, N_SYM_MAX=N_SYM_MAX, skip=skip,
            eq_params=eq_params, safety=safety
        ))
    return results


# ============================ Gráfico BER vs SNR ============================

def plot_ber_curve(results, title="BER vs SNR", floor=1e-12, x_max=None):
    snr_esn0 = np.array([r["SNR"] for r in results], dtype=float)
    ber_raw  = np.array([r["BER"] for r in results], dtype=float)

    ber_sim = np.where(np.isfinite(ber_raw), np.maximum(ber_raw, floor), np.nan)
    m = ~np.isnan(ber_sim) & np.isfinite(snr_esn0)
    snr_esn0, ber_sim = snr_esn0[m], ber_sim[m]

    if not snr_esn0.size:
        print("plot_ber_curve: No hay datos válidos para graficar.")
        return

    xmin = float(np.nanmin(snr_esn0))
    xmax_data = float(np.nanmax(snr_esn0))
    xmax = xmax_data if x_max is None else float(x_max)

    x_dense_esn0 = np.linspace(xmin, xmax, 600)
    ber_th = [_ber_qpsk_awgn_from_EsN0_dB(snr) for snr in x_dense_esn0]

    plt.figure(figsize=(6.5, 4.2))
    plt.semilogy(x_dense_esn0, ber_th, '-', linewidth=2, label='QPSK teoría (Es/N0)')
    plt.semilogy(snr_esn0, ber_sim, 'o-', linewidth=1, label='Simulación')
    plt.grid(True, which="both", linestyle=":")
    plt.xlabel("Es/N0 [dB]"); plt.ylabel("BER")
    plt.xlim(xmin, xmax)
    plt.ylim(max(floor, 1e-7), 1)
    try:
        import math
        xticks = np.arange(math.floor(xmin/2)*2, math.ceil(xmax/2)*2 + 0.1, 2)
        plt.xticks(xticks)
    except Exception:
        pass
    plt.title(title); plt.legend(); plt.tight_layout()


# ======================= Barrido rápido de ganancia =======================

def quick_agc_scan(snrs, *, target=0.7, N_SYM=2000, mode="rms",
                   clip=(0.25, 4.0), eq_params=None):
    """
    Estima multiplicador escalar que deja la entrada del ecualizador en 'target'.
    """
    if eq_params is None: eq_params = {}
    out = []
    for snr in snrs:
        sim = equalizerSimulator(
            N_SYM=N_SYM, N_PLOT=0, N_SKIP=0,
            CHAN_MODE=eq_params.get("CHAN_MODE", "fir"),
            H_TAPS=eq_params.get("H_TAPS", None),
            SNR_DB=snr, SEED_NOISE=eq_params.get("SEED_NOISE", 5678),
            L_EQ=eq_params.get("L_EQ", 31),
            PART_N=eq_params.get("PART_N", 16),
            CENTER_TAP=eq_params.get("CENTER_TAP", None),
            MU=eq_params.get("MU", 0.006),
            MU_SWITCH_ENABLE=False, MU_FINAL=0.0, N_SWITCH=0,
            USE_STABLE=False, STABLE_WIN=0, STABLE_TOL=0.0, STABLE_PATIENCE=0,
            seedI=eq_params.get("seedI", 0x17F),
            seedQ=eq_params.get("seedQ", 0x11D),
            NORM_H_POWER=eq_params.get("NORM_H_POWER", False),
            SNR_REF=eq_params.get("SNR_REF", "post")
        )
        sim.AGC_ENABLE = False            # medir nivel crudo (sin AGC)
        s = sim.gen_source()
        x = sim.pass_channel(s)
        mags = np.abs(x)
        meas = float(np.sqrt(np.mean(mags*mags))) if mode == "rms" else float(np.mean(mags))
        g_est = float(np.clip(target / max(meas, 1e-12), clip[0], clip[1]))
        out.append({"SNR": snr, "gain": g_est, "meas": meas})
        print(f"SNR={snr:>2} dB -> meas={meas:.4f}, gain≈{g_est:.3f}")
    return out


# =================================== Main ===================================

if __name__ == "__main__":

    RUN_SINGLE_SIM      = True
    RUN_ADAPTIVE_SWEEP  = True
    RUN_BER_VS_MU_SWEEP = False

    if RUN_SINGLE_SIM:
        print("\n" + "="*30)
        print("=== SIMULACIÓN ÚNICA (DEBUG) ===")
        print("="*30)
        sim = equalizerSimulator(
            N_SYM=20000, 
            N_PLOT=10000, 
            N_SKIP=0,
            CHAN_MODE="fir", 
            H_TAPS=None,
            SNR_DB=20, 
            SEED_NOISE=5678,
            L_EQ=31, 
            PART_N=16, 
            CENTER_TAP=None,
            MU=0.006, 
            MU_SWITCH_ENABLE=True, 
            MU_FINAL=0.0004, 
            N_SWITCH=500,
            USE_STABLE=False, 
            STABLE_WIN=300, 
            STABLE_TOL=1.0, 
            STABLE_PATIENCE=80,
            seedI=0x17F, 
            seedQ=0x11D, 
            NORM_H_POWER=False, 
            SNR_REF="post"
        ).run()
        
        sim.plot(
            weights=True, weights_smoothing_window=None,
            profile=False, chan_profile=True, freq=True, conv=True,
            time_in=False, const_in=True, time_out=False, const_out=True, const_dec=True
        )
        print("CENTER_TAP =", sim.CENTER_TAP, "k0 =", sim.k0)
        res = sim.ber(skip=5000, win=4*sim.PART_N, mN=8)
        print("BER:", res)

    if RUN_ADAPTIVE_SWEEP:
        SNRS = [0, 2, 4, 6, 8, 10, 12, 14, 16]
        EQ_PARAMS = dict(
            CHAN_MODE="fir", 
            H_TAPS=None, 
            L_EQ=31, 
            PART_N=16, 
            CENTER_TAP=None,
            MU=0.009, 
            MU_SWITCH_ENABLE=True, 
            MU_FINAL=0.0003, 
            N_SWITCH=500,
            USE_STABLE=False, 
            STABLE_WIN=300, 
            STABLE_TOL=1.0, 
            STABLE_PATIENCE=80,
            NORM_H_POWER=False, 
            SNR_REF="post", 
            seedI=0x17F, 
            seedQ=0x11D
        )
        results = sweep_snr_until_errors(
            SNRS, 
            E_TARGET=100, 
            N_SYM_MAX=1_000_000, 
            skip=20000,
            eq_params=EQ_PARAMS, 
            safety=1.5
        )
        print("\n--- Resultados del Barrido Adaptativo ---")
        for r in results:
            print(f"SNR(Es/N0)={r['SNR']:>2} dB | BER={r['BER']:.3e} | Nbits={r['Nbits']:<10} | Errores={r['E_total']}")
        plot_ber_curve(results, title="QPSK: BER teórica vs simulada (Adaptativo)", x_max=20)

    # (RUN_BER_VS_MU_SWEEP omitido para brevedad; usar tu versión si lo necesitás)

    plt.show()