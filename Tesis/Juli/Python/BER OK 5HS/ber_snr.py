# ber_snr.py
import numpy as np
import matplotlib.pyplot as plt
from math import erfc as _erfc_scalar
from simulator import equalizerSimulator  
from prbs_qpsk import prbs_qpsk_iq

# Teoria QPSK 
def _erfc_vec(x):
    x = np.asarray(x, dtype=float)
    return np.vectorize(_erfc_scalar)(x)

def _Q(x):
    # Q(x) = 0.5 * erfc(x / sqrt(2))
    return 0.5 * _erfc_vec(x / np.sqrt(2.0))

def ber_qpsk_theory_esn0(EsN0_dB):
    # QPSK: 2 bits/simbolo -> Eb/N0(dB) = Es/N0(dB) - 3.0103
    EbN0_dB  = np.asarray(EsN0_dB) - 10*np.log10(2.0)
    EbN0_lin = 10**(EbN0_dB/10.0)
    return _Q(np.sqrt(2.0 * EbN0_lin)) 

def _is_impulse(h, tol=1e-12):
    if h is None:
        return True
    h = np.asarray(h, np.complex128)
    return np.count_nonzero(np.abs(h) > tol) == 1

# Medicion simulada 
def ber_at_snr(snr_db, *, N_SYM=30000, skip=2000, runs=1, eq_params=None, reuse_src=None):
  
    if eq_params is None:
        eq_params = {}

    ber_sum = 0.0
    berI_sum = 0.0
    berQ_sum = 0.0
    nbits_sum = 0

    for r in range(runs):
        print(f"[SNR={snr_db} dB] run {r+1}/{runs}…", flush=True)
        sim = equalizerSimulator(
            # sim
            N_SYM             =  N_SYM,
            N_PLOT            =  0,
            N_SKIP            =  0,

            # channel (Es/N0)
            CHAN_MODE         =  eq_params.get("CHAN_MODE", "fir"),
            H_TAPS           =  [0,0,0,0,0,0,1,0,0,0,0,0,0],  # impulso
            SNR_DB            =  snr_db,  # Es/N0(dB)
            SEED_NOISE        =  eq_params.get("SEED_NOISE", 5678 + r),

            # eq
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

            # PRBS
            seedI             =  eq_params.get("seedI", 0x17F + r),
            seedQ             =  eq_params.get("seedQ", 0x11D + r),
        )

        # reutilizar fuente para todo el sweep 
        if reuse_src is not None:
            sI, sQ = reuse_src
            sim.set_source(sI, sQ)
            s = sim.sI + 1j*sim.sQ
        else:
            s = sim.gen_source()

        sim.pass_channel(s)
        sim.equalize()

        # BER medida
        force_lag = 0 if (sim.CHAN_MODE == "ideal" or _is_impulse(sim.H_TAPS)) else None

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
    if eq_params is None:
        eq_params = {}

    src = None
    if reuse_src:
        # usar misma PRBS en todo el sweep para menor varianza
        sI, sQ = prbs_qpsk_iq(N_SYM, 0x17F, 0x11D)
        src = (sI, sQ)

    results = []
    for i, snr in enumerate(snr_list, 1):
        print(f"\n=== {i}/{len(snr_list)} : SNR={snr} dB ===", flush=True)
        results.append(ber_at_snr(snr, N_SYM=N_SYM, skip=skip, runs=runs,
                                  eq_params=eq_params, reuse_src=src))
    return results

def plot_ber_curve(results, title="QPSK: BER teorica vs simulada", xmax=None, floor=1e-12):
    snr_esn0 = np.array([r["SNR"] for r in results], dtype=float)
    ber_raw  = np.array([r["BER"] for r in results], dtype=float)

    ber_sim = np.where(np.isfinite(ber_raw), np.maximum(ber_raw, floor), np.nan)
    m = ~np.isnan(ber_sim) & np.isfinite(snr_esn0)
    snr_esn0, ber_sim = snr_esn0[m], ber_sim[m]

    # X máx dinámico 
    if xmax is None:
        xmax = float(np.nanmax(snr_esn0))
        xmax = np.ceil(xmax)  # redondeo al entero superior

    # Teoría hasta xmax
    x_dense = np.linspace(0, xmax, 800)
    ber_th  = ber_qpsk_theory_esn0(x_dense)

    ymin = min(floor, np.min(ber_sim), np.min(ber_th))
    ymin = 10**np.floor(np.log10(ymin))  # a la década

    plt.figure(figsize=(6.5, 4.2))
    plt.semilogy(x_dense, ber_th, '-', lw=2, label='QPSK teoria (BER vs Es/N0)')
    plt.semilogy(snr_esn0, ber_sim, 'o-', lw=1, label='QPSK simulada')
    plt.grid(True, which="both", linestyle=":")
    plt.xlabel("Es/N0 [dB]"); plt.ylabel("BER")
    plt.xlim(0, xmax); plt.ylim(ymin, 1)
    plt.title(title); plt.legend(); plt.tight_layout()


# main
if __name__ == "__main__":
    # Barrido 0..12 dB (paso 1)
    SNRS = list(range(0, 13, 1))  # Es/N0(dB)

    EQ_PARAMS = dict(
        CHAN_MODE        = "fir",
        # H_TAPS por default
        L_EQ             = 31,
        PART_N           = 8,
        CENTER_TAP       = None,
        MU               = 0.015,
        MU_SWITCH_ENABLE = True,
        MU_FINAL         = 0.006,
        N_SWITCH         = 200,
        USE_STABLE       = False,
        STABLE_WIN       = 300,
        STABLE_TOL       = 1.0,
        STABLE_PATIENCE  = 80,
    )

    results = sweep_snr(
        SNRS,
        N_SYM     = 30000,
        skip      = 2000,
        runs      = 1,
        eq_params = EQ_PARAMS,
        reuse_src = True
    )

    for r in results:
        print(f"SNR(Es/N0)={r['SNR']:>2} dB | BER={r['BER']:.3e} | Nbits={r['Nbits']}")

    plot_ber_curve(results, title="QPSK: BER teorica vs simulada ")
    plt.show()
