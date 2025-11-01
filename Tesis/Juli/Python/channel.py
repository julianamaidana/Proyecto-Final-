import numpy as np

def channel(
    x,
    mode="fir",
    h=None,
    SNR_dB=None,
    seed=5678,
    return_det=False,
    *,
    norm_h_power=False,   #  normaliza ||h||^2 = 1 si True
    snr_ref="post"        #  "post"  o "pre"
):
    """
    Parámetros
    
    norm_h_power : bool
        Si True y mode="fir", normaliza h para que sum(|h|^2)=1 (potencia unitaria).
    snr_ref : {"post","pre"}
        - "post": SNR referida a la salida del canal (y_det).  (comportamiento clásico)
        - "pre" : SNR referida a la señal de TX (x). Internamente se compensa por ||h||^2.
    """
    rng = np.random.default_rng(seed)

    # --- Construcción del canal determinístico ---
    if mode == "ideal" or h is None:
        y_det = x.copy()
        Gh_eff = 1.0  # ganancia de potencia efectiva del "canal"
    elif mode == "fir":
        h = np.asarray(h, np.complex128)

        # Potencia del canal antes de normalizar (por claridad)
        Gh = np.sum(np.abs(h)**2) + 1e-15

        if norm_h_power:
            # Normalizar a potencia unitaria
            h = h / np.sqrt(Gh)
            Gh_eff = 1.0
        else:
            Gh_eff = Gh

        y_full = np.convolve(x, h, mode="full")
        Lh, N = len(h), len(x)
        y_det = y_full[Lh - 1 : Lh - 1 + N]
    else:
        raise ValueError("mode debe ser 'ideal' o 'fir'")

    # --- Ruido AWGN ---
    if SNR_dB is None:
        y = y_det
    else:
        EsN0 = 10.0 ** (SNR_dB / 10.0)

        if snr_ref == "post":
            # SNR referida a la potencia post-canal (salida determinística)
            Es_ref = np.mean(np.abs(y_det)**2) + 1e-12
        elif snr_ref == "pre":
            # SNR referida a la potencia de TX; se compensa por ||h||^2 efectiva
            Es_tx  = np.mean(np.abs(x)**2) + 1e-12
            Es_ref = Es_tx * Gh_eff
        else:
            raise ValueError("snr_ref debe ser 'post' o 'pre'")

        N0     = Es_ref / EsN0
        sigma2 = N0 / 2.0
        noise  = np.sqrt(sigma2) * (
            rng.standard_normal(y_det.shape) + 1j * rng.standard_normal(y_det.shape)
        )
        y = y_det + noise

    return (y, y_det) if return_det else y
