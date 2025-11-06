import numpy as np

def _slice_lag(b_src, b_hat, lag):
    # alinea b_src y b_hat con lag
    if lag >= 0:
        L = min(len(b_src) - lag, len(b_hat))
        if L <= 0: return np.empty(0, np.uint8), np.empty(0, np.uint8)
        return b_src[lag:lag+L], b_hat[:L]
    off = -lag
    L = min(len(b_src), len(b_hat) - off)
    if L <= 0: return np.empty(0, np.uint8), np.empty(0, np.uint8)
    return b_src[:L], b_hat[off:off+L]

def _xforms_bits(Ib, Qb):
    # 8 ambigüedades QPSK (rot/ conj) en bits
    Ib = np.asarray(Ib, dtype=np.uint8); Qb = np.asarray(Qb, dtype=np.uint8)
    out = []
    out.append(("rot0",        Ib,         Qb))
    out.append(("rot0+conj",   Ib,         Qb ^ 1))
    out.append(("rot90",       Qb ^ 1,     Ib))
    out.append(("rot90+conj",  Qb,         Ib))
    out.append(("rot180",      Ib ^ 1,     Qb ^ 1))
    out.append(("rot180+conj", Ib ^ 1,     Qb))
    out.append(("rot270",      Qb,         Ib ^ 1))
    out.append(("rot270+conj", Qb ^ 1,     Ib ^ 1))
    return out

def ber(bI_src, bQ_src, dI_sym, dQ_sym,
        *, lag, N=None, win=8, mN=0, skip=0, min_overlap=256):
    # bits fuente y decididos
    bI_src = np.asarray(bI_src, dtype=np.uint8)
    bQ_src = np.asarray(bQ_src, dtype=np.uint8)
    bI_hat0 = (np.asarray(dI_sym) < 0).astype(np.uint8)
    bQ_hat0 = (np.asarray(dQ_sym) < 0).astype(np.uint8)

    # lags candidatos: bases (lag +- m.N) y ventana fina (+-win)
    bases = {int(lag)}
    if (N is not None) and (N > 0) and (mN > 0):
        NN = int(N)
        for m in range(-int(mN), int(mN) + 1):
            bases.add(int(lag) + m * NN)
    cand_lags = []
    for b0 in sorted(bases):
        for d in range(-int(win), int(win) + 1):
            cand_lags.append(b0 + d)
    seen = set()
    cand_lags = [x for x in cand_lags if not (x in seen or seen.add(x))]

    # barrido: (xform, lag) -> errores; quedarse con el mínimo
    best = None
    for name, Ih, Qh in _xforms_bits(bI_hat0, bQ_hat0):
        for Lg in cand_lags:
            sI, hI = _slice_lag(bI_src, Ih, Lg)
            sQ, hQ = _slice_lag(bQ_src, Qh, Lg)
            Lc = min(len(sI), len(hI), len(sQ), len(hQ))
            if Lc < max(min_overlap, skip + 1):
                continue
            if skip > 0:
                sI = sI[skip:]; sQ = sQ[skip:]; hI = hI[skip:]; hQ = hQ[skip:]
                Lc = min(len(sI), len(hI), len(sQ), len(hQ))
                if Lc <= 0: continue
                sI = sI[:Lc]; sQ = sQ[:Lc]; hI = hI[:Lc]; hQ = hQ[:Lc]
            errI = int(np.count_nonzero(sI ^ hI))
            errQ = int(np.count_nonzero(sQ ^ hQ))
            tot = errI + errQ
            if (best is None) or (tot < best["tot"]):
                best = {"tot": tot, "errI": errI, "errQ": errQ, "Lc": Lc,
                        "lag_used": int(Lg), "xform_used": name}

    # salida
    if (best is None) or (best["Lc"] <= 0):
        return {"BER": np.nan, "BER_I": np.nan, "BER_Q": np.nan, "Nbits": 0,
                "lag_used": None, "xform_used": None}

    bits = 2 * best["Lc"]
    return {
        "BER":   best["tot"] / bits,
        "BER_I": best["errI"] / best["Lc"],
        "BER_Q": best["errQ"] / best["Lc"],
        "Nbits": bits,
        "lag_used": best["lag_used"],
        "xform_used": best["xform_used"],
    }
