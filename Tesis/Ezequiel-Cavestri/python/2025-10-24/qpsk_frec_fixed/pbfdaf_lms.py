import numpy as np
from fixedpoint import q, q_vec, q_add, q_sub, to_fx, FX_NARROW, FX_WIDE, cmul_iq
from fft_iq import twiddles_iq, fft_iq, ifft_iq
from prbs_qpsk import QPSK_A

def ffe_lms_freq_iq(xI, xQ, L_eq, part_N, mu, center_index=None, mu_step=None):
    M = int(L_eq); N = int(part_N)
    if N <= 0: raise ValueError("part_N debe ser > 0")
    if center_index is None: center_index = (M - 1) // 2

    Nfft = 2 * N
    K = int(np.ceil(M / N))

    WR,  WQ  = twiddles_iq(Nfft, inverse=False)
    WRi, WQi = twiddles_iq(Nfft, inverse=True)

    # taps tiempo (impulso en CENTER_TAP)
    wI = np.zeros(M, dtype=np.float64)
    wQ = np.zeros(M, dtype=np.float64)
    if 0 <= center_index < M: wI[center_index] = 1.0

    # particiones en tiempo
    w_partsI = np.zeros((K, N), dtype=np.float64)
    w_partsQ = np.zeros((K, N), dtype=np.float64)
    for k in range(K):
        s = k * N; e = min(s + N, M)
        w_partsI[k, :e - s] = wI[s:e]
        w_partsQ[k, :e - s] = wQ[s:e]

    # espectros por particion
    W_partsI = np.zeros((K, Nfft), dtype=np.float64)
    W_partsQ = np.zeros((K, Nfft), dtype=np.float64)
    tmpI = np.zeros(Nfft, dtype=np.float64)
    tmpQ = np.zeros(Nfft, dtype=np.float64)
    for k in range(K):
        tmpI[:] = 0.0; tmpQ[:] = 0.0
        tmpI[:N] = w_partsI[k]; tmpQ[:N] = w_partsQ[k]      # TIEMPO
        fft_iq(tmpI, tmpQ, WR, WQ)
        W_partsI[k, :], W_partsQ[k, :] = tmpI.copy(), tmpQ.copy()

    # overlap-save 50%
    overlapI = np.zeros(N, dtype=np.float64)
    overlapQ = np.zeros(N, dtype=np.float64)

    # historial X por particion
    X_histI = np.zeros((K, Nfft), dtype=np.float64)
    X_histQ = np.zeros((K, Nfft), dtype=np.float64)

    # buffers
    num_blocks = (len(xI) - N) // N + 1 if len(xI) >= N else 0
    if num_blocks <= 0: raise ValueError("señal muy corta para el N elegido.")

    yI = np.zeros(num_blocks * N, dtype=np.float64)
    yQ = np.zeros(num_blocks * N, dtype=np.float64)
    yhatI = np.zeros(num_blocks * N, dtype=np.float64)
    yhatQ = np.zeros(num_blocks * N, dtype=np.float64)
    eI = np.zeros(num_blocks * N, dtype=np.float64)
    eQ = np.zeros(num_blocks * N, dtype=np.float64)
    W_histI = np.zeros((num_blocks, M), dtype=np.float64)
    W_histQ = np.zeros((num_blocks, M), dtype=np.float64)

    x_blkI = np.zeros(Nfft, dtype=np.float64)
    x_blkQ = np.zeros(Nfft, dtype=np.float64)
    e_blk_padI = np.zeros(Nfft, dtype=np.float64)
    e_blk_padQ = np.zeros(Nfft, dtype=np.float64)

    out_idx = 0
    for b in range(num_blocks):
        # bloque 2N con solape
        start, stop = b * N, b * N + N
        x_newI = xI[start:stop]; x_newQ = xQ[start:stop]
        if len(x_newI) < N:
            x_newI = np.pad(x_newI, (0, N - len(x_newI)))
            x_newQ = np.pad(x_newQ, (0, N - len(x_newQ)))

        x_blkI[:N] = overlapI; x_blkQ[:N] = overlapQ
        x_blkI[N:] = x_newI;   x_blkQ[N:] = x_newQ

        # FFT del bloque
        X0I = q_vec(x_blkI, FX_NARROW); X0Q = q_vec(x_blkQ, FX_NARROW)
        fft_iq(X0I, X0Q, WR, WQ)
        if K > 1:
            X_histI[1:] = X_histI[:-1]; X_histQ[1:] = X_histQ[:-1]
        X_histI[0] = X0I; X_histQ[0] = X0Q

        # Y(f) = sum_k Wk * Xk
        YI = np.zeros(Nfft, dtype=np.float64)
        YQ = np.zeros(Nfft, dtype=np.float64)
        for k in range(K):
            wIr = W_partsI[k]; wQr = W_partsQ[k]
            xIr = X_histI[k];  xQr = X_histQ[k]
            for i in range(Nfft):
                yi, yq = cmul_iq(wIr[i], wQr[i], xIr[i], xQr[i], FX_WIDE, FX_WIDE)
                YI[i] = q_add(YI[i], yi, FX_WIDE)
                YQ[i] = q_add(YQ[i], yq, FX_WIDE)
        YI = q_vec(YI, FX_NARROW); YQ = q_vec(YQ, FX_NARROW)

        # IFFT y overlap-save
        ifft_iq(YI, YQ, WRi, WQi)
        y_blkI = q_vec(YI[N:], FX_NARROW)
        y_blkQ = q_vec(YQ[N:], FX_NARROW)

        # slicer y error
        d_blkI = np.where(y_blkI >= 0, +QPSK_A, -QPSK_A).astype(np.float64)
        d_blkQ = np.where(y_blkQ >= 0, +QPSK_A, -QPSK_A).astype(np.float64)
        d_blkI = q_vec(d_blkI, FX_NARROW); d_blkQ = q_vec(d_blkQ, FX_NARROW)
        e_blkI = q_vec(d_blkI - y_blkI, FX_NARROW)
        e_blkQ = q_vec(d_blkQ - y_blkQ, FX_NARROW)

        # FFT del error
        e_blk_padI[:] = 0.0; e_blk_padQ[:] = 0.0
        e_blk_padI[N:] = e_blkI; e_blk_padQ[N:] = e_blkQ
        EkI = q_vec(e_blk_padI, FX_NARROW); EkQ = q_vec(e_blk_padQ, FX_NARROW)
        fft_iq(EkI, EkQ, WR, WQ)

        # mu dinámico 
        mu_k  = mu_step(np.mean(e_blkI) + 1j*np.mean(e_blkQ)) if mu_step is not None else mu
        mu_fx = to_fx(mu_k, FX_WIDE)

        # update por partición
        for k in range(K):
            XkI = X_histI[k]; XkQ = X_histQ[k]
            # conj(Xk)*Ek  
            PI = (XkI * EkI) + (XkQ * EkQ)               # Re{ conj(X) * E }
            PQ = (XkI * EkQ) - (XkQ * EkI)               # Im{ conj(X) * E } = Xi*Er - Xr*Ei

            GI = q_vec(PI, FX_NARROW).copy()
            GQ = q_vec(PQ, FX_NARROW).copy()
            ifft_iq(GI, GQ, WRi, WQi)      # gradiente en tiempo
            gI = q_vec(GI[:N], FX_NARROW)
            gQ = q_vec(GQ[:N], FX_NARROW)

            # w_new = w_old + mu*g (en TIEMPO)
            for i in range(N):
                step_i = q(mu_fx * to_fx(gI[i], FX_WIDE), FX_WIDE)
                step_q = q(mu_fx * to_fx(gQ[i], FX_WIDE), FX_WIDE)
                w_partsI[k, i] = q(to_fx(w_partsI[k, i], FX_WIDE) + to_fx(step_i, FX_WIDE), FX_NARROW)
                w_partsQ[k, i] = q(to_fx(w_partsQ[k, i], FX_WIDE) + to_fx(step_q, FX_WIDE), FX_NARROW)

            # refrescar espectro de la partición 
            tmpI[:] = 0.0; tmpQ[:] = 0.0
            tmpI[:N] = w_partsI[k]; tmpQ[:N] = w_partsQ[k]
            fft_iq(tmpI, tmpQ, WR, WQ)
            W_partsI[k, :], W_partsQ[k, :] = tmpI.copy(), tmpQ.copy()

        # preparar siguiente bloque (overlap)
        overlapI[:] = x_blkI[N:]; overlapQ[:] = x_blkQ[N:]

        # guardar salidas
        yI[out_idx:out_idx+N]    = y_blkI
        yQ[out_idx:out_idx+N]    = y_blkQ
        yhatI[out_idx:out_idx+N] = d_blkI
        yhatQ[out_idx:out_idx+N] = d_blkQ
        eI[out_idx:out_idx+N]    = e_blkI
        eQ[out_idx:out_idx+N]    = e_blkQ
        out_idx += N

        # reconstruir w_full para trazas 
        wI[:] = 0.0; wQ[:] = 0.0
        for kk in range(K):
            s = kk * N; e = min(s + N, M)
            wI[s:e] = w_partsI[kk, :e - s]
            wQ[s:e] = w_partsQ[kk, :e - s]
        W_histI[b, :] = wI
        W_histQ[b, :] = wQ

    # taps finales
    w_finI = np.zeros(M, dtype=np.float64)
    w_finQ = np.zeros(M, dtype=np.float64)
    for k in range(K):
        s = k * N; e = min(s + N, M)
        w_finI[s:e] = w_partsI[k, :e - s]
        w_finQ[s:e] = w_partsQ[k, :e - s]

    k0 = int(center_index)

    return (yI, yQ, yhatI, yhatQ, eI, eQ,
            W_histI, W_histQ,
            w_finI, w_finQ,
            k0)
