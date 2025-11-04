import math
import numpy as np
from fixedpoint import q, to_fx, q_add, q_sub, cmul_iq, FX_NARROW, FX_WIDE

# FFT/IFFT radix-2

def bitrev_idx(N):
    # indices en orden bit-reversed
    m = int(math.log2(N))
    out = np.arange(N, dtype=np.int64)
    for i in range(N):
        r = 0; x = i
        for _ in range(m): r = (r << 1) | (x & 1); x >>= 1
        out[i] = r
    return out

def twiddles_iq(N, inverse=False):
    WR = np.zeros(N//2, dtype=np.float64)
    WQ = np.zeros(N//2, dtype=np.float64)
    sgn = +1.0 if inverse else -1.0
    for k in range(N//2):
        ang = 2.0 * math.pi * k / N
        # usar palabra ancha para la tabla
        WR[k] = q(math.cos(ang), FX_WIDE)
        WQ[k] = q(sgn * math.sin(ang), FX_WIDE)
    return WR, WQ

def fft_iq(I, Q, WR, WQ):
    # FFT in-place (I/Q reales) con butterfly radix-2
    N = len(I)
    assert (N & (N - 1)) == 0 and N > 1
    br = bitrev_idx(N)
    I[:] = I[br]; Q[:] = Q[br]          # reorden bit-reversed
    m = 2
    while m <= N:
        half = m // 2; step = N // m
        for k0 in range(0, N, m):
            for j in range(half):
                i0 = k0 + j; i1 = i0 + half; idx = j * step
                # t = W * x1
                tr, tq = cmul_iq(I[i1], Q[i1], WR[idx], WQ[idx], FX_WIDE, FX_WIDE)
                # a = x0 + t ; b = x0 - t
                ar = q_add(I[i0], tr, FX_WIDE); aq = q_add(Q[i0], tq, FX_WIDE)
                brv = q_sub(I[i0], tr, FX_WIDE); bq = q_sub(Q[i0], tq, FX_WIDE)
                # almacenar con estrechamiento
                I[i0], Q[i0] = q(ar,  FX_WIDE), q(aq,  FX_WIDE)
                I[i1], Q[i1] = q(brv, FX_WIDE), q(bq,  FX_WIDE)

        m <<= 1

def ifft_iq(I, Q, WRi, WQi):
    # IFFT in-place + escala 1/N
    N = len(I)
    assert (N & (N - 1)) == 0 and N > 1
    br = bitrev_idx(N)
    I[:] = I[br]; Q[:] = Q[br]
    m = 2
    while m <= N:
        half = m // 2; step = N // m
        for k0 in range(0, N, m):
            for j in range(half):
                i0 = k0 + j; i1 = i0 + half; idx = j * step
                # t = W_conj * x1
                tr, tq = cmul_iq(I[i1], Q[i1], WRi[idx], WQi[idx], FX_WIDE, FX_WIDE)
                # a = x0 + t ; b = x0 - t
                ar = q_add(I[i0], tr, FX_WIDE); aq = q_add(Q[i0], tq, FX_WIDE)
                brv = q_sub(I[i0], tr, FX_WIDE); bq = q_sub(Q[i0], tq, FX_WIDE)
                I[i0], Q[i0] = q(ar,  FX_WIDE), q(aq,  FX_WIDE)
                I[i1], Q[i1] = q(brv, FX_WIDE), q(bq,  FX_WIDE)
        m <<= 1
    # normalización 1/N
    invN = to_fx(1.0 / N, FX_WIDE)
    for i in range(N):
        I[i] = q(invN * to_fx(I[i], FX_WIDE), FX_NARROW)
        Q[i] = q(invN * to_fx(Q[i], FX_WIDE), FX_NARROW)
