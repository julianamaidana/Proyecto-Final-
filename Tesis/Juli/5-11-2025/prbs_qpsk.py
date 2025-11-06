import numpy as np
from fixedpoint import q, FX_NARROW

# PRBS9 + QPSK

def prbs9(N, seed):
    if seed == 0 or seed >= (1 << 9):
        raise ValueError("Seed PRBS9 debe ser 9 bits y no nula.")
    st = seed & 0x1FF
    out = np.empty(N, dtype=np.uint8)
    for i in range(N):
        out[i] = (st >> 8) & 1
        fb = ((st >> 8) ^ (st >> 4)) & 1
        st = ((st << 1) & 0x1FF) | fb
    return out

QPSK_A = q(1.0 / np.sqrt(2.0), FX_NARROW)

def prbs_qpsk_iq(Nsym, seedI=0x17F, seedQ=0x11D):
    bI = prbs9(Nsym, seedI)
    bQ = prbs9(Nsym, seedQ)
    sI = np.where(bI == 0, +QPSK_A, -QPSK_A).astype(np.float64)
    sQ = np.where(bQ == 0, +QPSK_A, -QPSK_A).astype(np.float64)
    return sI, sQ