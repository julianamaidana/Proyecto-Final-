# pbfdaf_lms.py
# FFE_Engine: motor por bloques (overlap-save, LMS/NLMS en frecuencia)

import numpy as np
from fft_iq import fft_iq, ifft_iq, twiddles_iq   # se usan twiddles precomputados
from prbs_qpsk import QPSK_A  
__all__ = ["FFE_Engine"]

class FFE_Engine:
    """
    Ecualizador FFE en frecuencia por bloques (overlap-save, 50%).
    Mantiene estado entre llamadas a process_block(...).

    Parámetros
    ----------
    L_eq : int          # longitud total del FFE (taps en tiempo)
    part_N : int        # tamaño de partición (N); FFT usa 2N
    mu_init : float     # paso de adaptación inicial
    center_index : int  # índice del tap "centro" (impulso inicial)
    mu_step : callable  # opcional, recibe un error complejo medio y devuelve μ
    """
    def __init__(self, L_eq, part_N, mu_init, center_index=None, mu_step=None):
        # Dimensiones/estructura
        self.M = int(L_eq)
        self.N = int(part_N)
        if self.N <= 0:
            raise ValueError("part_N debe ser > 0")
        self.Nfft = 2 * self.N
        self.K = int(np.ceil(self.M / self.N))
        self.center = (self.M - 1) // 2 if center_index is None else int(center_index)

        # Twiddles
        self.WR,  self.WQ  = twiddles_iq(self.Nfft, inverse=False)
        self.WRi, self.WQi = twiddles_iq(self.Nfft, inverse=True)

        # Taps iniciales: impulso re=1 en 'center'
        wI = np.zeros(self.M, dtype=np.float64)
        wQ = np.zeros(self.M, dtype=np.float64)
        if 0 <= self.center < self.M:
            wI[self.center] = 1.0

        # Particiones (tiempo) de tamaño N
        self.w_partsI = np.zeros((self.K, self.N), dtype=np.float64)
        self.w_partsQ = np.zeros((self.K, self.N), dtype=np.float64)
        for k in range(self.K):
            s = k * self.N
            e = min(s + self.N, self.M)
            self.w_partsI[k, :e - s] = wI[s:e]
            self.w_partsQ[k, :e - s] = wQ[s:e]

        # Espectros por partición
        self.W_partsI = np.zeros((self.K, self.Nfft), dtype=np.float64)
        self.W_partsQ = np.zeros((self.K, self.Nfft), dtype=np.float64)
        tmpI = np.zeros(self.Nfft, dtype=np.float64)
        tmpQ = np.zeros(self.Nfft, dtype=np.float64)
        for k in range(self.K):
            tmpI[:] = 0.0; tmpQ[:] = 0.0
            tmpI[:self.N] = self.w_partsI[k]
            tmpQ[:self.N] = self.w_partsQ[k]
            fft_iq(tmpI, tmpQ, self.WR, self.WQ)
            self.W_partsI[k] = tmpI
            self.W_partsQ[k] = tmpQ

        # Estado overlap-save y buffers
        self.overlapI = np.zeros(self.N, dtype=np.float64)
        self.overlapQ = np.zeros(self.N, dtype=np.float64)
        self.X_histI  = np.zeros((self.K, self.Nfft), dtype=np.float64)
        self.X_histQ  = np.zeros((self.K, self.Nfft), dtype=np.float64)

        self.x_blkI = np.zeros(self.Nfft, dtype=np.float64)
        self.x_blkQ = np.zeros(self.Nfft, dtype=np.float64)
        self.e_blk_padI = np.zeros(self.Nfft, dtype=np.float64)
        self.e_blk_padQ = np.zeros(self.Nfft, dtype=np.float64)

        # Adaptación (NLMS suavizado)
        self.mu_init = float(mu_init)
        self.mu_step = mu_step if mu_step is not None else (lambda e: self.mu_init)
        self.pwr_ema = 0.0
        self.alpha = 0.9
        self.eps = 1e-3

    def process_block(self, x_newI, x_newQ):
        """
        Procesa un bloque de N muestras de entrada (I/Q).
        Devuelve (y_blkI, y_blkQ, yhat_blkI, yhat_blkQ, e_blkI, e_blkQ).
        """
        x_newI = np.asarray(x_newI, dtype=np.float64)
        x_newQ = np.asarray(x_newQ, dtype=np.float64)
        if x_newI.size != self.N or x_newQ.size != self.N:
            raise ValueError("process_block: se espera bloque de longitud N (part_N).")

        N = self.N; Nfft = self.Nfft; K = self.K

        # Construir bloque 2N con solape
        self.x_blkI[:N] = self.overlapI; self.x_blkQ[:N] = self.overlapQ
        self.x_blkI[N:] = x_newI;        self.x_blkQ[N:] = x_newQ

        # Potencia del bloque (para NLMS)
        p_blk = float(np.mean(x_newI**2 + x_newQ**2))
        self.pwr_ema = self.alpha * self.pwr_ema + (1.0 - self.alpha) * max(p_blk, 1e-12)

        # FFT de X0 y shift del historial
        X0I = self.x_blkI.copy(); X0Q = self.x_blkQ.copy()
        fft_iq(X0I, X0Q, self.WR, self.WQ)
        if K > 1:
            self.X_histI[1:] = self.X_histI[:-1]
            self.X_histQ[1:] = self.X_histQ[:-1]
        self.X_histI[0] = X0I; self.X_histQ[0] = X0Q

        # Y(f) = sum_k Wk * Xk   (a+jb)*(c+jd)
        YI = np.zeros(Nfft, dtype=np.float64)
        YQ = np.zeros(Nfft, dtype=np.float64)
        for k in range(K):
            wIr = self.W_partsI[k]; wQr = self.W_partsQ[k]
            XkI = self.X_histI[k];  XkQ = self.X_histQ[k]
            YI += (XkI * wIr) - (XkQ * wQr)
            YQ += (XkI * wQr) + (XkQ * wIr)

        # IFFT y salida útil (últimos N)
        Yti = YI.copy(); Ytq = YQ.copy()
        ifft_iq(Yti, Ytq, self.WRi, self.WQi)
        y_blkI = Yti[N:].astype(np.float64, copy=False)
        y_blkQ = Ytq[N:].astype(np.float64, copy=False)

        # Slicer QPSK (±1)
        #yhat_blkI = np.where(y_blkI >= 0.0, +1.0, -1.0)
        #yhat_blkQ = np.where(y_blkQ >= 0.0, +1.0, -1.0)
        yhat_blkI = np.where(y_blkI >= 0.0, +QPSK_A, -QPSK_A)
        yhat_blkQ = np.where(y_blkQ >= 0.0, +QPSK_A, -QPSK_A)
        # Error
        e_blkI = (yhat_blkI - y_blkI)
        e_blkQ = (yhat_blkQ - y_blkQ)

        # FFT del error (colocado en la mitad superior)
        self.e_blk_padI[:] = 0.0; self.e_blk_padQ[:] = 0.0
        self.e_blk_padI[N:] = e_blkI; self.e_blk_padQ[N:] = e_blkQ
        EkI = self.e_blk_padI.copy(); EkQ = self.e_blk_padQ.copy()
        fft_iq(EkI, EkQ, self.WR, self.WQ)

        # Paso NLMS
        mu_k  = float(self.mu_step(np.mean(e_blkI) + 1j*np.mean(e_blkQ)))
        mu_eff = mu_k / (self.pwr_ema + self.eps)

        # Gradiente por partición y actualización de W_parts
        tmpI = np.zeros(Nfft, dtype=np.float64)
        tmpQ = np.zeros(Nfft, dtype=np.float64)
        for k in range(K):
            XkI = self.X_histI[k]; XkQ = self.X_histQ[k]
            # conj(Xk)*Ek = (Xr - jXi)*(Er + jEi)
            PI = (XkI * EkI) + (XkQ * EkQ)
            PQ = (XkI * EkQ) - (XkQ * EkI)

            GI = PI.copy(); GQ = PQ.copy()
            ifft_iq(GI, GQ, self.WRi, self.WQi)
            gI = GI[:N]; gQ = GQ[:N]

            # update en tiempo: w_new = w_old + mu_eff * g
            self.w_partsI[k, :N] += mu_eff * gI
            self.w_partsQ[k, :N] += mu_eff * gQ

            # actualizar espectro de la partición
            tmpI[:] = 0.0; tmpQ[:] = 0.0
            tmpI[:N] = self.w_partsI[k, :N]
            tmpQ[:N] = self.w_partsQ[k, :N]
            fft_iq(tmpI, tmpQ, self.WR, self.WQ)
            self.W_partsI[k] = tmpI
            self.W_partsQ[k] = tmpQ

        # Avanzar solape
        self.overlapI[:] = x_newI
        self.overlapQ[:] = x_newQ

        return y_blkI, y_blkQ, yhat_blkI, yhat_blkQ, e_blkI, e_blkQ

    # Utilidad para reconstruir taps finales (para trazas)
    def get_final_taps(self):
        wI = np.zeros(self.M, dtype=np.float64)
        wQ = np.zeros(self.M, dtype=np.float64)
        for k in range(self.K):
            s = k * self.N; e = min(s + self.N, self.M)
            wI[s:e] = self.w_partsI[k, :e - s]
            wQ[s:e] = self.w_partsQ[k, :e - s]
        return wI, wQ, self.center
