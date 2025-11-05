import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Plots

def plot_weights_labeled(W_hist, center_index, smoothing_window=None):
    """
    Grafica la evolución de los coeficientes del FFE.
    
    smoothing_window (int, opcional): 
        Si se provee, aplica una media móvil de esta ventana
        para mostrar la tendencia en lugar del ruido bloque-a-bloque.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    L = W_hist.shape[1]

    # Aplicar media móvil si se especificó
    if smoothing_window is not None and smoothing_window > 1:
        win = int(smoothing_window)
        # Preparamos arrays vacíos para los datos suavizados
        W_hist_real_smooth = np.empty_like(W_hist.real)
        W_hist_imag_smooth = np.empty_like(W_hist.imag)
        
        # Aplicamos la media móvil a cada tap (columna)
        for i in range(L):
            W_hist_real_smooth[:, i] = pd.Series(W_hist[:, i].real).rolling(
                window=win, min_periods=1, center=True).mean()
            W_hist_imag_smooth[:, i] = pd.Series(W_hist[:, i].imag).rolling(
                window=win, min_periods=1, center=True).mean()
        
        # Sobrescribimos los datos a graficar con la versión suavizada
        plot_data_real = W_hist_real_smooth
        plot_data_imag = W_hist_imag_smooth
        ax.set_title(f"Evolución de coeficientes FFE (Tendencia con ventana de {win} bloques)")
    else:
        # Comportamiento original: graficar punto por punto
        plot_data_real = W_hist.real
        plot_data_imag = W_hist.imag
        ax.set_title("Evolución de coeficientes del FFE")

    # Graficar los datos (originales o suavizados)
    for i in range(L):
        ls = "-" if i == center_index else "--"
        ax.plot(plot_data_real[:, i], linestyle=ls, linewidth=1,
                label=f"Re w[{i}]" + (" (c)" if i == center_index else ""))
        ax.plot(plot_data_imag[:, i], linestyle=":", linewidth=1,
                label=f"Im w[{i}]")
    
    ax.set_xlabel("bloque"); ax.set_ylabel("valor del tap")
    ax.grid(True)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              borderaxespad=0., fontsize=8, ncol=1)
    
    plt.subplots_adjust(right=0.78)
    fig.tight_layout()
    plt.show()

# helper: límites simétricos (misma escala Re/Im y entre curvas)
def sym_ylim(*arrs, margin=0.05):
    m = 0.0
    for a in arrs:
        if a is None: 
            continue
        aa = np.asarray(a)
        if aa.size:
            m = max(m, float(np.max(np.abs(aa))))
    if m <= 0.0: m = 1.0
    m *= (1.0 + margin)
    return (-m, m)

def plot_filter_profile(w, center_index, ylims_re=None, ylims_im=None):
    # perfil del FFE (Re/Im). Si se pasan ylims_*, usa esa escala (para matchear con h).
    n = np.arange(len(w))

    # Real
    plt.figure(figsize=(8, 3.6))
    plt.stem(n, w.real)
    plt.axvline(center_index, color="r", linestyle=":", linewidth=1)
    if ylims_re is None:
        ymax = 1.05 * max(1.0, np.max(np.abs(np.concatenate([w.real, w.imag]))))
        plt.ylim(-ymax, ymax)
    else:
        plt.ylim(ylims_re)
    plt.title("Perfil del FFE (Re)")
    plt.xlabel("índice de tap"); plt.ylabel("amplitud")
    plt.grid(True); plt.tight_layout(); plt.show()

    # Imaginario
    plt.figure(figsize=(8, 3.6))
    plt.stem(n, w.imag)
    plt.axvline(center_index, color="r", linestyle=":", linewidth=1)
    if ylims_im is None:
        ymax = 1.05 * max(1.0, np.max(np.abs(np.concatenate([w.real, w.imag]))))
        plt.ylim(-ymax, ymax)
    else:
        plt.ylim(ylims_im)
    plt.title("Perfil del FFE (Im)")
    plt.xlabel("índice de tap"); plt.ylabel("amplitud")
    plt.grid(True); plt.tight_layout(); plt.show()

def plot_channel_profile(h, center_index=None, ylims=None, title="Perfil del canal h[n]"):

    h = np.asarray(h, np.complex128)
    n = np.arange(len(h))

    # limites simétricos comunes
    if ylims is None:
        ymax = 1.05 * np.max(np.abs(np.r_[h.real, h.imag]))
        if ymax <= 0: ymax = 1.0
        ylims = (-ymax, ymax)

    # Re
    plt.figure(figsize=(8, 3.6))
    plt.stem(n, h.real)
    if center_index is not None: plt.axvline(center_index, color="r", linestyle=":", linewidth=1)
    plt.ylim(ylims)
    plt.title(title + " (Re)")
    plt.xlabel("índice de tap"); plt.ylabel("amplitud")
    plt.grid(True); plt.tight_layout(); plt.show()

    # Im
    plt.figure(figsize=(8, 3.6))
    plt.stem(n, h.imag)
    if center_index is not None: plt.axvline(center_index, color="r", linestyle=":", linewidth=1)
    plt.ylim(ylims)
    plt.title(title + " (Im)")
    plt.xlabel("índice de tap"); plt.ylabel("amplitud")
    plt.grid(True); plt.tight_layout(); plt.show()

    return ylims



def combined_response(w, h):
    return np.convolve(w, np.asarray(h, np.complex128))

def plot_combined_response(w, h):
    g = combined_response(w, h)
    n = np.arange(len(g))
    plt.figure(figsize=(8, 3.6)); plt.stem(n, np.abs(g))
    m0 = np.argmax(np.abs(g))
    plt.axvline(m0, color="r", linestyle=":", linewidth=1)
    plt.title("Convolución filtro–canal |w * h|")
    plt.xlabel("m"); plt.ylabel("magnitud")
    plt.grid(True); plt.tight_layout(); plt.show()
    return g

def plot_time_iq_complex(x_seq, n_show, title):
    k = np.arange(min(n_show, len(x_seq)))
    x = x_seq[:len(k)]
    plt.figure(figsize=(10, 5.6))
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(k, x.real, '.', markersize=2)
    ax1.axhline(0, color="k", linewidth=0.8)
    ax1.set_title(title + " (Re)")
    ax1.set_xlabel("n"); ax1.set_ylabel("amplitud")
    ax1.grid(True)
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(k, x.imag, '.', markersize=2)
    ax2.axhline(0, color="k", linewidth=0.8)
    ax2.set_title(title + " (Im)")
    ax2.set_xlabel("n"); ax2.set_ylabel("amplitud")
    ax2.grid(True)
    plt.tight_layout(); plt.show()

def plot_constellation(x_seq, title="Constelación", tail=5000):
    # últimas tail muestras (default: 5000)
    n_tail = min(tail, len(x_seq))
    x_tail = x_seq[-n_tail:]

    plt.figure(figsize=(4.8, 4.8))
    plt.scatter(x_tail.real, x_tail.imag, s=10, alpha=0.5)
    a = 1.6/np.sqrt(2)
    plt.axhline(0, color="k", linewidth=0.8); plt.axvline(0, color="k", linewidth=0.8)
    plt.xlim([-a, a]); plt.ylim([-a, a])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("Re"); plt.ylabel("Im")
    plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()


# FFT cplx -> espectro y eje f (ciclos/muestra)
def _fft_cplx(x, pad_n=4096, full=True):
    x = np.asarray(x, np.complex128).ravel()
    X = np.fft.fft(x, n=pad_n)
    if full:
        X = np.fft.fftshift(X)
        f = np.linspace(-0.5, 0.5, pad_n, endpoint=False)
    else:
        half = pad_n // 2 + 1
        X = X[:half]
        f = np.linspace(0.0, 0.5, half, endpoint=True)
    return X, f

# |X| en dB
def _mag_db_fft_cplx_full(x, pad_n=4096, eps=1e-12):
    x = np.asarray(x, np.complex128).ravel()
    X = np.fft.fft(x, n=pad_n)
    f = np.linspace(0.0, 1.0, pad_n, endpoint=False)   # 0..1
    return 20*np.log10(np.abs(X)+eps), f

# plot 0..1
def plot_freq_responses_db(w_final, h_float, pad_n=4096, title="H(f), W(f) y H·W (dB)"):
    Hw_db, f  = _mag_db_fft_cplx_full(h_float, pad_n)
    Ww_db, _  = _mag_db_fft_cplx_full(w_final, pad_n)
    HW_db, _  = _mag_db_fft_cplx_full(np.convolve(w_final, h_float), pad_n)

    plt.figure(figsize=(9,5))
    plt.plot(f, Hw_db, label="|H(f)| dB (canal)")
    plt.plot(f, Ww_db, label="|W(f)| dB (FFE)")
    plt.plot(f, HW_db, label="|H(f)·W(f)| dB")
    plt.xlabel("f (ciclos/muestra) [0…1]")
    plt.ylabel("magnitud [dB]")
    plt.title(title)
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

def plot_ber_vs_mu(mus, ber_mean, ber_std=None, title=None, annotate_min=True):
    """
    Grafica BER (eje Y) vs paso de adaptación μ (eje X, log).
    - mus: array de μ (>0)
    - ber_mean: BER promedio por μ
    - ber_std (opcional): desvío/errores por μ para sombrear
    """
    mus = np.asarray(mus)
    ber_mean = np.asarray(ber_mean)

    idx_min = int(np.nanargmin(ber_mean))

    plt.figure(figsize=(7.5, 4.5))
    plt.loglog(mus, ber_mean, 'o-', linewidth=1)
    if ber_std is not None:
        lo = np.maximum(1e-12, ber_mean - ber_std)
        hi = ber_mean + ber_std
        plt.fill_between(mus, lo, hi, alpha=0.15, step=None)

    plt.grid(True, which='both', linestyle=':')
    plt.xlabel('Paso de adaptación μ')
    plt.ylabel('BER')
    if title:
        plt.title(title)

    if annotate_min:
        plt.plot(mus[idx_min], ber_mean[idx_min], 's', markersize=6)
        plt.annotate(f'μ*={mus[idx_min]:.2e}\nBER={ber_mean[idx_min]:.2g}',
                     (mus[idx_min], ber_mean[idx_min]),
                     textcoords='offset points', xytext=(8, 8))

    plt.tight_layout()
    plt.show()
    return idx_min, mus[idx_min], float(ber_mean[idx_min])
