import numpy as np
import matplotlib.pyplot as plt

# Plots

def plot_weights_labeled(W_hist, center_index):
    import numpy as np
    fig, ax = plt.subplots(figsize=(10, 4))
    L = W_hist.shape[1]
    for i in range(L):
        ls = "-" if i == center_index else "--"
        ax.plot(W_hist[:, i].real, linestyle=ls, linewidth=1,
                label=f"Re w[{i}]" + (" (c)" if i == center_index else ""))
        ax.plot(W_hist[:, i].imag, linestyle=":", linewidth=1,
                label=f"Im w[{i}]")
    ax.set_title("Evolución de coeficientes del FFE")
    ax.set_xlabel("bloque"); ax.set_ylabel("valor del tap")
    ax.grid(True)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              borderaxespad=0., fontsize=8, ncol=1)
    # deja espacio a la derecha para la leyenda
    plt.subplots_adjust(right=0.78)
    fig.tight_layout()
    plt.show()

def plot_filter_profile(w, center_index):
    import numpy as np
    ymax = np.max(np.abs(np.concatenate([w.real, w.imag])))
    ymax = 1.05 * (ymax if ymax > 0 else 1.0) 

    n = np.arange(len(w))

    # Real
    plt.figure(figsize=(8, 3.6))
    plt.stem(n, w.real)
    plt.axvline(center_index, color="r", linestyle=":", linewidth=1)
    plt.ylim(-ymax, ymax)  
    plt.title("Perfil del FFE (Re)")
    plt.xlabel("índice de tap"); plt.ylabel("amplitud")
    plt.grid(True); plt.tight_layout(); plt.show()

    # Imaginario 
    plt.figure(figsize=(8, 3.6))
    plt.stem(n, w.imag)
    plt.axvline(center_index, color="r", linestyle=":", linewidth=1)
    plt.ylim(-ymax, ymax) 
    plt.title("Perfil del FFE (Im)")
    plt.xlabel("índice de tap"); plt.ylabel("amplitud")
    plt.grid(True); plt.tight_layout(); plt.show()

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
    # últimas 'tail' muestras (default: 5000)
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