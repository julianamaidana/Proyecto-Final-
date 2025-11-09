import numpy as np
import matplotlib.pyplot as plt
from math import erfc as _erfc_scalar

# erfc vectorizada sin SciPy
def erfc(x):
    x = np.asarray(x, dtype=float)
    return np.vectorize(_erfc_scalar)(x)

# Q(x) = 0.5 * erfc(x / sqrt(2))
def Q(x):
    return 0.5 * erfc(x / np.sqrt(2.0))

# curva BER teorica QPSK vs Eb/N0 (dB)
snr_db  = np.linspace(-2, 12, 281)    # Eb/N0 en dB
snr_lin = 10**(snr_db/10.0)           # Eb/N0 lineal
ber = Q(np.sqrt(2.0 * snr_lin))       # QPSK = BPSK en BER

plt.semilogy(snr_db, ber, label='QPSK BER teorica')
plt.grid(True, which='both', ls=':')
plt.xlabel(r'$E_b/N_0$ [dB]')
plt.ylabel('BER')
plt.ylim(1e-7, 1)
plt.xlim(snr_db[0], snr_db[-1])
plt.title('QPSK en AWGN (BER teorica)')
plt.legend()
plt.show()
