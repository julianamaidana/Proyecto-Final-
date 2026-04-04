"""
ber_curva.py — Curva BER vs Es/N0 para tesis
=============================================
Compara tres curvas:
  1. QPSK teórica AWGN
  2. RTL sin ecualizador (i_we=0, W=identidad)
  3. RTL con ecualizador (i_we=fft_w_valid, LMS activo)

INSTRUCCIONES:
  1. Correr simulación con i_we=1'b0 para cada sigma_scale → llenar RTL_NO_ECUAL
  2. Correr simulación con i_we=fft_w_valid → llenar RTL_ECUAL
  3. python ber_curva.py

El BER a ingresar es el valor de:
  [BER]  BER=0.XXX (x1e-3)
del resumen final de cada corrida.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import erfc

# ===== PARÁMETROS DEL SISTEMA =====
SIGMA_GNG = 2**11   # varianza unitaria del GNG
SIGMA_F   = 10      # shift en top_ch
QPSK_A    = 91      # amplitud QPSK en Q7

def sigma_to_EsN0_dB(sc):
    noise_std = SIGMA_GNG * sc / (2**SIGMA_F)
    N0 = 2 * noise_std**2
    Es = 2 * QPSK_A**2
    return 10 * np.log10(Es / N0)

# ===== COMPLETAR CON LOS RESULTADOS DE VIVADO =====
# Formato: (sigma_scale, BER_medido)
# BER_medido = el número de "[BER]  BER=0.XXX (x1e-3)" × 0.001

RTL_NO_ECUAL = [
    # (sigma_scale, BER)   ← i_we=1'b0 en top_global.v
    # Ejemplos:
    # (4,  0.020)
    (3, 0.002), # 0.002
    (4, 0.008), # 0.008
    (5, 0.018), # 0.018
    (8, 0.045), # 0.045
]

RTL_ECUAL = [
    # (sigma_scale, BER)   ← i_we=fft_w_valid en top_global.v
    # Ejemplos:
    # (4,  0.008),
    (3, 0.506), # 0.506
    (4, 0.376), # 0.376
    (5, 0.457), # 0.457
    (6, 0.509), # 0.509
    (8, 0.447), # 0.447
]

# ===== CURVA TEÓRICA QPSK AWGN =====
esn0_range = np.linspace(0, 22, 500)
ber_theory = [0.5 * erfc(np.sqrt(10**(x/10)/2)) for x in esn0_range]

# ===== PLOT =====
fig, ax = plt.subplots(figsize=(9, 6))

ax.semilogy(esn0_range, ber_theory, 'b-', lw=2, label='QPSK teórica (AWGN)')

if RTL_NO_ECUAL:
    x = [sigma_to_EsN0_dB(sc) for sc, _ in RTL_NO_ECUAL]
    y = [ber for _, ber in RTL_NO_ECUAL]
    ax.semilogy(x, y, 'r^--', lw=2, ms=8, label='RTL — sin ecualizador (canal ISI)')

if RTL_ECUAL:
    x = [sigma_to_EsN0_dB(sc) for sc, _ in RTL_ECUAL]
    y = [ber for _, ber in RTL_ECUAL]
    ax.semilogy(x, y, 'go-', lw=2, ms=8, label='RTL — con ecualizador PBFDAF-LMS')

ax.set_xlabel('Es/N0 [dB]', fontsize=13)
ax.set_ylabel('BER', fontsize=13)
ax.set_title('QPSK-OFDM: BER vs Es/N0\nCanal ISI + Ecualizador PBFDAF-LMS (RTL Q17.10)',
            fontsize=12)
ax.set_xlim([0, 22])
ax.set_ylim([1e-5, 1])
ax.grid(True, which='both', alpha=0.4)
ax.legend(fontsize=11)

if not RTL_NO_ECUAL and not RTL_ECUAL:
    print("RTL_NO_ECUAL y RTL_ECUAL vacíos.")
    print("Tabla de sigma_scale → Es/N0 para las corridas:")
    print(f"{'sigma_scale':>12}  {'Es/N0 (dB)':>12}")
    for sc in [3, 4, 5, 6, 8, 10, 12, 16, 20]:
        print(f"  {sc:>10}  {sigma_to_EsN0_dB(sc):>12.2f}")

plt.tight_layout()
plt.savefig('ber_curva_tesis.png', dpi=150)
plt.show()
print("Guardado: ber_curva_tesis.png")
