# generate_vectors_hex.py
import numpy as np
from prbs_qpsk import prbs9

# --- Configuración ---
N_SYM = 1022      # Cantidad de muestras a comparar
SEED_I = 0x17F    # Semilla Canal I
SEED_Q = 0x11D    # Semilla Canal Q
FILENAME = "verilog_ref.mem"

# Valor de amplitud Q9.7 (+90 y -90)
QPSK_A_FX_INT = 90

def generate_hex_file():
    # 1. Generar bits PRBS9
    bI = prbs9(N_SYM, seed=SEED_I)
    bQ = prbs9(N_SYM, seed=SEED_Q)

    # 2. Mapeo a Enteros (+90 o -90)
    # Bit 0 -> +90, Bit 1 -> -90
    sI_int = np.where(bI == 0, +QPSK_A_FX_INT, -QPSK_A_FX_INT).astype(int)
    sQ_int = np.where(bQ == 0, +QPSK_A_FX_INT, -QPSK_A_FX_INT).astype(int)

    print(f"Generando {FILENAME} con {N_SYM} vectores...")

    # 3. Escribir archivo en formato HEX para Verilog ($readmemh)
    with open(FILENAME, "w") as f:
        for i in range(N_SYM):
            # Convertir a 16 bits complemento a 2 usando máscara 0xFFFF
            val_I = sI_int[i] & 0xFFFF
            val_Q = sQ_int[i] & 0xFFFF
            
            # Escribir línea: HHHH HHHH
            f.write(f"{val_I:04x} {val_Q:04x}\n")

    print("¡Listo! Archivo generado correctamente.")
    print("Copia 'verilog_ref.mem' a tu carpeta de simulación de Vivado (sim_1/behav/xsim/).")

if __name__ == "__main__":
    generate_hex_file()