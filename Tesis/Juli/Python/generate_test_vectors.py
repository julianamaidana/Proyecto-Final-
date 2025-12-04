# generate_test_vectors.py
import numpy as np
from prbs_qpsk import prbs9, QPSK_A
import math

# --- Configuraciones ---
N_SYM = 1022      # Número de símbolos a generar (ej. dos períodos PRBS9)
SEED_I = 0x17F    # Semilla para el canal I (prbs_qpsk.py)
SEED_Q = 0x11D    # Semilla para el canal Q (prbs_qpsk.py)
FILENAME = "qpsk_prbs_test_vector.txt"

# --- Valor de Punto Fijo (Q9.7)
QPSK_A_FX_INT = 90

def generate_qpsk_prbs_vector():
    """Genera el vector de prueba de bits (bI, bQ) y amplitudes Q9.7 (sI, sQ)."""

    # 1. Generar bits PRBS9 (0 o 1)
    bI = prbs9(N_SYM, seed=SEED_I)
    bQ = prbs9(N_SYM, seed=SEED_Q)

    # 2. Mapeo QPSK a entero de punto fijo (Q9.7)
    #   Bit 0 -> +QPSK_A_FX_INT
    #   Bit 1 -> -QPSK_A_FX_INT
    sI_int = np.where(bI == 0, +QPSK_A_FX_INT, -QPSK_A_FX_INT).astype(np.int64)
    sQ_int = np.where(bQ == 0, +QPSK_A_FX_INT, -QPSK_A_FX_INT).astype(np.int64)
    
    # 3. Empaquetar datos para el archivo
    # La columna 'n' es opcional pero útil para indexar
    n = np.arange(N_SYM)
    test_data = np.vstack([n, bI, bQ, sI_int, sQ_int]).T
    
    # 4. Guardar en un formato legible para Verilog
    header = "n,bI_source,bQ_source,sI_Q9_7,sQ_Q9_7"
    np.savetxt(FILENAME, test_data, fmt='%d', header=header, delimiter=',')

    print("--- Vector Matching Generado ---")
    print(f"Archivo: {FILENAME}")
    print(f"Símbolos: {N_SYM}")
    print(f"Amplitud Q9.7: {QPSK_A_FX_INT} (equiv. a {QPSK_A:.6f} flotante)")

    # Mostrar las primeras 10 líneas para verificación
    print("\nPrimeras 10 líneas (Índice, bI, bQ, sI, sQ):")
    for row in test_data[:10]:
        print(f"  {row[0]:<3} | {row[1]:<2} {row[2]:<2} | {row[3]:>4} {row[4]:>4}")
    print("------------------------------")


if __name__ == "__main__":
    generate_qpsk_prbs_vector()