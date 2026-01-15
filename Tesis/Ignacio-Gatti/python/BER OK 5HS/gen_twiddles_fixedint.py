import math
from tool._fixedInt import DeFixedInt

FX_WIDE = {"N": 17, "F": 10, "SIGNED": True, "ROUND": "round_even", "SAT": "saturate"}

def fx_obj(val, cfg):
    obj = DeFixedInt(cfg["N"], cfg["F"],
                     signedMode=('S' if cfg["SIGNED"] else 'U'),
                     roundMode=cfg["ROUND"],
                     saturateMode=cfg["SAT"])
    obj.value = float(val)   # <-- acá se aplica round_even + saturate
    return obj

def fx_int(val, cfg):
    """
    Devuelve el entero cuantizado (escalado por 2^F) en rango signed.
    Intenta leer el 'raw' del objeto; si tu DeFixedInt no lo expone,
    cae a reconstruirlo desde fValue (menos ideal, pero funciona).
    """
    o = fx_obj(val, cfg)

    # 1) Intentar atributos típicos (depende de tu implementación de DeFixedInt)
    for attr in ("intValue", "iValue", "raw", "_value", "valueInt", "int_value"):
        if hasattr(o, attr):
            v = getattr(o, attr)
            v = v() if callable(v) else v
            if isinstance(v, int):
                return v

    # 2) Fallback: reconstruir desde el float cuantizado (o.fValue)
    #    (si tu DeFixedInt devuelve exactamente múltiplos de 2^-F, esto coincide)
    return int(round(o.fValue * (1 << cfg["F"])))

def to_hex_tc(x, W):
    """signed int -> W-bit 2's complement hex (sin 0x)"""
    mask = (1 << W) - 1
    return f"{(x & mask):0{(W+3)//4}X}"

def gen_twiddles(NFFT, cfg, inverse=False):
    """
    Devuelve listas de enteros (wr_int, wi_int) para k=0..NFFT/2-1
    FFT: wi = -sin
    IFFT: wi = +sin  (conjugado)
    """
    W, F = cfg["N"], cfg["F"]
    sgn = +1.0 if inverse else -1.0
    wr = []
    wi = []
    for k in range(NFFT // 2):
        ang = 2.0 * math.pi * k / NFFT
        wr.append(fx_int(math.cos(ang), cfg))
        wi.append(fx_int(sgn * math.sin(ang), cfg))
    return wr, wi

def write_memh(fname, arr, W):
    with open(fname, "w") as f:
        for v in arr:
            f.write(to_hex_tc(v, W) + "\n")

def write_case(fname, arr, W, name):
    with open(fname, "w") as f:
        f.write(f"// {name} LUT, W={W} bits (2's complement)\n")
        f.write("case (addr)\n")
        for k, v in enumerate(arr):
            f.write(f"  {k}: {name} = {W}'sh{to_hex_tc(v, W)};\n")
        f.write(f"  default: {name} = '0;\n")
        f.write("endcase\n")

if __name__ == "__main__":
    # ======= elegí tu NFFT real =======
    # En tu PBFDAF suele ser NFFT = 2*part_N (por ej part_N=64 => NFFT=128)
    NFFT = 128

    TW_W = FX_WIDE["N"]

    # FFT twiddles (cos, -sin)
    wr_fft, wi_fft = gen_twiddles(NFFT, FX_WIDE, inverse=False)
    write_memh("tw_re_fft.mem", wr_fft, TW_W)
    write_memh("tw_im_fft.mem", wi_fft, TW_W)

    # Opcional: cases para pegar en Verilog
    write_case("tw_re_fft_case.vh", wr_fft, TW_W, "tw_re")
    write_case("tw_im_fft_case.vh", wi_fft, TW_W, "tw_im")

    print("OK: tw_re_fft.mem / tw_im_fft.mem y .vh generados.")
