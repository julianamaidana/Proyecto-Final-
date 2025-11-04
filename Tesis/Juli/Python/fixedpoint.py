import numpy as np
from tool._fixedInt import DeFixedInt

# Cuantizacion

FX_NARROW = {"N": 20, "F": 10, "SIGNED": True, "ROUND": "round_even", "SAT": "saturate"}
FX_WIDE   = {"N": 30, "F": 15, "SIGNED": True, "ROUND": "round_even", "SAT": "saturate"}

def _fx_make(val, cfg):
    obj = DeFixedInt(cfg["N"], cfg["F"], signedMode=('S' if cfg["SIGNED"] else 'U'),
                     roundMode=cfg["ROUND"], saturateMode=cfg["SAT"])
    obj.value = float(val)
    return obj

def to_fx(x, cfg):
    if isinstance(x, DeFixedInt):
        return _fx_make(x.fValue, cfg)
    return _fx_make(x, cfg)

def q(x, cfg):
    return to_fx(x, cfg).fValue

def q_add(a, b, cfg): return q(to_fx(a, cfg) + to_fx(b, cfg), cfg)

def q_sub(a, b, cfg): return q(to_fx(a, cfg) - to_fx(b, cfg), cfg)

def q_mul(a, b, cfg): return q(to_fx(a, cfg) * to_fx(b, cfg), cfg)

def q_vec(arr, cfg):
    a = np.asarray(arr, dtype=np.float64)
    out = np.empty_like(a)
    for i, v in enumerate(a): out[i] = q(v, cfg)
    return out

def q_vec_iq(I, Q, cfg):
    return q_vec(I, cfg), q_vec(Q, cfg)

def cmul_iq(aI, aQ, bI, bQ, cfg_op, cfg_out):
    rI = q_sub(q_mul(aI, bI, cfg_op), q_mul(aQ, bQ, cfg_op), cfg_out)
    rQ = q_add(q_mul(aI, bQ, cfg_op), q_mul(aQ, bI, cfg_op), cfg_out)
    return rI, rQ
