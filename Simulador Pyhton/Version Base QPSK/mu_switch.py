from collections import deque
import numpy as np

def make_mu_switch(enable=True, mu_init=2e-3, mu_final=2e-4,
                   n_switch=None, use_stable=False,
                   win=200, tol=1e-2, patience=100):
    state = {"k": 0, "switched": False, "buf": deque(maxlen=max(2, int(win))), "ok": 0}

    def step(e_k):
        if not enable or state["switched"]:
            state["k"] += 1
            return mu_final if state["switched"] else mu_init
        if (n_switch is not None) and (state["k"] >= n_switch):
            state["switched"] = True; state["k"] += 1; return mu_final
        if use_stable:
            e2 = float(e_k.real * e_k.real + e_k.imag * e_k.imag)
            state["buf"].append(e2)
            if len(state["buf"]) == state["buf"].maxlen:
                rng_rel = np.std(state["buf"]) / (np.mean(state["buf"]) + 1e-12)
                state["ok"] = state["ok"] + 1 if rng_rel < tol else 0
                if state["ok"] >= patience:
                    state["switched"] = True; state["k"] += 1; return mu_final
        state["k"] += 1
        return mu_init

    step.state = state
    return step