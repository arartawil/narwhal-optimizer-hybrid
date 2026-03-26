"""
Ablation Study for NWOA-NM (Nelder-Mead Hybrid)
=================================================
Tests the contribution of each local search component:
  1. NWOA (base, no LS)
  2. NWOA-NM Full (freq=10, cand=3, budget=50)
  3. NWOA-NM (freq=50 — less frequent LS)
  4. NWOA-NM (cand=1 — refine only the best)

Functions: F1 (unimodal), F5 (multimodal), F9 (composition)
Dim=10, 30 runs, 1000 iter, pop=30
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heurilab import BenchmarkConfig, BenchmarkSuite, run_experiment
from opfunu.cec_based import cec2022

from narwhal_optimizer import NarwhalOptimizer
from Hybrid.nwoa_nelder_mead import NWOA_NelderMead


# ─────────────────────────────────────────────
#  Wrappers for each ablation variant
# ─────────────────────────────────────────────

class NWOA_Base:
    """Base NWOA — no local search."""
    def __init__(self, pop_size, dim, lb, ub, max_iter, obj_func):
        self.algo = NarwhalOptimizer(
            objective_function=obj_func,
            dim=dim, lb=lb, ub=ub,
            n_agents=pop_size, max_iter=max_iter,
        )
    def optimize(self):
        return self.algo.optimize()


class NWOA_NM_Full:
    """NWOA-NM Full: freq=10, cand=3, budget=50."""
    def __init__(self, pop_size, dim, lb, ub, max_iter, obj_func):
        self.algo = NWOA_NelderMead(
            objective_function=obj_func,
            dim=dim, lb=lb, ub=ub,
            n_agents=pop_size, max_iter=max_iter,
            ls_frequency=10, ls_candidates=3, ls_max_evals=50,
        )
    def optimize(self):
        return self.algo.optimize()


class NWOA_NM_FreqLow:
    """NWOA-NM with less frequent LS: freq=50."""
    def __init__(self, pop_size, dim, lb, ub, max_iter, obj_func):
        self.algo = NWOA_NelderMead(
            objective_function=obj_func,
            dim=dim, lb=lb, ub=ub,
            n_agents=pop_size, max_iter=max_iter,
            ls_frequency=50, ls_candidates=3, ls_max_evals=50,
        )
    def optimize(self):
        return self.algo.optimize()


class NWOA_NM_Cand1:
    """NWOA-NM refining only the best solution: cand=1."""
    def __init__(self, pop_size, dim, lb, ub, max_iter, obj_func):
        self.algo = NWOA_NelderMead(
            objective_function=obj_func,
            dim=dim, lb=lb, ub=ub,
            n_agents=pop_size, max_iter=max_iter,
            ls_frequency=10, ls_candidates=1, ls_max_evals=50,
        )
    def optimize(self):
        return self.algo.optimize()


# ─────────────────────────────────────────────
#  Benchmark: F1, F5, F9 — D=10
# ─────────────────────────────────────────────
DIM = 10

ABLATION_FUNCTIONS = [
    (cec2022.F12022,  "CEC22_F1"),   # Unimodal
    (cec2022.F52022,  "CEC22_F5"),   # Multimodal
    (cec2022.F92022,  "CEC22_F9"),   # Composition
]


def build_ablation_suite(dim):
    benchmarks = []
    for FuncClass, name in ABLATION_FUNCTIONS:
        bench = FuncClass(ndim=dim)
        def make_obj(b):
            return lambda x: b.evaluate(x)
        benchmarks.append(BenchmarkConfig(
            name=name,
            obj_func=make_obj(bench),
            lb=float(bench.lb[0]),
            ub=float(bench.ub[0]),
            dim=dim,
        ))
    return BenchmarkSuite(category="CEC2022_Ablation", benchmarks=benchmarks)


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    algorithms = [
        ("NWOA",            NWOA_Base),
        ("NWOA-NM_Full",    NWOA_NM_Full),
        ("NWOA-NM_Freq50",  NWOA_NM_FreqLow),
        ("NWOA-NM_Cand1",   NWOA_NM_Cand1),
    ]

    suites = [build_ablation_suite(DIM)]

    run_experiment(
        algorithms=algorithms,
        benchmark_suites=suites,
        output_dir="Hybrid/Ablation_Study",
        pop_size=30,
        max_iter=1000,
        dim=DIM,
        n_runs=30,
        run_engineering=False,
    )
