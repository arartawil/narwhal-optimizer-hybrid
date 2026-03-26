"""
Ablation Study for NWOA-HC (Hill Climbing Hybrid)
===================================================
  1. NWOA (base, no LS)
  2. NWOA-HC Full (freq=10, cand=3, budget=50)
  3. NWOA-HC (freq=50 — less frequent LS)
  4. NWOA-HC (cand=1 — refine only the best)

Functions: F1 (unimodal), F5 (multimodal), F9 (composition)
Dim=10, 30 runs, 1000 iter, pop=30
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heurilab import BenchmarkConfig, BenchmarkSuite, run_experiment
from opfunu.cec_based import cec2022

from narwhal_optimizer import NarwhalOptimizer
from Hybrid.nwoa_hill_climbing import NWOA_HillClimbing


class NWOA_Base:
    def __init__(self, pop_size, dim, lb, ub, max_iter, obj_func):
        self.algo = NarwhalOptimizer(
            objective_function=obj_func, dim=dim, lb=lb, ub=ub,
            n_agents=pop_size, max_iter=max_iter,
        )
    def optimize(self):
        return self.algo.optimize()


class NWOA_HC_Full:
    """NWOA-HC Full: freq=10, cand=3, budget=50."""
    def __init__(self, pop_size, dim, lb, ub, max_iter, obj_func):
        self.algo = NWOA_HillClimbing(
            objective_function=obj_func, dim=dim, lb=lb, ub=ub,
            n_agents=pop_size, max_iter=max_iter,
            ls_frequency=10, ls_candidates=3, ls_max_evals=50,
        )
    def optimize(self):
        return self.algo.optimize()


class NWOA_HC_FreqLow:
    """NWOA-HC with less frequent LS: freq=50."""
    def __init__(self, pop_size, dim, lb, ub, max_iter, obj_func):
        self.algo = NWOA_HillClimbing(
            objective_function=obj_func, dim=dim, lb=lb, ub=ub,
            n_agents=pop_size, max_iter=max_iter,
            ls_frequency=50, ls_candidates=3, ls_max_evals=50,
        )
    def optimize(self):
        return self.algo.optimize()


class NWOA_HC_Cand1:
    """NWOA-HC refining only the best solution: cand=1."""
    def __init__(self, pop_size, dim, lb, ub, max_iter, obj_func):
        self.algo = NWOA_HillClimbing(
            objective_function=obj_func, dim=dim, lb=lb, ub=ub,
            n_agents=pop_size, max_iter=max_iter,
            ls_frequency=10, ls_candidates=1, ls_max_evals=50,
        )
    def optimize(self):
        return self.algo.optimize()


DIM = 10

ABLATION_FUNCTIONS = [
    (cec2022.F12022,  "CEC22_F1"),
    (cec2022.F52022,  "CEC22_F5"),
    (cec2022.F92022,  "CEC22_F9"),
]


def build_ablation_suite(dim):
    benchmarks = []
    for FuncClass, name in ABLATION_FUNCTIONS:
        bench = FuncClass(ndim=dim)
        def make_obj(b):
            return lambda x: b.evaluate(x)
        benchmarks.append(BenchmarkConfig(
            name=name, obj_func=make_obj(bench),
            lb=float(bench.lb[0]), ub=float(bench.ub[0]), dim=dim,
        ))
    return BenchmarkSuite(category="CEC2022_Ablation_HC", benchmarks=benchmarks)


if __name__ == "__main__":
    algorithms = [
        ("NWOA",            NWOA_Base),
        ("NWOA-HC_Full",    NWOA_HC_Full),
        ("NWOA-HC_Freq50",  NWOA_HC_FreqLow),
        ("NWOA-HC_Cand1",   NWOA_HC_Cand1),
    ]
    suites = [build_ablation_suite(DIM)]
    run_experiment(
        algorithms=algorithms, benchmark_suites=suites,
        output_dir="Hybrid/Ablation_Study_HC",
        pop_size=30, max_iter=1000, dim=DIM, n_runs=30,
        run_engineering=False,
    )
