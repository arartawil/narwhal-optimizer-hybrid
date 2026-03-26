"""
CEC2022 Benchmark Runner for NWOA Variants (using heurilab runner)
===================================================================
Uses heurilab v2.0.3 run_experiment() for real-time CSV output,
convergence plots, box plots, Excel reports, and Wilcoxon/Friedman tests.

Outputs saved to: Hybrid/CEC2022_Results/
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heurilab import BenchmarkConfig, BenchmarkSuite, run_experiment
from opfunu.cec_based import cec2022

from narwhal_optimizer import NarwhalOptimizer
from Hybrid.nwoa_nelder_mead import NWOA_NelderMead
from Hybrid.nwoa_hill_climbing import NWOA_HillClimbing
from Hybrid.nwoa_pattern_search import NWOA_PatternSearch
from Hybrid.nwoa_simulated_annealing import NWOA_SimulatedAnnealing


# ─────────────────────────────────────────────
#  Wrappers: adapt NWOA classes to heurilab interface
#  heurilab expects: __init__(pop_size, dim, lb, ub, max_iter, obj_func)
#                    optimize() -> (best_sol, best_fit, convergence)
# ─────────────────────────────────────────────

class NWOA_Wrapper:
    """Wrapper for base NarwhalOptimizer."""
    def __init__(self, pop_size, dim, lb, ub, max_iter, obj_func):
        self.algo = NarwhalOptimizer(
            objective_function=obj_func,
            dim=dim, lb=lb, ub=ub,
            n_agents=pop_size, max_iter=max_iter,
        )
    def optimize(self):
        return self.algo.optimize()


class NWOA_NM_Wrapper:
    """Wrapper for NWOA + Nelder-Mead."""
    def __init__(self, pop_size, dim, lb, ub, max_iter, obj_func):
        self.algo = NWOA_NelderMead(
            objective_function=obj_func,
            dim=dim, lb=lb, ub=ub,
            n_agents=pop_size, max_iter=max_iter,
            ls_frequency=10, ls_candidates=3, ls_max_evals=50,
        )
    def optimize(self):
        return self.algo.optimize()


class NWOA_HC_Wrapper:
    """Wrapper for NWOA + Hill Climbing."""
    def __init__(self, pop_size, dim, lb, ub, max_iter, obj_func):
        self.algo = NWOA_HillClimbing(
            objective_function=obj_func,
            dim=dim, lb=lb, ub=ub,
            n_agents=pop_size, max_iter=max_iter,
            ls_frequency=10, ls_candidates=3, ls_max_evals=50,
        )
    def optimize(self):
        return self.algo.optimize()


class NWOA_PS_Wrapper:
    """Wrapper for NWOA + Pattern Search."""
    def __init__(self, pop_size, dim, lb, ub, max_iter, obj_func):
        self.algo = NWOA_PatternSearch(
            objective_function=obj_func,
            dim=dim, lb=lb, ub=ub,
            n_agents=pop_size, max_iter=max_iter,
            ls_frequency=10, ls_candidates=3, ls_max_evals=50,
        )
    def optimize(self):
        return self.algo.optimize()


class NWOA_SA_Wrapper:
    """Wrapper for NWOA + Simulated Annealing."""
    def __init__(self, pop_size, dim, lb, ub, max_iter, obj_func):
        self.algo = NWOA_SimulatedAnnealing(
            objective_function=obj_func,
            dim=dim, lb=lb, ub=ub,
            n_agents=pop_size, max_iter=max_iter,
            ls_frequency=10, ls_candidates=3, ls_max_evals=50,
        )
    def optimize(self):
        return self.algo.optimize()


# ─────────────────────────────────────────────
#  CEC2022 Benchmark Suite
# ─────────────────────────────────────────────
DIM = 20

CEC2022_FUNC_CLASSES = [
    (cec2022.F12022,  "CEC22_F1"),
    (cec2022.F22022,  "CEC22_F2"),
    (cec2022.F32022,  "CEC22_F3"),
    (cec2022.F42022,  "CEC22_F4"),
    (cec2022.F52022,  "CEC22_F5"),
    (cec2022.F62022,  "CEC22_F6"),
    (cec2022.F72022,  "CEC22_F7"),
    (cec2022.F82022,  "CEC22_F8"),
    (cec2022.F92022,  "CEC22_F9"),
    (cec2022.F102022, "CEC22_F10"),
    (cec2022.F112022, "CEC22_F11"),
    (cec2022.F122022, "CEC22_F12"),
]

# Select which functions to run (change slice to run subset)
SELECTED_FUNCTIONS = [CEC2022_FUNC_CLASSES[7]]  # F8 only


def build_cec2022_suite(dim):
    """Build heurilab BenchmarkSuite from CEC2022 functions."""
    benchmarks = []
    for FuncClass, name in SELECTED_FUNCTIONS:
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
    return BenchmarkSuite(category="CEC2022", benchmarks=benchmarks)


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    algorithms = [
        ("NWOA",    NWOA_Wrapper),
        ("NWOA-NM", NWOA_NM_Wrapper),
        ("NWOA-HC", NWOA_HC_Wrapper),
        ("NWOA-PS", NWOA_PS_Wrapper),
        ("NWOA-SA", NWOA_SA_Wrapper),
    ]

    suites = [build_cec2022_suite(DIM)]

    run_experiment(
        algorithms=algorithms,
        benchmark_suites=suites,
        output_dir="Hybrid/CEC2022_Results",
        pop_size=30,
        max_iter=1000,
        dim=DIM,
        n_runs=30,
        run_engineering=False,
    )
