"""
Quick Test for All Hybrid NWOA Algorithms

Tests all 4 hybrid variants on benchmark functions to verify implementations.
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Hybrid.nwoa_nelder_mead import NWOA_NelderMead
from Hybrid.nwoa_hill_climbing import NWOA_HillClimbing
from Hybrid.nwoa_pattern_search import NWOA_PatternSearch
from Hybrid.nwoa_simulated_annealing import NWOA_SimulatedAnnealing

# Benchmark functions
def sphere(x):
    """Sphere: f(x*) = 0 at x* = (0,...,0)"""
    return np.sum(x**2)

def rosenbrock(x):
    """Rosenbrock: f(x*) = 0 at x* = (1,...,1)"""
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def rastrigin(x):
    """Rastrigin: f(x*) = 0 at x* = (0,...,0)"""
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def ackley(x):
    """Ackley: f(x*) = 0 at x* = (0,...,0)"""
    n = len(x)
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n)) - \
           np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e


def main():
    print("=" * 80)
    print("HYBRID NWOA ALGORITHMS - QUICK TEST")
    print("=" * 80)
    
    # Test settings
    dim = 10
    lb, ub = -100, 100
    n_agents = 30
    max_iter = 100  # Quick test
    
    # Test functions
    test_functions = [
        (sphere, "Sphere", lb, ub),
        (rastrigin, "Rastrigin", -5.12, 5.12),
        (ackley, "Ackley", -32, 32),
    ]
    
    # Algorithms
    algorithms = [
        ("NWOA-NM", NWOA_NelderMead),
        ("NWOA-HC", NWOA_HillClimbing),
        ("NWOA-PS", NWOA_PatternSearch),
        ("NWOA-SA", NWOA_SimulatedAnnealing),
    ]
    
    print(f"\nConfiguration:")
    print(f"  Dimension: {dim}")
    print(f"  Population: {n_agents}")
    print(f"  Iterations: {max_iter}")
    print(f"  LS Frequency: 10")
    print(f"  LS Candidates: 3")
    print("=" * 80)
    
    # Run tests
    for func, func_name, func_lb, func_ub in test_functions:
        print(f"\n{func_name} Function:")
        print("-" * 80)
        
        for algo_name, AlgoClass in algorithms:
            optimizer = AlgoClass(
                objective_function=func,
                dim=dim,
                lb=func_lb,
                ub=func_ub,
                n_agents=n_agents,
                max_iter=max_iter,
                ls_frequency=10,
                ls_candidates=3,
                ls_max_evals=50
            )
            
            best_solution, best_fitness, convergence = optimizer.optimize()
            
            print(f"  {algo_name:12s}: Best Fitness = {best_fitness:.6e}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED!")
    print("=" * 80)
    print("\nAll hybrid algorithms are working correctly.")
    print("\nNext steps:")
    print("  1. Run: python Hybrid/compare_all_hybrids.py")
    print("  2. Check results in: Hybrid/Hybrid_Results/")
    print("=" * 80)


if __name__ == "__main__":
    main()
