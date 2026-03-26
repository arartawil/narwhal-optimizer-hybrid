"""
Quick Comparison: Hybrid NWOA vs Original NWOA

Fast comparison on selected benchmark functions to see improvement.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from narwhal_optimizer import NarwhalOptimizer
from Hybrid.nwoa_nelder_mead import NWOA_NelderMead
from Hybrid.nwoa_hill_climbing import NWOA_HillClimbing
from Hybrid.nwoa_pattern_search import NWOA_PatternSearch
from Hybrid.nwoa_simulated_annealing import NWOA_SimulatedAnnealing

sns.set_style("whitegrid")

# Benchmark functions
def sphere(x):
    return np.sum(x**2)

def rosenbrock(x):
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def ackley(x):
    n = len(x)
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n)) - \
           np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e

def griewank(x):
    return 1 + np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))


def run_test(algo_name, AlgoClass, func, dim, lb, ub, runs=10):
    """Run algorithm multiple times and return statistics."""
    results = []
    convergences = []
    
    for run in range(runs):
        if algo_name == "NWOA":
            optimizer = AlgoClass(
                objective_function=func,
                dim=dim,
                lb=lb,
                ub=ub,
                n_agents=30,
                max_iter=200
            )
        else:
            optimizer = AlgoClass(
                objective_function=func,
                dim=dim,
                lb=lb,
                ub=ub,
                n_agents=30,
                max_iter=200,
                ls_frequency=10,
                ls_candidates=3,
                ls_max_evals=50
            )
        
        best_solution, best_fitness, convergence = optimizer.optimize()
        results.append(best_fitness)
        if run == 0:
            convergences.append(convergence)
    
    return {
        'mean': np.mean(results),
        'std': np.std(results),
        'min': np.min(results),
        'max': np.max(results),
        'median': np.median(results),
        'convergence': convergences[0] if convergences else None
    }


def main():
    print("=" * 90)
    print("QUICK COMPARISON: HYBRID NWOA vs ORIGINAL NWOA")
    print("=" * 90)
    
    dim = 30
    runs = 10
    
    test_functions = [
        (sphere, "Sphere", -100, 100),
        (rastrigin, "Rastrigin", -5.12, 5.12),
        (ackley, "Ackley", -32, 32),
        (rosenbrock, "Rosenbrock", -30, 30),
        (griewank, "Griewank", -600, 600),
    ]
    
    algorithms = [
        ("NWOA", NarwhalOptimizer),
        ("NWOA-NM", NWOA_NelderMead),
        ("NWOA-HC", NWOA_HillClimbing),
        ("NWOA-PS", NWOA_PatternSearch),
        ("NWOA-SA", NWOA_SimulatedAnnealing),
    ]
    
    print(f"\nConfiguration:")
    print(f"  Dimension: {dim}")
    print(f"  Runs per function: {runs}")
    print(f"  Population: 30")
    print(f"  Iterations: 200")
    print("=" * 90)
    
    all_results = []
    convergence_data = {}
    
    # Run tests
    for func, func_name, lb, ub in test_functions:
        print(f"\n{func_name} Function [{lb}, {ub}]:")
        print("-" * 90)
        
        for algo_name, AlgoClass in algorithms:
            print(f"  Running {algo_name}...", end=' ')
            
            stats = run_test(algo_name, AlgoClass, func, dim, lb, ub, runs)
            
            all_results.append({
                'Algorithm': algo_name,
                'Function': func_name,
                'Mean': stats['mean'],
                'Std': stats['std'],
                'Median': stats['median'],
                'Min': stats['min'],
                'Max': stats['max']
            })
            
            convergence_data[f"{algo_name}_{func_name}"] = stats['convergence']
            
            print(f"Mean: {stats['mean']:.6e}, Std: {stats['std']:.6e}")
    
    # Create results DataFrame
    df = pd.DataFrame(all_results)
    
    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    print(df.to_string(index=False))
    
    # Calculate improvement over NWOA
    print("\n" + "=" * 90)
    print("IMPROVEMENT OVER ORIGINAL NWOA (%)")
    print("=" * 90)
    
    for func_name in [f[1] for f in test_functions]:
        print(f"\n{func_name}:")
        nwoa_mean = df[(df['Algorithm'] == 'NWOA') & (df['Function'] == func_name)]['Mean'].values[0]
        
        for algo_name in ["NWOA-NM", "NWOA-HC", "NWOA-PS", "NWOA-SA"]:
            algo_mean = df[(df['Algorithm'] == algo_name) & (df['Function'] == func_name)]['Mean'].values[0]
            improvement = ((nwoa_mean - algo_mean) / nwoa_mean) * 100
            
            if improvement > 0:
                print(f"  {algo_name:12s}: +{improvement:6.2f}% better")
            else:
                print(f"  {algo_name:12s}: {improvement:6.2f}% worse")
    
    # Save results
    output_dir = "Hybrid/Hybrid_Results"
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f"{output_dir}/quick_comparison.csv", index=False)
    
    # Generate plots
    print("\n" + "=" * 90)
    print("GENERATING PLOTS...")
    print("=" * 90)
    
    # 1. Performance comparison bar plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, (func, func_name, lb, ub) in enumerate(test_functions):
        func_data = df[df['Function'] == func_name]
        
        ax = axes[idx]
        x = np.arange(len(func_data))
        
        bars = ax.bar(x, func_data['Mean'], yerr=func_data['Std'], 
                      capsize=5, alpha=0.7, edgecolor='black')
        
        # Color bars
        colors = ['blue', 'green', 'orange', 'red', 'purple']
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xlabel('Algorithm', fontsize=10)
        ax.set_ylabel('Mean Best Fitness', fontsize=10)
        ax.set_title(f'{func_name}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(func_data['Algorithm'], rotation=45, ha='right')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Remove extra subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/quick_comparison_bars.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Convergence curves
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    algo_colors = {'NWOA': 'blue', 'NWOA-NM': 'green', 'NWOA-HC': 'orange', 
                   'NWOA-PS': 'red', 'NWOA-SA': 'purple'}
    
    for idx, (func, func_name, lb, ub) in enumerate(test_functions):
        ax = axes[idx]
        
        for algo_name, _ in algorithms:
            key = f"{algo_name}_{func_name}"
            if key in convergence_data and convergence_data[key] is not None:
                ax.plot(convergence_data[key], label=algo_name, 
                       linewidth=2, color=algo_colors[algo_name])
        
        ax.set_xlabel('Iteration', fontsize=10)
        ax.set_ylabel('Best Fitness', fontsize=10)
        ax.set_title(f'{func_name} - Convergence', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/quick_comparison_convergence.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Overall ranking
    rankings = []
    for func_name in [f[1] for f in test_functions]:
        func_data = df[df['Function'] == func_name].sort_values('Mean')
        for rank, row in enumerate(func_data.iterrows(), 1):
            rankings.append({'Algorithm': row[1]['Algorithm'], 'Rank': rank})
    
    rank_df = pd.DataFrame(rankings)
    avg_ranks = rank_df.groupby('Algorithm')['Rank'].mean().sort_values()
    
    plt.figure(figsize=(10, 6))
    colors_list = [algo_colors[algo] for algo in avg_ranks.index]
    bars = plt.barh(avg_ranks.index, avg_ranks.values, color=colors_list, alpha=0.7, edgecolor='black')
    plt.xlabel('Average Rank (Lower is Better)', fontsize=12)
    plt.ylabel('Algorithm', fontsize=12)
    plt.title('Overall Performance Ranking', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (algo, rank) in enumerate(avg_ranks.items()):
        plt.text(rank + 0.05, i, f'{rank:.2f}', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/quick_comparison_ranking.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n" + "=" * 90)
    print("OVERALL RANKING (Average Rank Across All Functions)")
    print("=" * 90)
    for algo, rank in avg_ranks.items():
        print(f"  {algo:12s}: {rank:.2f}")
    
    print("\n" + "=" * 90)
    print(f"Results saved to: {output_dir}/")
    print("  - quick_comparison.csv")
    print("  - quick_comparison_bars.png")
    print("  - quick_comparison_convergence.png")
    print("  - quick_comparison_ranking.png")
    print("=" * 90)
    print("\nCOMPARISON COMPLETED!")
    print("=" * 90)


if __name__ == "__main__":
    main()
