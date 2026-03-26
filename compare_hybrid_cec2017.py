"""
Compare NWOA-LocalSearch Hybrid vs Original NWOA on CEC2017 Benchmark

This script compares the performance of the hybrid NWOA-LS algorithm
against the original NWOA on CEC2017 test functions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from narwhal_optimizer import NarwhalOptimizer
from Hybrid.nwoa_local_search import NWOA_LocalSearch
from cec2017_functions import cec17_test_func

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# CEC2017 Configuration
CEC_DIM = 10
CEC_RUNS = 30
CEC_MAX_FES = 10000 * CEC_DIM  # Standard CEC2017

# Functions to test (CEC2017 F1-F10)
FUNCTIONS = list(range(1, 11))

def run_algorithm(algorithm_name, func_num, run_num):
    """Run a single optimization and return results."""
    
    # Define bounds (CEC2017 standard)
    lb, ub = -100, 100
    
    # Create objective function
    def objective(x):
        return cec17_test_func(x, func_num, CEC_DIM)
    
    # Calculate population and iterations based on max FEs
    n_agents = 30
    max_iter = CEC_MAX_FES // n_agents
    
    if algorithm_name == "NWOA":
        optimizer = NarwhalOptimizer(
            objective_function=objective,
            dim=CEC_DIM,
            lb=lb,
            ub=ub,
            n_agents=n_agents,
            max_iter=max_iter
        )
    elif algorithm_name == "NWOA-LS-NM":
        optimizer = NWOA_LocalSearch(
            objective_function=objective,
            dim=CEC_DIM,
            lb=lb,
            ub=ub,
            n_agents=n_agents,
            max_iter=max_iter,
            local_search_method='nelder-mead',
            ls_frequency=10,
            ls_candidates=3,
            ls_max_evals=50
        )
    elif algorithm_name == "NWOA-LS-HC":
        optimizer = NWOA_LocalSearch(
            objective_function=objective,
            dim=CEC_DIM,
            lb=lb,
            ub=ub,
            n_agents=n_agents,
            max_iter=max_iter,
            local_search_method='hill-climbing',
            ls_frequency=10,
            ls_candidates=3,
            ls_max_evals=50
        )
    elif algorithm_name == "NWOA-LS-PS":
        optimizer = NWOA_LocalSearch(
            objective_function=objective,
            dim=CEC_DIM,
            lb=lb,
            ub=ub,
            n_agents=n_agents,
            max_iter=max_iter,
            local_search_method='pattern-search',
            ls_frequency=10,
            ls_candidates=3,
            ls_max_evals=50
        )
    
    # Optimize
    best_solution, best_fitness, convergence = optimizer.optimize()
    
    return {
        'algorithm': algorithm_name,
        'function': f'F{func_num}',
        'run': run_num,
        'best_fitness': best_fitness,
        'convergence': convergence
    }


def main():
    print("=" * 80)
    print("NWOA-LocalSearch vs Original NWOA - CEC2017 Benchmark Comparison")
    print("=" * 80)
    print(f"Dimension: {CEC_DIM}")
    print(f"Max FEs: {CEC_MAX_FES:,}")
    print(f"Runs per function: {CEC_RUNS}")
    print(f"Functions: F1-F10")
    print("=" * 80)
    
    # Algorithms to compare
    algorithms = [
        "NWOA",
        "NWOA-LS-NM",    # Nelder-Mead
        "NWOA-LS-HC",    # Hill Climbing
        "NWOA-LS-PS"     # Pattern Search
    ]
    
    # Store all results
    all_results = []
    convergence_data = {}
    
    # Run experiments
    for func_num in FUNCTIONS:
        print(f"\nTesting F{func_num}...")
        
        for algo in algorithms:
            print(f"  Running {algo}...")
            
            for run in range(CEC_RUNS):
                result = run_algorithm(algo, func_num, run)
                all_results.append(result)
                
                # Store convergence for first run
                if run == 0:
                    key = f"{algo}_F{func_num}"
                    convergence_data[key] = result['convergence']
                
                if (run + 1) % 10 == 0:
                    print(f"    Completed {run + 1}/{CEC_RUNS} runs")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Calculate statistics
    stats_df = df.groupby(['algorithm', 'function'])['best_fitness'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('median', 'median'),
        ('min', 'min'),
        ('max', 'max')
    ]).reset_index()
    
    # Save detailed results
    output_dir = "Hybrid/Hybrid_Results"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/BoxPlots", exist_ok=True)
    os.makedirs(f"{output_dir}/Convergence", exist_ok=True)
    
    df.to_csv(f"{output_dir}/detailed_results.csv", index=False)
    stats_df.to_csv(f"{output_dir}/summary_statistics.csv", index=False)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(stats_df.to_string(index=False))
    
    # Statistical comparison (Wilcoxon signed-rank test)
    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISON (Wilcoxon Signed-Rank Test)")
    print("=" * 80)
    
    nwoa_results = df[df['algorithm'] == 'NWOA']
    
    for algo in ["NWOA-LS-NM", "NWOA-LS-HC", "NWOA-LS-PS"]:
        algo_results = df[df['algorithm'] == algo]
        
        wins = 0
        losses = 0
        ties = 0
        p_values = []
        
        for func_num in FUNCTIONS:
            nwoa_func = nwoa_results[nwoa_results['function'] == f'F{func_num}']['best_fitness'].values
            algo_func = algo_results[algo_results['function'] == f'F{func_num}']['best_fitness'].values
            
            # Wilcoxon test
            try:
                stat, p_value = stats.wilcoxon(nwoa_func, algo_func)
                p_values.append(p_value)
                
                # Count wins/losses (α = 0.05)
                if p_value < 0.05:
                    if np.median(algo_func) < np.median(nwoa_func):
                        wins += 1
                    else:
                        losses += 1
                else:
                    ties += 1
            except:
                ties += 1
        
        print(f"\n{algo} vs NWOA:")
        print(f"  Wins: {wins}, Losses: {losses}, Ties: {ties}")
        print(f"  Avg p-value: {np.mean(p_values):.4f}")
    
    # Generate plots
    print("\n" + "=" * 80)
    print("GENERATING PLOTS...")
    print("=" * 80)
    
    # 1. Box plots for each function
    for func_num in FUNCTIONS:
        func_data = df[df['function'] == f'F{func_num}']
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=func_data, x='algorithm', y='best_fitness')
        plt.title(f'F{func_num} - Fitness Distribution Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Algorithm', fontsize=12)
        plt.ylabel('Best Fitness', fontsize=12)
        plt.yscale('log')
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/BoxPlots/F{func_num}_boxplot.png", dpi=300)
        plt.close()
    
    # 2. Convergence curves
    for func_num in FUNCTIONS:
        plt.figure(figsize=(10, 6))
        
        for algo in algorithms:
            key = f"{algo}_F{func_num}"
            if key in convergence_data:
                plt.plot(convergence_data[key], label=algo, linewidth=2)
        
        plt.title(f'F{func_num} - Convergence Curves', fontsize=14, fontweight='bold')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Best Fitness', fontsize=12)
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/Convergence/F{func_num}_convergence.png", dpi=300)
        plt.close()
    
    # 3. Overall performance comparison
    plt.figure(figsize=(14, 8))
    pivot_mean = stats_df.pivot(index='function', columns='algorithm', values='mean')
    
    x = np.arange(len(FUNCTIONS))
    width = 0.2
    
    for i, algo in enumerate(algorithms):
        plt.bar(x + i * width, pivot_mean[algo], width, label=algo)
    
    plt.xlabel('Function', fontsize=12)
    plt.ylabel('Mean Best Fitness', fontsize=12)
    plt.title('Mean Performance Comparison Across All Functions', fontsize=14, fontweight='bold')
    plt.xticks(x + width * 1.5, [f'F{i}' for i in FUNCTIONS])
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/overall_comparison.png", dpi=300)
    plt.close()
    
    # Generate summary report
    with open(f"{output_dir}/Comparison_Summary.txt", 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("NWOA-LocalSearch vs Original NWOA - CEC2017 Comparison Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dimension: {CEC_DIM}\n")
        f.write(f"Max FEs: {CEC_MAX_FES:,}\n")
        f.write(f"Runs per function: {CEC_RUNS}\n")
        f.write(f"Functions tested: F1-F10\n\n")
        f.write("=" * 80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("=" * 80 + "\n")
        f.write(stats_df.to_string(index=False))
        f.write("\n\n")
        
        # Add rankings
        f.write("=" * 80 + "\n")
        f.write("ALGORITHM RANKINGS (By Mean Performance)\n")
        f.write("=" * 80 + "\n")
        
        for func_num in FUNCTIONS:
            func_stats = stats_df[stats_df['function'] == f'F{func_num}'].sort_values('mean')
            f.write(f"\nF{func_num}:\n")
            for idx, row in func_stats.iterrows():
                f.write(f"  {row['algorithm']:15s}: {row['mean']:.6e}\n")
    
    print("\nResults saved to:", output_dir)
    print("\nComparison completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
