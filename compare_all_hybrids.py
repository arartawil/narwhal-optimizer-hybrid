"""
Compare All Hybrid NWOA Algorithms on CEC2017 Benchmark

Comprehensive comparison of:
- NWOA (original)
- NWOA-Nelder-Mead
- NWOA-Hill-Climbing
- NWOA-Pattern-Search
- NWOA-Simulated-Annealing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from narwhal_optimizer import NarwhalOptimizer
from Hybrid.nwoa_nelder_mead import NWOA_NelderMead
from Hybrid.nwoa_hill_climbing import NWOA_HillClimbing
from Hybrid.nwoa_pattern_search import NWOA_PatternSearch
from Hybrid.nwoa_simulated_annealing import NWOA_SimulatedAnnealing
from cec2017_functions import CEC2017

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Configuration
CEC_DIM = 10
CEC_RUNS = 30
CEC_MAX_FES = 150 * 30  # 150 iterations × 30 population
FUNCTIONS = list(range(1, 31))  # F1-F30 (all CEC2017 functions)


def run_algorithm(algorithm_name, func_num, run_num):
    """Run a single optimization."""
    lb, ub = -100, 100
    
    cec = CEC2017(dim=CEC_DIM)
    
    def objective(x):
        return cec(x, func_num)
    
    n_agents = 30
    max_iter = CEC_MAX_FES // n_agents
    
    # Select algorithm
    if algorithm_name == "NWOA":
        optimizer = NarwhalOptimizer(
            objective_function=objective, dim=CEC_DIM, lb=lb, ub=ub,
            n_agents=n_agents, max_iter=max_iter
        )
    elif algorithm_name == "NWOA-NM":
        optimizer = NWOA_NelderMead(
            objective_function=objective, dim=CEC_DIM, lb=lb, ub=ub,
            n_agents=n_agents, max_iter=max_iter,
            ls_frequency=10, ls_candidates=3, ls_max_evals=50
        )
    elif algorithm_name == "NWOA-HC":
        optimizer = NWOA_HillClimbing(
            objective_function=objective, dim=CEC_DIM, lb=lb, ub=ub,
            n_agents=n_agents, max_iter=max_iter,
            ls_frequency=10, ls_candidates=3, ls_max_evals=50
        )
    elif algorithm_name == "NWOA-PS":
        optimizer = NWOA_PatternSearch(
            objective_function=objective, dim=CEC_DIM, lb=lb, ub=ub,
            n_agents=n_agents, max_iter=max_iter,
            ls_frequency=10, ls_candidates=3, ls_max_evals=50
        )
    elif algorithm_name == "NWOA-SA":
        optimizer = NWOA_SimulatedAnnealing(
            objective_function=objective, dim=CEC_DIM, lb=lb, ub=ub,
            n_agents=n_agents, max_iter=max_iter,
            ls_frequency=10, ls_candidates=3, ls_max_evals=50
        )
    
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
    print("ALL HYBRID NWOA ALGORITHMS - CEC2017 COMPREHENSIVE COMPARISON")
    print("=" * 80)
    print(f"Dimension: {CEC_DIM}")
    print(f"Max FEs: {CEC_MAX_FES:,}")
    print(f"Runs per function: {CEC_RUNS}")
    print(f"Functions: F1-F10")
    print("=" * 80)
    
    algorithms = ["NWOA", "NWOA-NM", "NWOA-HC", "NWOA-PS", "NWOA-SA"]
    
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
    
    # Save results
    output_dir = "Hybrid/Hybrid_Results"
    os.makedirs(f"{output_dir}/BoxPlots", exist_ok=True)
    os.makedirs(f"{output_dir}/Convergence", exist_ok=True)
    
    df.to_csv(f"{output_dir}/detailed_results.csv", index=False)
    stats_df.to_csv(f"{output_dir}/summary_statistics.csv", index=False)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(stats_df.to_string(index=False))
    
    # Statistical tests
    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISON (vs Original NWOA)")
    print("=" * 80)
    
    nwoa_results = df[df['algorithm'] == 'NWOA']
    
    for algo in ["NWOA-NM", "NWOA-HC", "NWOA-PS", "NWOA-SA"]:
        algo_results = df[df['algorithm'] == algo]
        wins, losses, ties = 0, 0, 0
        
        for func_num in FUNCTIONS:
            nwoa_func = nwoa_results[nwoa_results['function'] == f'F{func_num}']['best_fitness'].values
            algo_func = algo_results[algo_results['function'] == f'F{func_num}']['best_fitness'].values
            
            try:
                stat, p_value = stats.wilcoxon(nwoa_func, algo_func)
                if p_value < 0.05:
                    if np.median(algo_func) < np.median(nwoa_func):
                        wins += 1
                    else:
                        losses += 1
                else:
                    ties += 1
            except:
                ties += 1
        
        print(f"\n{algo}: Wins={wins}, Losses={losses}, Ties={ties}")
    
    # Generate plots
    print("\n" + "=" * 80)
    print("GENERATING PLOTS...")
    print("=" * 80)
    
    # Box plots for each function
    for func_num in FUNCTIONS:
        func_data = df[df['function'] == f'F{func_num}']
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=func_data, x='algorithm', y='best_fitness', order=algorithms)
        plt.title(f'F{func_num} - Fitness Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Algorithm', fontsize=12)
        plt.ylabel('Best Fitness', fontsize=12)
        plt.yscale('log')
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/BoxPlots/F{func_num}_boxplot.png", dpi=300)
        plt.close()
    
    # Convergence curves
    for func_num in FUNCTIONS:
        plt.figure(figsize=(12, 6))
        
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
    
    # Overall performance heatmap
    pivot_mean = stats_df.pivot(index='function', columns='algorithm', values='mean')
    pivot_mean = pivot_mean[algorithms]  # Reorder columns
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(np.log10(pivot_mean), annot=True, fmt='.2f', cmap='YlOrRd', 
                xticklabels=algorithms, yticklabels=[f'F{i}' for i in FUNCTIONS])
    plt.title('Mean Performance (log10 scale)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_heatmap.png", dpi=300)
    plt.close()
    
    # Ranking analysis
    print("\n" + "=" * 80)
    print("OVERALL RANKING (Average Rank Across All Functions)")
    print("=" * 80)
    
    rankings = []
    for func_num in FUNCTIONS:
        func_stats = stats_df[stats_df['function'] == f'F{func_num}'].sort_values('mean')
        for rank, (idx, row) in enumerate(func_stats.iterrows(), 1):
            rankings.append({'algorithm': row['algorithm'], 'function': f'F{func_num}', 'rank': rank})
    
    rank_df = pd.DataFrame(rankings)
    avg_ranks = rank_df.groupby('algorithm')['rank'].mean().sort_values()
    
    print("\nAlgorithm Rankings (lower is better):")
    for algo, avg_rank in avg_ranks.items():
        print(f"  {algo:12s}: {avg_rank:.2f}")
    
    # Save summary report
    with open(f"{output_dir}/Comparison_Summary.txt", 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ALL HYBRID NWOA ALGORITHMS - COMPREHENSIVE COMPARISON\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dimension: {CEC_DIM}\n")
        f.write(f"Max FEs: {CEC_MAX_FES:,}\n")
        f.write(f"Runs: {CEC_RUNS}\n")
        f.write(f"Functions: F1-F10\n\n")
        f.write("=" * 80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("=" * 80 + "\n")
        f.write(stats_df.to_string(index=False))
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("AVERAGE RANKINGS\n")
        f.write("=" * 80 + "\n")
        for algo, avg_rank in avg_ranks.items():
            f.write(f"{algo:12s}: {avg_rank:.2f}\n")
    
    print(f"\nResults saved to: {output_dir}/")
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
