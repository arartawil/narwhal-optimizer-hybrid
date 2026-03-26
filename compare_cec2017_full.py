"""
Comprehensive Comparison: Hybrid NWOA vs Original NWOA on CEC2017

Full CEC2017 benchmark with 30 runs per function.
Generates box plots and convergence curves.
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

# Configuration
CEC_DIM = 10
CEC_RUNS = 30
N_AGENTS = 30
MAX_ITER = 50
FUNCTIONS = list(range(1, 31))  # All CEC2017 functions F1-F30


def run_algorithm(algo_name, AlgoClass, func_num, run_num):
    """Run a single optimization."""
    lb, ub = -100, 100
    
    # Create CEC2017 instance
    cec = CEC2017(dim=CEC_DIM)
    
    def objective(x):
        return cec(x, func_num)
    
    if algo_name == "NWOA":
        optimizer = AlgoClass(
            objective_function=objective,
            dim=CEC_DIM,
            lb=lb,
            ub=ub,
            n_agents=N_AGENTS,
            max_iter=MAX_ITER
        )
    else:
        optimizer = AlgoClass(
            objective_function=objective,
            dim=CEC_DIM,
            lb=lb,
            ub=ub,
            n_agents=N_AGENTS,
            max_iter=MAX_ITER,
            ls_frequency=10,
            ls_candidates=3,
            ls_max_evals=50
        )
    
    best_solution, best_fitness, convergence = optimizer.optimize()
    
    return {
        'algorithm': algo_name,
        'function': f'F{func_num}',
        'run': run_num,
        'best_fitness': best_fitness,
        'convergence': convergence
    }


def main():
    print("=" * 90)
    print("COMPREHENSIVE CEC2017 COMPARISON: HYBRID NWOA vs ORIGINAL NWOA")
    print("=" * 90)
    print(f"Dimension: {CEC_DIM}")
    print(f"Population Size: {N_AGENTS}")
    print(f"Iterations: {MAX_ITER}")
    print(f"Runs per function: {CEC_RUNS}")
    print(f"Functions: F1-F30 (All CEC2017)")
    print("=" * 90)
    
    algorithms = [
        ("NWOA", NarwhalOptimizer),
        ("NWOA-NM", NWOA_NelderMead),
        ("NWOA-HC", NWOA_HillClimbing),
        ("NWOA-PS", NWOA_PatternSearch),
        ("NWOA-SA", NWOA_SimulatedAnnealing),
    ]
    
    all_results = []
    convergence_data = {}
    
    # Run experiments
    total_experiments = len(FUNCTIONS) * len(algorithms) * CEC_RUNS
    current_exp = 0
    
    for func_num in FUNCTIONS:
        print(f"\n{'='*90}")
        print(f"Testing F{func_num}...")
        print(f"{'='*90}")
        
        for algo_name, AlgoClass in algorithms:
            print(f"  Running {algo_name}...", end=' ', flush=True)
            
            for run in range(CEC_RUNS):
                result = run_algorithm(algo_name, AlgoClass, func_num, run)
                all_results.append(result)
                
                # Store first run convergence
                if run == 0:
                    key = f"{algo_name}_F{func_num}"
                    convergence_data[key] = result['convergence']
                
                current_exp += 1
                if (run + 1) % 10 == 0:
                    progress = (current_exp / total_experiments) * 100
                    print(f"{run+1}/{CEC_RUNS}", end=' ', flush=True)
            
            print("✓")
    
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
    output_dir = "Hybrid/Hybrid_Results/CEC2017_Full"
    os.makedirs(f"{output_dir}/BoxPlots", exist_ok=True)
    os.makedirs(f"{output_dir}/Convergence", exist_ok=True)
    
    df.to_csv(f"{output_dir}/detailed_results.csv", index=False)
    stats_df.to_csv(f"{output_dir}/summary_statistics.csv", index=False)
    
    print("\n" + "=" * 90)
    print("SUMMARY STATISTICS (First 10 Functions)")
    print("=" * 90)
    print(stats_df[stats_df['function'].isin([f'F{i}' for i in range(1, 11)])].to_string(index=False))
    
    # Statistical comparison
    print("\n" + "=" * 90)
    print("STATISTICAL COMPARISON (Wilcoxon Signed-Rank Test vs NWOA)")
    print("=" * 90)
    
    nwoa_results = df[df['algorithm'] == 'NWOA']
    comparison_summary = []
    
    for algo_name in ["NWOA-NM", "NWOA-HC", "NWOA-PS", "NWOA-SA"]:
        algo_results = df[df['algorithm'] == algo_name]
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
        
        comparison_summary.append({
            'Algorithm': algo_name,
            'Wins': wins,
            'Losses': losses,
            'Ties': ties
        })
        print(f"{algo_name}: Wins={wins}, Losses={losses}, Ties={ties}")
    
    # Generate plots
    print("\n" + "=" * 90)
    print("GENERATING BOX PLOTS...")
    print("=" * 90)
    
    # Box plots for each function
    for func_num in FUNCTIONS:
        func_data = df[df['function'] == f'F{func_num}']
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=func_data, x='algorithm', y='best_fitness', 
                   order=[a[0] for a in algorithms])
        plt.title(f'CEC2017 F{func_num} - Fitness Distribution (30 Runs)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Algorithm', fontsize=12)
        plt.ylabel('Best Fitness', fontsize=12)
        plt.yscale('log')
        plt.xticks(rotation=15)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/BoxPlots/F{func_num}_boxplot.png", dpi=300)
        plt.close()
        
        if func_num % 5 == 0:
            print(f"  Generated box plots for F1-F{func_num}")
    
    print("\n" + "=" * 90)
    print("GENERATING CONVERGENCE CURVES...")
    print("=" * 90)
    
    # Convergence curves
    algo_colors = {
        'NWOA': '#1f77b4',
        'NWOA-NM': '#2ca02c',
        'NWOA-HC': '#ff7f0e',
        'NWOA-PS': '#d62728',
        'NWOA-SA': '#9467bd'
    }
    
    for func_num in FUNCTIONS:
        plt.figure(figsize=(12, 6))
        
        for algo_name, _ in algorithms:
            key = f"{algo_name}_F{func_num}"
            if key in convergence_data:
                plt.plot(convergence_data[key], label=algo_name, 
                        linewidth=2.5, color=algo_colors[algo_name], alpha=0.8)
        
        plt.title(f'CEC2017 F{func_num} - Convergence Curves', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Best Fitness (log scale)', fontsize=12)
        plt.yscale('log')
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/Convergence/F{func_num}_convergence.png", dpi=300)
        plt.close()
        
        if func_num % 5 == 0:
            print(f"  Generated convergence curves for F1-F{func_num}")
    
    # Overall ranking
    print("\n" + "=" * 90)
    print("CALCULATING OVERALL RANKINGS...")
    print("=" * 90)
    
    rankings = []
    for func_num in FUNCTIONS:
        func_stats = stats_df[stats_df['function'] == f'F{func_num}'].sort_values('mean')
        for rank, (idx, row) in enumerate(func_stats.iterrows(), 1):
            rankings.append({'Algorithm': row['algorithm'], 'Function': f'F{func_num}', 'Rank': rank})
    
    rank_df = pd.DataFrame(rankings)
    avg_ranks = rank_df.groupby('Algorithm')['Rank'].mean().sort_values()
    
    plt.figure(figsize=(10, 6))
    colors_list = [algo_colors[algo] for algo in avg_ranks.index]
    bars = plt.barh(avg_ranks.index, avg_ranks.values, color=colors_list, 
                   alpha=0.7, edgecolor='black', linewidth=2)
    plt.xlabel('Average Rank (Lower is Better)', fontsize=12, fontweight='bold')
    plt.ylabel('Algorithm', fontsize=12, fontweight='bold')
    plt.title('Overall Performance Ranking Across All CEC2017 Functions', 
             fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (algo, rank) in enumerate(avg_ranks.items()):
        plt.text(rank + 0.05, i, f'{rank:.2f}', va='center', 
                fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/overall_ranking.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Summary comparison bar chart
    fig, axes = plt.subplots(5, 6, figsize=(24, 18))
    axes = axes.flatten()
    
    for idx, func_num in enumerate(FUNCTIONS):
        func_data = stats_df[stats_df['function'] == f'F{func_num}']
        ax = axes[idx]
        
        x = np.arange(len(func_data))
        colors = [algo_colors[algo] for algo in func_data['algorithm']]
        
        bars = ax.bar(x, func_data['mean'], yerr=func_data['std'], 
                     capsize=3, alpha=0.7, edgecolor='black', color=colors)
        
        ax.set_title(f'F{func_num}', fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(func_data['algorithm'], rotation=45, ha='right', fontsize=7)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Mean Performance Comparison - All CEC2017 Functions (30 Runs)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/all_functions_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Improvement percentage heatmap
    improvement_data = []
    for func_num in FUNCTIONS:
        nwoa_mean = stats_df[(stats_df['algorithm'] == 'NWOA') & 
                             (stats_df['function'] == f'F{func_num}')]['mean'].values[0]
        
        for algo_name in ["NWOA-NM", "NWOA-HC", "NWOA-PS", "NWOA-SA"]:
            algo_mean = stats_df[(stats_df['algorithm'] == algo_name) & 
                                (stats_df['function'] == f'F{func_num}')]['mean'].values[0]
            improvement = ((nwoa_mean - algo_mean) / nwoa_mean) * 100
            improvement_data.append({
                'Algorithm': algo_name,
                'Function': f'F{func_num}',
                'Improvement': improvement
            })
    
    improvement_df = pd.DataFrame(improvement_data)
    improvement_pivot = improvement_df.pivot(index='Function', columns='Algorithm', values='Improvement')
    
    plt.figure(figsize=(10, 16))
    sns.heatmap(improvement_pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
               cbar_kws={'label': 'Improvement over NWOA (%)'}, linewidths=0.5)
    plt.title('Improvement over Original NWOA - All CEC2017 Functions (%)', 
             fontsize=14, fontweight='bold')
    plt.xlabel('Hybrid Algorithm', fontsize=12, fontweight='bold')
    plt.ylabel('CEC2017 Function', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/improvement_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate comprehensive report
    with open(f"{output_dir}/Comparison_Report.txt", 'w') as f:
        f.write("=" * 90 + "\n")
        f.write("COMPREHENSIVE CEC2017 COMPARISON: HYBRID NWOA vs ORIGINAL NWOA\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"Dimension: {CEC_DIM}\n")
        f.write(f"Population Size: {N_AGENTS}\n")
        f.write(f"Iterations: {MAX_ITER}\n")
        f.write(f"Runs per function: {CEC_RUNS}\n")
        f.write(f"Functions tested: F1-F30 (All CEC2017)\n\n")
        
        f.write("=" * 90 + "\n")
        f.write("OVERALL RANKINGS (Average Rank)\n")
        f.write("=" * 90 + "\n")
        for algo, rank in avg_ranks.items():
            f.write(f"{algo:12s}: {rank:.2f}\n")
        
        f.write("\n" + "=" * 90 + "\n")
        f.write("STATISTICAL COMPARISON (vs NWOA)\n")
        f.write("=" * 90 + "\n")
        for item in comparison_summary:
            f.write(f"{item['Algorithm']:12s}: Wins={item['Wins']:2d}, "
                   f"Losses={item['Losses']:2d}, Ties={item['Ties']:2d}\n")
        
        f.write("\n" + "=" * 90 + "\n")
        f.write("DETAILED STATISTICS\n")
        f.write("=" * 90 + "\n")
        f.write(stats_df.to_string(index=False))
        
        f.write("\n\n" + "=" * 90 + "\n")
        f.write("AVERAGE IMPROVEMENT OVER NWOA (%)\n")
        f.write("=" * 90 + "\n")
        avg_improvement = improvement_df.groupby('Algorithm')['Improvement'].mean().sort_values(ascending=False)
        for algo, imp in avg_improvement.items():
            f.write(f"{algo:12s}: {imp:+7.2f}%\n")
    
    print("\n" + "=" * 90)
    print("OVERALL RANKING (Average Rank Across All CEC2017 Functions)")
    print("=" * 90)
    for algo, rank in avg_ranks.items():
        print(f"  {algo:12s}: {rank:.2f}")
    
    print("\n" + "=" * 90)
    print("AVERAGE IMPROVEMENT OVER ORIGINAL NWOA")
    print("=" * 90)
    avg_improvement = improvement_df.groupby('Algorithm')['Improvement'].mean().sort_values(ascending=False)
    for algo, imp in avg_improvement.items():
        print(f"  {algo:12s}: {imp:+7.2f}%")
    
    print("\n" + "=" * 90)
    print(f"ALL RESULTS SAVED TO: {output_dir}/")
    print("=" * 90)
    print("\nGenerated Files:")
    print(f"  - detailed_results.csv (All {len(df)} runs)")
    print(f"  - summary_statistics.csv")
    print(f"  - BoxPlots/ (30 box plots, one per function)")
    print(f"  - Convergence/ (30 convergence curves)")
    print(f"  - overall_ranking.png")
    print(f"  - all_functions_comparison.png")
    print(f"  - improvement_heatmap.png")
    print(f"  - Comparison_Report.txt")
    print("=" * 90)
    print("\nCOMPREHENSIVE COMPARISON COMPLETED!")
    print("=" * 90)


if __name__ == "__main__":
    main()
