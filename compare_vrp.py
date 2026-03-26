"""
Compare Original NWOA vs Hybrid variants on Vehicle Routing Problem (VRP)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time

# Import VRP problem
from vrp_problem import get_benchmark_vrp, create_vrp_objective

# Import algorithms
from narwhal_optimizer import NarwhalOptimizer
from Hybrid.nwoa_nelder_mead import NWOA_NelderMead
from Hybrid.nwoa_hill_climbing import NWOA_HillClimbing
from Hybrid.nwoa_pattern_search import NWOA_PatternSearch
from Hybrid.nwoa_simulated_annealing import NWOA_SimulatedAnnealing

# Configuration
VRP_PROBLEM = 'medium'  # 'small', 'medium', 'large'
N_RUNS = 30
MAX_ITER = 200
POPULATION = 50

plt.rcParams['figure.figsize'] = (12, 8)


def run_algorithm(algorithm_name, vrp, run_num):
    """Run a single optimization on VRP"""
    
    objective = create_vrp_objective(vrp)
    dim = vrp.n_customers
    lb = 0.0
    ub = 1.0
    
    # Algorithm configurations
    if algorithm_name == 'NWOA':
        optimizer = NarwhalOptimizer(
            objective_function=objective,
            dim=dim,
            lb=lb,
            ub=ub,
            n_agents=POPULATION,
            max_iter=MAX_ITER
        )
    elif algorithm_name == 'NWOA-NM':
        optimizer = NWOA_NelderMead(
            objective_function=objective,
            dim=dim,
            lb=lb,
            ub=ub,
            n_agents=POPULATION,
            max_iter=MAX_ITER,
            ls_frequency=10,
            ls_candidates=3,
            ls_max_evals=50
        )
    elif algorithm_name == 'NWOA-HC':
        optimizer = NWOA_HillClimbing(
            objective_function=objective,
            dim=dim,
            lb=lb,
            ub=ub,
            n_agents=POPULATION,
            max_iter=MAX_ITER,
            ls_frequency=10,
            ls_candidates=3,
            ls_max_evals=50
        )
    elif algorithm_name == 'NWOA-PS':
        optimizer = NWOA_PatternSearch(
            objective_function=objective,
            dim=dim,
            lb=lb,
            ub=ub,
            n_agents=POPULATION,
            max_iter=MAX_ITER,
            ls_frequency=10,
            ls_candidates=3,
            ls_max_evals=50
        )
    elif algorithm_name == 'NWOA-SA':
        optimizer = NWOA_SimulatedAnnealing(
            objective_function=objective,
            dim=dim,
            lb=lb,
            ub=ub,
            n_agents=POPULATION,
            max_iter=MAX_ITER,
            ls_frequency=10,
            ls_candidates=3,
            ls_max_evals=50
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    # Run optimization
    start_time = time.time()
    best_solution, best_fitness, convergence = optimizer.optimize()
    elapsed_time = time.time() - start_time
    
    return {
        'solution': best_solution,
        'fitness': best_fitness,
        'convergence': convergence,
        'time': elapsed_time
    }


def main():
    print("=" * 80)
    print("VEHICLE ROUTING PROBLEM - NWOA VARIANTS COMPARISON")
    print("=" * 80)
    
    # Create VRP instance
    vrp = get_benchmark_vrp(VRP_PROBLEM)
    
    print(f"\nVRP Configuration:")
    print(f"  Problem Size: {VRP_PROBLEM.upper()}")
    print(f"  Customers: {vrp.n_customers}")
    print(f"  Vehicles: {vrp.n_vehicles}")
    print(f"  Vehicle Capacity: {vrp.vehicle_capacity}")
    print(f"  Total Demand: {np.sum(vrp.demands)}")
    print(f"\nAlgorithm Settings:")
    print(f"  Population: {POPULATION}")
    print(f"  Max Iterations: {MAX_ITER}")
    print(f"  Independent Runs: {N_RUNS}")
    print("=" * 80)
    
    # Algorithms to compare
    algorithms = ['NWOA', 'NWOA-NM', 'NWOA-HC', 'NWOA-PS', 'NWOA-SA']
    
    # Results storage
    results = {algo: {'fitness': [], 'time': [], 'convergence': [], 'solutions': []} 
               for algo in algorithms}
    
    # Run experiments
    for algo_name in algorithms:
        print(f"\nRunning {algo_name}...")
        
        for run in range(N_RUNS):
            result = run_algorithm(algo_name, vrp, run)
            
            results[algo_name]['fitness'].append(result['fitness'])
            results[algo_name]['time'].append(result['time'])
            results[algo_name]['convergence'].append(result['convergence'])
            results[algo_name]['solutions'].append(result['solution'])
            
            if (run + 1) % 10 == 0:
                print(f"  Completed {run + 1}/{N_RUNS} runs")
        
        print(f"  Completed {N_RUNS}/{N_RUNS} runs")
    
    # Create results directory
    results_dir = "Hybrid/VRP_Results"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f"{results_dir}/BoxPlots", exist_ok=True)
    os.makedirs(f"{results_dir}/Convergence", exist_ok=True)
    os.makedirs(f"{results_dir}/Routes", exist_ok=True)
    
    # Statistical analysis
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    stats_data = []
    for algo in algorithms:
        fitness_values = results[algo]['fitness']
        time_values = results[algo]['time']
        
        stats_data.append({
            'Algorithm': algo,
            'Mean_Distance': np.mean(fitness_values),
            'Std_Distance': np.std(fitness_values),
            'Median_Distance': np.median(fitness_values),
            'Min_Distance': np.min(fitness_values),
            'Max_Distance': np.max(fitness_values),
            'Mean_Time': np.mean(time_values),
            'Std_Time': np.std(time_values)
        })
    
    stats_df = pd.DataFrame(stats_data)
    print(stats_df.to_string(index=False))
    
    # Save detailed results
    detailed_results = []
    for algo in algorithms:
        for run in range(N_RUNS):
            detailed_results.append({
                'Algorithm': algo,
                'Run': run + 1,
                'Distance': results[algo]['fitness'][run],
                'Time': results[algo]['time'][run]
            })
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv(f"{results_dir}/detailed_results.csv", index=False)
    
    # Statistical comparison
    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISON (vs Original NWOA)")
    print("=" * 80)
    
    nwoa_fitness = results['NWOA']['fitness']
    
    comparison_summary = []
    for algo in algorithms[1:]:  # Skip NWOA itself
        algo_fitness = results[algo]['fitness']
        
        # Wilcoxon signed-rank test
        statistic, p_value = stats.wilcoxon(nwoa_fitness, algo_fitness)
        
        wins = sum(1 for a, b in zip(algo_fitness, nwoa_fitness) if a < b)
        losses = sum(1 for a, b in zip(algo_fitness, nwoa_fitness) if a > b)
        ties = N_RUNS - wins - losses
        
        print(f"\n{algo}: Wins={wins}, Losses={losses}, Ties={ties}")
        print(f"  Wilcoxon p-value: {p_value:.6f}")
        
        comparison_summary.append({
            'Algorithm': algo,
            'Wins': wins,
            'Losses': losses,
            'Ties': ties,
            'P_Value': p_value,
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })
    
    comparison_df = pd.DataFrame(comparison_summary)
    
    # Generate plots
    print("\n" + "=" * 80)
    print("GENERATING PLOTS...")
    print("=" * 80)
    
    # 1. Box Plot
    plt.figure(figsize=(12, 8))
    fitness_data = [results[algo]['fitness'] for algo in algorithms]
    bp = plt.boxplot(fitness_data, labels=algorithms, patch_artist=True)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(algorithms)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.ylabel('Total Distance', fontsize=12, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=12, fontweight='bold')
    plt.title(f'VRP Solution Quality Comparison ({VRP_PROBLEM.upper()} Problem)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/BoxPlots/vrp_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Convergence Curves
    plt.figure(figsize=(12, 8))
    for algo in algorithms:
        avg_convergence = np.mean(results[algo]['convergence'], axis=0)
        plt.plot(avg_convergence, label=algo, linewidth=2)
    
    plt.xlabel('Iteration', fontsize=12, fontweight='bold')
    plt.ylabel('Best Distance', fontsize=12, fontweight='bold')
    plt.title(f'Convergence Comparison ({VRP_PROBLEM.upper()} VRP)', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/Convergence/convergence_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Computation Time Comparison
    plt.figure(figsize=(12, 8))
    time_data = [results[algo]['time'] for algo in algorithms]
    bp = plt.boxplot(time_data, labels=algorithms, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.ylabel('Computation Time (seconds)', fontsize=12, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=12, fontweight='bold')
    plt.title(f'Computational Efficiency Comparison ({VRP_PROBLEM.upper()} Problem)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/computation_time.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Visualize best solutions
    print("\nVisualizing best routes for each algorithm...")
    for algo in algorithms:
        best_idx = np.argmin(results[algo]['fitness'])
        best_solution = results[algo]['solutions'][best_idx]
        best_fitness = results[algo]['fitness'][best_idx]
        
        vrp.visualize_solution(
            best_solution,
            title=f"{algo} - Best Solution (Distance: {best_fitness:.2f})",
            save_path=f"{results_dir}/Routes/{algo}_best_route.png"
        )
    
    # 5. Algorithm Rankings
    rankings = []
    for run in range(N_RUNS):
        run_fitness = {algo: results[algo]['fitness'][run] for algo in algorithms}
        sorted_algos = sorted(run_fitness.items(), key=lambda x: x[1])
        for rank, (algo, _) in enumerate(sorted_algos, 1):
            rankings.append({'Algorithm': algo, 'Run': run + 1, 'Rank': rank})
    
    rankings_df = pd.DataFrame(rankings)
    avg_rankings = rankings_df.groupby('Algorithm')['Rank'].mean().sort_values()
    
    print("\n" + "=" * 80)
    print("OVERALL RANKING (Average Rank Across All Runs)")
    print("=" * 80)
    print("\nAlgorithm Rankings (lower is better):")
    for algo, rank in avg_rankings.items():
        print(f"  {algo:15s}: {rank:.2f}")
    
    # 6. Ranking Bar Chart
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(avg_rankings)), avg_rankings.values, color=colors)
    plt.xticks(range(len(avg_rankings)), avg_rankings.index, rotation=15)
    plt.ylabel('Average Rank', fontsize=12, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=12, fontweight='bold')
    plt.title(f'Algorithm Ranking ({VRP_PROBLEM.upper()} VRP)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, avg_rankings.values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/algorithm_rankings.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save summary
    with open(f"{results_dir}/Comparison_Summary.txt", 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("VEHICLE ROUTING PROBLEM - ALGORITHM COMPARISON SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Problem: {VRP_PROBLEM.upper()}\n")
        f.write(f"Customers: {vrp.n_customers}\n")
        f.write(f"Vehicles: {vrp.n_vehicles}\n")
        f.write(f"Vehicle Capacity: {vrp.vehicle_capacity}\n")
        f.write(f"Population: {POPULATION}\n")
        f.write(f"Max Iterations: {MAX_ITER}\n")
        f.write(f"Independent Runs: {N_RUNS}\n\n")
        f.write("=" * 80 + "\n")
        f.write("STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        f.write(stats_df.to_string(index=False))
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("WILCOXON COMPARISON (vs NWOA)\n")
        f.write("=" * 80 + "\n\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("AVERAGE RANKINGS\n")
        f.write("=" * 80 + "\n\n")
        for algo, rank in avg_rankings.items():
            f.write(f"{algo:15s}: {rank:.2f}\n")
    
    print(f"\nResults saved to: {results_dir}/")
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
