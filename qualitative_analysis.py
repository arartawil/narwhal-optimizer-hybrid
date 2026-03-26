"""
Qualitative Analysis Plots for NWOA Hybrid Paper
==================================================
Generates 4 publication-quality IEEE-style plots:
1. Convergence Curves (F1, F4, F9, F20)
2. Population Diversity (F1, F4, F9, F20)
3. Exploration vs Exploitation Balance (F4, F9)
4. Box Plots (F1, F4, F9, F15, F20, F29)

All plots: 300 DPI, Times New Roman, IEEE style
Output: Hybrid/qualitative_plots/
"""

import sys
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ─────────────────────────────────────────────
#  SETUP
# ─────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE))

OUTPUT_DIR = os.path.join(BASE, "qualitative_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# IEEE style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

# Algorithm colors and markers
ALGO_STYLES = {
    'NWOA':    {'color': '#1f77b4', 'marker': 'o', 'ls': '-'},
    'NWOA-NM': {'color': '#ff7f0e', 'marker': 's', 'ls': '--'},
    'NWOA-HC': {'color': '#2ca02c', 'marker': '^', 'ls': '-.'},
    'NWOA-PS': {'color': '#d62728', 'marker': 'D', 'ls': ':'},
    'NWOA-SA': {'color': '#9467bd', 'marker': 'v', 'ls': '-'},
}

ALGORITHMS = list(ALGO_STYLES.keys())

# CEC2017 benchmark functions (from heurilab)
from heurilab import CEC2017_FUNCTIONS

# ─────────────────────────────────────────────
#  LOAD CEC2017 RAW DATA
# ─────────────────────────────────────────────
CEC2017_FILE = os.path.join(BASE, "Hybrid", "Hybrid_Results", "detailed_results.csv")


def load_cec2017_data():
    """Load CEC2017 raw results with convergence curves."""
    df = pd.read_csv(CEC2017_FILE)
    print(f"Loaded CEC2017: {len(df)} rows")
    return df


def parse_convergence(conv_str):
    """Parse convergence string from CSV."""
    cleaned = conv_str.replace('np.float64(', '').replace(')', '')
    return np.array(eval(cleaned))


# ─────────────────────────────────────────────
#  PLOT 1: CONVERGENCE CURVES
# ─────────────────────────────────────────────

def plot_convergence_curves(df):
    """Plot convergence curves for F1, F4, F9, F20."""
    print("\n[1/4] Generating convergence curves...")
    funcs = ['F1', 'F4', 'F9', 'F20']
    func_titles = {
        'F1': 'F1 (Unimodal)',
        'F4': 'F4 (Multimodal)',
        'F9': 'F9 (Hybrid)',
        'F20': 'F20 (Composition)',
    }

    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))
    axes = axes.flatten()

    for idx, func in enumerate(funcs):
        ax = axes[idx]

        for algo in ALGORITHMS:
            sub = df[(df['algorithm'] == algo) & (df['function'] == func)]
            if sub.empty:
                continue

            # Parse all 30 convergence curves
            all_conv = []
            for _, row in sub.iterrows():
                conv = parse_convergence(row['convergence'])
                all_conv.append(conv)

            # Average over runs
            max_len = min(len(c) for c in all_conv)
            all_conv = np.array([c[:max_len] for c in all_conv])
            mean_conv = np.mean(all_conv, axis=0)

            style = ALGO_STYLES[algo]
            ax.semilogy(range(len(mean_conv)), mean_conv,
                        color=style['color'], linestyle=style['ls'],
                        linewidth=1.2, label=algo,
                        marker=style['marker'], markersize=3,
                        markevery=max(1, len(mean_conv) // 8))

        ax.set_title(func_titles[func], fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Mean Best Fitness (log)')
        ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray')

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(OUTPUT_DIR, f'convergence_curves.{ext}'))
    plt.close()
    print("  Saved: convergence_curves.pdf / .png")


# ─────────────────────────────────────────────
#  PLOT 2: POPULATION DIVERSITY
# ─────────────────────────────────────────────

def compute_diversity(agents):
    """Compute mean pairwise Euclidean distance (population diversity)."""
    n = len(agents)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += np.linalg.norm(agents[i] - agents[j])
            count += 1
    return total / count


def run_with_diversity(AlgoClass, obj_func, dim, lb, ub, n_agents, max_iter, is_hybrid=False):
    """
    Run algorithm and record diversity + exploration ratio at each iteration.
    Works generically with all NWOA variants by monkey-patching the optimize loop.
    """
    if is_hybrid:
        algo = AlgoClass(
            objective_function=obj_func, dim=dim, lb=lb, ub=ub,
            n_agents=n_agents, max_iter=max_iter,
            ls_frequency=10, ls_candidates=3, ls_max_evals=50,
        )
    else:
        algo = AlgoClass(
            objective_function=obj_func, dim=dim, lb=lb, ub=ub,
            n_agents=n_agents, max_iter=max_iter,
        )

    # Initialize population
    algo.initialize_population()
    algo.convergence_curve = [algo.best_fitness]

    diversity = [compute_diversity(algo.agents)]
    exploration_ratios = []

    for t in range(max_iter):
        prev_agents = algo.agents.copy()
        prev_best = algo.best_agent.copy() if algo.best_agent is not None else None

        # Update prey energy
        algo.update_prey_energy(t)

        # --- Run one iteration based on algorithm type ---
        if is_hybrid:
            # Hybrid variants use echolocation/sonar/tusk phases
            for i in range(algo.n_agents):
                if algo.prey_energy > 0.5:
                    new_agent = algo.echolocation_phase(algo.agents[i], t)
                elif 0.2 < algo.prey_energy <= 0.5:
                    new_agent = algo.sonar_communication(algo.agents[i], t)
                else:
                    new_agent = algo.tusk_stunning(algo.agents[i], t)

                new_fitness = algo.objective_function(new_agent)
                if new_fitness < algo.fitness[i]:
                    algo.agents[i] = new_agent
                    algo.fitness[i] = new_fitness
                    if new_fitness < algo.best_fitness:
                        algo.best_agent = new_agent.copy()
                        algo.best_fitness = new_fitness

            # Apply local search periodically (if applicable)
            if hasattr(algo, 'ls_frequency') and (t + 1) % algo.ls_frequency == 0:
                sorted_indices = np.argsort(algo.fitness)
                best_indices = sorted_indices[:algo.ls_candidates]
                ls_method = None
                for method_name in ['nelder_mead_search', 'hill_climbing_search',
                                     'pattern_search', 'simulated_annealing_search']:
                    if hasattr(algo, method_name):
                        ls_method = getattr(algo, method_name)
                        break
                if ls_method:
                    for idx in best_indices:
                        ref_sol, ref_fit = ls_method(algo.agents[idx])
                        if ref_fit < algo.fitness[idx]:
                            algo.agents[idx] = ref_sol
                            algo.fitness[idx] = ref_fit
                            if ref_fit < algo.best_fitness:
                                algo.best_agent = ref_sol.copy()
                                algo.best_fitness = ref_fit
        else:
            # Base NWOA uses exploration/exploitation phases
            a = 2 - (2 * t / max_iter)
            prev_fitness = algo.best_fitness
            exp_ratio = algo.exploration_ratio(t, prev_fitness)

            for i in range(algo.n_agents):
                r1 = np.random.rand()
                if r1 < exp_ratio:
                    new_pos = algo.exploration_phase(algo.agents[i], t, a)
                else:
                    new_pos = algo.exploitation_phase(algo.agents[i], t, a)

                new_pos = algo.bound_position(new_pos)
                new_fit = algo.objective_function(new_pos)

                if new_fit < algo.fitness[i]:
                    algo.agents[i] = new_pos
                    algo.fitness[i] = new_fit
                    if new_fit < algo.best_fitness:
                        algo.best_agent = new_pos.copy()
                        algo.best_fitness = new_fit

        # Record convergence
        algo.convergence_curve.append(algo.best_fitness)

        # Record diversity
        diversity.append(compute_diversity(algo.agents))

        # Record exploration ratio: fraction of agents that moved away from best
        if prev_best is not None:
            moving_away = 0
            for i in range(algo.n_agents):
                dist_before = np.linalg.norm(prev_agents[i] - prev_best)
                dist_after = np.linalg.norm(algo.agents[i] - algo.best_agent)
                if dist_after > dist_before:
                    moving_away += 1
            exploration_ratios.append(moving_away / algo.n_agents)
        else:
            exploration_ratios.append(0.5)

    return diversity, exploration_ratios


def plot_diversity(diversity_data):
    """Plot population diversity for F1, F4, F9, F20."""
    print("\n[2/4] Generating diversity plots...")
    funcs = ['F1', 'F4', 'F9', 'F20']
    func_titles = {
        'F1': 'F1 (Unimodal)',
        'F4': 'F4 (Multimodal)',
        'F9': 'F9 (Hybrid)',
        'F20': 'F20 (Composition)',
    }

    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))
    axes = axes.flatten()

    for idx, func in enumerate(funcs):
        ax = axes[idx]
        for algo in ALGORITHMS:
            key = (func, algo)
            if key not in diversity_data:
                continue
            div = diversity_data[key]
            style = ALGO_STYLES[algo]
            ax.plot(range(len(div)), div,
                    color=style['color'], linestyle=style['ls'],
                    linewidth=1.2, label=algo,
                    marker=style['marker'], markersize=3,
                    markevery=max(1, len(div) // 8))

        ax.set_title(func_titles[func], fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Mean Pairwise Distance')
        ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray')

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(OUTPUT_DIR, f'diversity_plots.{ext}'))
    plt.close()
    print("  Saved: diversity_plots.pdf / .png")


# ─────────────────────────────────────────────
#  PLOT 3: EXPLORATION vs EXPLOITATION
# ─────────────────────────────────────────────

def plot_exploration_exploitation(exp_data):
    """Plot exploration ratio for F4 and F9."""
    print("\n[3/4] Generating exploration vs exploitation plots...")
    funcs = ['F4', 'F9']
    func_titles = {'F4': 'F4 (Multimodal)', 'F9': 'F9 (Hybrid)'}

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    for idx, func in enumerate(funcs):
        ax = axes[idx]
        for algo in ALGORITHMS:
            key = (func, algo)
            if key not in exp_data:
                continue
            exp = exp_data[key]
            style = ALGO_STYLES[algo]

            # Smooth with moving average
            window = 5
            smoothed = np.convolve(exp, np.ones(window)/window, mode='valid')

            ax.plot(range(len(smoothed)), smoothed,
                    color=style['color'], linestyle=style['ls'],
                    linewidth=1.2, label=algo)

        ax.set_title(func_titles[func], fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Exploration Ratio')
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.legend(loc='lower right', framealpha=0.9, edgecolor='gray')

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(OUTPUT_DIR, f'exploration_exploitation.{ext}'))
    plt.close()
    print("  Saved: exploration_exploitation.pdf / .png")


# ─────────────────────────────────────────────
#  PLOT 4: BOX PLOTS
# ─────────────────────────────────────────────

def plot_boxplots(df):
    """Box plots for F1, F4, F9, F15, F20, F29."""
    print("\n[4/4] Generating box plots...")
    funcs = ['F1', 'F4', 'F9', 'F15', 'F20', 'F29']
    func_titles = {
        'F1': 'F1', 'F4': 'F4', 'F9': 'F9',
        'F15': 'F15', 'F20': 'F20', 'F29': 'F29',
    }

    fig, axes = plt.subplots(2, 3, figsize=(7, 5))
    axes = axes.flatten()

    colors = [ALGO_STYLES[a]['color'] for a in ALGORITHMS]

    for idx, func in enumerate(funcs):
        ax = axes[idx]
        data = []
        labels = []

        for algo in ALGORITHMS:
            sub = df[(df['algorithm'] == algo) & (df['function'] == func)]
            if not sub.empty:
                data.append(sub['best_fitness'].values)
                labels.append(algo)

        if not data:
            continue

        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        widths=0.6, showfliers=True,
                        flierprops=dict(marker='o', markersize=3, alpha=0.5))

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(func_titles[func], fontweight='bold')
        ax.set_ylabel('Best Fitness')
        ax.tick_params(axis='x', rotation=45)

        # Use log scale if range is large
        vals = np.concatenate(data)
        if vals.max() / max(vals.min(), 1e-30) > 100:
            ax.set_yscale('log')

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(OUTPUT_DIR, f'boxplots.{ext}'))
    plt.close()
    print("  Saved: boxplots.pdf / .png")


# ─────────────────────────────────────────────
#  RUN DIVERSITY & EXPLORATION EXPERIMENTS
# ─────────────────────────────────────────────

def run_diversity_experiments():
    """Run experiments to collect diversity and exploration data."""
    from narwhal_optimizer import NarwhalOptimizer
    from Hybrid.nwoa_nelder_mead import NWOA_NelderMead
    from Hybrid.nwoa_hill_climbing import NWOA_HillClimbing
    from Hybrid.nwoa_pattern_search import NWOA_PatternSearch
    from Hybrid.nwoa_simulated_annealing import NWOA_SimulatedAnnealing

    algo_classes = {
        'NWOA':    (NarwhalOptimizer, False),
        'NWOA-NM': (NWOA_NelderMead, True),
        'NWOA-HC': (NWOA_HillClimbing, True),
        'NWOA-PS': (NWOA_PatternSearch, True),
        'NWOA-SA': (NWOA_SimulatedAnnealing, True),
    }

    # Functions to test — CEC2017_FUNCTIONS is list of (name, func, lb, ub, dim)
    func_indices = {'F1': 0, 'F4': 3, 'F9': 8, 'F20': 19}
    DIM = 30
    N_AGENTS = 30
    MAX_ITER = 150
    N_RUNS = 5  # average over 5 runs for smoother curves

    diversity_data = {}
    exploration_data = {}

    for func_name, fi in func_indices.items():
        bench = CEC2017_FUNCTIONS[fi]
        obj_func = bench[1]
        lb = bench[2]
        ub = bench[3]

        for algo_name, (AlgoClass, is_hybrid) in algo_classes.items():
            print(f"  Running: {algo_name} on {func_name}...", flush=True)

            all_div = []
            all_exp = []

            for run in range(N_RUNS):
                np.random.seed(run * 42 + fi)
                div, exp = run_with_diversity(
                    AlgoClass, obj_func, DIM, lb, ub, N_AGENTS, MAX_ITER, is_hybrid
                )
                all_div.append(div)
                all_exp.append(exp)

            # Average over runs
            min_div_len = min(len(d) for d in all_div)
            min_exp_len = min(len(e) for e in all_exp)
            mean_div = np.mean([d[:min_div_len] for d in all_div], axis=0)
            mean_exp = np.mean([e[:min_exp_len] for e in all_exp], axis=0)

            diversity_data[(func_name, algo_name)] = mean_div
            exploration_data[(func_name, algo_name)] = mean_exp

    return diversity_data, exploration_data


def save_diversity_csv(diversity_data, exploration_data):
    """Save diversity and exploration data to CSV."""
    # Diversity CSV
    div_rows = []
    for (func, algo), vals in diversity_data.items():
        for i, v in enumerate(vals):
            div_rows.append({'Function': func, 'Algorithm': algo, 'Iteration': i, 'Diversity': v})
    df_div = pd.DataFrame(div_rows)
    div_path = os.path.join(OUTPUT_DIR, 'diversity_data.csv')
    df_div.to_csv(div_path, index=False)
    print(f"  Saved: {div_path}")

    # Exploration CSV
    exp_rows = []
    for (func, algo), vals in exploration_data.items():
        for i, v in enumerate(vals):
            exp_rows.append({'Function': func, 'Algorithm': algo, 'Iteration': i, 'ExplorationRatio': v})
    df_exp = pd.DataFrame(exp_rows)
    exp_path = os.path.join(OUTPUT_DIR, 'exploration_data.csv')
    df_exp.to_csv(exp_path, index=False)
    print(f"  Saved: {exp_path}")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("  Qualitative Analysis — IEEE Publication Plots")
    print("="*60)

    # Load existing CEC2017 data
    df = load_cec2017_data()

    # Plot 1: Convergence curves (from existing data)
    plot_convergence_curves(df)

    # Plot 4: Box plots (from existing data)
    plot_boxplots(df)

    # Plots 2 & 3: Need fresh runs for diversity & exploration tracking
    print("\nRunning diversity & exploration experiments (5 runs each)...")
    diversity_data, exploration_data = run_diversity_experiments()

    # Save CSV data
    save_diversity_csv(diversity_data, exploration_data)

    # Plot 2: Diversity
    plot_diversity(diversity_data)

    # Plot 3: Exploration vs Exploitation
    plot_exploration_exploitation(exploration_data)

    print(f"\n{'='*60}")
    print(f"  All plots saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")
