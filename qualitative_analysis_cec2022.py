"""
Qualitative Analysis Plots for NWOA Hybrid Paper — CEC2022
============================================================
Generates 4 publication-quality IEEE-style plots:
1. Convergence Curves (F1, F5, F8, F9)
2. Population Diversity (F1, F5, F8, F9)
3. Exploration vs Exploitation Balance (F5, F9)
4. Box Plots (F1, F3, F5, F8, F9, F12)

All plots: 300 DPI, Times New Roman, IEEE style
Output: Hybrid/qualitative_plots_cec2022/
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
import seaborn as sns

# ─────────────────────────────────────────────
#  SETUP
# ─────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE))

OUTPUT_DIR = os.path.join(BASE, "qualitative_plots_cec2022")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# IEEE style
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

ALGO_STYLES = {
    'NWOA':    {'color': '#1f77b4', 'marker': 'o', 'ls': '-'},
    'NWOA-NM': {'color': '#ff7f0e', 'marker': 's', 'ls': '--'},
    'NWOA-HC': {'color': '#2ca02c', 'marker': '^', 'ls': '-.'},
    'NWOA-PS': {'color': '#d62728', 'marker': 'D', 'ls': ':'},
    'NWOA-SA': {'color': '#9467bd', 'marker': 'v', 'ls': '-'},
}
ALGORITHMS = list(ALGO_STYLES.keys())

from opfunu.cec_based import cec2022

# ─────────────────────────────────────────────
#  LOAD CEC2022 RAW DATA (D=10 and D=20)
# ─────────────────────────────────────────────

CEC2022_D10_FILES = [
    os.path.join(BASE, "2022_10_d", "CEC2022_Results_F1_F7", "CSV Data", "raw_runs.csv"),
    os.path.join(BASE, "2022_10_d", "CEC2022_Results_F8", "CSV Data", "raw_runs.csv"),
    os.path.join(BASE, "2022_10_d", "CEC2022_Results_F9", "CSV Data", "raw_runs.csv"),
    os.path.join(BASE, "2022_10_d", "CEC2022_Results_10", "CSV Data", "raw_runs.csv"),
    os.path.join(BASE, "2022_10_d", "CEC2022_Results_11", "CSV Data", "raw_runs.csv"),
    os.path.join(BASE, "2022_10_d", "CEC2022_Results_12", "CSV Data", "raw_runs.csv"),
]

CEC2022_D20_FILES = [
    os.path.join(BASE, "CEC2022_20_d", "CEC2022_Results_F1", "CSV Data", "raw_runs.csv"),
    os.path.join(BASE, "CEC2022_20_d", "CEC2022_Results_F2", "CSV Data", "raw_runs.csv"),
    os.path.join(BASE, "CEC2022_20_d", "CEC2022_Results_F3_F4", "CSV Data", "raw_runs.csv"),
    os.path.join(BASE, "CEC2022_20_d", "CEC2022_Results_F5_F6", "CSV Data", "raw_runs.csv"),
    os.path.join(BASE, "CEC2022_20_d", "CEC2022_Results_F7_F9", "CSV Data", "raw_runs.csv"),
    os.path.join(BASE, "CEC2022_20_d", "CSV Data_F10", "raw_runs.csv"),
    os.path.join(BASE, "CEC2022_20_d", "CEC2022_Results_F11_F12", "CSV Data", "raw_runs.csv"),
    os.path.join(BASE, "CEC2022_20_d", "CEC2022_Results_F8", "CSV Data", "raw_runs.csv"),
]


def load_and_merge(file_list, label):
    frames = []
    for f in file_list:
        if os.path.exists(f):
            df = pd.read_csv(f)
            frames.append(df)
    if not frames:
        print(f"  No data for {label}")
        return None
    merged = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {label}: {len(merged)} rows, "
          f"Functions: {merged['Benchmark'].nunique()}, "
          f"Algorithms: {merged['Algorithm'].nunique()}")
    return merged


# ─────────────────────────────────────────────
#  PLOT 1: CONVERGENCE CURVES
# ─────────────────────────────────────────────

def plot_convergence_curves(df, dim_label):
    """Convergence curves for F1 (unimodal), F5 (multimodal), F8 (hybrid), F9 (composition)."""
    print(f"\n[1/4] Convergence curves ({dim_label})...")
    funcs = ['CEC22_F1', 'CEC22_F5', 'CEC22_F8', 'CEC22_F9']
    func_titles = {
        'CEC22_F1': 'F1 (Unimodal)',
        'CEC22_F5': 'F5 (Multimodal)',
        'CEC22_F8': 'F8 (Hybrid)',
        'CEC22_F9': 'F9 (Composition)',
    }

    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))
    axes = axes.flatten()

    # Get convergence columns
    conv_cols = [c for c in df.columns if c.startswith('Conv_')]
    conv_cols_sorted = sorted(conv_cols, key=lambda x: int(x.split('_')[1]))

    for idx, func in enumerate(funcs):
        ax = axes[idx]
        sub_func = df[df['Benchmark'] == func]
        if sub_func.empty:
            ax.set_title(f"{func_titles.get(func, func)} - No Data")
            continue

        for algo in ALGORITHMS:
            sub = sub_func[sub_func['Algorithm'] == algo]
            if sub.empty:
                continue

            # Get convergence data
            conv_data = sub[conv_cols_sorted].values
            mean_conv = np.mean(conv_data, axis=0)

            # Remove trailing NaNs
            valid = ~np.isnan(mean_conv)
            mean_conv = mean_conv[valid]

            style = ALGO_STYLES[algo]
            iters = range(len(mean_conv))
            ax.semilogy(iters, mean_conv,
                        color=style['color'], linestyle=style['ls'],
                        linewidth=1.2, label=algo,
                        marker=style['marker'], markersize=3,
                        markevery=max(1, len(mean_conv) // 8))

        ax.set_title(func_titles.get(func, func), fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Mean Best Fitness (log)')
        ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray')

    plt.suptitle(f'CEC2022 Convergence Curves ({dim_label})', fontweight='bold', y=1.02)
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(OUTPUT_DIR, f'convergence_curves_{dim_label}.{ext}'))
    plt.close()
    print(f"  Saved: convergence_curves_{dim_label}.pdf / .png")


# ─────────────────────────────────────────────
#  PLOT 2: POPULATION DIVERSITY
# ─────────────────────────────────────────────

def compute_diversity(agents):
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
    """Run algorithm and record diversity + exploration ratio."""
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

    algo.initialize_population()
    algo.convergence_curve = [algo.best_fitness]

    diversity = [compute_diversity(algo.agents)]
    exploration_ratios = []

    for t in range(max_iter):
        prev_agents = algo.agents.copy()
        prev_best = algo.best_agent.copy() if algo.best_agent is not None else None

        algo.update_prey_energy(t)

        if is_hybrid:
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

        algo.convergence_curve.append(algo.best_fitness)
        diversity.append(compute_diversity(algo.agents))

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


def run_diversity_experiments_cec2022(dim):
    """Run diversity experiments on CEC2022 functions."""
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

    CEC2022_FUNCS = {
        'CEC22_F1':  cec2022.F12022,
        'CEC22_F5':  cec2022.F52022,
        'CEC22_F8':  cec2022.F82022,
        'CEC22_F9':  cec2022.F92022,
    }

    N_AGENTS = 30
    MAX_ITER = 150
    N_RUNS = 5

    diversity_data = {}
    exploration_data = {}

    for func_name, FuncClass in CEC2022_FUNCS.items():
        bench = FuncClass(ndim=dim)
        lb = bench.lb
        ub = bench.ub

        def make_obj(b):
            return lambda x: b.evaluate(x)
        obj_func = make_obj(bench)

        for algo_name, (AlgoClass, is_hybrid) in algo_classes.items():
            print(f"  Running: {algo_name} on {func_name} (D={dim})...", flush=True)

            all_div = []
            all_exp = []

            for run in range(N_RUNS):
                np.random.seed(run * 42)
                div, exp = run_with_diversity(
                    AlgoClass, obj_func, dim, lb, ub, N_AGENTS, MAX_ITER, is_hybrid
                )
                all_div.append(div)
                all_exp.append(exp)

            min_div_len = min(len(d) for d in all_div)
            min_exp_len = min(len(e) for e in all_exp)
            mean_div = np.mean([d[:min_div_len] for d in all_div], axis=0)
            mean_exp = np.mean([e[:min_exp_len] for e in all_exp], axis=0)

            diversity_data[(func_name, algo_name)] = mean_div
            exploration_data[(func_name, algo_name)] = mean_exp

    return diversity_data, exploration_data


def plot_diversity(diversity_data, dim_label):
    print(f"\n[2/4] Diversity plots ({dim_label})...")
    funcs = ['CEC22_F1', 'CEC22_F5', 'CEC22_F8', 'CEC22_F9']
    func_titles = {
        'CEC22_F1': 'F1 (Unimodal)',
        'CEC22_F5': 'F5 (Multimodal)',
        'CEC22_F8': 'F8 (Hybrid)',
        'CEC22_F9': 'F9 (Composition)',
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

        ax.set_title(func_titles.get(func, func), fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Mean Pairwise Distance')
        ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray')

    plt.suptitle(f'CEC2022 Population Diversity ({dim_label})', fontweight='bold', y=1.02)
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(OUTPUT_DIR, f'diversity_plots_{dim_label}.{ext}'))
    plt.close()
    print(f"  Saved: diversity_plots_{dim_label}.pdf / .png")


def plot_exploration_exploitation(exp_data, dim_label):
    print(f"\n[3/4] Exploration vs exploitation ({dim_label})...")
    funcs = ['CEC22_F5', 'CEC22_F9']
    func_titles = {'CEC22_F5': 'F5 (Multimodal)', 'CEC22_F9': 'F9 (Composition)'}

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    for idx, func in enumerate(funcs):
        ax = axes[idx]
        for algo in ALGORITHMS:
            key = (func, algo)
            if key not in exp_data:
                continue
            exp = exp_data[key]
            style = ALGO_STYLES[algo]

            window = 5
            smoothed = np.convolve(exp, np.ones(window)/window, mode='valid')
            ax.plot(range(len(smoothed)), smoothed,
                    color=style['color'], linestyle=style['ls'],
                    linewidth=1.2, label=algo)

        ax.set_title(func_titles.get(func, func), fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Exploration Ratio')
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.legend(loc='lower right', framealpha=0.9, edgecolor='gray')

    plt.suptitle(f'CEC2022 Exploration vs Exploitation ({dim_label})', fontweight='bold', y=1.05)
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(OUTPUT_DIR, f'exploration_exploitation_{dim_label}.{ext}'))
    plt.close()
    print(f"  Saved: exploration_exploitation_{dim_label}.pdf / .png")


def plot_boxplots(df, dim_label):
    print(f"\n[4/4] Box plots ({dim_label})...")
    funcs = ['CEC22_F1', 'CEC22_F3', 'CEC22_F5', 'CEC22_F8', 'CEC22_F9', 'CEC22_F12']
    func_titles = {
        'CEC22_F1': 'F1', 'CEC22_F3': 'F3', 'CEC22_F5': 'F5',
        'CEC22_F8': 'F8', 'CEC22_F9': 'F9', 'CEC22_F12': 'F12',
    }

    # Filter to only available functions
    available = df['Benchmark'].unique()
    funcs = [f for f in funcs if f in available]

    n_funcs = len(funcs)
    if n_funcs == 0:
        print("  No matching functions found!")
        return

    cols = min(3, n_funcs)
    rows = (n_funcs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7, 2.5 * rows))
    if n_funcs == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = [ALGO_STYLES[a]['color'] for a in ALGORITHMS]

    for idx, func in enumerate(funcs):
        ax = axes[idx]
        data = []
        labels = []

        for algo in ALGORITHMS:
            sub = df[(df['Algorithm'] == algo) & (df['Benchmark'] == func)]
            if not sub.empty:
                data.append(sub['BestFitness'].values)
                labels.append(algo)

        if not data:
            continue

        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        widths=0.6, showfliers=True,
                        flierprops=dict(marker='o', markersize=3, alpha=0.5))

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(func_titles.get(func, func), fontweight='bold')
        ax.set_ylabel('Best Fitness')
        ax.tick_params(axis='x', rotation=45)

        vals = np.concatenate(data)
        if vals.max() / max(vals.min(), 1e-30) > 100:
            ax.set_yscale('log')

    # Hide unused axes
    for i in range(n_funcs, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f'CEC2022 Box Plots ({dim_label})', fontweight='bold', y=1.02)
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(OUTPUT_DIR, f'boxplots_{dim_label}.{ext}'))
    plt.close()
    print(f"  Saved: boxplots_{dim_label}.pdf / .png")


def save_csv(diversity_data, exploration_data, dim_label):
    div_rows = []
    for (func, algo), vals in diversity_data.items():
        for i, v in enumerate(vals):
            div_rows.append({'Function': func, 'Algorithm': algo, 'Iteration': i, 'Diversity': v})
    pd.DataFrame(div_rows).to_csv(os.path.join(OUTPUT_DIR, f'diversity_data_{dim_label}.csv'), index=False)

    exp_rows = []
    for (func, algo), vals in exploration_data.items():
        for i, v in enumerate(vals):
            exp_rows.append({'Function': func, 'Algorithm': algo, 'Iteration': i, 'ExplorationRatio': v})
    pd.DataFrame(exp_rows).to_csv(os.path.join(OUTPUT_DIR, f'exploration_data_{dim_label}.csv'), index=False)
    print(f"  Saved: diversity_data_{dim_label}.csv, exploration_data_{dim_label}.csv")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Qualitative Analysis — CEC2022 — IEEE Publication Plots")
    print("=" * 60)

    for dim, files, dim_label in [(10, CEC2022_D10_FILES, "D10"), (20, CEC2022_D20_FILES, "D20")]:
        print(f"\n{'='*40}")
        print(f"  Processing CEC2022 {dim_label}")
        print(f"{'='*40}")

        # Load raw data
        df = load_and_merge(files, f"CEC2022_{dim_label}")

        if df is not None:
            # Plot 1: Convergence (from existing data)
            plot_convergence_curves(df, dim_label)

            # Plot 4: Box plots (from existing data)
            plot_boxplots(df, dim_label)

        # Plots 2 & 3: Fresh runs for diversity & exploration
        print(f"\nRunning diversity experiments ({dim_label}, 5 runs each)...")
        diversity_data, exploration_data = run_diversity_experiments_cec2022(dim)

        save_csv(diversity_data, exploration_data, dim_label)
        plot_diversity(diversity_data, dim_label)
        plot_exploration_exploitation(exploration_data, dim_label)

    print(f"\n{'='*60}")
    print(f"  All CEC2022 plots saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")
