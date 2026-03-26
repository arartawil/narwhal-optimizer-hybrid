"""
Comprehensive Statistical Tests for CEC2022 Results
=====================================================
1. Friedman test — ranks all algorithms across all functions
2. Post-hoc Wilcoxon signed-rank with Bonferroni correction
3. Effect sizes — Cliff's delta
4. Critical difference diagram
5. All results saved to Excel (.xlsx)
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  1. LOAD & MERGE ALL RAW RESULTS
# ─────────────────────────────────────────────

BASE = os.path.dirname(os.path.abspath(__file__))

# CEC2022 D=10 files
CEC2022_D10_FILES = [
    os.path.join(BASE, "2022_10_d", "CEC2022_Results_F1_F7", "CSV Data", "raw_runs.csv"),
    os.path.join(BASE, "2022_10_d", "CEC2022_Results_F8", "CSV Data", "raw_runs.csv"),
    os.path.join(BASE, "2022_10_d", "CEC2022_Results_F9", "CSV Data", "raw_runs.csv"),
    os.path.join(BASE, "2022_10_d", "CEC2022_Results_10", "CSV Data", "raw_runs.csv"),
    os.path.join(BASE, "2022_10_d", "CEC2022_Results_11", "CSV Data", "raw_runs.csv"),
    os.path.join(BASE, "2022_10_d", "CEC2022_Results_12", "CSV Data", "raw_runs.csv"),
]

# CEC2022 D=20 files
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
    """Load and merge multiple raw_runs CSV files."""
    frames = []
    for f in file_list:
        if os.path.exists(f):
            df = pd.read_csv(f)
            frames.append(df)
            print(f"  Loaded: {os.path.basename(os.path.dirname(os.path.dirname(f)))} "
                  f"({len(df)} rows)")
        else:
            print(f"  WARNING: Missing file: {f}")
    if not frames:
        print(f"  No data found for {label}!")
        return None
    merged = pd.concat(frames, ignore_index=True)
    print(f"  Total {label}: {len(merged)} rows, "
          f"Functions: {merged['Benchmark'].nunique()}, "
          f"Algorithms: {merged['Algorithm'].nunique()}")
    return merged


def build_fitness_matrix(df):
    """
    Build a matrix: rows = functions, columns = algorithms.
    Each cell = mean best fitness across 30 runs.
    Also returns raw data dict for pairwise tests.
    """
    functions = sorted(df['Benchmark'].unique())
    algorithms = sorted(df['Algorithm'].unique())

    # Mean fitness matrix (for Friedman)
    mean_matrix = pd.DataFrame(index=functions, columns=algorithms, dtype=float)
    # Raw runs dict: {(func, algo): [30 fitness values]}
    raw_dict = {}

    for func in functions:
        for algo in algorithms:
            vals = df[(df['Benchmark'] == func) & (df['Algorithm'] == algo)]['BestFitness'].values
            mean_matrix.loc[func, algo] = np.mean(vals)
            raw_dict[(func, algo)] = vals

    return mean_matrix, raw_dict, functions, algorithms


# ─────────────────────────────────────────────
#  2. FRIEDMAN TEST
# ─────────────────────────────────────────────

def friedman_test(mean_matrix, algorithms):
    """Run Friedman test and return ranks."""
    # Rank each row (function) — rank 1 = best (lowest fitness)
    rank_matrix = mean_matrix.rank(axis=1, method='average')
    avg_ranks = rank_matrix.mean()

    # Friedman chi-square statistic
    n = len(mean_matrix)  # number of functions
    k = len(algorithms)   # number of algorithms
    data_for_friedman = [mean_matrix[algo].values for algo in algorithms]
    stat, p_value = stats.friedmanchisquare(*data_for_friedman)

    return rank_matrix, avg_ranks, stat, p_value


# ─────────────────────────────────────────────
#  3. POST-HOC WILCOXON WITH BONFERRONI
# ─────────────────────────────────────────────

def posthoc_wilcoxon_bonferroni(raw_dict, functions, algorithms):
    """
    Pairwise Wilcoxon signed-rank test with Bonferroni correction.
    For each pair of algorithms, aggregate mean fitness per function,
    then run Wilcoxon.
    """
    n_pairs = len(list(combinations(algorithms, 2)))
    results = []

    for a1, a2 in combinations(algorithms, 2):
        vals_a1 = np.array([np.mean(raw_dict[(f, a1)]) for f in functions])
        vals_a2 = np.array([np.mean(raw_dict[(f, a2)]) for f in functions])

        try:
            stat, p_val = stats.wilcoxon(vals_a1, vals_a2, alternative='two-sided')
        except ValueError:
            stat, p_val = np.nan, 1.0

        p_corrected = min(p_val * n_pairs, 1.0)  # Bonferroni

        # Win/Tie/Loss
        wins = np.sum(vals_a1 < vals_a2)
        ties = np.sum(vals_a1 == vals_a2)
        losses = np.sum(vals_a1 > vals_a2)

        results.append({
            'Algorithm_1': a1,
            'Algorithm_2': a2,
            'W_statistic': stat,
            'p_value': p_val,
            'p_corrected': p_corrected,
            'Significant (p<0.05)': 'Yes' if p_corrected < 0.05 else 'No',
            'Wins(A1)': wins,
            'Ties': ties,
            'Losses(A1)': losses,
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
#  4. CLIFF'S DELTA EFFECT SIZE
# ─────────────────────────────────────────────

def cliffs_delta(x, y):
    """Compute Cliff's delta effect size."""
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return 0.0, 'negligible'
    count = 0
    for xi in x:
        for yj in y:
            if xi < yj:
                count += 1
            elif xi > yj:
                count -= 1
    delta = count / (n1 * n2)

    # Interpret
    abs_d = abs(delta)
    if abs_d < 0.147:
        magnitude = 'negligible'
    elif abs_d < 0.33:
        magnitude = 'small'
    elif abs_d < 0.474:
        magnitude = 'medium'
    else:
        magnitude = 'large'

    return delta, magnitude


def compute_effect_sizes(raw_dict, functions, algorithms):
    """Compute Cliff's delta for all pairs across all functions."""
    results = []

    for a1, a2 in combinations(algorithms, 2):
        for func in functions:
            vals_a1 = raw_dict[(func, a1)]
            vals_a2 = raw_dict[(func, a2)]
            delta, magnitude = cliffs_delta(vals_a1, vals_a2)
            results.append({
                'Function': func,
                'Algorithm_1': a1,
                'Algorithm_2': a2,
                'Cliffs_Delta': delta,
                'Magnitude': magnitude,
            })

    return pd.DataFrame(results)


def compute_effect_size_summary(effect_df, algorithms):
    """Summarize effect sizes per algorithm pair."""
    summary = []
    for a1, a2 in combinations(algorithms, 2):
        sub = effect_df[(effect_df['Algorithm_1'] == a1) & (effect_df['Algorithm_2'] == a2)]
        avg_delta = sub['Cliffs_Delta'].mean()
        counts = sub['Magnitude'].value_counts()
        summary.append({
            'Algorithm_1': a1,
            'Algorithm_2': a2,
            'Avg_Cliffs_Delta': avg_delta,
            'Large': counts.get('large', 0),
            'Medium': counts.get('medium', 0),
            'Small': counts.get('small', 0),
            'Negligible': counts.get('negligible', 0),
        })
    return pd.DataFrame(summary)


# ─────────────────────────────────────────────
#  5. CRITICAL DIFFERENCE DIAGRAM
# ─────────────────────────────────────────────

def plot_critical_difference(avg_ranks, n_functions, n_algorithms, title, output_path):
    """
    Plot a Critical Difference (CD) diagram.
    Uses Nemenyi critical difference.
    """
    k = n_algorithms
    n = n_functions

    # Nemenyi critical value (q_alpha for alpha=0.05)
    # Approximate q_alpha values for k algorithms
    q_alpha_table = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164
    }
    q_alpha = q_alpha_table.get(k, 2.728)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))

    # Sort algorithms by average rank
    sorted_algos = avg_ranks.sort_values()
    algo_names = sorted_algos.index.tolist()
    ranks = sorted_algos.values

    fig, ax = plt.subplots(1, 1, figsize=(12, 4 + 0.3 * k))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(0, k + 1)
    ax.invert_yaxis()

    # Draw rank axis at top
    ax.set_title(f"{title}\nCD = {cd:.3f} (Nemenyi, p<0.05)", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Average Rank", fontsize=12)

    # Draw horizontal lines for each algorithm
    for i, (name, rank) in enumerate(zip(algo_names, ranks)):
        y = i + 0.5
        ax.plot(rank, y, 'ko', markersize=8)
        ax.annotate(f"  {name} ({rank:.2f})", (rank, y),
                    fontsize=10, va='center', ha='left')
        ax.axhline(y=y, color='lightgray', linewidth=0.5, linestyle='--')

    # Draw CD bar
    ax.plot([1, 1 + cd], [-0.2, -0.2], 'r-', linewidth=3)
    ax.text(1 + cd / 2, -0.5, f"CD={cd:.2f}", ha='center', fontsize=10, color='red')

    # Draw connections for non-significant differences
    for i in range(len(algo_names)):
        for j in range(i + 1, len(algo_names)):
            if abs(ranks[i] - ranks[j]) < cd:
                y_mid = (i + j) / 2 + 0.5
                ax.plot([ranks[i], ranks[j]], [y_mid, y_mid],
                        'b-', linewidth=2, alpha=0.4)

    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  CD diagram saved: {output_path}")


# ─────────────────────────────────────────────
#  6. SAVE ALL TO EXCEL
# ─────────────────────────────────────────────

def save_to_excel(output_path, sheets_dict):
    """Save multiple DataFrames to a single Excel file."""
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet_name, df in sheets_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=True)
    print(f"  Excel saved: {output_path}")


# ─────────────────────────────────────────────
#  7. RUN FULL ANALYSIS
# ─────────────────────────────────────────────

def run_analysis(df, label, output_dir):
    """Run full statistical analysis on a merged DataFrame."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Statistical Analysis: {label}")
    print(f"{'='*60}")

    # Build matrices
    mean_matrix, raw_dict, functions, algorithms = build_fitness_matrix(df)
    print(f"  Functions: {len(functions)}, Algorithms: {len(algorithms)}")

    # 1. Friedman test
    print("\n  [1/4] Friedman test...")
    rank_matrix, avg_ranks, chi2, p_val = friedman_test(mean_matrix, algorithms)
    print(f"    Chi-square = {chi2:.4f}, p-value = {p_val:.6e}")
    print(f"    Average ranks:")
    for algo in avg_ranks.sort_values().index:
        print(f"      {algo}: {avg_ranks[algo]:.3f}")

    friedman_df = pd.DataFrame({
        'Algorithm': avg_ranks.index,
        'Avg_Rank': avg_ranks.values,
        'Chi2': chi2,
        'p_value': p_val,
        'Significant (p<0.05)': 'Yes' if p_val < 0.05 else 'No',
    }).set_index('Algorithm')

    # 2. Post-hoc Wilcoxon
    print("\n  [2/4] Post-hoc Wilcoxon (Bonferroni)...")
    wilcoxon_df = posthoc_wilcoxon_bonferroni(raw_dict, functions, algorithms)
    sig_count = (wilcoxon_df['Significant (p<0.05)'] == 'Yes').sum()
    print(f"    {sig_count}/{len(wilcoxon_df)} pairs are significant")

    # 3. Effect sizes
    print("\n  [3/4] Cliff's delta effect sizes...")
    effect_df = compute_effect_sizes(raw_dict, functions, algorithms)
    effect_summary_df = compute_effect_size_summary(effect_df, algorithms)

    # 4. Critical difference diagram
    print("\n  [4/4] Critical difference diagram...")
    cd_path = os.path.join(output_dir, f"critical_difference_{label}.png")
    plot_critical_difference(avg_ranks, len(functions), len(algorithms), label, cd_path)

    # Save to Excel
    excel_path = os.path.join(output_dir, f"statistical_tests_{label}.xlsx")
    save_to_excel(excel_path, {
        'Friedman_Ranks': friedman_df,
        'Rank_Matrix': rank_matrix,
        'Wilcoxon_PostHoc': wilcoxon_df.set_index(['Algorithm_1', 'Algorithm_2']),
        'Cliffs_Delta_Detail': effect_df.set_index(['Function', 'Algorithm_1', 'Algorithm_2']),
        'Cliffs_Delta_Summary': effect_summary_df.set_index(['Algorithm_1', 'Algorithm_2']),
        'Mean_Fitness': mean_matrix,
    })

    print(f"\n  Done: {label}")
    return friedman_df, wilcoxon_df, effect_summary_df


# ─────────────────────────────────────────────
#  CEC2017 OLD DATA LOADER
# ─────────────────────────────────────────────

CEC2017_FILE = os.path.join(BASE, "Hybrid", "Hybrid_Results", "detailed_results.csv")


def load_cec2017():
    """Load old CEC2017 detailed_results.csv and rename columns to match."""
    if not os.path.exists(CEC2017_FILE):
        print(f"  WARNING: CEC2017 file not found: {CEC2017_FILE}")
        return None
    df = pd.read_csv(CEC2017_FILE, usecols=['algorithm', 'function', 'run', 'best_fitness'])
    df = df.rename(columns={
        'algorithm': 'Algorithm',
        'function': 'Benchmark',
        'best_fitness': 'BestFitness',
        'run': 'Run',
    })
    print(f"  Loaded CEC2017: {len(df)} rows, "
          f"Functions: {df['Benchmark'].nunique()}, "
          f"Algorithms: {df['Algorithm'].nunique()}")
    return df


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    output_dir = os.path.join(BASE, "Statistical_Tests")
    os.makedirs(output_dir, exist_ok=True)

    # CEC2017 (30 functions)
    print("\nLoading CEC2017 data...")
    df_cec17 = load_cec2017()
    if df_cec17 is not None:
        run_analysis(df_cec17, "CEC2017_30F", output_dir)

    # CEC2022 D=10
    print("\nLoading CEC2022 D=10 data...")
    df_d10 = load_and_merge(CEC2022_D10_FILES, "CEC2022_D10")
    if df_d10 is not None:
        run_analysis(df_d10, "CEC2022_D10", output_dir)

    # CEC2022 D=20
    print("\nLoading CEC2022 D=20 data...")
    df_d20 = load_and_merge(CEC2022_D20_FILES, "CEC2022_D20")
    if df_d20 is not None:
        run_analysis(df_d20, "CEC2022_D20", output_dir)

    print(f"\n{'='*60}")
    print(f"  All statistical tests complete!")
    print(f"  Results in: {output_dir}")
    print(f"{'='*60}")
