"""
Generate LaTeX tables from paper metrics for easy insertion into the paper.

Usage:
    python scripts/generate_latex_tables.py paper_metrics/paper_metrics.json
"""

import json
import argparse
import numpy as np


ATTRIBUTE_NAMES = ['Smiling', 'Young', 'Male', 'Eyeglasses', 'Mustache']


def generate_comparison_table(results):
    """
    Generate LaTeX table comparing FCLF vs. Linear Steering baseline.

    Table includes:
    - Linear probe accuracy per attribute
    - k-NN purity
    - Centroid distance
    """
    fclf = results['comparison']['fclf']
    linear = results['comparison']['linear_steering']

    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Comparison of FCLF vs. Linear Steering Baseline}")
    latex.append("\\label{tab:method_comparison}")
    latex.append("\\begin{tabular}{lcccc}")
    latex.append("\\toprule")
    latex.append("Metric & FCLF & Linear & $\\Delta$ & Win \\\\")
    latex.append("\\midrule")

    # Linear probe per attribute
    for attr in ATTRIBUTE_NAMES:
        fclf_acc = fclf['linear_probe'][attr]
        linear_acc = linear['linear_probe'][attr]
        delta = fclf_acc - linear_acc
        win = "\\checkmark" if delta > 0 else ""

        latex.append(f"{attr} Acc. & {fclf_acc:.3f} & {linear_acc:.3f} & {delta:+.3f} & {win} \\\\")

    # Average
    fclf_avg = np.mean([fclf['linear_probe'][a] for a in ATTRIBUTE_NAMES])
    linear_avg = np.mean([linear['linear_probe'][a] for a in ATTRIBUTE_NAMES])
    delta_avg = fclf_avg - linear_avg
    win_avg = "\\checkmark" if delta_avg > 0 else ""
    latex.append("\\midrule")
    latex.append(f"\\textbf{{Avg. Accuracy}} & \\textbf{{{fclf_avg:.3f}}} & \\textbf{{{linear_avg:.3f}}} & \\textbf{{{delta_avg:+.3f}}} & {win_avg} \\\\")

    # k-NN purity
    latex.append("\\midrule")
    fclf_knn = fclf['knn_purity']
    linear_knn = linear['knn_purity']
    delta_knn = fclf_knn - linear_knn
    win_knn = "\\checkmark" if delta_knn > 0 else ""
    latex.append(f"k-NN Purity & {fclf_knn:.3f} & {linear_knn:.3f} & {delta_knn:+.3f} & {win_knn} \\\\")

    # Centroid distance (lower is better)
    fclf_cent = fclf['centroid_distance']
    linear_cent = linear['centroid_distance']
    delta_cent = fclf_cent - linear_cent
    win_cent = "\\checkmark" if delta_cent < 0 else ""  # Lower is better!
    latex.append(f"Centroid Dist. & {fclf_cent:.3f} & {linear_cent:.3f} & {delta_cent:+.3f} & {win_cent} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    return "\n".join(latex)


def generate_leakage_table(leakage):
    """
    Generate LaTeX table for attribute leakage.

    Shows how much non-target attributes change during flow.
    """
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Attribute Leakage: Non-Target Attribute Stability}")
    latex.append("\\label{tab:attribute_leakage}")
    latex.append("\\begin{tabular}{lccc}")
    latex.append("\\toprule")
    latex.append("Attribute & Acc. Before & Acc. After & Leakage \\\\")
    latex.append("\\midrule")

    leakages = []
    for attr in ATTRIBUTE_NAMES:
        if 'note' in leakage[attr]:
            latex.append(f"{attr} & - & - & N/A \\\\")
        else:
            acc_start = leakage[attr]['accuracy_start']
            acc_end = leakage[attr]['accuracy_end']
            leak = leakage[attr]['leakage']
            leakages.append(leak)

            latex.append(f"{attr} & {acc_start:.3f} & {acc_end:.3f} & {leak:.4f} \\\\")

    # Overall
    latex.append("\\midrule")
    mean_leak = leakage['_overall']['mean_leakage']
    max_leak = leakage['_overall']['max_leakage']
    latex.append(f"\\textbf{{Mean Leakage}} & - & - & \\textbf{{{mean_leak:.4f}}} \\\\")
    latex.append(f"\\textbf{{Max Leakage}} & - & - & \\textbf{{{max_leak:.4f}}} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    return "\n".join(latex)


def generate_field_diagnostics_table(field_stats):
    """
    Generate LaTeX table for vector field diagnostics.
    """
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Vector Field Diagnostics}")
    latex.append("\\label{tab:field_diagnostics}")
    latex.append("\\begin{tabular}{lccc}")
    latex.append("\\toprule")
    latex.append("Metric & Mean & Std & Max \\\\")
    latex.append("\\midrule")

    div = field_stats['divergence']
    curl = field_stats['curl']

    latex.append(f"Divergence & {div['mean']:.4f} & {div['std']:.4f} & {div['max']:.4f} \\\\")
    latex.append(f"Curl Magnitude & {curl['mean']:.4f} & {curl['std']:.4f} & {curl['max']:.4f} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    return "\n".join(latex)


def generate_monotonic_auc_table(monotonic_frac):
    """
    Generate LaTeX table for monotonic AUC progress.
    """
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Monotonic AUC Progress Along Flow Paths}")
    latex.append("\\label{tab:monotonic_auc}")
    latex.append("\\begin{tabular}{lc}")
    latex.append("\\toprule")
    latex.append("Attribute & Fraction Monotonic \\\\")
    latex.append("\\midrule")

    for attr in ATTRIBUTE_NAMES:
        frac = monotonic_frac[attr]
        latex.append(f"{attr} & {frac:.1%} \\\\")

    # Average
    avg_frac = np.mean([monotonic_frac[a] for a in ATTRIBUTE_NAMES])
    latex.append("\\midrule")
    latex.append(f"\\textbf{{Average}} & \\textbf{{{avg_frac:.1%}}} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    return "\n".join(latex)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('metrics_file', type=str, help='Path to paper_metrics.json')
    parser.add_argument('--output', type=str, default=None, help='Output .tex file (optional)')
    args = parser.parse_args()

    # Load metrics
    with open(args.metrics_file, 'r') as f:
        results = json.load(f)

    # Generate tables
    print("Generating LaTeX tables...\n")

    table1 = generate_comparison_table(results)
    table2 = generate_leakage_table(results['attribute_leakage'])
    table3 = generate_field_diagnostics_table(results['field_diagnostics'])
    table4 = generate_monotonic_auc_table(results['monotonic_auc_fraction'])

    # Combine
    all_tables = "\n\n% " + "="*70 + "\n\n"
    all_tables += "% TABLE 1: Method Comparison\n"
    all_tables += table1
    all_tables += "\n\n% " + "="*70 + "\n\n"
    all_tables += "% TABLE 2: Attribute Leakage\n"
    all_tables += table2
    all_tables += "\n\n% " + "="*70 + "\n\n"
    all_tables += "% TABLE 3: Field Diagnostics\n"
    all_tables += table3
    all_tables += "\n\n% " + "="*70 + "\n\n"
    all_tables += "% TABLE 4: Monotonic AUC Progress\n"
    all_tables += table4

    # Print to console
    print(all_tables)

    # Save to file if specified
    if args.output:
        with open(args.output, 'w') as f:
            f.write(all_tables)
        print(f"\n✅ Saved to: {args.output}")
    else:
        # Save to same directory as metrics file
        import os
        output_path = args.metrics_file.replace('.json', '_tables.tex')
        with open(output_path, 'w') as f:
            f.write(all_tables)
        print(f"\n✅ Saved to: {output_path}")


if __name__ == '__main__':
    main()
