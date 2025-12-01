"""
Generate comprehensive Markdown evaluation report.

Combines results from all evaluation scripts into a single shareable document.

Usage:
    python scripts/create_evaluation_report.py \
        --base_dir outputs/v4_projection \
        --output outputs/v4_projection/EVALUATION_REPORT.md
"""

import os
import argparse
import json
from datetime import datetime


def load_json(path):
    """Load JSON file, return None if not found."""
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def format_metric(value, precision=4):
    """Format metric with appropriate precision."""
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def generate_report(base_dir, output_path):
    """Generate comprehensive evaluation report."""

    # Load all results
    paper_metrics_path = os.path.join(base_dir, 'paper_metrics', 'paper_metrics.json')
    flow_depth_path = os.path.join(base_dir, 'flow_depth', 'flow_depth_analysis.json')
    config_path = os.path.join(base_dir, 'checkpoints', 'fclf_best.pt')

    paper_metrics = load_json(paper_metrics_path)
    flow_depth = load_json(flow_depth_path)

    # Load config if available
    config_info = None
    if os.path.exists(config_path):
        import torch
        ckpt = torch.load(config_path, map_location='cpu')
        if 'config' in ckpt:
            config_info = ckpt['config']
            epoch_info = ckpt.get('epoch', 'N/A')

    # Start building report
    report = []
    report.append("# FCLF Evaluation Report")
    report.append("")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Model:** {os.path.basename(base_dir)}")
    report.append("")
    report.append("---")
    report.append("")

    # ====================
    # 1. Training Configuration
    # ====================
    report.append("## 1. Training Configuration")
    report.append("")

    if config_info:
        report.append("### Model Architecture")
        report.append("```yaml")
        report.append(f"embedding_dim: {config_info['model']['embedding_dim']}")
        report.append(f"num_attributes: {config_info['model']['num_attributes']}")
        report.append(f"hidden_dim: {config_info['model']['hidden_dim']}")
        if 'projection_radius' in config_info['model']:
            report.append(f"projection_radius: {config_info['model']['projection_radius']}")
        report.append("```")
        report.append("")

        report.append("### Training Hyperparameters")
        report.append("```yaml")
        report.append(f"num_epochs: {config_info['training']['num_epochs']}")
        report.append(f"batch_size: {config_info['training']['batch_size']}")
        report.append(f"learning_rate: {config_info['training']['learning_rate']}")
        report.append(f"alpha: {config_info['training']['alpha']}")
        report.append("```")
        report.append("")

        report.append("### Loss Configuration")
        report.append("```yaml")
        report.append(f"temperature: {config_info['loss']['temperature']}")
        report.append(f"lambda_contrastive: {config_info['loss']['lambda_contrastive']}")
        report.append(f"lambda_curl: {config_info['loss']['lambda_curl']}")
        report.append(f"lambda_div: {config_info['loss']['lambda_div']}")
        report.append(f"lambda_identity: {config_info['loss']['lambda_identity']}")
        report.append("```")
        report.append("")

    # ====================
    # 2. Training Curves
    # ====================
    report.append("## 2. Training Progress")
    report.append("")

    plots_dir = os.path.join(base_dir, 'plots')
    if os.path.exists(os.path.join(plots_dir, 'training_curves.png')):
        report.append("### Loss Curves")
        report.append("")
        report.append(f"![Training Curves](plots/training_curves.png)")
        report.append("")

    if os.path.exists(os.path.join(plots_dir, 'loss_breakdown.png')):
        report.append("### Loss Component Breakdown")
        report.append("")
        report.append(f"![Loss Breakdown](plots/loss_breakdown.png)")
        report.append("")

    # ====================
    # 3. Flow Depth Analysis
    # ====================
    report.append("## 3. Flow Depth Analysis (K Optimization)")
    report.append("")
    report.append("Following cs229.ipynb approach: train fresh classifier at each K value.")
    report.append("")

    if flow_depth:
        analysis = flow_depth.get('optimal_K_analysis', {})

        report.append("### Optimal K Recommendation")
        report.append("")
        report.append(f"**Recommended K:** {analysis.get('optimal_K', 'N/A')}")
        report.append("")
        report.append(f"**Reasoning:** {analysis.get('reasoning', 'N/A')}")
        report.append("")

        report.append("### K Value Comparison")
        report.append("")
        report.append("| K | Mean Test Acc | Mean Test AUC | Within-Class Dist | Between-Class Dist | Geometry Ratio |")
        report.append("|---|---------------|---------------|-------------------|--------------------|----|")

        for result in flow_depth.get('results', []):
            K = result['K']
            acc = format_metric(result['mean_test_acc'])
            auc = format_metric(result['mean_test_auc'])
            within = format_metric(result['within_class_dist'])
            between = format_metric(result['between_class_dist'])
            ratio = format_metric(result['geometry_ratio'], 2)

            report.append(f"| {K} | {acc} | {auc} | {within} | {between} | {ratio} |")

        report.append("")

        # Add plots
        flow_depth_dir = os.path.join(base_dir, 'flow_depth')
        if os.path.exists(os.path.join(flow_depth_dir, 'accuracy_vs_K.png')):
            report.append("### Per-Attribute Accuracy vs K")
            report.append("")
            report.append(f"![Accuracy vs K](flow_depth/accuracy_vs_K.png)")
            report.append("")

        if os.path.exists(os.path.join(flow_depth_dir, 'geometry_vs_K.png')):
            report.append("### Geometry Evolution vs K")
            report.append("")
            report.append(f"![Geometry vs K](flow_depth/geometry_vs_K.png)")
            report.append("")

    # ====================
    # 4. Method Comparison
    # ====================
    report.append("## 4. Method Comparison: FCLF vs Linear Steering")
    report.append("")

    if paper_metrics and 'comparison' in paper_metrics:
        fclf = paper_metrics['comparison']['fclf']
        linear = paper_metrics['comparison']['linear_steering']

        report.append("### Per-Attribute Linear Probe Accuracy")
        report.append("")
        report.append("| Attribute | FCLF | Linear | Δ |")
        report.append("|-----------|------|--------|---|")

        for attr in ['Smiling', 'Young', 'Male', 'Eyeglasses', 'Mustache']:
            fclf_acc = format_metric(fclf['linear_probe'][attr])
            lin_acc = format_metric(linear['linear_probe'][attr])
            diff = fclf['linear_probe'][attr] - linear['linear_probe'][attr]
            diff_str = f"{diff:+.4f}"

            report.append(f"| {attr} | {fclf_acc} | {lin_acc} | {diff_str} |")

        # Mean
        fclf_mean = sum(fclf['linear_probe'].values()) / 5
        lin_mean = sum(linear['linear_probe'].values()) / 5
        diff_mean = fclf_mean - lin_mean
        report.append(f"| **MEAN** | **{fclf_mean:.4f}** | **{lin_mean:.4f}** | **{diff_mean:+.4f}** |")
        report.append("")

        report.append("### Overall Metrics")
        report.append("")
        report.append("| Metric | FCLF | Linear | Δ |")
        report.append("|--------|------|--------|---|")

        metrics_to_show = [
            ('k-NN Purity', 'knn_purity', 4),
            ('Centroid Distance', 'centroid_distance', 4),
            ('Within-Class Distance', 'within_class_distance', 4),
            ('Between-Class Distance', 'between_class_distance', 4),
            ('Geometry Ratio', 'geometry_ratio', 2)
        ]

        for label, key, prec in metrics_to_show:
            if key in fclf and key in linear:
                fclf_val = format_metric(fclf[key], prec)
                lin_val = format_metric(linear[key], prec)
                diff = fclf[key] - linear[key]
                diff_str = f"{diff:+.{prec}f}"
                report.append(f"| {label} | {fclf_val} | {lin_val} | {diff_str} |")

        report.append("")

    # ====================
    # 5. Attribute Leakage
    # ====================
    report.append("## 5. Attribute Leakage")
    report.append("")
    report.append("Measures how much NON-target attributes change during flow (should be ~0).")
    report.append("")

    if paper_metrics and 'attribute_leakage' in paper_metrics:
        leakage = paper_metrics['attribute_leakage']

        report.append("| Attribute | Start Acc | End Acc | Leakage |")
        report.append("|-----------|-----------|---------|---------|")

        for attr in ['Smiling', 'Young', 'Male', 'Eyeglasses', 'Mustache']:
            if attr in leakage and 'note' not in leakage[attr]:
                start = format_metric(leakage[attr]['accuracy_start'])
                end = format_metric(leakage[attr]['accuracy_end'])
                leak = format_metric(leakage[attr]['leakage'])
                report.append(f"| {attr} | {start} | {end} | {leak} |")

        if '_overall' in leakage:
            mean_leak = format_metric(leakage['_overall']['mean_leakage'])
            max_leak = format_metric(leakage['_overall']['max_leakage'])
            report.append(f"| **Mean** | - | - | **{mean_leak}** |")
            report.append(f"| **Max** | - | - | **{max_leak}** |")

        report.append("")

    # ====================
    # 6. AUC Along Path
    # ====================
    report.append("## 6. AUC Progression Along Flow Path")
    report.append("")

    paper_metrics_dir = os.path.join(base_dir, 'paper_metrics')
    if os.path.exists(os.path.join(paper_metrics_dir, 'auc_curves.png')):
        report.append(f"![AUC Curves](paper_metrics/auc_curves.png)")
        report.append("")

    if paper_metrics and 'monotonic_auc_fraction' in paper_metrics:
        mono = paper_metrics['monotonic_auc_fraction']
        report.append("### Monotonic AUC Increase")
        report.append("")
        report.append("Fraction of trajectories with monotonically increasing AUC:")
        report.append("")
        report.append("| Attribute | Monotonic Fraction |")
        report.append("|-----------|-------------------|")

        for attr in ['Smiling', 'Young', 'Male', 'Eyeglasses', 'Mustache']:
            if attr in mono:
                frac = format_metric(mono[attr], 2)
                report.append(f"| {attr} | {frac} |")

        report.append("")

    # ====================
    # 7. Field Diagnostics
    # ====================
    report.append("## 7. Vector Field Diagnostics")
    report.append("")

    if paper_metrics and 'field_diagnostics' in paper_metrics:
        diag = paper_metrics['field_diagnostics']

        report.append("| Metric | Mean | Std | Max |")
        report.append("|--------|------|-----|-----|")

        if 'divergence' in diag:
            div = diag['divergence']
            report.append(f"| Divergence | {format_metric(div['mean'])} | {format_metric(div['std'])} | {format_metric(div['max'])} |")

        if 'curl' in diag:
            curl = diag['curl']
            report.append(f"| Curl | {format_metric(curl['mean'])} | {format_metric(curl['std'])} | {format_metric(curl['max'])} |")

        report.append("")

    # ====================
    # 8. Visualizations
    # ====================
    report.append("## 8. Qualitative Results")
    report.append("")

    flipbook_dir = os.path.join(base_dir, 'paper_metrics', 'flipbooks')
    if os.path.exists(flipbook_dir):
        report.append("### Nearest-Neighbor Flipbooks")
        report.append("")
        report.append(f"Flipbook visualizations showing flow trajectories can be found in:")
        report.append(f"```")
        report.append(f"{flipbook_dir}")
        report.append(f"```")
        report.append("")

    # ====================
    # 9. Summary
    # ====================
    report.append("## 9. Summary & Recommendations")
    report.append("")

    if flow_depth and 'optimal_K_analysis' in flow_depth:
        opt_k = flow_depth['optimal_K_analysis']['optimal_K']
        best_acc = flow_depth['optimal_K_analysis']['best_accuracy']

        report.append(f"- **Optimal flow depth:** K = {opt_k} (test accuracy: {best_acc:.4f})")

    if paper_metrics and 'comparison' in paper_metrics:
        fclf_purity = paper_metrics['comparison']['fclf'].get('knn_purity', 0)
        fclf_geo = paper_metrics['comparison']['fclf'].get('geometry_ratio', 0)

        report.append(f"- **Clustering quality:** k-NN purity = {fclf_purity:.4f}, geometry ratio = {fclf_geo:.2f}")

    if paper_metrics and 'attribute_leakage' in paper_metrics:
        mean_leak = paper_metrics['attribute_leakage']['_overall']['mean_leakage']
        report.append(f"- **Attribute specificity:** Mean leakage = {mean_leak:.4f} (lower is better)")

    report.append("")
    report.append("---")
    report.append("")
    report.append("*Report generated by `scripts/create_evaluation_report.py`*")

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"✅ Evaluation report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive evaluation report')
    parser.add_argument('--base_dir', type=str, required=True,
                       help='Base directory containing all evaluation results')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for Markdown report')
    args = parser.parse_args()

    if not os.path.exists(args.base_dir):
        print(f"Error: Base directory not found: {args.base_dir}")
        return

    print("="*80)
    print("GENERATING COMPREHENSIVE EVALUATION REPORT")
    print("="*80)
    print(f"\nBase directory: {args.base_dir}")
    print(f"Output: {args.output}")

    # Generate report
    generate_report(args.base_dir, args.output)

    print("\n" + "="*80)
    print("REPORT GENERATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
