"""
Analyze and visualize comprehensive metrics results.
Creates publication-ready tables and plots.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_metrics(json_path):
    """Load metrics from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def print_summary(metrics):
    """Print a formatted summary of all metrics."""
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*80)

    # 1. Linear Probe Accuracy
    print("\n📊 1. LINEAR PROBE ACCURACY (Per Attribute)")
    print("-" * 80)
    print(f"{'Attribute':<15} {'Original':>12} {'Flowed':>12} {'Improvement':>15} {'Status':>10}")
    print("-" * 80)

    attr_names = ['Smiling', 'Young', 'Male', 'Eyeglasses', 'Mustache']
    total_improvement = 0

    for i, name in enumerate(attr_names):
        metrics_attr = metrics['linear_probe'][f'attr_{i}']
        orig = metrics_attr['original_accuracy']
        flow = metrics_attr['flowed_accuracy']
        imp = metrics_attr['improvement']
        total_improvement += imp

        status = "✅" if imp > 0 else "❌"
        print(f"{name:<15} {orig:>12.3f} {flow:>12.3f} {imp:>+15.3f} {status:>10}")

    avg_improvement = total_improvement / 5
    print("-" * 80)
    print(f"{'AVERAGE':<15} {'':>12} {'':>12} {avg_improvement:>+15.3f}")

    # Interpretation
    print("\n💡 Interpretation:")
    if avg_improvement > 0.05:
        print("   ✅ EXCELLENT: Target attributes became MORE linearly separable after flow")
    elif avg_improvement > 0:
        print("   ✅ GOOD: Slight improvement in linear separability")
    else:
        print("   ⚠️  WARNING: Attributes became LESS separable (unexpected!)")

    # 2. Clustering Quality
    print("\n\n📊 2. CLUSTERING QUALITY")
    print("-" * 80)
    print(f"{'Metric':<30} {'Original→Target':>20} {'Flowed→Target':>20} {'Improvement':>15}")
    print("-" * 80)

    orig_target = metrics['clustering']['original_by_target']
    flow_target = metrics['clustering']['flowed_by_target']

    # Silhouette
    sil_orig = orig_target['silhouette']
    sil_flow = flow_target['silhouette']
    sil_imp = sil_flow - sil_orig
    print(f"{'Silhouette Score':<30} {sil_orig:>20.3f} {sil_flow:>20.3f} {sil_imp:>+15.3f}")

    # Calinski-Harabasz (higher is better)
    ch_orig = orig_target['calinski_harabasz']
    ch_flow = flow_target['calinski_harabasz']
    ch_imp = ch_flow - ch_orig
    print(f"{'Calinski-Harabasz':<30} {ch_orig:>20.1f} {ch_flow:>20.1f} {ch_imp:>+15.1f}")

    # Davies-Bouldin (lower is better)
    db_orig = orig_target['davies_bouldin']
    db_flow = flow_target['davies_bouldin']
    db_imp = db_orig - db_flow  # Note: reversed (lower is better)
    print(f"{'Davies-Bouldin':<30} {db_orig:>20.3f} {db_flow:>20.3f} {db_imp:>+15.3f}")

    print("\n💡 Interpretation:")
    if sil_flow > 0.3 and ch_imp > 0 and db_imp > 0:
        print("   ✅ EXCELLENT: Embeddings cluster well by target attributes")
    elif sil_flow > 0.15:
        print("   ✅ GOOD: Moderate clustering by target attributes")
    else:
        print("   ⚠️  WEAK: Limited clustering improvement")

    # 3. k-NN Purity
    print("\n\n📊 3. k-NN CLASS PURITY (k=10)")
    print("-" * 80)
    knn_orig = metrics['knn_purity_k10']['original_by_target']
    knn_flow = metrics['knn_purity_k10']['flowed_by_target']
    knn_imp = knn_flow - knn_orig

    print(f"Original → Target: {knn_orig:.3f}")
    print(f"Flowed → Target:   {knn_flow:.3f}")
    print(f"Improvement:       {knn_imp:+.3f}")

    print("\n💡 Interpretation:")
    print(f"   After flow, {knn_flow*100:.1f}% of k-nearest neighbors share target attributes")
    if knn_flow > 0.8:
        print("   ✅ EXCELLENT: Very tight clustering")
    elif knn_flow > 0.6:
        print("   ✅ GOOD: Reasonable clustering")
    else:
        print("   ⚠️  WEAK: Poor local clustering")

    # 4. Centroid Distance
    print("\n\n📊 4. DISTANCE TO CLASS CENTROIDS")
    print("-" * 80)
    cent_orig = metrics['centroid_distance']['original_to_target']
    cent_flow = metrics['centroid_distance']['flowed_to_target']
    cent_imp = cent_orig - cent_flow
    cent_pct = (cent_imp / cent_orig) * 100

    print(f"Original → Target: {cent_orig:.4f}")
    print(f"Flowed → Target:   {cent_flow:.4f}")
    print(f"Improvement:       {cent_imp:+.4f} ({cent_pct:+.1f}%)")

    print("\n💡 Interpretation:")
    if cent_pct > 80:
        print(f"   ✅ EXCELLENT: {cent_pct:.0f}% reduction in distance to target centroids")
    elif cent_pct > 50:
        print(f"   ✅ GOOD: {cent_pct:.0f}% reduction")
    else:
        print(f"   ⚠️  MODERATE: {cent_pct:.0f}% reduction")

    # 5. Path Quality
    print("\n\n📊 5. PATH QUALITY METRICS")
    print("-" * 80)
    mono_frac = metrics['path_quality']['fraction_monotonic']
    cosine = metrics['path_quality']['mean_cosine_similarity']
    efficiency = metrics['path_quality']['mean_path_efficiency']

    print(f"Monotonic Progress:    {mono_frac:.1%}")
    print(f"Cosine Similarity:     {cosine:.3f}")
    print(f"Path Efficiency:       {efficiency:.3f}")

    print("\n💡 Interpretation:")
    if mono_frac > 0.7:
        print(f"   ✅ EXCELLENT: {mono_frac:.0%} of paths monotonically approach target")
    elif mono_frac > 0.5:
        print(f"   ✅ GOOD: {mono_frac:.0%} monotonic paths")
    else:
        print(f"   ⚠️  WEAK: Only {mono_frac:.0%} monotonic")

    if cosine > 0.8:
        print(f"   ✅ EXCELLENT: Very smooth paths (cosine {cosine:.2f})")
    elif cosine > 0.5:
        print(f"   ✅ GOOD: Moderately smooth paths")
    else:
        print(f"   ⚠️  WEAK: Erratic paths")

    if efficiency > 0.8:
        print(f"   ✅ EXCELLENT: Highly efficient paths ({efficiency:.0%} of straight-line)")
    elif efficiency > 0.6:
        print(f"   ✅ GOOD: Reasonable efficiency")
    else:
        print(f"   ⚠️  WEAK: Inefficient paths")

    # Overall Assessment
    print("\n\n" + "="*80)
    print("OVERALL ASSESSMENT")
    print("="*80)

    score = 0
    total = 0

    # Scoring
    if sil_flow > 0.2: score += 1
    total += 1

    if knn_flow > 0.7: score += 1
    total += 1

    if cent_pct > 50: score += 1
    total += 1

    if mono_frac > 0.6: score += 1
    total += 1

    if cosine > 0.7: score += 1
    total += 1

    percentage = (score / total) * 100

    print(f"\nPassed {score}/{total} quality checks ({percentage:.0f}%)")

    if percentage >= 80:
        print("\n🎉 EXCELLENT: Model is working very well!")
        print("   ✅ Strong attribute transfer")
        print("   ✅ Good path quality")
        print("   ✅ Ready for publication")
    elif percentage >= 60:
        print("\n✅ GOOD: Model is working reasonably well")
        print("   Some room for improvement")
    else:
        print("\n⚠️  NEEDS IMPROVEMENT")
        print("   Consider: more training, hyperparameter tuning, or architecture changes")

    print("\n" + "="*80 + "\n")


def create_visualizations(metrics, output_dir='comprehensive_metrics'):
    """Create publication-ready visualizations."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    # 1. Linear Probe Comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    attr_names = ['Smiling', 'Young', 'Male', 'Eyeglasses', 'Mustache']
    orig_accs = [metrics['linear_probe'][f'attr_{i}']['original_accuracy'] for i in range(5)]
    flow_accs = [metrics['linear_probe'][f'attr_{i}']['flowed_accuracy'] for i in range(5)]

    x = np.arange(len(attr_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, orig_accs, width, label='Original Embeddings', color='steelblue')
    bars2 = ax.bar(x + width/2, flow_accs, width, label='Flowed Embeddings', color='coral')

    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Linear Probe Accuracy: Original vs Flowed Embeddings', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(attr_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/linear_probe_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/linear_probe_comparison.png")
    plt.close()

    # 2. Clustering Metrics Summary
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Silhouette
    metrics_names = ['Original→Target', 'Flowed→Target']
    sil_scores = [
        metrics['clustering']['original_by_target']['silhouette'],
        metrics['clustering']['flowed_by_target']['silhouette']
    ]
    axes[0].bar(metrics_names, sil_scores, color=['steelblue', 'coral'])
    axes[0].set_ylabel('Silhouette Score', fontsize=11)
    axes[0].set_title('Silhouette Score\n(Higher = Better)', fontweight='bold')
    axes[0].axhline(0, color='black', linestyle='--', linewidth=0.5)
    axes[0].grid(axis='y', alpha=0.3)

    # Calinski-Harabasz
    ch_scores = [
        metrics['clustering']['original_by_target']['calinski_harabasz'],
        metrics['clustering']['flowed_by_target']['calinski_harabasz']
    ]
    axes[1].bar(metrics_names, ch_scores, color=['steelblue', 'coral'])
    axes[1].set_ylabel('Calinski-Harabasz Score', fontsize=11)
    axes[1].set_title('Calinski-Harabasz\n(Higher = Better)', fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    # Davies-Bouldin
    db_scores = [
        metrics['clustering']['original_by_target']['davies_bouldin'],
        metrics['clustering']['flowed_by_target']['davies_bouldin']
    ]
    axes[2].bar(metrics_names, db_scores, color=['steelblue', 'coral'])
    axes[2].set_ylabel('Davies-Bouldin Score', fontsize=11)
    axes[2].set_title('Davies-Bouldin\n(Lower = Better)', fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/clustering_metrics.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/clustering_metrics.png")
    plt.close()

    # 3. Summary Dashboard
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # k-NN Purity
    ax1 = fig.add_subplot(gs[0, 0])
    knn_vals = [
        metrics['knn_purity_k10']['original_by_target'],
        metrics['knn_purity_k10']['flowed_by_target']
    ]
    ax1.bar(['Original', 'Flowed'], knn_vals, color=['steelblue', 'coral'])
    ax1.set_ylabel('k-NN Purity (k=10)')
    ax1.set_title('k-NN Class Purity', fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)

    # Centroid Distance
    ax2 = fig.add_subplot(gs[0, 1])
    cent_vals = [
        metrics['centroid_distance']['original_to_target'],
        metrics['centroid_distance']['flowed_to_target']
    ]
    ax2.bar(['Original', 'Flowed'], cent_vals, color=['steelblue', 'coral'])
    ax2.set_ylabel('Mean Distance')
    ax2.set_title('Distance to Target Centroids', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Path Quality
    ax3 = fig.add_subplot(gs[0, 2])
    path_metrics = ['Monotonic\nFraction', 'Cosine\nSimilarity', 'Path\nEfficiency']
    path_vals = [
        metrics['path_quality']['fraction_monotonic'],
        metrics['path_quality']['mean_cosine_similarity'],
        metrics['path_quality']['mean_path_efficiency']
    ]
    bars = ax3.bar(path_metrics, path_vals, color='mediumseagreen')
    ax3.set_ylabel('Score')
    ax3.set_title('Path Quality Metrics', fontweight='bold')
    ax3.set_ylim(0, 1)
    ax3.grid(axis='y', alpha=0.3)

    # Add values on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    # Summary text
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')

    summary_text = f"""
    📊 COMPREHENSIVE EVALUATION SUMMARY

    Linear Probe:       Original vs Flowed (avg improvement: {np.mean([metrics['linear_probe'][f'attr_{i}']['improvement'] for i in range(5)]):.3f})
    Silhouette:         {metrics['clustering']['original_by_target']['silhouette']:.3f} → {metrics['clustering']['flowed_by_target']['silhouette']:.3f}
    k-NN Purity:        {metrics['knn_purity_k10']['original_by_target']:.3f} → {metrics['knn_purity_k10']['flowed_by_target']:.3f}
    Centroid Distance:  {metrics['centroid_distance']['original_to_target']:.4f} → {metrics['centroid_distance']['flowed_to_target']:.4f}

    Monotonic Paths:    {metrics['path_quality']['fraction_monotonic']:.1%}
    Path Smoothness:    {metrics['path_quality']['mean_cosine_similarity']:.3f}
    Path Efficiency:    {metrics['path_quality']['mean_path_efficiency']:.3f}
    """

    ax4.text(0.5, 0.5, summary_text, ha='center', va='center',
            fontsize=11, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.savefig(f'{output_dir}/summary_dashboard.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/summary_dashboard.png")
    plt.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_metrics.py <path_to_comprehensive_metrics.json>")
        sys.exit(1)

    json_path = sys.argv[1]
    metrics = load_metrics(json_path)

    # Print summary
    print_summary(metrics)

    # Create visualizations
    import os
    output_dir = os.path.dirname(json_path)
    create_visualizations(metrics, output_dir)

    print(f"\n✅ Analysis complete! Check {output_dir}/ for visualizations.")
