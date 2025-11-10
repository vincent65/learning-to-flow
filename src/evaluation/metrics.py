"""
Evaluation metrics for FCLF.
"""

import torch
import numpy as np
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from typing import Dict, Tuple


def compute_silhouette_score(
    embeddings: np.ndarray,
    attributes: np.ndarray
) -> Dict[str, float]:
    """
    Compute silhouette score for embeddings grouped by attributes.

    Args:
        embeddings: [N, dim] embeddings
        attributes: [N, num_attrs] binary attribute vectors

    Returns:
        Dictionary with silhouette scores per attribute and overall
    """
    num_attrs = attributes.shape[1]
    scores = {}

    # Overall score using all attributes as labels
    # Convert attribute vectors to single label
    labels = (attributes * (2 ** np.arange(num_attrs))).sum(axis=1)

    # Only compute if we have more than one cluster
    if len(np.unique(labels)) > 1:
        overall_score = silhouette_score(embeddings, labels)
        scores['overall'] = float(overall_score)
    else:
        scores['overall'] = 0.0

    # Per-attribute scores
    for i in range(num_attrs):
        attr_labels = attributes[:, i]
        if len(np.unique(attr_labels)) > 1:
            score = silhouette_score(embeddings, attr_labels)
            scores[f'attr_{i}'] = float(score)
        else:
            scores[f'attr_{i}'] = 0.0

    return scores


def compute_cluster_purity(
    embeddings: np.ndarray,
    attributes: np.ndarray,
    n_clusters: int = 10
) -> float:
    """
    Compute cluster purity using k-means clustering.

    Args:
        embeddings: [N, dim] embeddings
        attributes: [N, num_attrs] attribute vectors
        n_clusters: Number of clusters for k-means

    Returns:
        Cluster purity score
    """
    from sklearn.cluster import KMeans

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Convert attributes to single labels
    num_attrs = attributes.shape[1]
    true_labels = (attributes * (2 ** np.arange(num_attrs))).sum(axis=1)

    # Compute purity
    total_correct = 0
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        if cluster_mask.sum() == 0:
            continue

        # Most common true label in this cluster
        cluster_true_labels = true_labels[cluster_mask]
        most_common = np.bincount(cluster_true_labels.astype(int)).argmax()
        correct = (cluster_true_labels == most_common).sum()
        total_correct += correct

    purity = total_correct / len(true_labels)
    return float(purity)


def train_attribute_classifier(
    train_embeddings: np.ndarray,
    train_attributes: np.ndarray,
    val_embeddings: np.ndarray,
    val_attributes: np.ndarray
) -> Tuple[float, Dict[str, float]]:
    """
    Train a simple classifier to predict attributes from embeddings.

    Args:
        train_embeddings: [N, dim] training embeddings
        train_attributes: [N, num_attrs] training attributes
        val_embeddings: [M, dim] validation embeddings
        val_attributes: [M, num_attrs] validation attributes

    Returns:
        (overall_accuracy, per_attribute_accuracy_dict)
    """
    num_attrs = train_attributes.shape[1]

    per_attr_acc = {}
    all_preds = []
    all_true = []

    for i in range(num_attrs):
        # Train classifier for this attribute
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(train_embeddings, train_attributes[:, i])

        # Predict on validation
        val_pred = clf.predict(val_embeddings)
        val_true = val_attributes[:, i]

        # Accuracy
        acc = accuracy_score(val_true, val_pred)
        per_attr_acc[f'attr_{i}'] = float(acc)

        all_preds.append(val_pred)
        all_true.append(val_true)

    # Overall accuracy (exact match on all attributes)
    all_preds = np.stack(all_preds, axis=1)
    all_true = np.stack(all_true, axis=1)

    exact_match = (all_preds == all_true).all(axis=1).mean()

    return float(exact_match), per_attr_acc


def compute_trajectory_smoothness(trajectories: np.ndarray) -> Dict[str, float]:
    """
    Compute smoothness of flow trajectories.

    Measures L2 distance between consecutive steps.

    Args:
        trajectories: [N, num_steps, dim] trajectories

    Returns:
        Dictionary with smoothness metrics
    """
    # Compute distances between consecutive steps
    diffs = np.diff(trajectories, axis=1)  # [N, num_steps-1, dim]
    distances = np.linalg.norm(diffs, axis=2)  # [N, num_steps-1]

    metrics = {
        'mean_step_distance': float(distances.mean()),
        'std_step_distance': float(distances.std()),
        'max_step_distance': float(distances.max()),
        'min_step_distance': float(distances.min())
    }

    return metrics


def compute_attribute_transfer_success(
    original_attributes: np.ndarray,
    target_attributes: np.ndarray,
    predicted_attributes: np.ndarray
) -> Dict[str, float]:
    """
    Measure how successfully attributes were transferred.

    Args:
        original_attributes: [N, num_attrs] original attributes
        target_attributes: [N, num_attrs] target attributes
        predicted_attributes: [N, num_attrs] predicted attributes after flow

    Returns:
        Dictionary with transfer success metrics
    """
    # Per-attribute transfer success
    num_attrs = original_attributes.shape[1]
    per_attr_success = {}

    for i in range(num_attrs):
        # Only consider samples where target differs from original
        changed_mask = original_attributes[:, i] != target_attributes[:, i]

        if changed_mask.sum() == 0:
            per_attr_success[f'attr_{i}'] = 0.0
            continue

        # Success = predicted matches target
        success = (predicted_attributes[changed_mask, i] == target_attributes[changed_mask, i]).mean()
        per_attr_success[f'attr_{i}'] = float(success)

    # Overall transfer success
    changed_mask = (original_attributes != target_attributes).any(axis=1)
    if changed_mask.sum() > 0:
        exact_match = (predicted_attributes[changed_mask] == target_attributes[changed_mask]).all(axis=1).mean()
        overall_success = float(exact_match)
    else:
        overall_success = 0.0

    return {
        'overall': overall_success,
        **per_attr_success
    }


def evaluate_all_metrics(
    original_embeddings: np.ndarray,
    flowed_embeddings: np.ndarray,
    original_attributes: np.ndarray,
    target_attributes: np.ndarray,
    trajectories: np.ndarray = None
) -> Dict:
    """
    Compute all evaluation metrics.

    Args:
        original_embeddings: [N, dim] original embeddings
        flowed_embeddings: [N, dim] flowed embeddings
        original_attributes: [N, num_attrs] original attributes
        target_attributes: [N, num_attrs] target attributes
        trajectories: [N, num_steps, dim] optional trajectories

    Returns:
        Dictionary with all metrics
    """
    results = {}

    # Silhouette scores
    print("Computing silhouette scores...")
    results['silhouette_original'] = compute_silhouette_score(
        original_embeddings, original_attributes
    )
    results['silhouette_flowed'] = compute_silhouette_score(
        flowed_embeddings, target_attributes
    )

    # Cluster purity
    print("Computing cluster purity...")
    results['purity_original'] = compute_cluster_purity(
        original_embeddings, original_attributes
    )
    results['purity_flowed'] = compute_cluster_purity(
        flowed_embeddings, target_attributes
    )

    # Trajectory smoothness
    if trajectories is not None:
        print("Computing trajectory smoothness...")
        results['smoothness'] = compute_trajectory_smoothness(trajectories)

    return results
