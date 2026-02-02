# src/evaluation/metrics.py
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from matching.retrieval import PlaceRetrievalSystem


def compute_pose_distance(pose1: Dict[str, float], pose2: Dict[str, float]) -> float:
    """
    Compute Euclidean distance between two poses in x-y plane.
    
    Args:
        pose1: First pose dictionary with 'x' and 'y' keys
        pose2: Second pose dictionary with 'x' and 'y' keys
        
    Returns:
        Euclidean distance in meters
    """
    x1, y1 = pose1.get('x', 0), pose1.get('y', 0)
    x2, y2 = pose2.get('x', 0), pose2.get('y', 0)
    
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def define_ground_truth(
    query_pose: Dict[str, float],
    database_poses: List[Dict[str, float]],
    positive_threshold: float = 5.0,
    negative_threshold: float = 20.0
) -> np.ndarray:
    """
    Define ground truth labels for database entries based on pose distance.
    
    Args:
        query_pose: Query pose dictionary
        database_poses: List of database pose dictionaries
        positive_threshold: Distance threshold for positive match (meters)
        negative_threshold: Distance threshold for negative match (meters)
        
    Returns:
        Binary array: 1 for positive match, 0 for negative match, -1 for ambiguous
    """
    labels = np.zeros(len(database_poses), dtype=int)
    
    for i, db_pose in enumerate(database_poses):
        distance = compute_pose_distance(query_pose, db_pose)
        
        if distance <= positive_threshold:
            labels[i] = 1  # Positive match
        elif distance >= negative_threshold:
            labels[i] = 0  # Negative match
        else:
            labels[i] = -1  # Ambiguous (between thresholds)
    
    return labels


def compute_precision_recall(
    retrieved_indices: List[int],
    ground_truth_labels: np.ndarray,
    k: Optional[int] = None
) -> Tuple[float, float]:
    """
    Compute precision and recall for retrieved results.
    
    Args:
        retrieved_indices: List of retrieved database indices
        ground_truth_labels: Ground truth binary labels (1=positive, 0=negative, -1=ambiguous)
        k: Number of top results to consider (None = all)
        
    Returns:
        Tuple of (precision, recall)
    """
    if k is not None:
        retrieved_indices = retrieved_indices[:k]
    
    if len(retrieved_indices) == 0:
        return 0.0, 0.0
    
    # Filter out ambiguous labels
    valid_mask = ground_truth_labels != -1
    if not np.any(valid_mask):
        return 0.0, 0.0
    
    # Count true positives and false positives
    retrieved_labels = ground_truth_labels[retrieved_indices]
    valid_retrieved = retrieved_labels != -1
    
    if not np.any(valid_retrieved):
        return 0.0, 0.0
    
    true_positives = np.sum(retrieved_labels[valid_retrieved] == 1)
    false_positives = np.sum(retrieved_labels[valid_retrieved] == 0)
    
    # Precision = TP / (TP + FP)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    
    # Recall = TP / (TP + FN)
    total_positives = np.sum(ground_truth_labels[valid_mask] == 1)
    recall = true_positives / total_positives if total_positives > 0 else 0.0
    
    return precision, recall


def compute_f1_score(precision: float, recall: float) -> float:
    """
    Compute F1 score from precision and recall.
    
    Args:
        precision: Precision value
        recall: Recall value
        
    Returns:
        F1 score
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def compute_average_precision(
    similarities: np.ndarray,
    ground_truth_labels: np.ndarray,
    k: Optional[int] = None
) -> float:
    """
    Compute Average Precision (AP) using precision-recall curve.
    
    Args:
        similarities: Similarity scores for retrieved results
        ground_truth_labels: Ground truth binary labels
        k: Number of top results to consider (None = all)
        
    Returns:
        Average Precision score
    """
    if k is not None:
        similarities = similarities[:k]
        ground_truth_labels = ground_truth_labels[:k]
    
    # Filter out ambiguous labels
    valid_mask = ground_truth_labels != -1
    if not np.any(valid_mask):
        return 0.0
    
    similarities = similarities[valid_mask]
    labels = ground_truth_labels[valid_mask]
    
    if len(similarities) == 0:
        return 0.0
    
    # Sort by similarity (descending)
    sorted_indices = np.argsort(similarities)[::-1]
    sorted_labels = labels[sorted_indices]
    
    # Compute precision at each recall level
    precisions = []
    recalls = []
    true_positives = 0
    total_positives = np.sum(sorted_labels == 1)
    
    if total_positives == 0:
        return 0.0
    
    for i in range(len(sorted_labels)):
        if sorted_labels[i] == 1:
            true_positives += 1
        
        precision = true_positives / (i + 1)
        recall = true_positives / total_positives
        
        precisions.append(precision)
        recalls.append(recall)
    
    # Compute AP using trapezoidal rule
    if len(precisions) == 0:
        return 0.0
    
    # Use sklearn's precision_recall_curve for more accurate AP
    try:
        ap = auc(recalls, precisions)
    except:
        # Fallback: simple average
        ap = np.mean(precisions) if len(precisions) > 0 else 0.0
    
    return ap


def compute_roc_curve(
    similarities: np.ndarray,
    ground_truth_labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve.
    
    Args:
        similarities: Similarity scores
        ground_truth_labels: Ground truth binary labels
        
    Returns:
        Tuple of (fpr, tpr, thresholds)
    """
    # Filter out ambiguous labels
    valid_mask = ground_truth_labels != -1
    if not np.any(valid_mask):
        return np.array([]), np.array([]), np.array([])
    
    similarities = similarities[valid_mask]
    labels = ground_truth_labels[valid_mask]
    
    if len(similarities) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Convert labels to binary (0/1)
    binary_labels = (labels == 1).astype(int)
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(binary_labels, similarities)
    
    return fpr, tpr, thresholds


def compute_pr_curve(
    similarities: np.ndarray,
    ground_truth_labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Precision-Recall curve.
    
    Args:
        similarities: Similarity scores
        ground_truth_labels: Ground truth binary labels
        
    Returns:
        Tuple of (precision, recall, thresholds)
    """
    # Filter out ambiguous labels
    valid_mask = ground_truth_labels != -1
    if not np.any(valid_mask):
        return np.array([]), np.array([]), np.array([])
    
    similarities = similarities[valid_mask]
    labels = ground_truth_labels[valid_mask]
    
    if len(similarities) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Convert labels to binary (0/1)
    binary_labels = (labels == 1).astype(int)
    
    # Compute PR curve
    precision, recall, thresholds = precision_recall_curve(binary_labels, similarities)
    
    return precision, recall, thresholds


def evaluate_system(
    retrieval_system: PlaceRetrievalSystem,
    query_descriptors: List[np.ndarray],
    query_poses: List[Dict[str, float]],
    query_metadata: List[Dict[str, Any]],
    database_poses: List[Dict[str, float]],
    positive_threshold: float = 5.0,
    negative_threshold: float = 20.0,
    k: int = 5,
    similarity_thresholds: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Evaluate the place recognition system.
    
    Args:
        retrieval_system: PlaceRetrievalSystem instance
        query_descriptors: List of query descriptors
        query_poses: List of query pose dictionaries
        query_metadata: List of query metadata dictionaries
        database_poses: List of database pose dictionaries
        positive_threshold: Distance threshold for positive match (meters)
        negative_threshold: Distance threshold for negative match (meters)
        k: Number of neighbors to retrieve
        similarity_thresholds: List of similarity thresholds for curve generation
        
    Returns:
        Dictionary with evaluation results including metrics and curves
    """
    results = {
        'per_query_results': [],
        'overall_metrics': {},
        'pr_curves': [],
        'roc_curves': []
    }
    
    all_precisions = []
    all_recalls = []
    all_f1_scores = []
    all_aps = []
    
    # Evaluate each query
    for i, (query_desc, query_pose, query_meta) in enumerate(zip(
        query_descriptors, query_poses, query_metadata
    )):
        # Define ground truth for this query
        gt_labels = define_ground_truth(
            query_pose, database_poses,
            positive_threshold, negative_threshold
        )
        
        # Query the system for top-k (for metrics)
        retrieved = retrieval_system.query(
            query_desc,
            method='knn',
            k=k,
            query_metadata=query_meta
        )
        
        if len(retrieved) == 0:
            continue
        
        # Extract indices and similarities for top-k
        retrieved_indices = [r[0] for r in retrieved]
        retrieved_similarities_topk = np.array([r[1] for r in retrieved])
        retrieved_gt_labels_topk = gt_labels[retrieved_indices]
        
        # Compute metrics at top-k
        precision, recall = compute_precision_recall(
            retrieved_indices, gt_labels, k=k
        )
        f1 = compute_f1_score(precision, recall)
        ap = compute_average_precision(retrieved_similarities_topk, retrieved_gt_labels_topk, k=k)
        
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1_scores.append(f1)
        all_aps.append(ap)
        
        # Store per-query results
        results['per_query_results'].append({
            'query_id': i,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'average_precision': ap,
            'num_retrieved': len(retrieved),
            'num_positive': np.sum(gt_labels == 1)
        })
        
        # Compute similarities with ALL database entries for proper PR/ROC curves
        all_similarities = []
        all_gt_labels = []
        
        # Get all descriptors from retrieval system
        for db_idx in range(retrieval_system.get_database_size()):
            db_desc = retrieval_system.descriptors[db_idx]
            sim, _ = retrieval_system.compute_similarity(query_desc, db_desc, handle_rotation=True)
            all_similarities.append(sim)
            all_gt_labels.append(gt_labels[db_idx])
        
        all_similarities = np.array(all_similarities)
        all_gt_labels = np.array(all_gt_labels)
        
        # Compute PR and ROC curves using ALL database entries
        pr_precision, pr_recall, _ = compute_pr_curve(
            all_similarities, all_gt_labels
        )
        roc_fpr, roc_tpr, _ = compute_roc_curve(
            all_similarities, all_gt_labels
        )
        
        # Only store if we have valid curves
        if len(pr_precision) > 0 and len(pr_recall) > 0:
            results['pr_curves'].append({
                'query_id': i,
                'precision': pr_precision,
                'recall': pr_recall
            })
        
        if len(roc_fpr) > 0 and len(roc_tpr) > 0:
            results['roc_curves'].append({
                'query_id': i,
                'fpr': roc_fpr,
                'tpr': roc_tpr
            })
    
    # Compute overall metrics
    if len(all_precisions) > 0:
        results['overall_metrics'] = {
            'mean_precision': np.mean(all_precisions),
            'mean_recall': np.mean(all_recalls),
            'mean_f1_score': np.mean(all_f1_scores),
            'mean_average_precision': np.mean(all_aps),
            'std_precision': np.std(all_precisions),
            'std_recall': np.std(all_recalls),
            'std_f1_score': np.std(all_f1_scores),
            'std_average_precision': np.std(all_aps),
            'num_queries': len(all_precisions)
        }
    else:
        results['overall_metrics'] = {
            'mean_precision': 0.0,
            'mean_recall': 0.0,
            'mean_f1_score': 0.0,
            'mean_average_precision': 0.0,
            'num_queries': 0
        }
    
    return results
