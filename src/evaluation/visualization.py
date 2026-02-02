# src/evaluation/visualization.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional
import os


def plot_precision_recall_curve(
    evaluation_results: Dict[str, Any],
    save_path: Optional[str] = None,
    show_plot: bool = True
):
    """
    Plot Precision-Recall curves from evaluation results.
    
    Args:
        evaluation_results: Results dictionary from evaluate_system
        save_path: Path to save the figure (optional)
        show_plot: Whether to display the plot
    """
    plt.figure(figsize=(10, 6))
    
    pr_curves = evaluation_results.get('pr_curves', [])
    
    if len(pr_curves) > 0:
        # Plot individual query curves
        for curve_data in pr_curves:
            recall = curve_data['recall']
            precision = curve_data['precision']
            plt.plot(recall, precision, alpha=0.3, linewidth=0.5, color='gray')
        
        # Compute and plot average PR curve
        all_recalls = []
        all_precisions = []
        
        for curve_data in pr_curves:
            recall = curve_data['recall']
            precision = curve_data['precision']
            if len(recall) > 0 and len(precision) > 0:
                all_recalls.append(recall)
                all_precisions.append(precision)
        
        if len(all_recalls) > 0:
            # Interpolate to common recall values (0 to 1)
            # sklearn's precision_recall_curve returns arrays in descending threshold order
            # We need to reverse them to get increasing recall
            common_recall = np.linspace(0, 1, 100)
            
            interpolated_precisions = []
            for recall, precision in zip(all_recalls, all_precisions):
                if len(recall) > 1:
                    # Reverse arrays to get increasing recall (sklearn returns descending)
                    recall_sorted = recall[::-1]
                    precision_sorted = precision[::-1]
                    # Remove duplicates and ensure monotonic recall
                    unique_indices = np.where(np.diff(recall_sorted) >= 0)[0]
                    if len(unique_indices) > 0:
                        recall_sorted = recall_sorted[unique_indices]
                        precision_sorted = precision_sorted[unique_indices]
                    # Interpolate precision at common recall values
                    interp_precision = np.interp(common_recall, recall_sorted, precision_sorted, 
                                                left=precision_sorted[0] if len(precision_sorted) > 0 else 1.0,
                                                right=precision_sorted[-1] if len(precision_sorted) > 0 else 0.0)
                    interpolated_precisions.append(interp_precision)
            
            if len(interpolated_precisions) > 0:
                mean_precision = np.mean(interpolated_precisions, axis=0)
                std_precision = np.std(interpolated_precisions, axis=0)
                
                plt.plot(common_recall, mean_precision, 'b-', linewidth=2, label='Mean PR Curve')
                plt.fill_between(
                    common_recall,
                    mean_precision - std_precision,
                    mean_precision + std_precision,
                    alpha=0.2,
                    color='blue'
                )
    else:
        # Fallback: plot from per-query results
        per_query = evaluation_results.get('per_query_results', [])
        if len(per_query) > 0:
            recalls = [r['recall'] for r in per_query]
            precisions = [r['precision'] for r in per_query]
            plt.scatter(recalls, precisions, alpha=0.5, label='Query Results')
            
            # Plot mean
            mean_recall = np.mean(recalls)
            mean_precision = np.mean(precisions)
            plt.scatter([mean_recall], [mean_precision], color='red', s=100, marker='*', label='Mean')
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved PR curve to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_roc_curve(
    evaluation_results: Dict[str, Any],
    save_path: Optional[str] = None,
    show_plot: bool = True
):
    """
    Plot ROC curves from evaluation results.
    
    Args:
        evaluation_results: Results dictionary from evaluate_system
        save_path: Path to save the figure (optional)
        show_plot: Whether to display the plot
    """
    plt.figure(figsize=(10, 6))
    
    roc_curves = evaluation_results.get('roc_curves', [])
    
    if len(roc_curves) > 0:
        # Plot individual query curves
        for curve_data in roc_curves:
            fpr = curve_data['fpr']
            tpr = curve_data['tpr']
            if len(fpr) > 0 and len(tpr) > 0:
                plt.plot(fpr, tpr, alpha=0.3, linewidth=0.5, color='gray')
        
        # Compute and plot average ROC curve
        all_fprs = []
        all_tprs = []
        
        for curve_data in roc_curves:
            fpr = curve_data['fpr']
            tpr = curve_data['tpr']
            if len(fpr) > 0 and len(tpr) > 0:
                all_fprs.append(fpr)
                all_tprs.append(tpr)
        
        if len(all_fprs) > 0:
            # Interpolate to common FPR values (0 to 1)
            # sklearn's roc_curve returns arrays in increasing FPR order
            common_fpr = np.linspace(0, 1, 100)
            
            interpolated_tprs = []
            for fpr, tpr in zip(all_fprs, all_tprs):
                if len(fpr) > 1:
                    # Ensure FPR is sorted (should be, but just in case)
                    sort_idx = np.argsort(fpr)
                    fpr_sorted = fpr[sort_idx]
                    tpr_sorted = tpr[sort_idx]
                    # Interpolate TPR at common FPR values
                    interp_tpr = np.interp(common_fpr, fpr_sorted, tpr_sorted,
                                          left=0.0, right=tpr_sorted[-1] if len(tpr_sorted) > 0 else 1.0)
                    interpolated_tprs.append(interp_tpr)
            
            if len(interpolated_tprs) > 0:
                mean_tpr = np.mean(interpolated_tprs, axis=0)
                std_tpr = np.std(interpolated_tprs, axis=0)
                
                plt.plot(common_fpr, mean_tpr, 'b-', linewidth=2, label='Mean ROC Curve')
                plt.fill_between(
                    common_fpr,
                    mean_tpr - std_tpr,
                    mean_tpr + std_tpr,
                    alpha=0.2,
                    color='blue'
                )
    
    # Plot diagonal (random classifier)
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curve to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_metrics_summary(
    evaluation_results: Dict[str, Any],
    save_path: Optional[str] = None,
    show_plot: bool = True
):
    """
    Plot summary of evaluation metrics.
    
    Args:
        evaluation_results: Results dictionary from evaluate_system
        save_path: Path to save the figure (optional)
        show_plot: Whether to display the plot
    """
    overall_metrics = evaluation_results.get('overall_metrics', {})
    
    if not overall_metrics:
        print("No overall metrics found in evaluation results")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot of mean metrics
    metrics = ['mean_precision', 'mean_recall', 'mean_f1_score', 'mean_average_precision']
    metric_labels = ['Precision', 'Recall', 'F1 Score', 'Average Precision']
    values = [overall_metrics.get(m, 0.0) for m in metrics]
    stds = [overall_metrics.get(f'std_{m.split("_")[1]}', 0.0) for m in metrics]
    
    axes[0].bar(metric_labels, values, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Mean Evaluation Metrics', fontsize=14)
    axes[0].set_ylim([0, 1])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Box plot of per-query metrics
    per_query = evaluation_results.get('per_query_results', [])
    if len(per_query) > 0:
        precisions = [r['precision'] for r in per_query]
        recalls = [r['recall'] for r in per_query]
        f1_scores = [r['f1_score'] for r in per_query]
        aps = [r['average_precision'] for r in per_query]
        
        data = [precisions, recalls, f1_scores, aps]
        axes[1].boxplot(data, labels=metric_labels)
        axes[1].set_ylabel('Score', fontsize=12)
        axes[1].set_title('Per-Query Metrics Distribution', fontsize=14)
        axes[1].set_ylim([0, 1])
        axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved metrics summary to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_performance_analysis(
    evaluation_results: Dict[str, Any],
    save_dir: Optional[str] = None,
    show_plots: bool = True
):
    """
    Generate all performance analysis plots.
    
    Args:
        evaluation_results: Results dictionary from evaluate_system
        save_dir: Directory to save plots (optional)
        show_plots: Whether to display plots
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        pr_path = os.path.join(save_dir, 'precision_recall_curve.png')
        roc_path = os.path.join(save_dir, 'roc_curve.png')
        summary_path = os.path.join(save_dir, 'metrics_summary.png')
    else:
        pr_path = None
        roc_path = None
        summary_path = None
    
    # Generate all plots
    plot_precision_recall_curve(evaluation_results, save_path=pr_path, show_plot=show_plots)
    plot_roc_curve(evaluation_results, save_path=roc_path, show_plot=show_plots)
    plot_metrics_summary(evaluation_results, save_path=summary_path, show_plot=show_plots)
    
    print("Performance analysis plots generated!")
