"""
Evaluation Metrics for Omni-Memory.

Standard metrics for multimodal memory evaluation:
- Accuracy (for QA tasks)
- Recall@K (for retrieval tasks)
- Token Efficiency (unique to our system)
- Latency metrics
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Result of a metric computation."""
    name: str
    value: float
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "details": self.details,
        }


def compute_accuracy(
    predictions: List[str],
    ground_truths: List[str],
    normalize: bool = True,
) -> MetricResult:
    """
    Compute accuracy for QA tasks.
    
    Args:
        predictions: Predicted answers
        ground_truths: Ground truth answers
        normalize: Whether to normalize text before comparison
        
    Returns:
        MetricResult with accuracy value
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")
    
    if not predictions:
        return MetricResult(name="accuracy", value=0.0)
    
    correct = 0
    total = len(predictions)
    
    for pred, gt in zip(predictions, ground_truths):
        if normalize:
            pred = pred.lower().strip()
            gt = gt.lower().strip()
        
        if pred == gt:
            correct += 1
        # Also check if prediction contains ground truth (for longer answers)
        elif gt in pred or pred in gt:
            correct += 0.5  # Partial credit
    
    accuracy = correct / total
    
    return MetricResult(
        name="accuracy",
        value=accuracy,
        details={
            "correct": correct,
            "total": total,
        }
    )


def compute_recall_at_k(
    retrieved_ids: List[List[str]],
    relevant_ids: List[List[str]],
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, MetricResult]:
    """
    Compute Recall@K for retrieval tasks.
    
    Args:
        retrieved_ids: List of retrieved item IDs per query
        relevant_ids: List of relevant item IDs per query
        k_values: K values to compute
        
    Returns:
        Dict mapping "recall@k" to MetricResult
    """
    if len(retrieved_ids) != len(relevant_ids):
        raise ValueError("Retrieved and relevant lists must have same length")
    
    results = {}
    
    for k in k_values:
        recalls = []
        
        for retrieved, relevant in zip(retrieved_ids, relevant_ids):
            if not relevant:
                continue
            
            retrieved_at_k = set(retrieved[:k])
            relevant_set = set(relevant)
            
            hits = len(retrieved_at_k & relevant_set)
            recall = hits / len(relevant_set)
            recalls.append(recall)
        
        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
        
        results[f"recall@{k}"] = MetricResult(
            name=f"recall@{k}",
            value=avg_recall,
            details={
                "k": k,
                "num_queries": len(recalls),
            }
        )
    
    return results


def compute_token_efficiency(
    token_counts: List[int],
    baseline_token_counts: List[int],
) -> MetricResult:
    """
    Compute token efficiency compared to baseline.
    
    Token efficiency = 1 - (our_tokens / baseline_tokens)
    Higher is better (more tokens saved).
    
    Args:
        token_counts: Our system's token usage per query
        baseline_token_counts: Baseline system's token usage
        
    Returns:
        MetricResult with efficiency value
    """
    if len(token_counts) != len(baseline_token_counts):
        raise ValueError("Token count lists must have same length")
    
    if not token_counts:
        return MetricResult(name="token_efficiency", value=0.0)
    
    our_total = sum(token_counts)
    baseline_total = sum(baseline_token_counts)
    
    if baseline_total == 0:
        efficiency = 0.0
    else:
        efficiency = 1 - (our_total / baseline_total)
    
    return MetricResult(
        name="token_efficiency",
        value=efficiency,
        details={
            "our_tokens": our_total,
            "baseline_tokens": baseline_total,
            "tokens_saved": baseline_total - our_total,
            "avg_our_tokens": our_total / len(token_counts),
            "avg_baseline_tokens": baseline_total / len(baseline_token_counts),
        }
    )


def compute_latency_metrics(
    latencies_ms: List[float],
) -> Dict[str, MetricResult]:
    """
    Compute latency statistics.
    
    Args:
        latencies_ms: List of latencies in milliseconds
        
    Returns:
        Dict with mean, p50, p95, p99 latency metrics
    """
    if not latencies_ms:
        return {}
    
    sorted_latencies = sorted(latencies_ms)
    n = len(sorted_latencies)
    
    mean_latency = sum(latencies_ms) / n
    p50_latency = sorted_latencies[n // 2]
    p95_latency = sorted_latencies[int(n * 0.95)]
    p99_latency = sorted_latencies[int(n * 0.99)]
    
    return {
        "mean_latency_ms": MetricResult(
            name="mean_latency_ms",
            value=mean_latency,
            details={"num_samples": n}
        ),
        "p50_latency_ms": MetricResult(
            name="p50_latency_ms",
            value=p50_latency,
        ),
        "p95_latency_ms": MetricResult(
            name="p95_latency_ms",
            value=p95_latency,
        ),
        "p99_latency_ms": MetricResult(
            name="p99_latency_ms",
            value=p99_latency,
        ),
    }


def compute_f1_score(
    predictions: List[str],
    ground_truths: List[str],
) -> MetricResult:
    """
    Compute token-level F1 score for QA tasks.
    
    Used for tasks where partial answer overlap matters.
    """
    f1_scores = []
    
    for pred, gt in zip(predictions, ground_truths):
        pred_tokens = set(pred.lower().split())
        gt_tokens = set(gt.lower().split())
        
        if not gt_tokens:
            continue
        
        common = pred_tokens & gt_tokens
        
        if not common:
            f1_scores.append(0.0)
            continue
        
        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall = len(common) / len(gt_tokens)
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        f1_scores.append(f1)
    
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    
    return MetricResult(
        name="f1_score",
        value=avg_f1,
        details={
            "num_samples": len(f1_scores),
        }
    )


def compute_mrr(
    retrieved_ids: List[List[str]],
    relevant_ids: List[List[str]],
) -> MetricResult:
    """
    Compute Mean Reciprocal Rank for retrieval tasks.
    
    MRR = average of 1/rank of first relevant item.
    """
    reciprocal_ranks = []
    
    for retrieved, relevant in zip(retrieved_ids, relevant_ids):
        relevant_set = set(relevant)
        
        rank = None
        for i, rid in enumerate(retrieved, 1):
            if rid in relevant_set:
                rank = i
                break
        
        if rank:
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    return MetricResult(
        name="mrr",
        value=mrr,
        details={
            "num_queries": len(reciprocal_ranks),
        }
    )


def compute_memory_retrieval_metrics(
    queries: List[str],
    retrieved_summaries: List[List[str]],
    relevant_summaries: List[List[str]],
    token_counts: List[int],
    latencies_ms: List[float],
    baseline_token_counts: Optional[List[int]] = None,
) -> Dict[str, MetricResult]:
    """
    Compute comprehensive metrics for memory retrieval.
    
    Combines:
    - Retrieval quality (Recall@K, MRR)
    - Efficiency (token savings)
    - Latency
    """
    results = {}
    
    # Recall@K
    recall_results = compute_recall_at_k(
        retrieved_summaries,
        relevant_summaries,
        k_values=[1, 5, 10]
    )
    results.update(recall_results)
    
    # MRR
    results["mrr"] = compute_mrr(retrieved_summaries, relevant_summaries)
    
    # Token efficiency (if baseline provided)
    if baseline_token_counts:
        results["token_efficiency"] = compute_token_efficiency(
            token_counts,
            baseline_token_counts
        )
    
    # Token usage stats
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
    results["avg_tokens"] = MetricResult(
        name="avg_tokens",
        value=avg_tokens,
        details={"total_tokens": sum(token_counts)}
    )
    
    # Latency
    latency_results = compute_latency_metrics(latencies_ms)
    results.update(latency_results)
    
    return results
