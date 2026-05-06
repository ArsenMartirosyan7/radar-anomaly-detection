from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def safe_average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def point_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = (scores > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": safe_auc(y_true, scores),
        "pr_auc": safe_average_precision(y_true, scores),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total_points": int(len(y_true)),
        "anomaly_points": int(y_true.sum()),
        "predicted_anomaly_points": int(y_pred.sum()),
    }


def extract_events(binary: np.ndarray) -> List[Tuple[int, int]]:
    """
    Converts binary labels into event intervals.

    Example:
        [0, 0, 1, 1, 1, 0, 1]
        -> [(2, 5), (6, 7)]

    End index is exclusive.
    """
    binary = binary.astype(int)
    events: List[Tuple[int, int]] = []

    in_event = False
    start = 0

    for i, value in enumerate(binary):
        if value == 1 and not in_event:
            start = i
            in_event = True

        elif value == 0 and in_event:
            events.append((start, i))
            in_event = False

    if in_event:
        events.append((start, len(binary)))

    return events


def intervals_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def event_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    true_events = extract_events(y_true)
    pred_events = extract_events(y_pred)

    detected_true_events = 0
    detection_delays = []

    for true_start, true_end in true_events:
        overlapping_preds = [
            (pred_start, pred_end)
            for pred_start, pred_end in pred_events
            if intervals_overlap((true_start, true_end), (pred_start, pred_end))
        ]

        if overlapping_preds:
            detected_true_events += 1
            first_detection = min(
                max(pred_start, true_start)
                for pred_start, _ in overlapping_preds
            )
            detection_delays.append(first_detection - true_start)

    correct_pred_events = 0

    for pred_event in pred_events:
        if any(intervals_overlap(pred_event, true_event) for true_event in true_events):
            correct_pred_events += 1

    event_recall = detected_true_events / len(true_events) if true_events else 0.0
    event_precision = correct_pred_events / len(pred_events) if pred_events else 0.0

    if event_precision + event_recall == 0:
        event_f1 = 0.0
    else:
        event_f1 = 2 * event_precision * event_recall / (
            event_precision + event_recall
        )

    return {
        "true_events": int(len(true_events)),
        "predicted_events": int(len(pred_events)),
        "detected_true_events": int(detected_true_events),
        "correct_predicted_events": int(correct_pred_events),
        "event_precision": float(event_precision),
        "event_recall": float(event_recall),
        "event_f1": float(event_f1),
        "mean_detection_delay": float(np.mean(detection_delays))
        if detection_delays
        else float("nan"),
        "median_detection_delay": float(np.median(detection_delays))
        if detection_delays
        else float("nan"),
    }


def interval_iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    intersection = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])

    if union == 0:
        return 0.0

    return intersection / union


def strict_event_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Stricter event metric.

    A predicted event is correct only if its IoU overlap with a true event
    is at least iou_threshold.
    """
    true_events = extract_events(y_true)
    pred_events = extract_events(y_pred)

    detected_true_events = 0

    for true_event in true_events:
        best_iou = 0.0

        for pred_event in pred_events:
            best_iou = max(best_iou, interval_iou(true_event, pred_event))

        if best_iou >= iou_threshold:
            detected_true_events += 1

    correct_pred_events = 0

    for pred_event in pred_events:
        best_iou = 0.0

        for true_event in true_events:
            best_iou = max(best_iou, interval_iou(pred_event, true_event))

        if best_iou >= iou_threshold:
            correct_pred_events += 1

    precision = correct_pred_events / len(pred_events) if pred_events else 0.0
    recall = detected_true_events / len(true_events) if true_events else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "strict_event_precision": float(precision),
        "strict_event_recall": float(recall),
        "strict_event_f1": float(f1),
        "strict_event_iou_threshold": float(iou_threshold),
    }