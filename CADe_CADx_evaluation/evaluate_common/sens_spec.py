"""Utility functions for computing sensitivity, specificity, and accuracy at a
given decision threshold.
"""
import numpy as np
from typing import Tuple


def closest_value(input_list: list, input_value: float) -> float:
    """Return the value in *input_list* closest to *input_value*.

    Parameters
    ----------
    input_list:
        Sequence of numeric values (typically a threshold array from
        :func:`sklearn.metrics.roc_curve`).
    input_value:
        Target value to look up.

    Returns
    -------
    float
        The element of *input_list* whose absolute difference to
        *input_value* is minimal.
    """
    arr = np.asarray(input_list)
    i = (np.abs(arr - input_value)).argmin()
    return arr[i]


def sens_spec(y_labels: np.ndarray, y_predictions: np.ndarray, threshold: float) -> Tuple[float, float]:
    """Compute sensitivity (recall) and specificity at a fixed threshold.

    Parameters
    ----------
    y_labels:
        Binary ground-truth array (0 = negative, 1 = positive).
    y_predictions:
        Numeric prediction scores, same length as *y_labels*.
    threshold:
        Decision threshold; samples with score >= threshold are classified
        as positive.

    Returns
    -------
    sensitivity : float
        True Positive Rate = TP / (TP + FN).
    specificity : float
        True Negative Rate = TN / (TN + FP).
    """
    prediction_binary = (y_predictions >= threshold).astype(int)
    sensitivity = (prediction_binary * y_labels).sum() / (y_labels.sum())
    specificity = ((1 - prediction_binary) * (1 - y_labels)).sum() / ((1 - y_labels).sum())
    return sensitivity, specificity


def accuracy(y_labels: np.ndarray, y_predictions: np.ndarray, threshold: float) -> float:
    """Compute overall accuracy at a fixed threshold.

    Parameters
    ----------
    y_labels:
        Binary ground-truth array (0 = negative, 1 = positive).
    y_predictions:
        Numeric prediction scores, same length as *y_labels*.
    threshold:
        Decision threshold.

    Returns
    -------
    float
        Fraction of correctly classified samples = (TP + TN) / N.
    """
    prediction_binary = (y_predictions >= threshold).astype(int)
    accuracy = ((prediction_binary * y_labels).sum() +  ((1 - prediction_binary) * (1 - y_labels)).sum() )/ ((y_labels.sum()) + ((1 - y_labels).sum()))
    return accuracy
