import numpy as np
from typing import Tuple

def closest_value(input_list: list, input_value: float) -> float:

    arr = np.asarray(input_list)
    i = (np.abs(arr - input_value)).argmin()
    return arr[i]


def sens_spec(y_labels: np.ndarray, y_predictions: np.ndarray, threshold: float) -> Tuple[float, float]:
    
    prediction_binary = (y_predictions >= threshold).astype(int)
    sensitivity = (prediction_binary * y_labels).sum() / (y_labels.sum())
    specificity = ((1 - prediction_binary) * (1 - y_labels)).sum() / ((1 - y_labels).sum())
    return sensitivity, specificity

def accuracy(y_labels: np.ndarray, y_predictions: np.ndarray, threshold: float) -> Tuple[float, float]:
    
    prediction_binary = (y_predictions >= threshold).astype(int)
    accuracy = ((prediction_binary * y_labels).sum() +  ((1 - prediction_binary) * (1 - y_labels)).sum() )/ ((y_labels.sum()) + ((1 - y_labels).sum()))
    return accuracy
