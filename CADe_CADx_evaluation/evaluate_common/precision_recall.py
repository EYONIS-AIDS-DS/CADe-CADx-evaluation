from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_precision_recall(
    y_labels: np.ndarray,
    y_predictions: np.ndarray,
    expdir_analysis: str,
    set_name: str,
) -> None:

    average_precision = average_precision_score(y_labels, y_predictions)
    precision, recall, _ = precision_recall_curve(y_labels, y_predictions)
    plt.figure(figsize=(9.6, 7.2))
    plt.step(recall, precision, color="b", alpha=0.2, where="post")
    plt.fill_between(recall, precision, step="post", alpha=0.2, color="b")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title("Precision-Recall - AUC: {0:0.4f}".format(average_precision))
    plt.savefig(os.path.join(expdir_analysis,"Precision_Recall_curve_model_" + str(set_name) + ".png",))
    plt.savefig(os.path.join(expdir_analysis, "Precision_Recall_curve_model_" + str(set_name)+ ".svg",))
    plt.close()
