import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Tuple

#from lcseval.evaluate_common.logger import logger
from evaluate_common.logger import logger


def roc_curve_rectangle(
    y_labels: np.ndarray,
    y_predictions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:

    fpr = []
    tpr = []
    thresholds = np.linspace(0.0, 1.0, len(y_labels))
    # get number of positive and negative examples in the dataset
    P = sum(y_labels)
    N = len(y_labels) - P

    # iterate through all thresholds and determine fraction of true positives
    # and false positives found at this threshold
    for thresh in thresholds:
        FP = 0
        TP = 0
        for i in range(len(y_predictions)):
            if y_predictions[i] >= thresh:
                if y_labels[i] == 1:
                    TP = TP + 1
                if y_labels[i] == 0:
                    FP = FP + 1
        fpr.append(FP / N)
        tpr.append(TP / P)
    # Initialize the area under the curve
    auc_rect = 0.0
    # Iterate through the values and calculate the area using Riemann midpoint summation method
    for i in range(len(fpr) - 1):
        midpoint_height = (tpr[i] + tpr[i + 1]) / 2  # Riemann midpoint height
        auc_rect += midpoint_height * (fpr[i] - fpr[i + 1])
    fpr = np.array(fpr)
    tpr = np.array(tpr)
    thresholds = np.array(thresholds)

    return fpr, tpr, thresholds, auc_rect


def compute_roc_auc(
    y_labels: np.ndarray,
    y_predictions: np.ndarray,
    use_scikit_trapezoidal_roc: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:

    if use_scikit_trapezoidal_roc:
        fpr, tpr, thresholds = roc_curve(y_labels, y_predictions, drop_intermediate=False)
        test_auc_model = roc_auc_score(y_labels, y_predictions)
    else:
        fpr, tpr, thresholds, test_auc_model = roc_curve_rectangle(y_labels, y_predictions)

    return fpr, tpr, thresholds, test_auc_model


def plot_roc_op(name_analysis: str, expdir_analysis: str, set_name: str, operating_point_labels: list) -> None:

    # load the results needed to plot the figure... they are generated and saved by roc_confidence_interval.py
    roc_fpr_tpr = pd.read_csv(os.path.join(expdir_analysis,"roc_fpr_tpr_" + str(set_name) + ".csv",))
    operating_points_scores = pd.read_csv(os.path.join(expdir_analysis, "operating_point_performances_" + str(set_name) + ".csv",))
    df_results_auc = pd.read_csv(os.path.join(expdir_analysis,"roc_CI_bootstrap_hanley_"+ str(set_name)+ ".csv",))
    

    operating_point_thresholds = operating_points_scores["Threshold"].tolist()
    operating_point_sens = operating_points_scores["Sensitivity"].tolist()
    operating_point_spec = operating_points_scores["Specificty"].tolist()
    
    if len(operating_point_thresholds) == 1:
        color_list = ["Black",]
    elif len(operating_point_thresholds) == 2:
        color_list = ["Red", "Black"]   
    elif len(operating_point_thresholds) == 3:
        color_list = ["Green","Red", "Black"]
    elif len(operating_point_thresholds) == 7:
        color_list = ["#abbedfff", "#70ab4dff", "#ffff07ff", "#ec7e31ff", "#ff0100ff", "#ff0100ff", "#000000ff"]
    else:
        color_list = plt.cm.viridis(np.linspace(0, 1, len(operating_point_thresholds)))

    fpr = roc_fpr_tpr["fpr"].to_numpy()
    tpr = roc_fpr_tpr["tpr"].to_numpy()

    list_of_operating_point_absyssa = []
    list_of_operating_point_ordinate = []
    label_list = []
    label_list.append("ROC AUC = %0.3f" % df_results_auc["AUC"].iloc[0])
    

    for k, operating_point in enumerate(operating_point_thresholds):
        list_of_operating_point_ordinate.append(operating_point_sens[k])
        list_of_operating_point_absyssa.append(1 - operating_point_spec[k])
        label_list.append(str(operating_point_labels[k])
                            + ": Thresh: "
                            + str(int(1000 * operating_point) / 1000)
                            + " Sens: "
                            + str(int(1000 * operating_point_sens[k]) / 10)
                            + "%"
                            + " Spec: "
                            + str(int(1000 * operating_point_spec[k]) / 10)
                            + "%")

    scatter = []

    plt.figure(figsize=(9.6, 7.2))
    plt.title(f"Receiver Operating Characteristic ({str(name_analysis)})")
    for i in range(0, len(operating_point_thresholds)):
        scatter.append(plt.scatter(list_of_operating_point_absyssa[i], list_of_operating_point_ordinate[i], s=100, c=color_list[i],))
     
    if "radiologist" in name_analysis:  
        ROC = plt.plot(fpr, tpr, "b", label="ROC AUC = %0.3f" % df_results_auc["AUC"].iloc[0], marker="o", markersize=5,)
    else:           
        ROC = plt.plot(fpr, tpr, "b", label="ROC AUC = %0.3f" % df_results_auc["AUC"].iloc[0])

    plt.legend((ROC[0], *scatter), tuple(label_list), scatterpoints=1, loc="lower right", fontsize=10,)    

    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.savefig(os.path.join(expdir_analysis, "roc_curve_with_op_" + str(set_name) + ".svg",))
    plt.savefig(os.path.join(expdir_analysis, "roc_curve_with_op_" + str(set_name) + ".png",))
    plt.close()


def plot_distribution_proba_malignant_benign(features: pd.DataFrame, 
                                             prediction_1: str, 
                                             expdir_analysis: str, 
                                             set_name: str, 
                                             data_name: str,
                                             label_name: str,
                                             plot_fp: bool,) -> None:

    benign_test_nodule_set = features.loc[(features[label_name] == 0)]
    cancer_test_nodule_set = features.loc[(features[label_name] == 1)]
    if plot_fp:
        fp_detect = features.loc[(features.detection_status == "FP")]
        fp_detect_proba = fp_detect[prediction_1].to_numpy()
    benign_test_proba = benign_test_nodule_set[prediction_1].to_numpy()
    cancer_test_proba = cancer_test_nodule_set[prediction_1].to_numpy()

    bins = np.linspace(0, 1, 100)
    fig, (ax) = plt.subplots(1, 1, figsize=[9.6, 7.2])

    plt.title("Distribution of malignancy predictions")
    values_bins_canc, bins_canc, bars = plt.hist(cancer_test_proba, bins, alpha=0.5, color="#FF0000", label="malignant")
    plt.hist(benign_test_proba, bins, alpha=0.5, color="#00AEEF", label="benign")
    if plot_fp:
        values_bins_fp, bins_fp, bars = ax.hist(fp_detect_proba, bins, alpha=0.5, color="#008000")
    plt.ylabel("Number of "+ str(data_name))
    plt.xlabel("Predicted malignancy probability")
    plt.xlim([0, 1])

    _ = inset_axes( ax, width="80%", height="80%",)
    plt.hist(cancer_test_proba, bins, alpha=0.5, color="#FF0000", label="malignant")
    plt.hist(benign_test_proba, bins, alpha=0.5, color="#00AEEF", label="benign")
    if plot_fp:
        plt.hist(fp_detect_proba, bins, alpha=0.5, color="#008000", label="detection FP")
    plt.legend(loc="upper center")
    plt.xlim([0, 1])
    plt.ylim([0, values_bins_canc.max() + 1])
    plt.savefig(os.path.join(expdir_analysis, "distribution_proba_" + str(set_name) + ".png",))
    plt.savefig(os.path.join(expdir_analysis,"distribution_proba_" + str(set_name) + ".svg",))
    plt.close()