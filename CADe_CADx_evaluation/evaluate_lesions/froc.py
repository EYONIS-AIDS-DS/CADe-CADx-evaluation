import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import resample
from tqdm import tqdm
from typing import Tuple, List
#from lcseval.evaluate_common.logger import logger
#from lcseval.evaluate_common.sens_spec import closest_value, sens_spec
from evaluate_common.logger import logger
from evaluate_common.sens_spec import closest_value, sens_spec


def compute_fp_fn(features: pd.DataFrame,
                  threshold_detection: float,
                  y_labels: np.ndarray,
                  y_predictions: np.ndarray,) -> Tuple[float, float]:

    sensitivity, specificity = sens_spec(y_labels, y_predictions, threshold_detection)

    number_patient_nodule = len(list(features["patient_id"].unique()))
    number_benign = list(y_labels).count(0)
    fp_per_scan = (1 - specificity) * number_benign / number_patient_nodule
    
    return fp_per_scan, sensitivity




def compute_froc_ci_sens_fp_per_scan(features: pd.DataFrame,
                                     y_labels: np.ndarray,
                                     y_predictions: np.ndarray,
                                     value_tresh: List[float],
                                     nb_bootstrap_samples: int,
                                     confidence_threshold: float,
                                     expdir_analysis :str,
                                     set_name: str,
                                     type_of_op : str,) -> Tuple[float, ...]:
    
    sens_nested_list = []
    fp_per_scan_nested_list = []
    count = 0
    for i in tqdm(range(nb_bootstrap_samples)):
        try:
            resampled_predictions, resampled_labels = resample(y_predictions, y_labels, random_state=(42 + i))
            sens_list = []
            fp_per_scan_list = []
            for value in value_tresh:
                fp_per_scan, sensitivity = compute_fp_fn(features, value, resampled_labels, resampled_predictions,)
                sens_list.append(sensitivity)
                fp_per_scan_list.append(fp_per_scan)

            sens_nested_list.append(sens_list)
            fp_per_scan_nested_list.append(fp_per_scan_list)

        except Exception:
            count = count + 1
            logger.error("Number of failed bootstrapp (2small sample)", count)

    # Convert nested list to 2D NumPy array
    sens_array = np.array(sens_nested_list)
    fp_per_scan_array = np.array(fp_per_scan_nested_list)
    
    # save the 2D array of sens and fp_per_scan    
    np.save(os.path.join(expdir_analysis, str(type_of_op) + "sensitivity_array_5000bootstrap_for_each_OP_FROC_"+ str(set_name)+ ".npy",), sens_array)
    np.save(os.path.join(expdir_analysis, str(type_of_op) + "FP_per_SCAN_array_5000bootstrap_for_each_OP_FROC_"+ str(set_name)+ ".npy",), fp_per_scan_array)


    # Compute mean for each OP
    sens_mean_per_op = [np.mean(sens_array[:, i]) for i in range(sens_array.shape[1])]
    fp_per_scan_mean_per_op = [np.mean(fp_per_scan_array[:, i]) for i in range(fp_per_scan_array.shape[1])]

    # Compute sens_fp_per_scan CI lower and upper bounds for each OP
    lower_sens_per_op = [np.percentile(sens_array[:, i], 100 - (100 + 100 * confidence_threshold) / 2) for i in range(sens_array.shape[1])]
    upper_sens_per_op = [np.percentile(sens_array[:, i], (100 + 100 * confidence_threshold) / 2) for i in range(sens_array.shape[1])]

    lower_fp_per_scan_per_op = [np.percentile(fp_per_scan_array[:, i], 100 - (100 + 100 * confidence_threshold) / 2) for i in range(fp_per_scan_array.shape[1])]
    upper_fp_per_scan_per_op = [np.percentile(fp_per_scan_array[:, i], (100 + 100 * confidence_threshold) / 2) for i in range(fp_per_scan_array.shape[1])]

    return (sens_mean_per_op,
            lower_sens_per_op,
            upper_sens_per_op,
            fp_per_scan_mean_per_op,
            lower_fp_per_scan_per_op,
            upper_fp_per_scan_per_op,)




# COMPUTE FROC WITH 2 OP : one at 0.5 and one at 1 FP/scan    
def compute_froc_with_2_op( features: pd.DataFrame,
                            y_labels: np.ndarray,
                            y_predictions: np.ndarray,
                            expdir_analysis: str,
                            set_name: str,
                            name_analysis: str,
                            nb_bootstrap_samples: int,
                            confidence_threshold: float,
                            fast_computation: bool = False,):
    
    
    operating_point_thresholds_sens = [0.5, 1]
    if fast_computation:
        thresholds = np.linspace(0.0000001, 0.9999999, num=1000)
    else:
        thresholds = np.linspace(0.0000001, 0.9999999, num=500000)
    
 
    list_fp_per_scan = []
    list_sensitivity = []
    fp_per_scan_per_op = [0] * len(operating_point_thresholds_sens)

    for threshold_detect in tqdm(thresholds):
        fp_per_scan, sensitivity = compute_fp_fn(features, threshold_detect, y_labels, y_predictions)
        list_fp_per_scan.append(fp_per_scan)
        list_sensitivity.append(sensitivity)
    
    value_tresh = []
    sens_per_op = [] 
        
    for i, op_thresh in enumerate(operating_point_thresholds_sens):
        value = closest_value(list_fp_per_scan, op_thresh)
        index =list_fp_per_scan.index(value)
        test_sens = list_sensitivity[index]
        #test_sens, _ = sens_spec(y_labels, y_predictions, value)
        sens_per_op.append(test_sens)
        value_thre = thresholds[index] 
        value_tresh.append(value_thre)
        fp_per_scan_per_op[i] = value
    
    
    #for threshold_detect in tqdm(thresholds):    
    #    for i, op_thresh in enumerate(operating_point_thresholds_sens):
    #        if threshold_detect == op_thresh:
    #           fp_per_scan_per_op[i] = fp_per_scan


    label_list = []  


    for k, operating_point in enumerate(operating_point_thresholds_sens):
        label_list.append(
            "Operating Point "
            + str(k + 1)
            + "at : "
            + str(int(1000 * operating_point) / 1000)
            + "FP∕scan, Sens: "
            + str(int(1000 * sens_per_op[k]) / 10)
            + "%"
            + " FP/scans: "
            + str(int(1000 * fp_per_scan_per_op[k]) / 1000)
        )
    
  
    scatter = []
    color_list = ["#666666ff","#f0e68c",]
    plt.figure(figsize=(9.6, 7.2))
    plt.title("Free-Response ROC")
    plt.plot(list_fp_per_scan, list_sensitivity, "b")
    
    if "radiologist" in name_analysis and not ("model_prediction" in name_analysis):   
        plt.plot(list_fp_per_scan, list_sensitivity, "r", marker="o", markersize=5,)
    else:
        plt.plot(list_fp_per_scan, list_sensitivity, "b")
          
    for i in range(0, len(operating_point_thresholds_sens)):
        scatter.append(plt.scatter(fp_per_scan_per_op[i], sens_per_op[i], s=100, c=color_list[i],))

    plt.legend((scatter[0], scatter[1],),
                tuple(label_list),
                scatterpoints=1,
                loc="lower right",
                fontsize=10,
                title="Operating Points of the FROC",)
    plt.grid(True, color = "grey", linewidth = "1.4", linestyle = "-.")
    plt.ylim([0, 1])
    plt.xlim([0, 2])
    plt.ylabel("Sensitivity")
    plt.xlabel("Average number of FP/scan")

    plt.savefig(
        os.path.join(
            expdir_analysis,
            "froc_curve_with_2_op_" + str(set_name) + ".svg",
        )
    )
    plt.savefig(
        os.path.join(
            expdir_analysis,
            "froc_curve_with_2_op_" + str(set_name) + ".png",
        )
    )
    plt.close()
    # Compute CI FROC
    type_of_op = "at_0.5_and_1_FP_per_scan"
    logger.info("Compute FROC CI for fp_per_scan and sens per OP")
    (
        sens_mean_per_op,
        lower_sens_per_op,
        upper_sens_per_op,
        fp_per_scan_mean_per_op,
        lower_fp_per_scan_per_op,
        upper_fp_per_scan_per_op,
    ) = compute_froc_ci_sens_fp_per_scan(
        features,
        y_labels,
        y_predictions,
        value_tresh,
        nb_bootstrap_samples,
        confidence_threshold,
        expdir_analysis, 
        set_name,
        type_of_op,
    )

    # then save the operating points scores
    operating_point_number = np.arange(start=1, stop=3, step=1, dtype=int)
    operating_points_scores = pd.DataFrame({"OP #": operating_point_number})
    operating_points_scores["Th."] = operating_point_thresholds_sens
    operating_points_scores["Sens."] = sens_per_op
    operating_points_scores["Sens. Mean"] = sens_mean_per_op
    operating_points_scores["Sens. Low CI"] = lower_sens_per_op
    operating_points_scores["Sens. High CI"] = upper_sens_per_op
    operating_points_scores["FP/scan"] = fp_per_scan_per_op
    operating_points_scores["FP/scan Mean"] = fp_per_scan_mean_per_op
    operating_points_scores["FP/scan Low CI"] = lower_fp_per_scan_per_op
    operating_points_scores["FP/scan High CI"] = upper_fp_per_scan_per_op
    operating_points_scores.to_csv(
        os.path.join(
            expdir_analysis,
            "operating_point_FROC_scores_at_0.5_and_1_FP_per_scan_"
            + str(set_name)
            + ".csv",
        ),
        index=False,
        sep=",",
    )
