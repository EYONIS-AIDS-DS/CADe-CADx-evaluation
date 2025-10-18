import os
from math import sqrt
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.utils import resample
from tqdm import tqdm

# from lcseval.evaluate_common.logger import logger
# from lcseval.evaluate_common.sens_spec import accuracy, closest_value, sens_spec
from evaluate_common.logger import logger
from evaluate_common.sens_spec import accuracy, closest_value, sens_spec
from typing import Tuple, List


def ci_roc_auc_hanley(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    AUC = roc_auc_score(y_true, y_score)
    N1 = sum(y_true == 1)
    N2 = sum(y_true != 1)

    # Cast N1 and N2 to float to prevent overflow
    N1 = float(N1)
    N2 = float(N2)
    
    Q1 = AUC / (2 - AUC)
    Q2 = 2 * AUC**2 / (1 + AUC)
    se_auc = sqrt(
        (AUC * (1 - AUC) + (N1 - 1) * (Q1 - AUC**2) + (N2 - 1) * (Q2 - AUC**2))
        / (N1 * N2)
    )
    lower = AUC - 1.96 * se_auc
    upper = AUC + 1.96 * se_auc
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return (lower, upper, AUC)


#############################################################################################################
################                  COMPUTE THE MAXIMUM YOUDEN INDEX                         ##################
################        COMPUTE Confidence Interval and BOOTSTRAP SAMPLES FOR AUC,         ##################
################            and SENS, SPEC, Accuracy  at Maximum Youden Index              ##################
#############################################################################################################


def ci_roc_auc_bootstrap(
    y_labels: np.ndarray,
    y_predictions: np.ndarray,
    nb_bootstrap_samples: int,
    confidence_threshold: float,
    expdir_analysis: str,
    set_name: str,
    operating_point_thresholds: Optional[list] = None,
    compute_sens_spec_ci: bool = False,
) -> Tuple[float, ...]:
    # Initialize variables
    auc_list = []
    sens_nested_list = []
    spec_nested_list = []
    acc_nested_list = []
    count = 0

    # Compute the threshold of the YOUDEN MAX
    fpr, tpr, thresholds = roc_curve(y_labels, y_predictions, drop_intermediate=False)
    max_youden_thresh = thresholds[(tpr - fpr).argmax()]
    test_sens = tpr[(tpr - fpr).argmax()]
    test_spec = 1 - fpr[(tpr - fpr).argmax()]
    operating_point_thresholds[-1] = max_youden_thresh

    for i in tqdm(range(nb_bootstrap_samples)):
        try:
            resampled_predictions, resampled_labels = resample(
                y_predictions, y_labels, random_state=(42 + i)
            )
             # Check if resampled_labels contains at least two classes
            if len(np.unique(resampled_labels)) < 2:
                count += 1
                logger.warning(f"Skipping bootstrap sample {i} due to only one class in resampled_labels.")
                continue
            test_auc_model = roc_auc_score(resampled_labels, resampled_predictions)
            auc_list.append(test_auc_model)

            if compute_sens_spec_ci:
                sens_list = []
                spec_list = []
                acc_list = []
                _, _, thresholds = roc_curve(
                    resampled_labels, resampled_predictions, drop_intermediate=False
                )
                for operating_point in operating_point_thresholds:
                    value = closest_value(thresholds, operating_point)
                    test_sens, test_spec = sens_spec(
                        resampled_labels, resampled_predictions, value
                    )
                    test_acc = accuracy(resampled_labels, resampled_predictions, value)
                    sens_list.append(test_sens)
                    spec_list.append(test_spec)
                    acc_list.append(test_acc)

                sens_nested_list.append(sens_list)
                spec_nested_list.append(spec_list)
                acc_nested_list.append(acc_list)

        except Exception as e:
            count = count + 1
            logger.error(e)
            logger.error(f"Number of failed bootstrapp (2small sample): {count}")

    auc_arr = np.array(auc_list)
    np.save(
        os.path.join(
            expdir_analysis,
            "AUC_array_5000_bootstrap_" + str(set_name) + ".npy",
        ),
        auc_arr,
    )

    # Measure mean and standard deviation
    mean_auc = np.mean(auc_arr)
    stdev_auc = np.std(auc_arr)

    # Calculate the confidence interval
    lower_auc = np.percentile(auc_arr, 100 - (100 + 100 * confidence_threshold) / 2)
    upper_auc = np.percentile(auc_arr, (100 + 100 * confidence_threshold) / 2)

    if not compute_sens_spec_ci:
        return (mean_auc, stdev_auc, lower_auc, upper_auc)
    else:
        # Convert nested list to 2D NumPy array
        sens_array = np.array(sens_nested_list)
        spec_array = np.array(spec_nested_list)
        acc_array = np.array(acc_nested_list)
        # save the 2D array of sens and fp_per_scan
        np.save(
            os.path.join(
                expdir_analysis,
                "sensitivity_array_5000_bootstrap_for_each_OP_ROC_"
                + str(set_name)
                + ".npy",
            ),
            sens_array,
        )
        np.save(
            os.path.join(
                expdir_analysis,
                "specificity_array_5000_bootstrap_for_each_OP_ROC_"
                + str(set_name)
                + ".npy",
            ),
            spec_array,
        )
        np.save(
            os.path.join(
                expdir_analysis,
                "accuracy_array_5000_bootstrap_for_each_OP_ROC_" + str(set_name) + ".npy",
            ),
            acc_array,
        )

        # Compute mean for each OP
        sens_mean_per_op = [np.mean(sens_array[:, i]) for i in range(sens_array.shape[1])]
        spec_mean_per_op = [np.mean(spec_array[:, i]) for i in range(spec_array.shape[1])]
        acc_mean_per_op = [np.mean(acc_array[:, i]) for i in range(acc_array.shape[1])]

        # Compute sens_spec CI lower and upper bounds for each OP
        lower_sens_per_op = [
            np.percentile(sens_array[:, i], 100 - (100 + 100 * confidence_threshold) / 2)
            for i in range(sens_array.shape[1])
        ]
        upper_sens_per_op = [
            np.percentile(sens_array[:, i], (100 + 100 * confidence_threshold) / 2)
            for i in range(sens_array.shape[1])
        ]

        lower_spec_per_op = [
            np.percentile(spec_array[:, i], 100 - (100 + 100 * confidence_threshold) / 2)
            for i in range(spec_array.shape[1])
        ]
        upper_spec_per_op = [
            np.percentile(spec_array[:, i], (100 + 100 * confidence_threshold) / 2)
            for i in range(spec_array.shape[1])
        ]

        lower_acc_per_op = [
            np.percentile(acc_array[:, i], 100 - (100 + 100 * confidence_threshold) / 2)
            for i in range(acc_array.shape[1])
        ]
        upper_acc_per_op = [
            np.percentile(acc_array[:, i], (100 + 100 * confidence_threshold) / 2)
            for i in range(acc_array.shape[1])
        ]

        return (
            mean_auc,
            stdev_auc,
            lower_auc,
            upper_auc,
            sens_mean_per_op,
            lower_sens_per_op,
            upper_sens_per_op,
            spec_mean_per_op,
            lower_spec_per_op,
            upper_spec_per_op,
            acc_mean_per_op,
            lower_acc_per_op,
            upper_acc_per_op,
        )


def sens_spec_per_op(
    y_labels: np.ndarray,
    y_predictions: np.ndarray,
    operating_point_thresholds: List[float] = None,
) -> Tuple[float, ...]:
    sens_per_op = []
    spec_per_op = []
    acc_per_op = []
    fpr, tpr, thresholds = roc_curve(y_labels, y_predictions, drop_intermediate=False)

    for operating_point in operating_point_thresholds:
        value = closest_value(thresholds, operating_point)
        test_sens, test_spec = sens_spec(y_labels, y_predictions, value)
        test_acc = accuracy(y_labels, y_predictions, value)
        sens_per_op.append(test_sens)
        spec_per_op.append(test_spec)
        acc_per_op.append(test_acc)

    return (sens_per_op, spec_per_op, acc_per_op, fpr, tpr, thresholds)


#############################################################################################################
################                  COMPUTE THE MAXIMUM YOUDEN INDEX                         ##################
################        COMPUTE Confidence Interval Via Hanley and bootstrap methods       ##################
################                  and compute BOOTSTRAP SAMPLES FOR AUC,                   ##################
################            and SENS, SPEC, Accuracy  at Maximum Youden Index              ##################
################            and SENS, SPEC, Accuracy  mean and CI at Maximum Youden Index  ##################
################                         over bootstrap samples                            ##################
################               and SENS, SPEC, for all operating points                    ##################
#############################################################################################################


def compute_ci_roc_auc(
    y_labels: np.ndarray,
    y_predictions: np.ndarray,
    operating_point_thresholds: list,
    operating_point_labels: list,
    expdir_analysis: str,
    set_name: str,
    compute_sens_spec_ci: bool,
    nb_bootstrap_samples: int,
    confidence_threshold: float,
) -> None:
    # Measure confidence interval of AUC using Hanley & McNeil method
    (lower_auc_hc, upper_auc_hc, AUC) = ci_roc_auc_hanley(y_labels, y_predictions)

    # Measure confidence interval using bootstrap method  # for fast computation without saving the CI of sens and spec at each OP
    if not compute_sens_spec_ci:
        (mean_auc_bs, stdev_auc_bs, lower_auc_bs, upper_auc_bs) = ci_roc_auc_bootstrap(
            y_labels,
            y_predictions,
            nb_bootstrap_samples,
            confidence_threshold,
            expdir_analysis,
            set_name,
        )
    else:
        (
            mean_auc_bs,
            stdev_auc_bs,
            lower_auc_bs,
            upper_auc_bs,
            sens_mean_per_op,
            lower_sens_bs_per_op,
            upper_sens_bs_per_op,
            spec_mean_per_op,
            lower_spec_bs_per_op,
            upper_spec_bs_per_op,
            acc_mean_per_op,
            lower_acc_bs_per_op,
            upper_acc_bs_per_op,
        ) = ci_roc_auc_bootstrap(
            y_labels,
            y_predictions,
            nb_bootstrap_samples,
            confidence_threshold,
            expdir_analysis,
            set_name,
            operating_point_thresholds=operating_point_thresholds,
            compute_sens_spec_ci=compute_sens_spec_ci,
        )
        (
            sens_per_op,
            spec_per_op,
            acc_per_op,
            fpr,
            tpr,
            thresholds,
        ) = sens_spec_per_op(
            y_labels,
            y_predictions,
            operating_point_thresholds=operating_point_thresholds,
        )

    # Save results to csv files : here just the AUC and its CI
    results_dict = {
        "nb_bootstrap_samples": nb_bootstrap_samples,
        "confidence_threshold": confidence_threshold,
        "lower_CI_bootstap": lower_auc_bs,
        "upper_CI_bootstap": upper_auc_bs,
        "mean_AUC_bootstap": mean_auc_bs,
        "stdev_AUC_bootstap": stdev_auc_bs,
        "AUC": AUC,
        "lower_CI_Hanley": lower_auc_hc,
        "upper_CI_Hanley": upper_auc_hc,
    }

    df_results = pd.DataFrame([results_dict])

    df_results.to_csv(
        os.path.join(
            expdir_analysis,
            "roc_CI_bootstrap_hanley_" + str(set_name) + ".csv",
        ),
        index=False,
        sep=",",
    )

    # Save results to csv files : here all sens and spec  and their CI at each OP and the  whole ROC curve that is used in ROC.py to plot the ROC curve
    if compute_sens_spec_ci:
        # first save the whole ROC curve (fpr and tpr)
        roc_fpr_tpr = pd.DataFrame({"fpr": fpr})
        roc_fpr_tpr["tpr"] = tpr
        roc_fpr_tpr["thresholds"] = thresholds
        roc_fpr_tpr.to_csv(
            os.path.join(
                expdir_analysis,
                "roc_fpr_tpr_" + str(set_name) + ".csv",
            ),
            index=False,
            sep=",",
        )
        # then save the operating points scores and CI
        operating_point_number = np.arange(
            start=1, stop=len(operating_point_thresholds) + 1, step=1, dtype=int
        )
        operating_points_scores = pd.DataFrame({"OP #": operating_point_number})
        operating_points_scores["label"] = operating_point_labels
        operating_points_scores["Threshold"] = operating_point_thresholds
        operating_points_scores["Specificty"] = spec_per_op
        operating_points_scores["Spec. Mean"] = spec_mean_per_op
        operating_points_scores["Spec. Low CI"] = lower_spec_bs_per_op
        operating_points_scores["Spec. High CI"] = upper_spec_bs_per_op
        operating_points_scores["Sensitivity"] = sens_per_op
        operating_points_scores["Sens. Mean"] = sens_mean_per_op
        operating_points_scores["Sens. Low CI"] = lower_sens_bs_per_op
        operating_points_scores["Sens. High CI"] = upper_sens_bs_per_op
        operating_points_scores["Accuracy"] = acc_per_op
        operating_points_scores["Acc. Mean"] = acc_mean_per_op
        operating_points_scores["Acc. Low CI"] = lower_acc_bs_per_op
        operating_points_scores["Acc. High CI"] = upper_acc_bs_per_op

        operating_points_scores.to_csv(
            os.path.join(
                expdir_analysis,
                "operating_point_performances_" + str(set_name) + ".csv",
            ),
            index=False,
            sep=",",
        )
