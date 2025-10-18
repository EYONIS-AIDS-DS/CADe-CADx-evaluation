import os
from pathlib import Path
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import config_paper as config

from evaluate_common.logger import logger
from evaluate_common.plot_score_distribution_benign_cancer import distribution_risk_malignant_benign
from evaluate_common.precision_recall import plot_precision_recall
from evaluate_common.roc import plot_distribution_proba_malignant_benign, plot_roc_op
from evaluate_common.roc_confidence_interval import compute_ci_roc_auc
from evaluate_common.sample_size_analysis import database_sample_sizes
from evaluate_lesions.froc import compute_froc_with_2_op
from evaluate_lesions.plot_diameter_prediction_distributions import compute_diameter_distributions



def plot_distribution_malignant_benign(df: pd.DataFrame, expdir: Path, set_name: str) -> None:
    

    # Define D'arcy ratios
    
    delta_volume_min = np.min(df['delta_volume'].to_numpy())
    delta_volume_max = np.max(df['delta_volume'].to_numpy())

    df['delta_volume_norm'] = (df['delta_volume'].to_numpy()-delta_volume_min)/(delta_volume_max-delta_volume_min)

    # ###################### Histograms ##############################
    labels = df["lesion_diagnosis"].to_numpy()
    idx_benign = np.where(labels==0)   
    idx_malign = np.where(labels==1)
    pred_var = df['model_prediction_evolution'].to_numpy()
   
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    plt.hist(pred_var[idx_benign], bins=np.arange(0,2.8,0.05), label='benign, n='+str(len(idx_benign[0])), color='b', alpha=0.5, density=True)
    plt.hist(pred_var[idx_malign], bins=np.arange(0,2.8,0.05), label='malign, n='+str(len(idx_malign[0])), color='r', alpha=0.5, density=True)
    plt.suptitle('Distribution of $S_{TP_{-1}} / Norm_{[0;1]} (S_{TP_{-1}}-S_{TP_{-2}})$')
    plt.title("Test pop1 subset: "+str(len(pred_var))+"/"+str(len(labels)))
    plt.xlim(0,2.5)
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.savefig(os.path.join(expdir,"lesions_hist_model_prediction_evolution" + str(set_name) + ".svg",))
    plt.close()

    pred_var = df['model_prediction_evolution'].to_numpy()

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    plt.hist(pred_var[idx_benign], bins=np.arange(0,0.015,0.00025), label='benign, n='+str(len(idx_benign[0])), color='b', alpha=0.5, density=True)
    plt.hist(pred_var[idx_malign], bins=np.arange(0,0.015,0.00025), label='malign, n='+str(len(idx_malign[0])), color='r', alpha=0.5, density=True)
    plt.suptitle('Distribution of $S_{TP_{-1}} / Norm_{[0;1]} (S_{TP_{-1}}-S_{TP_{-2}})$')
    plt.title("Test pop1 subset: "+str(len(pred_var))+"/"+str(len(labels)))
    plt.xlim(0,0.013)
    ax.legend(loc="upper left")
    fig.tight_layout()
    plt.savefig(os.path.join(expdir,"lesions_hist_model_prediction_evolution_zoom" + str(set_name) + ".svg",))
    plt.close()

    pred_var = df['delta_lad'].to_numpy()
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    plt.hist(pred_var[idx_benign], bins=np.arange(-8,16,0.5), label='benign, n='+str(len(idx_benign[0])), color='b', alpha=0.5, density=True)
    plt.hist(pred_var[idx_malign], bins=np.arange(-8,16,0.5), label='malign, n='+str(len(idx_malign[0])), color='r', alpha=0.5, density=True)
    plt.suptitle('Distribution of $\Delta$ Diameter $(T_{-1},T_{-2})$')
    plt.title("Test pop1 subset: "+str(len(pred_var))+"/"+str(len(labels)))
    plt.xlim(-8,14)
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.savefig(os.path.join(expdir,"lesions_hist_delta_lad" + str(set_name) + ".svg",))


    pred_var = df['delta_volume'].to_numpy()
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    plt.hist(pred_var[idx_benign], bins=np.arange(-600,3400,85), label='benign, n='+str(len(idx_benign[0])), color='b', alpha=0.5, density=True)
    plt.hist(pred_var[idx_malign], bins=np.arange(-600,3400,85), label='malign, n='+str(len(idx_malign[0])), color='r', alpha=0.5, density=True)
    plt.suptitle('Distribution of $\Delta$ Volume $(T_{-1},T_{-2})$')
    plt.title("Test pop1 subset: "+str(len(pred_var))+"/"+str(len(labels)))
    plt.xlim(-600,3400)
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.savefig(os.path.join(expdir,"lesions_hist_delta_volume" + str(set_name) + ".svg",))
    plt.close()

    pred_var = df['RDT'].to_numpy()
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    plt.hist(pred_var[idx_benign], bins=np.arange(-4,5,0.2), label='benign, n='+str(len(idx_benign[0])), color='b', alpha=0.5, density=True)
    plt.hist(pred_var[idx_malign], bins=np.arange(-4,5,0.2), label='malign, n='+str(len(idx_malign[0])), color='r', alpha=0.5, density=True)
    plt.suptitle('Distribution of Reciprocal Doubling Time')
    plt.title("Test pop1 subset: "+str(len(pred_var))+"/"+str(len(labels)))
    plt.xlim(-4,5)
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.savefig(os.path.join(expdir,"lesions_hist_RDT" + str(set_name) + ".svg",))
    plt.close()



def prepare_data_frame_paper(data_frame, name_analysis, set_name, label_name, prediction, expdir) -> pd.DataFrame:

    if "test1" in set_name :
        data_frame = data_frame
    elif set_name == "test2" :
        data_frame = data_frame.loc[data_frame.nlst_test_2]
    elif "test3" in set_name :  
        data_frame = data_frame.loc[data_frame.nlst_test_3]
    elif set_name == "test4" :
        data_frame = data_frame.loc[data_frame.nlst_test_4]  
    elif set_name == "test5" :
        data_frame = data_frame.loc[data_frame.nlst_test_5]  
    elif set_name == "test6" :                                       
        data_frame = data_frame.loc[data_frame.nlst_test_6]        

    if "4_radiologist_prediction" in name_analysis :
        #concatene the fields in data_frame["patient_id"] and data_frame["annotator_id"] to replace data_frame["patient_id"] 
        # this will allow correct compuation of mean FP/scan
        df = pd.read_csv(config.path_data_lesions_radiologists)   
        data_frame = df.copy()
        data_frame["detection_id"] = data_frame.apply(lambda row: str(row["patient_id"]) + "." + str(row["det_id"]), axis=1)
        data_frame["patient_id"] = data_frame["patient_id"].astype(str)
        for i, row in data_frame.iterrows():
            data_frame.at[i, "patient_id"] = str(row["patient_id"]) + "." + str(row["annotator_id"])
        data_frame[prediction] = data_frame[prediction].fillna(1)    
        # normalize the radiologist scores from 1-10 to 0-1 (probability like).... cosmetic: does not change AUC or ROC
        data_frame[prediction] = (data_frame[prediction]-1) /9

    ############ FIGURE 2 ############################################
    if name_analysis == "nndetection_CADex_prediction_test1" :
        df = pd.read_csv(config.path_data_lesions_nndetection_CADex)   
        data_frame = df.copy()  
    elif name_analysis == "model_CADe_prediction_test1" :
        df = pd.read_csv(config.path_data_lesions_model_CADe)   
        data_frame = df.copy()           
    elif name_analysis == "nndetection_baumgartner_CADe_prediction_test1" :
        df = pd.read_csv(config.path_data_lesions_nndetection_baumgartner_CADe)   
        data_frame = df.copy()    
    elif name_analysis == "nndetection_CADe_prediction_test1" :
        df = pd.read_csv(config.path_data_lesions_nndetection_CADe)   
        data_frame = df.copy()
    
    ############ FIGURE 3 ############################################
    if name_analysis == "model_prediction_4_10_mm_test3" or name_analysis == "4_radiologist_prediction_4_10_mm_test3":
        data_frame = data_frame.loc[(data_frame["diameter_lad_predicted"] >= 4) & (data_frame["diameter_lad_predicted"] < 10)]           
    elif name_analysis == "model_prediction_10_20_mm_test3" or name_analysis == "4_radiologist_prediction_10_20_mm_test3":
        data_frame = data_frame.loc[(data_frame["diameter_lad_predicted"] >= 10) & (data_frame["diameter_lad_predicted"] < 20)] 
    elif name_analysis == "model_prediction_20_30_mm_test3" or name_analysis == "4_radiologist_prediction_20_30_mm_test3":
        data_frame = data_frame.loc[(data_frame["diameter_lad_predicted"] >= 20) & (data_frame["diameter_lad_predicted"] <= 30)]   
   
    ############ FIGURE 4 ############################################

    if name_analysis == "lcs_model_test6_LCS_T-1" or name_analysis == "lcs_model_test6_LCS_evolution" or name_analysis == "delta_volume_test6" or name_analysis == "delta_lad_test6" or name_analysis == "RDT_test6" : 
        plot_distribution_malignant_benign(data_frame, expdir, set_name) 
    
    ############ FIGURE extended 5 ############################################ 
    if  "4_radiologist_prediction_radiologist" in name_analysis:
        data_frame = data_frame.loc[(data_frame["annotator_id"]== set_name)] 
          
    if  "radiologist" in name_analysis and prediction == "model_prediction":        
        path_data_lesions_radiologists = Path(config.path_data_lesions_radiologists)
        df_annot = pd.read_csv(path_data_lesions_radiologists)       
        list_patient_annot = df_annot.loc[(df_annot["annotator_id"]== set_name), "patient_id"].unique()            
        data_frame = data_frame.loc[data_frame["patient_id"].isin(list_patient_annot)] 
  

    ############ FIGURE extended 9 ############################################
    #if name_analysis == "test6_delta_volume" or name_analysis == "test6_delta_volume_day_norm" or name_analysis == "test6_RDT" or name_analysis == "test6_RDT_year_norm" or name_analysis == "test6_VDT" :  
    #    data_frame = data_frame.loc[((data_frame["RDT"].notnull()) & (data_frame["model_prediction_T-2"].notnull()) &  (data_frame["model_prediction_T-1"].notnull()))]              
         
    if name_analysis == "VDT_test6" :
        # invert 0 , 1 labels because VDT is decreasing with malignancy 
        if prediction == "VDT":           
            data_frame["lesion_diagnosis"] = data_frame["lesion_diagnosis"].replace([0],"benign")  # benign negative class
            data_frame["lesion_diagnosis"] = data_frame["lesion_diagnosis"].replace([1], "malignant")  # cancer positive class
            data_frame["label"] = data_frame["lesion_diagnosis"] # "serie_diagnosis" with 1 and 2
            data_frame["label"] = data_frame["label"].replace(["benign"],1)  # benign negative class
            data_frame["label"] = data_frame["label"].replace(["malignant"], 0)  # cancer positive class
        # remove all rows from the dataframe where    data_frame["VDT"] is inf
        data_frame = data_frame.loc[~(data_frame["VDT"] == np.inf)]

    if name_analysis == "VDT_median_corrected_test6" :
        # invert 0 , 1 labels because VDT is decreasing with malignancy 
        if prediction == "VDT_median_corrected":           
            data_frame["lesion_diagnosis"] = data_frame["lesion_diagnosis"].replace([0],"benign")  # benign negative class
            data_frame["lesion_diagnosis"] = data_frame["lesion_diagnosis"].replace([1], "malignant")  # cancer positive class
            data_frame["label"] = data_frame["lesion_diagnosis"] # "serie_diagnosis" with 1 and 2
            data_frame["label"] = data_frame["label"].replace(["benign"],1)  # benign negative class
            data_frame["label"] = data_frame["label"].replace(["malignant"], 0)  # cancer positive class
        # remove all rows from the dataframe where    data_frame["VDT"] is inf
        data_frame = data_frame.loc[~(data_frame["VDT"] == np.inf)]
        data_frame["VDT_median_corrected"] = np.square((data_frame["VDT"] - np.median(data_frame["VDT"])))
    
    if name_analysis == "VDT_NELSON_criteria_test6" :
        # invert 0 , 1 labels because VDT is decreasing with malignancy 
        data_frame = data_frame.loc[((data_frame["nelson_criteria"] >= 0))]             
        if prediction == "VDT_NELSON_criteria":           
            data_frame["lesion_diagnosis"] = data_frame["lesion_diagnosis"].replace([0],"benign")  # benign negative class
            data_frame["lesion_diagnosis"] = data_frame["lesion_diagnosis"].replace([1], "malignant")  # cancer positive class
            data_frame["label"] = data_frame["lesion_diagnosis"] # "serie_diagnosis" with 1 and 2
            data_frame["label"] = data_frame["label"].replace(["benign"],1)  # benign negative class
            data_frame["label"] = data_frame["label"].replace(["malignant"], 0)  # cancer positive class
        # remove all rows from the dataframe where    data_frame["VDT"] is inf
        data_frame = data_frame.loc[~(data_frame["VDT"] == np.inf)]
        data_frame["VDT_NELSON_criteria"] = data_frame["VDT"]  
    
    return data_frame  
  
   ###################################################################################################
   ##############################         EVALUATION           #######################################
   ################################################################################################### 

"""
   This function evaluates the performance of a model on a given dataset by computing various metrics such as sensitivity, specificity, and ROC AUC.
"""

def evaluate_lesions_main(fast_computation,
                        path_to_load_csv_lesion,
                        expdir,
                        set_name,
                        prediction,
                        label_name,
                        operating_point_thresholds,
                        operating_point_labels,
                        nb_bootstrap_samples,
                        confidence_threshold,) -> None:
    
    name_analysis = prediction + "_" + set_name
    expdir = expdir.joinpath(name_analysis)  
    expdir.mkdir(parents=True, exist_ok=True)
    # Load data and select lesions
    try:
        df = pd.read_csv(path_to_load_csv_lesion)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)

    data_frame = df.copy()
    data_frame = prepare_data_frame_paper(data_frame, name_analysis, set_name, label_name, prediction, expdir) 
    logger.info(f"DataFrame shape after preparation: {data_frame.shape}")
    # data_label and data_name are only used for sample size analysis and to provide information in the plots
    data_label = "detection_id"  # unique ID : series_uid or detection_id
    data_name = "lesions"  # lesions or series
   
    database_sample_sizes(data_frame, data_label, data_name, label_name, expdir, set_name)
    if name_analysis == "model_prediction_test1" :    
        diametter_used = 'diameter_lad_predicted'
        compute_diameter_distributions(data_frame, diametter_used,  prediction, expdir, set_name)

    # Fill all missing predictions of the selection with 0 if the predictions are missing (failed series, missing reports, report without findings)
    # these failed will count as FN if cancer....
    # NOTABLY ALL MALIGANT NODULES WITHOUT PREDICTIONS WILL BE COUNTED AS FALSE NEGATIVE
    # Define predictions and labels
    data_frame[prediction] = data_frame[prediction].fillna(0)
    y_predictions = data_frame[prediction].to_numpy()
    y_labels = data_frame[label_name].to_numpy()

    
    compute_sens_spec_ci = (True)  # internal variable to avoid long unsusefull computation

    compute_ci_roc_auc( y_labels,
                        y_predictions,
                        operating_point_thresholds,
                        operating_point_labels,
                        expdir,
                        set_name,
                        compute_sens_spec_ci,
                        nb_bootstrap_samples,
                        confidence_threshold,)    
    # Compute Precision Recall
    plot_precision_recall(y_labels, y_predictions, expdir, set_name)
    # Compute ROC
    plot_roc_op(name_analysis, expdir, set_name, operating_point_labels)
    plot_fp = False  # if true plot false positive distribution
    plot_distribution_proba_malignant_benign(data_frame, prediction, expdir, set_name, data_name, label_name, plot_fp)  

    # Compute FROC
    logger.info("Plot FROC and sens spec at 2 operating points")
    fast_computation = False # to avoid long unsusefull computation
       
    compute_froc_with_2_op( data_frame,
                            y_labels,
                            y_predictions,
                            expdir,
                            set_name,
                            name_analysis,
                            nb_bootstrap_samples,
                            confidence_threshold,
                            fast_computation=fast_computation,)
        
        
 # #########################################################################
# #########################################################################
# ######          ARGUMENTS of the PROGRAM               ##################
# #########################################################################
# #########################################################################


def parse_args(args):
    """
    Checks and parse args used
    """
    parser = argparse.ArgumentParser(description="")

    # #########################################################################
    # ######         INPUT AND OUTPUT PATHS AND DIRS         ##################
    # #########################################################################
    
    ##################
    # fast_computation
    ##################
    help_msg = "fast_computation"
    default = config.fast_computation
    parser.add_argument("fast_computation", help=help_msg, default=default, type=bool)
   
    ##################
    # path_to_load_csv_lesions
    ##################
    help_msg = "path to the CSV lesions"
    default = config.path_data_lesions
    parser.add_argument("path_to_load_csv_lesion", help=help_msg, default=default, type=str)

    ##################
    # expdir
    ##################
    help_msg = "name of the directory where to save outputs "
    default = config.path_model_evaluate_lesions
    parser.add_argument("expdir", help=help_msg, default=default, type=str)

    ##################
    # set_name
    ##################
    help_msg = "name of the test set, as a boolean variable in the CSV file pointing to a subset of the data"
    default = config.list_lesions_evaluations[0][0]
    parser.add_argument("set_name", help=help_msg, default=default, type=str)

    ##################
    # prediction
    ##################
    help_msg = "name of the prediction, as a string  variable in the CSV file pointing to probability-prediction inference"
    default = config.list_lesions_evaluations[0][1]
    parser.add_argument("prediction", help=help_msg, default=default, type=str)

    ##################
    # label_name
    ##################
    help_msg = "name of the variable label to predict in the CSV file"
    default = config.list_lesions_evaluations[0][2]
    parser.add_argument("label_name", help=help_msg, default=default, type=str)

    ##################
    # operating_point_thresholds
    ##################
    help_msg = "list of operating point thresholds "
    default = config.list_lesions_evaluations[0][3] 
    parser.add_argument("operating_point_thresholds", nargs="+", help=help_msg, default=default, type=float,)
    
    ##################
    # operating_point_labels
    ##################
    help_msg = "list of operating point labels "
    default = config.list_lesions_evaluations[0][4]
    parser.add_argument("operating_point_labels", nargs="+", help=help_msg, default=default, type=float,)

    ##################
    # nb_bootstap_samples
    ##################
    help_msg = "nb_bootstrap_samples"
    default = config.nb_bootstrap_samples
    parser.add_argument("nb_bootstrap_samples", help=help_msg, default=default, type=int)

    ##################
    # confidence_threshold
    ##################
    help_msg = "confidence_threshold"
    default = config.confidence_threshold
    parser.add_argument("confidence_threshold", help=help_msg, default=default, type=int)
    
    return parser.parse_args(args)       


# #########################################################################
# #########################################################################
# ######          MAIN PROGRAM               ##############################
# #########################################################################
# #########################################################################

if __name__ == "__main__":

    logger.info("Start Evaluate Lesions")
    local_run = True  # Put False if you want to run the script from Command line
    if local_run:
        #######  PARAMETERS - ARGS   ##############################
        fast_computation = config.fast_computation
        path_to_load_csv_lesion = config.path_data_lesions
        expdir = config.path_model_evaluate_series
        set_name = config.list_lesions_evaluations[0][0]
        prediction = config.list_lesions_evaluations[0][1]
        label_name = config.list_lesions_evaluations[0][2]
        operating_point_thresholds =  config.list_lesions_evaluations[0][3]
        operating_point_labels = config.list_lesions_evaluations[0][4]        
        nb_bootstrap_samples = config.nb_bootstrap_samples
        confidence_threshold = config.confidence_threshold
    
    else:
        args = parse_args(sys.argv[1:])
        ######  INPUT AND OUTPUT PATHS AND DIRS  ##################
        fast_computation = args.fast_computation
        path_to_load_csv_lesion = Path(args.path_to_load_csv_lesion)
        expdir = Path(args.expdir)
        set_name = args.set_name
        prediction = args.prediction
        label_name = args.label_name
        operating_point_thresholds = args.operating_point_thresholds
        operating_point_labels = args.operating_point_labels
        nb_bootstrap_samples = args.nb_bootstrap_samples
        confidence_threshold = args.confidence_threshold

    # Run main
    evaluate_lesions_main(
        fast_computation,
        path_to_load_csv_lesion,
        expdir,
        set_name,
        prediction,
        label_name,
        operating_point_thresholds,
        operating_point_labels,
        nb_bootstrap_samples,
        confidence_threshold,
    )
