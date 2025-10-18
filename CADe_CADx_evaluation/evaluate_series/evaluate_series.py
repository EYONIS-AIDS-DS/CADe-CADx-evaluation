import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import config_paper as config
from evaluate_common.logger import logger
from evaluate_common.plot_score_distribution_benign_cancer import distribution_risk_malignant_benign
from evaluate_common.precision_recall import plot_precision_recall
from evaluate_common.roc import plot_distribution_proba_malignant_benign, plot_roc_op
from evaluate_common.roc_confidence_interval import compute_ci_roc_auc
from evaluate_common.sample_size_analysis import database_sample_sizes


def plot_number_annotation_per_annotator_distribution(expdir_analysis: str, set_name: str, label_name: str, data_frame: pd.DataFrame) -> None:
    #"#######################################################################"
    #"######   plot_number_annotation_per_annotator_distribution   ##########"
    #"#######################################################################"
    
    ranked_list_nb_cancer_annot = []
    ranked_list_nb_patient_annot = []
    anonym_ranked_list_annotators =[]
    annotator_number = 20 #only used if name_analysis == "annotator_analysis"
    for i in range(1, annotator_number + 1):
        anonym_ranked_list_annotators.append("Reader "+str(i))
        name_annot_selection = "radiologist_"+str(i)
        #count the number of True in the column data_frame[name_annot_selection]
        number_of_patient_annot = data_frame[name_annot_selection].sum()
        ranked_list_nb_patient_annot.append(number_of_patient_annot)
        #count the number of both True in the column data_frame[name_annot_selection] and data_frame["serie_diagnosis"] = "malignant"
        number_of_cancer_annot = data_frame.loc[(data_frame[name_annot_selection]) & (data_frame[label_name] == 1)].shape[0]
        ranked_list_nb_cancer_annot.append(number_of_cancer_annot)
        
    plt.hist(anonym_ranked_list_annotators, bins=len(anonym_ranked_list_annotators), weights=ranked_list_nb_patient_annot,  color = "lightblue", ec="red" , label='number of annotated patient', rwidth= 0.75)
    plt.hist(anonym_ranked_list_annotators, bins=len(anonym_ranked_list_annotators), weights=ranked_list_nb_cancer_annot, color = "black" , label='number of cancer patient', rwidth= 0.75)
    plt.legend(loc='upper right')
    plt.ylabel('Number of scans')
    plt.xticks(rotation=90)
    plt.savefig(Path(expdir_analysis) / f"distrib_num_annotation_per_annotator_{set_name}.svg")
    plt.savefig(Path(expdir_analysis) / f"distrib_num_annotation_per_annotator_{set_name}.png")
    seniority = [4, 2, 3, 3, 2, 3, 3, 2, 2, 3, 2, 5, 4, 6, 4, 3, 2, 1, 1, 3]
    #ranked_list_annotators: ['rmallilin', 'rgerman', 'vbalbuena', 'lpardo', 'jfhperalta', 'emalimban', 'fmiranda', 'ltabao', 'etan', 'igrano', 'jmanzana', 'jvaldez', 'surquiza', 'rzuniega', 'jkoon', 'cmperez', 'gtaverner', 'cbasa', 'agutierrez', 'ccorpus']
    plt.hist(anonym_ranked_list_annotators, bins=len(anonym_ranked_list_annotators), weights=seniority,  color = "lightblue", ec="red" , label='number of years postgraduate', rwidth= 0.75)
    plt.legend(loc='upper right')
    plt.ylabel('Number of years postgraduate')
    plt.xticks(rotation=90)
    plt.savefig(Path(expdir_analysis) / f"distrib_num_years_experience_per_annotator_{set_name}.svg")
    plt.savefig(Path(expdir_analysis) / f"distrib_num_years_experience_per_annotator_{set_name}.png")
    plt.close()
    

def distribution_cancer_stage(data_frame, expdir_analysis, label_name, set_name):
    list_cancer_stage= ["Stage IA", "Stage IB", "Stage IIA", "Stage IIB", "Stage IIIA", "Stage IIIB", "Stage IV", "NA"]
    malignant_nodule_set  = data_frame.loc[(data_frame[label_name] == 1) ]
    # replace blanks in malignant_nodule_set["cancer_stage"] by "Cannot be assessed"
    malignant_nodule_set.loc[:, "cancer_stage"] = malignant_nodule_set["cancer_stage"].replace("", "NA")
    # replace  values ["Stage IA", "Stage IB", "Stage IIA", "Stage IIB", "Stage IIIA", "Stage IIIB", "Stage IV", "NA"] in malignant_nodule_set["cancer_stage"] by [1,2,3,4,5,6,7,8]
    malignant_nodule_set.loc[:, "cancer_stage"] = malignant_nodule_set["cancer_stage"].replace(["Stage IA"], 1)
    # count the number of 1 in malignant_nodule_set["cancer_stage"]    
    malignant_nodule_set.loc[:, "cancer_stage"] = malignant_nodule_set["cancer_stage"].replace(["Stage IB"], 2)
    malignant_nodule_set.loc[:, "cancer_stage"] = malignant_nodule_set["cancer_stage"].replace(["Stage IIA"], 3)
    malignant_nodule_set.loc[:, "cancer_stage"] = malignant_nodule_set["cancer_stage"].replace(["Stage IIB"], 4)
    malignant_nodule_set.loc[:, "cancer_stage"] = malignant_nodule_set["cancer_stage"].replace(["Stage IIIA"], 5)
    malignant_nodule_set.loc[:, "cancer_stage"] = malignant_nodule_set["cancer_stage"].replace(["Stage IIIB"], 6)
    malignant_nodule_set.loc[:, "cancer_stage"] = malignant_nodule_set["cancer_stage"].replace(["Stage IV"], 7)
    malignant_nodule_set.loc[:, "cancer_stage"] = malignant_nodule_set["cancer_stage"].replace(["NA"], 8)
    weight = []
    weight.append((malignant_nodule_set["cancer_stage"] == 1).sum())
    weight.append((malignant_nodule_set["cancer_stage"] == 2).sum())
    weight.append((malignant_nodule_set["cancer_stage"] == 3).sum())
    weight.append((malignant_nodule_set["cancer_stage"] == 4).sum())
    weight.append((malignant_nodule_set["cancer_stage"] == 5).sum())
    weight.append((malignant_nodule_set["cancer_stage"] == 6).sum())
    weight.append((malignant_nodule_set["cancer_stage"] == 7).sum())
    weight.append((malignant_nodule_set["cancer_stage"] == 8).sum())

    plt.title('Distribution of cancer stages')
    plt.hist(list_cancer_stage, bins=len(list_cancer_stage), weights= weight ,  color = "lightblue", ec="red")
    plt.ylabel('Number of patients')
    plt.xlabel('Cancer stage')
    plt.xlim([0, 6])
    plt.xticks(rotation=90)
    plt.savefig(Path(expdir_analysis) / f"distribution_cancer_stage_{set_name}.svg")
    plt.savefig(Path(expdir_analysis) / f"distribution_cancer_stage_{set_name}.png")
    plt.close()
  

def prepare_data_frame_4_radiologist(data_frame,):

    for i in range(1, 5):
        data_frame = data_frame.copy()
        data_frame.loc[:,f"radiologist_max_malignancy_db_{i}"] = data_frame[f"radiologist_max_malignancy_db_{i}"].fillna(1)
    interim_frames = []
    for i in range(1, 5):
        df_interim = data_frame.copy()
        df_interim["4_radiologist_prediction"] = data_frame[f"radiologist_max_malignancy_db_{i}"]
        df_interim["patient_id"] = data_frame["patient_id"] * (10 ** (i - 1))
        interim_frames.append(df_interim)

    data_frame = pd.concat(interim_frames).reset_index(drop=True)

    return data_frame

def prepare_data_frame_lungrads_wo_prior(data_frame,):
    
    for i in range(1, 7):
        column = f"LungRADS_without_prior_lung_rads_{i}"
        data_frame = data_frame.copy()
        data_frame.loc[:, column] = (
            data_frame[column].replace({"4A": 4, "4B": 5, "4X": 6}).astype(int).fillna(1)
        )

    interim_frames = []
    for i in range(1, 7):
        df_interim = data_frame.copy()
        df_interim["lungrads_wo_prior_prediction"] = data_frame[f"LungRADS_without_prior_lung_rads_{i}"]
        df_interim["patient_id"] = data_frame["patient_id"] * (10 ** (i - 1))
        interim_frames.append(df_interim)

    data_frame = pd.concat(interim_frames).reset_index(drop=True)

    return data_frame

def prepare_data_frame_lungrads_with_prior(data_frame,):

    columns = [f"LungRADS_with_prior_lung_rads_{i}" for i in range(1, 7)]
    # Replace values and convert to integers
    for column in columns:
         data_frame = data_frame.copy()
         data_frame.loc[:, column] = (
            data_frame[column].replace({"4A": 4, "4B": 5, "4X": 6}).astype(int).fillna(1)
        )
    # Create interim DataFrames for each column
    interim_frames = []
    for i, column in enumerate(columns, start=1):
        df_interim = data_frame.copy()
        df_interim["lungrads_with_prior_prediction"] = data_frame[column]
        df_interim["patient_id"] = data_frame["patient_id"] * (10 ** (i - 1))
        interim_frames.append(df_interim)
    # Concatenate all interim DataFrames
    data_frame = pd.concat(interim_frames).reset_index(drop=True)
       
    return data_frame

def prepare_data_frame_kernel(data_frame, name_analysis,):

    list_sharp_siemens = [ "['I70f'; '1']","['I70f'; '2']","['I70f'; '3']","['I80s'; '1']","['I80s'; '2']","['I80s'; '3']","['I80s'; '4']","['I80s'; '5']","B70f","B70s","B75f","B75h","B75s","B80f","B80s","B90s","I70f","I80s","Hr68f"]
    list_sharp_GE = ["BONE","Bone2","BONEPLUS2","BODY FILTER/STANDARD","BODY FILTER / STANDARD","BONEPLUS","BODY FILTER/BONE","BODY FILTER/EXPERIMENTAL"]
    list_sharp_toshiba = ["FC55","FC56","FC59","FC65","FC57","FC58","FC80","FC81","FC82","FC86"] 
    list_average_siemens = ["I50f","B50s","['I50f'; '1']","['I50s'; '2']","I50s","['I50s'; '1']","B50f","['I50f'; '2']","['I50f'; '3']","['I50s'; '3']","['Bl54f'; '2']","['Bl54f'; '1']","['Br54d'; '2']","['Bl54d'; '1']","Bl54d","['Bl54d'; '2']","['Bl54d'; '3']","Br54d","['Br54d'; '1']","['Bl54f'; '3']","Bl54f","['Br54d'; '3']","['Bl56f'; '1']","['Bl56f'; '2']","['Bl56f'; '3']","['Bl57d'; '1']","['Bl57d'; '2']","['Bl57d'; '3']","['Bl57f'; '1']","['Bl57f'; '2']","['Bl57f'; '3']","['Bl60f'; '1']","['Bl60f'; '2']","['Bl60f'; '3']","['Bl64d'; '1']","['Bl64d'; '2']","['Bl64d'; '3']","['Bl64f'; '1']","['Bl64f'; '2']","['Bl64f'; '3']","['Bl64f'; '4']","['Bl64f'; '5']","['Br51f'; '1']","['Br51f'; '2']","['Br51f'; '3']","['Br58f'; '1']","['Br58f'; '2']","['Br58f'; '3']","['Br59d'; '1']","['Br59d'; '2']","['Br59d'; '3']","['Br59f'; '1']","['Br59f'; '2']","['Br59f'; '3']","['Br60f'; '1']","['Br60f'; '2']","['Br60f'; '3']","['Br64d'; '1']","['Br64d'; '2']","['Br64d'; '3']","['Br64f'; '1']","['Br64f'; '2']","['Br64f'; '3']","['Br64f'; '4']","['Br64f'; '5']","B60f","B60s","B65f","B65s","Bl56f","Bl57d","Bl57f","Bl57s","Bl60f","Bl64d","Bl64f","Br51f","Br58f","Br59d","Br59f","Br60f","Br64d","Br64f","Tx60f"]
    list_average_GE = ["CHEST","CHST","HD Lung","HD LUNG","LUNG"]
    list_average_toshiba = ["FC52","FC53 ","FC53","FC31","FC50","FC51","FC51 ","FC24","FC30","FC17","FC14","FC18","FC19","FC02","FC08","FC10","FC11 ","FC12","FC13","FC13-H","FL01","FL03","FL04"] 
    list_soft_siemens = ["B45s","B45f","Br49d","['Br49d'; '3']","Br49f","['Br49d'; '2']","['Br49d'; '1']","['Br49f'; '3']","['Br49f'; '1']","['Br49f'; '2']","['B44d'; '1']","['B44d'; '2']","['B44d'; '3']","['B44f'; '1']","['B44f'; '2']","['B44f'; '3']","['Bf37f'; '1']","['Bf37f'; '2']","['Bf37f'; '3']","['Br32d'; '1']","['Br32d'; '2']","['Br32d'; '3']","['Br32f'; '1']","['Br32f'; '2']","['Br32f'; '3']","['Br36d'; '1']","['Br36d'; '2']","['Br36f'; '1']","['Br36f'; '2']","['Br36f'; '3']","['Br36s'; '1']","['Br36s'; '2']","['Br36s'; '3']","['Br38f'; '1']","['Br38f'; '2']","['Br38f'; '3']","['Br40d'; '1']","['Br40d'; '2']","['Br40d'; '3']","['Br40f'; '1']","['Br40f'; '2']","['Br40f'; '3']","['Br44f'; '1']","['Br44f'; '2']","['Br44f'; '3']","['Bv36d'; '3']","['Bv40d'; '1']","['Bv40d'; '2']","['Bv40d'; '3']","['Bv40f'; '1']","['Bv40f'; '2']","['Bv40f'; '3']","['I26f'; '1']","['I26f'; '2']","['I26f'; '3']","['I30f'; '1']","['I30f'; '2']","['I30f'; '3']","['I30s'; '1']","['I30s'; '2']","['I30s'; '3']","['I31f'; '1']","['I31f'; '2']","['I31f'; '3']","['I31s'; '1']","['I31s'; '2']","['I31s'; '3']","['I40f'; '1']","['I40f'; '2']","['I40f'; '3']","['I41f'; '1']","['I41f'; '2']","['I41f'; '3']","['I41s'; '1']","['I41s'; '2']","['I41s'; '3']","['I41s'; '4']","['I44f'; '1']","['I44f'; '2']","['I44f'; '3']","B08f","B08s","B10f","B10s","B19f","B19s","B20f","B20s","B25f","B26f","B29f","B29s","B30","B30f","B30s","B31f","B31s","B35f","B35s","B36d","B36f","B36s","B39f","B39s","B40f","B40f","B40s","B41f","B44d","B44f","B46f","B46s","Bf37f","Br32d","Br32f","Br36d","Br36f","Br36s","Br38f","Br40d","Br40f","Br44f","Bv40d","Bv40f","I26f","I30f","I30s","I31f","I31s","I40f","I41f","I41s","I44f","T20f","T20s","Tr20f"]
    list_soft_GE = ["STANDARD","Detail2","Detail","Veo","EXPERIMENTAL","SOFT"]
    list_soft_toshiba = ["FC01"]  
    
    data_frame["manufacturer"] = data_frame["manufacturer"].replace(["Siemens Healthineers"], "SIEMENS")
    
    data_frame_siemens = data_frame.loc[(data_frame["manufacturer"] == "SIEMENS")]
    data_frame_GE = data_frame.loc[(data_frame["manufacturer"] == "GE MEDICAL SYSTEMS")]
    data_frame_toshiba = data_frame.loc[(data_frame["manufacturer"] == "TOSHIBA")]

    if name_analysis == "model_prediction_kernel_sharp_test1" :
        mask_sharp_siemens = data_frame_siemens['kernel'].isin(list_sharp_siemens)
        data_frame_sharp_siemens = data_frame_siemens[mask_sharp_siemens]
        mask_sharp_GE = data_frame_GE['kernel'].isin(list_sharp_GE)
        data_frame_sharp_GE = data_frame_GE[mask_sharp_GE]
        mask_sharp_toshiba = data_frame_toshiba['kernel'].isin(list_sharp_toshiba)
        data_frame_sharp_toshiba = data_frame_toshiba[mask_sharp_toshiba]
        data_frame = pd.concat([data_frame_sharp_siemens, data_frame_sharp_GE, data_frame_sharp_toshiba])

    
    if name_analysis == "model_prediction_kernel_average_test1" :                       
        mask_average_siemens = data_frame_siemens['kernel'].isin(list_average_siemens)
        data_frame_average_siemens = data_frame_siemens[mask_average_siemens]
        mask_average_GE = data_frame_GE['kernel'].isin(list_average_GE)
        data_frame_average_GE = data_frame_GE[mask_average_GE]
        mask_average_toshiba = data_frame_toshiba['kernel'].isin(list_average_toshiba)
        data_frame_average_toshiba = data_frame_toshiba[mask_average_toshiba]
        data_frame = pd.concat([data_frame_average_siemens, data_frame_average_GE, data_frame_average_toshiba])

        
    if name_analysis == "model_prediction_kernel_soft_test1" :                      
        mask_soft_siemens = data_frame_siemens['kernel'].isin(list_soft_siemens)
        data_frame_soft_siemens = data_frame_siemens[mask_soft_siemens]
        mask_soft_GE = data_frame_GE['kernel'].isin(list_soft_GE)
        data_frame_soft_GE = data_frame_GE[mask_soft_GE]
        mask_soft_toshiba = data_frame_toshiba['kernel'].isin(list_soft_toshiba)
        data_frame_soft_toshiba = data_frame_toshiba[mask_soft_toshiba]
        data_frame = pd.concat([data_frame_soft_siemens, data_frame_soft_GE, data_frame_soft_toshiba])  
    
    return data_frame    


def prepare_data_frame_paper(expdir, data_frame, name_analysis, set_name, label_name, prediction,):

    if "test1" in set_name :
        data_frame = data_frame
    elif set_name == "test2" :
        data_frame = data_frame.loc[data_frame.nlst_test_2]
    elif  "test3" in set_name  :
        data_frame = data_frame.loc[data_frame.nlst_test_3]
    elif set_name == "test4" :
        data_frame = data_frame.loc[data_frame.nlst_test_4]  
    elif set_name == "test5" :
        data_frame = data_frame.loc[data_frame.nlst_test_5] 
    elif set_name == "test6" :   
        data_frame = data_frame.loc[data_frame.nlst_test_6]         
    elif  set_name == "test6&3" or set_name == "VDT_sup_400_test6&3":
        data_frame = data_frame.loc[(data_frame.nlst_test_3 & data_frame.nlst_test_6)]   
    elif "radiologist" in set_name:    
        data_frame = data_frame.copy()       
        data_frame.loc[:,set_name] = data_frame[set_name].fillna(False)
        data_frame = data_frame.loc[data_frame[set_name] == True]      
        # Fill all missing predictions data_frame[name_prediction] of the selection with 1 if the predictions are missing ( = report without findings = nothing detected)
        data_frame = data_frame.copy()
        data_frame.loc[:,prediction] = data_frame[prediction].fillna(1) 
    else:    
        data_frame = data_frame
     
    ############ FIGURE 1 ############################################
  
    if "4_radiologist" in name_analysis:
        data_frame = prepare_data_frame_4_radiologist(data_frame)     
    
    if name_analysis == "lungrads_wo_prior_prediction_test4":
        data_frame = prepare_data_frame_lungrads_wo_prior(data_frame)
    elif name_analysis == "lungrads_with_prior_prediction_test5":
        data_frame = prepare_data_frame_lungrads_with_prior(data_frame)  
    ############ FIGURE 3 ############################################
    elif name_analysis == "model_prediction_4_10_mm_test3" or name_analysis == "4_radiologist_prediction_4_10_mm_test3":
        #filter data_frame where "max_lad_GT_serie" is greater or aqual to 4 mm and less than 10 mm
        data_frame = data_frame.loc[(data_frame["max_lad_GT_serie"] >= 4) & (data_frame["max_lad_GT_serie"] < 10)]    
    elif name_analysis == "model_prediction_10_20_mm_test3" or name_analysis == "4_radiologist_prediction_10_20_mm_test3":
        #filter data_frame where "max_lad_GT_serie" is greater or aqual to 10 mm and less than 20 mm
        data_frame = data_frame.loc[(data_frame["max_lad_GT_serie"] >= 10) & (data_frame["max_lad_GT_serie"] < 20)]  
    elif name_analysis == "model_prediction_20_30_mm_test3" or name_analysis == "4_radiologist_prediction_20_30_mm_test3":
        #filter data_frame where "max_lad_GT_serie" is greater or aqual to 20 mm and less than 30 mm
        data_frame = data_frame.loc[(data_frame["max_lad_GT_serie"] >= 20) & (data_frame["max_lad_GT_serie"] <= 30)]        
     
    elif name_analysis == "model_prediction_stage_1a_test3" or name_analysis == "model_prediction_stage_1a_test1" or name_analysis == "4_radiologist_prediction_stage_1a_test3":
        data_frame  = data_frame.loc[((data_frame[label_name] == 0) | ((data_frame[label_name] == 1) & (data_frame["cancer_stage"] == "Stage IA")) ) ]
    elif name_analysis == "model_prediction_stage_1_test3" or name_analysis == "model_prediction_stage_1_test1" or name_analysis == "4_radiologist_prediction_stage_1_test3":
        data_frame  = data_frame.loc[((data_frame[label_name] == 0) | ((data_frame[label_name] == 1) & (data_frame["cancer_stage"] == "Stage IA")) | ((data_frame[label_name] == 1) & (data_frame["cancer_stage"] == "Stage IB")) ) ]
    elif name_analysis == "model_prediction_stage_late_test3" or name_analysis == "model_prediction_stage_late_test1" or name_analysis == "4_radiologist_prediction_stage_late_test3":
        data_frame  = data_frame.loc[((data_frame[label_name] == 0) | ((data_frame[label_name] == 1) & (~data_frame["cancer_stage"].isin(["Stage IA","Stage IB", "Cannot be assessed"])))  ) ]   
    ############ FIGURE 4 ############################################
    elif name_analysis == "model_prediction_T-1_test6" or name_analysis == "model_prediction_evolution_test6" or name_analysis == "delta_volume_test6" or name_analysis == "RDT_test6" or name_analysis == "delta_lad_test6":  
        data_frame = data_frame.loc[((data_frame["RDT"].notnull()) & (data_frame["model_prediction_T-2"].notnull()) &  (data_frame["model_prediction_T-1"].notnull()))]                  
    elif name_analysis == "model_prediction_T-2_VDT_sup_400_test6&3"  or name_analysis == "4_radiologist_prediction_VDT_sup_400_test6&3":
        data_frame = data_frame = data_frame.loc[( (data_frame["VDT"] >= 400))]
   
    ############ FIGURE EXTENDED 2 ############################################   
    elif name_analysis == "model_prediction_manufacturer_siemens_test1" :
        data_frame = data_frame.loc[(data_frame["manufacturer"] == "SIEMENS")]  
    elif name_analysis == "model_prediction_manufacturer_GE_test1" :
        data_frame = data_frame.loc[(data_frame["manufacturer"] == "GE MEDICAL SYSTEMS")]  
    elif name_analysis == "model_prediction_manufacturer_toshiba_test1" :
        data_frame = data_frame.loc[(data_frame["manufacturer"] == "TOSHIBA")]       
                
    elif name_analysis == "model_prediction_thickness_0.5_1.5_test1":
        data_frame = data_frame.loc[(data_frame["slice_thickness"] >= 0.5) & (data_frame["slice_thickness"] < 1.5)]    
    elif name_analysis == "model_prediction_thickness_1.5_2.3_test1":
        data_frame = data_frame.loc[(data_frame["slice_thickness"] >= 1.5) & (data_frame["slice_thickness"] < 2.3)]  
    elif name_analysis == "model_prediction_thickness_2.3_3.5_test1":
        data_frame = data_frame.loc[(data_frame["slice_thickness"] >= 2.3) & (data_frame["slice_thickness"] <= 3.5)]   
        
    elif name_analysis == "model_prediction_age_55_62_test1":
        data_frame = data_frame.loc[(data_frame["age"] >= 55) & (data_frame["age"] < 62)]    
    elif name_analysis == "model_prediction_age_62_69_test1":
        data_frame = data_frame.loc[(data_frame["age"] >= 62) & (data_frame["age"] < 69)]  
    elif name_analysis == "model_prediction_age_69_84_test1":
        data_frame = data_frame.loc[(data_frame["age"] >= 69) & (data_frame["age"] <= 84)]    
  
    elif name_analysis == "model_prediction_gender_female_test1" :
        data_frame = data_frame.loc[(data_frame["gender"] == "female")]  
    elif name_analysis == "model_prediction_gender_male_test1" :
        data_frame = data_frame.loc[(data_frame["gender"] == "male")]  
 
    elif name_analysis == "model_prediction_with_copd_test1" :
        data_frame = data_frame.loc[(data_frame["copd"] == 1)]  
    elif name_analysis == "model_prediction_without_copd_test1" :
        data_frame = data_frame.loc[(data_frame["copd"] == 0)] 
        
    if  "model_prediction_kernel" in name_analysis:   
        data_frame = prepare_data_frame_kernel(data_frame,  name_analysis)

    ############ FIGURE EXTENDED 3 ############################################   
    if "_max_malignancy" in name_analysis:
        plot_number_annotation_per_annotator_distribution(expdir, set_name, label_name, data_frame)
        
    ############ FIGURE EXTENDED 9 ############################################   
    if "NODCATIII_test6"  in name_analysis :                
        data_frame = data_frame.loc[((data_frame["model_prediction_T-1"].notnull()) &  (data_frame["model_prediction_T-2"].notnull()))]  
        data_frame = data_frame.loc[((data_frame["nelson_criteria"] >= 0) & (data_frame["gt_volume_T-2"] >= 50) & (data_frame["gt_volume_T-2"] <= 500))]     
        # invert 0 , 1 labels because VDT is decreasing with malignancy 
        if prediction == "VDT":
            data_frame["label"] = data_frame["patient_diagnosis"] # "serie_diagnosis" with 1 and 2
            data_frame["label"] = data_frame["label"].replace(["benign"], 1)  # benign negative class
            data_frame["label"] = data_frame["label"].replace(["malignant"], 0)  # cancer positive class


    
    if name_analysis == "model_T-2_test6&3_subgroups" :     
        expdir = expdir.joinpath(name_analysis) 
        expdir.mkdir(parents=True, exist_ok=True)             
        data_frame = data_frame.loc[((data_frame["prediction_T-2"] >= -0.00001) &  (data_frame["prediction_T-1"] >= -0.00001))]  
        name_analysis_subgroup = "_all_negative_indeterminate_400_sup_VDT"
        #"_nelson_grow_criteria_sup_negative_600_sup_VDT"  #"_nelson_grow_criteria_sup_intermediate_400_600_VDT" #"_nelson_grow_criteria_sup_positive_400_inf_VDT"  #"_all_negative_indeterminate_400_sup_VDT"  #           
        #"nelson_grow_criteria_inf" # "_nelson_grow_criteria_sup" 
        #"_delta_lad_sup_1.5" #"_delta_lad_inf_1.5" # # 
        #"_VDT_T-1_NODCATIII"  
        name_analysis = name_analysis+ name_analysis_subgroup
        #data_frame = data_frame.loc[data_frame["delta_lad"] < 1.5]
        #data_frame = data_frame.loc[data_frame["delta_lad"] >= 1.5]
        #data_frame = data_frame.loc[(data_frame["nelson_criteria"] >= 0)]
        #data_frame = data_frame.loc[(data_frame["nelson_criteria"] < 0)]
        #data_frame = data_frame.loc[((data_frame["nelson_criteria"] >= 0)& (data_frame["VDT"] >400) & (data_frame["VDT"] <600))]
        #data_frame = data_frame.loc[( (data_frame["nelson_criteria"] >= 0) & (data_frame["VDT"] <= 400))]
        #data_frame = data_frame.loc[( (data_frame["nelson_criteria"] >= 0) & (data_frame["VDT"] >= 600))]
        data_frame = data_frame.loc[((data_frame["VDT"] >= 400))]
        #data_frame = data_frame.loc[((data_frame["nelson_criteria"] >= 0)& (data_frame["gt_volume_T-2"] >= 50) & (data_frame["gt_volume_T-2"] <= 500))]
        #prediction = "gt_RDT"
        prediction = 'prediction_T-2'
        
    if name_analysis == "model_T-1_test6&3_subgroups" :     
        expdir = expdir.joinpath(name_analysis) 
        expdir.mkdir(parents=True, exist_ok=True)             
        data_frame = data_frame.loc[((data_frame["prediction_T-2"] >= -0.00001) &  (data_frame["prediction_T-1"] >= -0.00001))]  
        name_analysis_subgroup = "_delta_lad_sup_1.5" #"_delta_lad_inf_1.5" # 
        # "_all_negative_indeterminate_400_sup_VDT"
        #"_nelson_grow_criteria_sup_negative_600_sup_VDT"  #"_nelson_grow_criteria_sup_intermediate_400_600_VDT" #"_nelson_grow_criteria_sup_positive_400_inf_VDT"  #"_all_negative_indeterminate_400_sup_VDT"  #           
        #"nelson_grow_criteria_inf" # "_nelson_grow_criteria_sup" 
        #"_delta_lad_sup_1.5" #"_delta_lad_inf_1.5" # # 
        #"_VDT_T-1_NODCATIII"  
        name_analysis = name_analysis+ name_analysis_subgroup
        #data_frame = data_frame.loc[data_frame["delta_lad"] < 1.5]
        data_frame = data_frame.loc[data_frame["delta_lad"] >= 1.5]
        #data_frame = data_frame.loc[(data_frame["nelson_criteria"] >= 0)]
        #data_frame = data_frame.loc[(data_frame["nelson_criteria"] < 0)]
        #data_frame = data_frame.loc[((data_frame["nelson_criteria"] >= 0)& (data_frame["VDT"] >400) & (data_frame["VDT"] <600))]
        #data_frame = data_frame.loc[( (data_frame["nelson_criteria"] >= 0) & (data_frame["VDT"] <= 400))]
        #data_frame = data_frame.loc[( (data_frame["nelson_criteria"] >= 0) & (data_frame["VDT"] >= 600))]
        #data_frame = data_frame.loc[((data_frame["VDT"] >= 400))]
        #data_frame = data_frame.loc[((data_frame["nelson_criteria"] >= 0)& (data_frame["gt_volume_T-2"] >= 50) & (data_frame["gt_volume_T-2"] <= 500))]
        #prediction = "gt_RDT"
        prediction = 'prediction_T-1'    
        
        ###############  cancer FN detgtection as FN CADex ############################  
        data_frame[prediction] = data_frame[prediction].fillna(0)  
    
    if name_analysis == "4_radiologist_T-1_test6&3_subgroups" :
        expdir = expdir.joinpath(name_analysis) 
        expdir.mkdir(parents=True, exist_ok=True)             
        data_frame = data_frame.loc[((data_frame["prediction_T-2"] >= -0.00001) &  (data_frame["prediction_T-1"] >= -0.00001))]  
        name_analysis_subgroup = "_nelson_grow_criteria_sup_positive_400_inf_VDT" #"_nelson_grow_criteria_sup_intermediate_400_600_VDT" #"_nelson_grow_criteria_sup_negative_600_sup_VDT"  #"_all_negative_indeterminate_400_sup_VDT"  #"_nelson_grow_criteria_sup_positive_400_inf_VDT" #
        #"nelson_grow_criteria_inf" #"_nelson_grow_criteria_sup" # 
        #"_delta_lad_sup_1.5" #"_delta_lad_inf_1.5" # # 
        #"_NODCATIII" 
        #"_all_negative_indeterminate_400_sup_VDT" #"_nelson_grow_criteria_sup_intermediate_400_600_VDT" #"_nelson_grow_criteria_sup_negative_600_sup_VDT" 
        name_analysis = name_analysis+ name_analysis_subgroup
        #data_frame = data_frame.loc[data_frame["delta_lad"] < 1.5]
        #data_frame = data_frame.loc[data_frame["delta_lad"] >= 1.5]
        #data_frame = data_frame.loc[(data_frame["nelson_criteria"] >= 0)]
        #data_frame = data_frame.loc[(data_frame["nelson_criteria"] < 0)]
        #data_frame = data_frame.loc[((data_frame["nelson_criteria"] >= 0)& (data_frame["VDT"] >400) & (data_frame["VDT"] <600))]
        data_frame = data_frame.loc[( (data_frame["nelson_criteria"] >= 0) & (data_frame["VDT"] <= 400))]
        #data_frame = data_frame.loc[( (data_frame["nelson_criteria"] >= 0) & (data_frame["VDT"] >= 600))]
        #data_frame = data_frame.loc[( (data_frame["VDT"] >= 400))]
        #data_frame = data_frame.loc[((data_frame["nelson_criteria"] >= 0) & (data_frame["gt_volume_T-2"] >= 50) & (data_frame["gt_volume_T-2"] <= 500))]
        
        data_frame["radiologist_max_malignancy_db_1"] = data_frame["radiologist_max_malignancy_db_1"].fillna(1)
        data_frame["radiologist_max_malignancy_db_2"] = data_frame["radiologist_max_malignancy_db_2"].fillna(1) 
        data_frame["radiologist_max_malignancy_db_3"] = data_frame["radiologist_max_malignancy_db_3"].fillna(1)
        data_frame["radiologist_max_malignancy_db_4"] = data_frame["radiologist_max_malignancy_db_4"].fillna(1)
        df_interim1 = data_frame.copy() 
        df_interim2 = data_frame.copy() 
        df_interim3 = data_frame.copy() 
        df_interim4 = data_frame.copy() 
        df_interim1["4_radiologist_prediction"] = data_frame["radiologist_max_malignancy_db_1"]
        df_interim2["4_radiologist_prediction"] = data_frame["radiologist_max_malignancy_db_2"] 
        df_interim3["4_radiologist_prediction"] = data_frame["radiologist_max_malignancy_db_3"] 
        df_interim4["4_radiologist_prediction"] = data_frame["radiologist_max_malignancy_db_4"]
        df_interim1["patient_id"] = data_frame["patient_id"]*1
        df_interim2["patient_id"] = data_frame["patient_id"]*10
        df_interim3["patient_id"] = data_frame["patient_id"]*100 
        df_interim4["patient_id"] = data_frame["patient_id"]*1000
        data_frame = pd.concat([df_interim1, df_interim2, df_interim3, df_interim4]).reset_index(drop=True)   
        prediction = "4_radiologist_prediction"     
    
    return data_frame    

  
  
   ###################################################################################################
   ##############################         EVALUATION           #######################################
   ################################################################################################### 

"""
   This function evaluates the performance of a model on a given dataset by computing various metrics such as sensitivity, specificity, and ROC AUC.
"""

def evaluate_serie_main(path_to_load_csv_serie,
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
    try:
        df = pd.read_csv(path_to_load_csv_serie)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)    
       
    data_frame = df.copy()   
    data_frame = prepare_data_frame_paper(expdir, data_frame, name_analysis, set_name, label_name, prediction,)        
    logger.info(f"DataFrame shape after preparation: {data_frame.shape}")
    # data_label and data_name are only used for sample size analysis and to provide information in the plots
    data_label = "series_uid"  # unique ID : series_uid or index_abs_detection
    data_name = "series"  # lesions or series
    database_sample_sizes(data_frame, data_label, data_name, label_name, expdir, set_name,)
    # Fill all missing predictions of the selection with 0 if the predictions are missing (failed series, missing reports, report without findings)
    # these failed will count as FN if cancer....

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
 
    plot_precision_recall(y_labels, y_predictions, expdir, set_name)
    # plot  distribution_risk_malignant_benign
    plot_roc_op(name_analysis, expdir, set_name, operating_point_labels)
    #distribution_risk_malignant_benign(data_frame, prediction, expdir, set_name)
    plot_fp = False  # if true plot the false positive distribution
    plot_distribution_proba_malignant_benign(data_frame, prediction, expdir, set_name, data_name, label_name, plot_fp)

            
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
    # path_to_load_csv_series
    ##################
    help_msg = "path to the CSV series"
    default = config.path_data_series 
    parser.add_argument("path_to_load_csv_serie", help=help_msg, default=default, type=str)
    ##################
    # expdir
    ##################
    help_msg = "name of the directory where to save outputs "
    default = config.path_model_evaluate_series
    parser.add_argument("expdir", help=help_msg, default=default, type=str)

    ##################
    # set_name
    ##################
    help_msg = "name of the test set, as a boolean variable in the CSV file deisgning a subset of the data"
    default = config.list_series_evaluations[0][0]
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
    help_msg = "name of the label GT column in the CSV file"
    default = config.list_series_evaluations[0][2]
    parser.add_argument("label_name", help=help_msg, default=default, type=str)

    ##################
    # operating_point_thresholds
    ##################
    help_msg = "list of operating point thresholds "
    default = config.list_series_evaluations[0][3]
    parser.add_argument("operating_point_thresholds", nargs="+", help=help_msg, default=default, type=float,)
    
    ##################
    # operating_point_labels
    ##################
    help_msg = "list of operating point labels "
    default = config.list_series_evaluations[0][4]
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

    logger.info("Start Evaluate Series")
    local_run = True  # Put False if you want to run the script from Command line else it will use the config parameters
    if local_run:
        #######  PARAMETERS - ARGS   ##############################
        path_to_load_csv_serie = config.path_data_series 
        expdir = config.path_model_evaluate_series
        set_name = config.list_series_evaluations[0][0]
        prediction = config.list_series_evaluations[0][1]
        label_name = config.list_series_evaluations[0][2]
        operating_point_thresholds =  config.list_series_evaluations[0][3]
        operating_point_labels = config.list_series_evaluations[0][4]        
        nb_bootstrap_samples = config.nb_bootstrap_samples
        confidence_threshold = config.confidence_threshold
    
    else:
        args = parse_args(sys.argv[1:])
        ######  INPUT AND OUTPUT PATHS AND DIRS  ##################
        path_to_load_csv_serie = Path(args.path_to_load_csv_series)
        expdir = Path(args.expdir)
        set_name = args.set_name
        prediction = args.prediction
        label_name = args.label_name
        operating_point_thresholds = args.operating_point_thresholds
        operating_point_labels = args.operating_point_labels
        nb_bootstrap_samples = args.nb_bootstrap_samples
        confidence_threshold = args.confidence_threshold
        

    evaluate_serie_main(
        path_to_load_csv_serie,
        expdir,
        set_name,
        prediction,
        label_name,
        operating_point_thresholds,
        operating_point_labels,
        nb_bootstrap_samples,
        confidence_threshold,
    )
