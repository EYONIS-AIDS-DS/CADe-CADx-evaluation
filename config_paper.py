from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent

############################################################################
###################               general              #####################
############################################################################
# general parameters

path_model_eval = PROJECT_ROOT / "data"  # root path to the evaluation results of the current experiment of the LCS_evaluation

##### ORCHESTRATOR (run.py) #####
# The principle is to use Boolean for the execution or not of the larger evaluation 
run_evaluate_series = True # run the evaluate series module
run_evaluate_lesions = True  # run the evaluate lesions module
run_statistical_tests = True  # run the statistical tests module

fast_computation = False  # if True, reduce the number of bootstrap samples for faster computation during testing/debugging and reduces FROC precision (set to False for full computation)

############################################################################
###################          evaluate_series            #####################
############################################################################
# parameters of the "evaluate_series" module
# path of the output files of "evaluate_series" module
path_data_series = path_model_eval / "data_series" / "series.csv"
path_model_evaluate_series = path_model_eval / "evaluate_series"
nb_bootstrap_samples = 5000 #5000 #50
if fast_computation:
    nb_bootstrap_samples = 50
confidence_threshold = 0.95

lungrads_model_thresholds = [3.053602733958832e-05, 0.0013636961100517, 0.022274809097966, 0.078009378336262, 0.3804615455610797, 0.6550206392280502, 0,]
lungrads_labels = ["LungRADS 1", "LungRADS 2", "LungRADS 3", "LungRADS 4A", "LungRADS 4B", "LungRADS 4X", "Youden Index Max",]
# an evaluation is a 5 tuple (set_name, prediction, label_name, operating_point_thresholds, operating_point_labels)
# set_name : is the name of the set to evaluate (test1, test2, test3, test4, test5, test6), it is a string that correspond to a column in the dataframe with boolean values defining a subset of the dataframe used in the paper, if unknown set_name, all the dataframe is used
# prediction : is the name of the column in the dataframe with the prediction scores of models, radiologs etc.
# label_name :  is the name of the column in the dataframe with the ground truth labels (0,1)
# operating_point_thresholds :  is a list of thresholds to evaluate the performance at these points
# operating_point_labels : is a list of labels for these operating points
list_series_evaluations = [("test1","model_prediction", "label",[0,],["Youden Index Max"]),]

list_series_evaluations_figure_1 = [
     ("test1", "model_prediction",               "label", [0,], ["Youden Index Max"]),
     ("test2", "model_prediction",               "label", [0,], ["Youden Index Max"]),
     ("test3", "model_prediction",               "label", [0,], ["Youden Index Max"]),
     ("test4", "model_prediction",               "label", lungrads_model_thresholds, lungrads_labels),
     ("test5", "model_prediction",               "label", lungrads_model_thresholds, lungrads_labels),
     ("test6", "model_prediction",               "label", [0,], ["Youden Index Max"]),
     ("test2", "ardila_prediction",              "label", [0,], ["Youden Index Max"]),
     ("test2", "liao_prediction",                "label", [0,], ["Youden Index Max"]),
     ("test2", "brock_prediction",               "label", [0,], ["Youden Index Max"]),
     ("test2", "sybil_prediction",               "label", [0,], ["Youden Index Max"]),
     ("test2", "mayo_prediction",                "label", [0,], ["Youden Index Max"]),
     ("test3", "4_radiologist_prediction",       "label", [0,], ["Youden Index Max"]),
     ("test4", "lungrads_wo_prior_prediction",   "label", [1,2,3,4,5,6,0], lungrads_labels),
     ("test5", "lungrads_with_prior_prediction", "label", [1,2,3,4,5,6,0], lungrads_labels), 
    ]

list_series_evaluations_figure_2 = [
     ("manufacturer_siemens_test1", "model_prediction", "label", [0,], ["Youden Index Max"]),
     ("manufacturer_GE_test1",      "model_prediction", "label", [0,], ["Youden Index Max"]),
     ("manufacturer_toshiba_test1", "model_prediction", "label", [0,], ["Youden Index Max"]),
     ("thickness_0.5_1.5_test1",    "model_prediction", "label", [0,], ["Youden Index Max"]),
     ("thickness_1.5_2.3_test1",    "model_prediction", "label", [0,], ["Youden Index Max"]),
     ("thickness_2.3_3.5_test1",    "model_prediction", "label", [0,], ["Youden Index Max"]),
     ("age_55_62_test1",            "model_prediction", "label", [0,], ["Youden Index Max"]),
     ("age_62_69_test1",            "model_prediction", "label", [0,], ["Youden Index Max"]),
     ("age_69_84_test1",            "model_prediction", "label", [0,], ["Youden Index Max"]),
     ("gender_female_test1",        "model_prediction", "label", [0,], ["Youden Index Max"]),
     ("gender_male_test1",          "model_prediction", "label", [0,], ["Youden Index Max"]),
     ("with_copd_test1",            "model_prediction", "label", [0,], ["Youden Index Max"]),
     ("without_copd_test1",         "model_prediction", "label", [0,], ["Youden Index Max"]),
     ("kernel_sharp_test1",         "model_prediction", "label", [0,], ["Youden Index Max"]),
     ("kernel_average_test1",       "model_prediction", "label", [0,], ["Youden Index Max"]),
     ("kernel_soft_test1",          "model_prediction", "label", [0,], ["Youden Index Max"]),
    ]
list_series_evaluations_figure_3 = [
     ("radiologist_1",  "radiologist_1_max_malignancy",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_2",  "radiologist_2_max_malignancy",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_3",  "radiologist_3_max_malignancy",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_4",  "radiologist_4_max_malignancy",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_5",  "radiologist_5_max_malignancy",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_6",  "radiologist_6_max_malignancy",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_7",  "radiologist_7_max_malignancy",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_8",  "radiologist_8_max_malignancy",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_9",  "radiologist_9_max_malignancy",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_10", "radiologist_10_max_malignancy", "label", [0,], ["Youden Index Max"]),
     ("radiologist_11", "radiologist_11_max_malignancy", "label", [0,], ["Youden Index Max"]),
     ("radiologist_12", "radiologist_12_max_malignancy", "label", [0,], ["Youden Index Max"]),
     ("radiologist_1",  "model_prediction",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_2",  "model_prediction",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_3",  "model_prediction",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_4",  "model_prediction",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_5",  "model_prediction",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_6",  "model_prediction",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_7",  "model_prediction",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_8",  "model_prediction",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_9",  "model_prediction",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_10", "model_prediction",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_11", "model_prediction",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_12", "model_prediction",  "label", [0,], ["Youden Index Max"]),
    ]

# list_series_evaluations_figure_2 = [("test1", "lad_pred_max_suspicious", "label", [0,], ["Youden Index Max"]),]
list_series_evaluations_figure_5 = [
    ("4_10_mm_test3",  "model_prediction",         "label",[0,],["Youden Index Max"]),
    ("10_20_mm_test3", "model_prediction",         "label",[0,],["Youden Index Max"]),
    ("20_30_mm_test3", "model_prediction",         "label",[0,],["Youden Index Max"]),
    ("4_10_mm_test3",  "4_radiologist_prediction", "label",[0,],["Youden Index Max"]),
    ("10_20_mm_test3", "4_radiologist_prediction", "label",[0,],["Youden Index Max"]),
    ("20_30_mm_test3", "4_radiologist_prediction", "label",[0,],["Youden Index Max"]),
     
    ("stage_1a_test3",   "model_prediction",         "label",[0,],["Youden Index Max"]),
    ("stage_1_test3",    "model_prediction",         "label",[0,],["Youden Index Max"]),
    ("stage_late_test3", "model_prediction",         "label",[0,],["Youden Index Max"]),
    ("stage_1a_test3",   "4_radiologist_prediction", "label",[0,],["Youden Index Max"]),
    ("stage_1_test3",    "4_radiologist_prediction", "label",[0,],["Youden Index Max"]),
    ("stage_late_test3", "4_radiologist_prediction", "label",[0,],["Youden Index Max"]),
    ("stage_1a_test1",   "model_prediction",         "label",[0,],["Youden Index Max"]),
    ("stage_1_test1",    "model_prediction",         "label",[0,],["Youden Index Max"]),
    ("stage_late_test1", "model_prediction",         "label",[0,],["Youden Index Max"]),
    ]
list_series_evaluations_figure_6 = [
    ("test6",               "model_prediction_T-1",       "label", [0,], ["Youden Index Max"]),
    ("test6",               "model_prediction_evolution", "label", [0,], ["Youden Index Max"]),
    ("test6",               "delta_volume",               "label", [0,], ["Youden Index Max"]),
    ("test6",               "delta_lad",                  "label", [1.5,0], ["1.5mm diameter","Youden max"]),
    ("test6",               "RDT",                        "label", [365/600,365/400,0], ["VDT 600 days","VDT 400 days","Youden max"]),
    ("VDT_sup_400_test6&3", "model_prediction_T-2",       "label", [0,], ["Youden Index Max"]),
    ("VDT_sup_400_test6&3", "4_radiologist_prediction",   "label", [0,], ["Youden Index Max"]),
    ("NODCATIII_test6",               "model_prediction_T-1",       "label", [0,], ["Youden Index Max"]),
    ("NODCATIII_test6",               "model_prediction_evolution", "label", [0,], ["Youden Index Max"]),
    ("NODCATIII_test6",               "VDT",                        "label", [400,600,0], ["VDT 600 days","VDT 400 days","Youden max"]),
    ]


# a list of evaluations is a 3 tuple (list_series_evaluations, figure_name, path_to_data)
list_series_evaluations_figure = [
    (list_series_evaluations_figure_1,          "figure_1",          path_data_series),
    (list_series_evaluations_figure_2,          "figure_2",          path_data_series),
    (list_series_evaluations_figure_3,          "figure_3",          path_data_series),
    (list_series_evaluations_figure_5,          "figure_5",          path_data_series),
    (list_series_evaluations_figure_6,          "figure_6",          path_data_series),
]

############################################################################
###################          evaluate_lesions            ###################
###################          evaluate_lesions            ###################
############################################################################
# parameters of the "evaluate_lesions" module
# path of the output files of  "evaluate_lesions" module

path_model_evaluate_lesions = path_model_eval / "evaluate_lesions"
path_data_lesions = path_model_eval / "data_lesions" / "lesions.csv"

path_data_lesions_longitudinal = path_model_eval / "data_lesions" / "lesions_longitudinal.csv"
path_data_lesions_radiologists = path_model_eval / "data_lesions" / "lesions_radiologists.csv"
path_data_lesions_nndetection_CADex = path_model_eval / "data_lesions" / "lesions_nndetection_CADex.csv"
path_data_lesions_model_CADe = path_model_eval / "data_lesions" / "lesions_model_CADe.csv"
path_data_lesions_nndetection_baumgartner_CADe = path_model_eval / "data_lesions" / "lesions_nndetection_baumgartner_CADe.csv"
path_data_lesions_nndetection_CADe = path_model_eval / "data_lesions" / "lesions_nndetection_CADe.csv"

list_lesions_evaluations = [("test1","model_prediction", "label",[0,],["Youden Index Max"]),]


# for size stratification
diameter_threshold_min = [4, 10, 20]
diameter_threshold_max = [10, 20, 32]

list_lesions_evaluations_figure_4 = [
    ("test3", "4_radiologist_prediction",               "label",       [0,], ["Youden Index Max"]), 
    ("test1", "nndetection_CADex_prediction",           "label",       [0,], ["Youden Index Max"]),
    ("test1", "model_CADe_prediction",                  "label_nodule",[0,], ["Youden Index Max"]), 
    ("test1", "nndetection_baumgartner_CADe_prediction","label_nodule",[0,], ["Youden Index Max"]), 
    ("test1", "nndetection_CADe_prediction",            "label_nodule",[0,], ["Youden Index Max"]),  
    ]
list_lesions_evaluations_figure_5 = [
    ("4_10_mm_test3",  "model_prediction",         "label",[0,],["Youden Index Max"]),
    ("10_20_mm_test3", "model_prediction",         "label",[0,],["Youden Index Max"]),
    ("20_30_mm_test3", "model_prediction",         "label",[0,],["Youden Index Max"]),
    ("4_10_mm_test3",  "4_radiologist_prediction", "label",[0,],["Youden Index Max"]),
    ("10_20_mm_test3", "4_radiologist_prediction", "label",[0,],["Youden Index Max"]),
    ("20_30_mm_test3", "4_radiologist_prediction", "label",[0,],["Youden Index Max"]),
    ]
list_lesions_evaluations_figure_6 = [
    ("test6",               "model_prediction_T-1",       "label", [0,], ["Youden Index Max"]),
    ("test6",               "model_prediction_evolution", "label", [0,], ["Youden Index Max"]),
    ("test6",               "delta_volume",               "label", [0,], ["Youden Index Max"]),
    ("test6",               "delta_lad",                  "label", [1.5,0], ["1.5mm diameter","Youden max"]),
    ("test6",               "RDT",                        "label", [365/600,365/400,0], ["VDT 600 days","VDT 400 days","Youden max"]),  
    ("test6",  "delta_volume",             "label", [0,], ["Youden Index Max"]),
    ("test6",  "delta_volume_day_norm",    "label", [0,], ["Youden Index Max"]),
    ("test6",  "RDT",                      "label", [0,], ["Youden Index Max"]),
    ("test6",  "RDT_year_norm",            "label", [0,], ["Youden Index Max"]),
    ("test6",  "VDT",                      "label", [0,], ["Youden Index Max"]),
    ("test6",  "VDT_median_corrected",     "label", [0,], ["Youden Index Max"]),
    ("test6",  "VDT_NELSON_criteria","label", [0,], ["Youden Index Max"]),
    ]
list_lesions_evaluations_figure_supplementary_3 = [
     ("radiologist_1",  "4_radiologist_prediction",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_2",  "4_radiologist_prediction",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_3",  "4_radiologist_prediction",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_4",  "4_radiologist_prediction",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_5",  "4_radiologist_prediction",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_6",  "4_radiologist_prediction",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_7",  "4_radiologist_prediction",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_8",  "4_radiologist_prediction",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_9",  "4_radiologist_prediction",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_10", "4_radiologist_prediction",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_11", "4_radiologist_prediction",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_12", "4_radiologist_prediction",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_1",  "model_prediction",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_2",  "model_prediction",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_3",  "model_prediction",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_4",  "model_prediction",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_5",  "model_prediction",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_6",  "model_prediction",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_7",  "model_prediction",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_8",  "model_prediction",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_9",  "model_prediction",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_10", "model_prediction",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_11", "model_prediction",  "label", [0,], ["Youden Index Max"]),
     ("radiologist_12", "model_prediction",  "label", [0,], ["Youden Index Max"]),
    ]    


# a list of evaluations is a 3 tuple (list_lesions_evaluations, figure_name, path_to_data)
list_lesions_evaluations_figure = [
    (list_lesions_evaluations_figure_4,          "figure_4",          path_data_lesions),
    (list_lesions_evaluations_figure_5,          "figure_5",          path_data_lesions),
    (list_lesions_evaluations_figure_6,          "figure_6",          path_data_lesions_longitudinal),
    (list_lesions_evaluations_figure_supplementary_3, "figure_supplementary_3", path_data_lesions),
]


############################################################################
###################         Statistical Tests          #####################
############################################################################
# parameters of the "Evalutae diameters" module
# path of output the "Evalutae diameters" module
analysis_figure = "figure_1"
path_model_statistical_tests = path_model_eval / "statistical_tests"
#

# input is structured as list of tuples of paths like [("name_comparison 1", "path1", "path2"), ("name_comparison 2", "path3", "path4")]
# of course tuple are ordered and the order of the H1 test is "path1"  greater than "path2"
#path_model_eval / "evaluate_series"
#list_of_tuples_of_pairs_of_bootstrap_paths = [
#    ("fake_comparaison_test_AUC", "/path" / "AUC_array_5000_bootstrap_test.npy", "/path" / "AUC_array_5000_bootstrap_test.npy",),
#    ("fake_comparaison_test_ACCURACY","/path/accuracy_array_5000_bootstrap_for_each_OP_ROC_test.npy","/path/accuracy_array_5000_bootstrap_for_each_OP_ROC_test.npy",),
#    ]

figure_name = "figure_1"
root = path_model_eval / "evaluate_series"/ figure_name
list_of_tuples_of_pairs_of_bootstrap_paths_figure_1 = [
    ("AUC_model_vs_4radiolog_test3",             root / "model_prediction_test3" / "AUC_array_5000_bootstrap_test3.npy", root / "4_radiologist_prediction_test3" / "AUC_array_5000_bootstrap_test3.npy",),
    ("AUC_model_vs_lungrads_wo_prior_test4",     root / "model_prediction_test4" / "AUC_array_5000_bootstrap_test4.npy", root / "lungrads_wo_prior_prediction_test4" / "AUC_array_5000_bootstrap_test4.npy",),
    ("AUC_model_vs_lungrads_with_prior_test5",   root / "model_prediction_test5" / "AUC_array_5000_bootstrap_test5.npy", root / "lungrads_with_prior_prediction_test5" / "AUC_array_5000_bootstrap_test5.npy",),
    ("AUC_model_vs_liao_test2",                  root / "model_prediction_test2" / "AUC_array_5000_bootstrap_test2.npy", root / "liao_prediction_test2" / "AUC_array_5000_bootstrap_test2.npy",),
    ("AUC_model_vs_brock_test2",                 root / "model_prediction_test2" / "AUC_array_5000_bootstrap_test2.npy", root / "brock_prediction_test2" / "AUC_array_5000_bootstrap_test2.npy",),
    ("AUC_model_vs_ardila_test2",                root / "model_prediction_test2" / "AUC_array_5000_bootstrap_test2.npy", root / "ardila_prediction_test2" / "AUC_array_5000_bootstrap_test2.npy",),
    ("AUC_model_vs_sybil_test2",                 root / "model_prediction_test2" / "AUC_array_5000_bootstrap_test2.npy", root / "sybil_prediction_test2" / "AUC_array_5000_bootstrap_test2.npy",),
    ("SENS_model_vs_lungrads_wo_prior_test4",    root / "model_prediction_test4" / "sensitivity_array_5000_bootstrap_for_each_OP_ROC_test4.npy", root / "lungrads_wo_prior_prediction_test4" / "sensitivity_array_5000_bootstrap_for_each_OP_ROC_test4.npy",),
    ("SENS_model_vs_lungrads_with_prior_test5",  root / "model_prediction_test5" / "sensitivity_array_5000_bootstrap_for_each_OP_ROC_test5.npy", root / "lungrads_with_prior_prediction_test5" / "sensitivity_array_5000_bootstrap_for_each_OP_ROC_test5.npy",),
    ("SPEC_model_vs_lungrads_wo_prior_test4",    root / "model_prediction_test4" / "specificity_array_5000_bootstrap_for_each_OP_ROC_test4.npy", root / "lungrads_wo_prior_prediction_test4" / "specificity_array_5000_bootstrap_for_each_OP_ROC_test4.npy",),
    ("SPEC_model_vs_lungrads_with_prior_test5",  root / "model_prediction_test5" / "specificity_array_5000_bootstrap_for_each_OP_ROC_test5.npy", root / "lungrads_with_prior_prediction_test5" / "specificity_array_5000_bootstrap_for_each_OP_ROC_test5.npy",),
]
figure_name = "figure_3"
root = path_model_eval / "evaluate_series"/ figure_name
list_of_tuples_of_pairs_of_bootstrap_paths_figure_3 = [
    ("AUC_radiolog_1_model_vs_radiolog", root / "model_prediction_radiologist_1" / "AUC_array_5000_bootstrap_radiologist_1.npy", root / "radiologist_1_max_malignancy_radiologist_1" / "AUC_array_5000_bootstrap_radiologist_1.npy"),
    ("AUC_radiolog_2_model_vs_radiolog", root / "model_prediction_radiologist_2" / "AUC_array_5000_bootstrap_radiologist_2.npy", root / "radiologist_2_max_malignancy_radiologist_2" / "AUC_array_5000_bootstrap_radiologist_2.npy"),
    ("AUC_radiolog_3_model_vs_radiolog", root / "model_prediction_radiologist_3" / "AUC_array_5000_bootstrap_radiologist_3.npy", root / "radiologist_3_max_malignancy_radiologist_3" / "AUC_array_5000_bootstrap_radiologist_3.npy"),
    ("AUC_radiolog_4_model_vs_radiolog", root / "model_prediction_radiologist_4" / "AUC_array_5000_bootstrap_radiologist_4.npy", root / "radiologist_4_max_malignancy_radiologist_4" / "AUC_array_5000_bootstrap_radiologist_4.npy"),
    ("AUC_radiolog_5_model_vs_radiolog", root / "model_prediction_radiologist_5" / "AUC_array_5000_bootstrap_radiologist_5.npy", root / "radiologist_5_max_malignancy_radiologist_5" / "AUC_array_5000_bootstrap_radiologist_5.npy"),
    ("AUC_radiolog_6_model_vs_radiolog", root / "model_prediction_radiologist_6" / "AUC_array_5000_bootstrap_radiologist_6.npy", root / "radiologist_6_max_malignancy_radiologist_6" / "AUC_array_5000_bootstrap_radiologist_6.npy"),
    ("AUC_radiolog_7_model_vs_radiolog", root / "model_prediction_radiologist_7" / "AUC_array_5000_bootstrap_radiologist_7.npy", root / "radiologist_7_max_malignancy_radiologist_7" / "AUC_array_5000_bootstrap_radiologist_7.npy"),
    ("AUC_radiolog_8_model_vs_radiolog", root / "model_prediction_radiologist_8" / "AUC_array_5000_bootstrap_radiologist_8.npy", root / "radiologist_8_max_malignancy_radiologist_8" / "AUC_array_5000_bootstrap_radiologist_8.npy"),
    ("AUC_radiolog_9_model_vs_radiolog", root / "model_prediction_radiologist_9" / "AUC_array_5000_bootstrap_radiologist_9.npy", root / "radiologist_9_max_malignancy_radiologist_9" / "AUC_array_5000_bootstrap_radiologist_9.npy"),
    ("AUC_radiolog_10_model_vs_radiolog", root / "model_prediction_radiologist_10" / "AUC_array_5000_bootstrap_radiologist_10.npy", root / "radiologist_10_max_malignancy_radiologist_10" / "AUC_array_5000_bootstrap_radiologist_10.npy"),
    ("AUC_radiolog_11_model_vs_radiolog", root / "model_prediction_radiologist_11" / "AUC_array_5000_bootstrap_radiologist_11.npy", root / "radiologist_11_max_malignancy_radiologist_11" / "AUC_array_5000_bootstrap_radiologist_11.npy"),
    ("AUC_radiolog_12_model_vs_radiolog", root / "model_prediction_radiologist_12" / "AUC_array_5000_bootstrap_radiologist_12.npy", root / "radiologist_12_max_malignancy_radiologist_12" / "AUC_array_5000_bootstrap_radiologist_12.npy"),
] 
figure_name = "figure_4"
root = path_model_eval / "evaluate_lesions"/ figure_name
list_of_tuples_of_pairs_of_bootstrap_paths_figure_4 = [
    ("SENS_at_0.5_and_1_FP_per_scan__model_vs_4radiolog_test3",             root / "model_prediction_test3/at_0.5_and_1_FP_per_scansensitivity_array_5000bootstrap_for_each_OP_FROC_test3.npy",      root / "4_radiologist_prediction_test3/at_0.5_and_1_FP_per_scansensitivity_array_5000bootstrap_for_each_OP_FROC_test3.npy",),
    ("SENS_at_0.5_and_1_FP_per_scan__model_vs_nnDetection_CADex_test1",     root / "model_prediction_test1/at_0.5_and_1_FP_per_scansensitivity_array_5000bootstrap_for_each_OP_FROC_test1.npy",      root / "nndetection_CADex_prediction_test1/at_0.5_and_1_FP_per_scansensitivity_array_5000bootstrap_for_each_OP_FROC_test1.npy",),
    ("SENS_at_0.5_and_1_FP_per_scan__model_CADe_vs_nnDetection_CADe_test1", root / "model_CADe_prediction_test1/at_0.5_and_1_FP_per_scansensitivity_array_5000bootstrap_for_each_OP_FROC_test1.npy", root / "nndetection_CADe_prediction_test1/at_0.5_and_1_FP_per_scansensitivity_array_5000bootstrap_for_each_OP_FROC_test1.npy",),
    ("SENS_at_0.5_and_1_FP_per_scan__model_CADe_vs_Baumgartner_CADe_test1", root / "model_CADe_prediction_test1/at_0.5_and_1_FP_per_scansensitivity_array_5000bootstrap_for_each_OP_FROC_test1.npy", root / "nndetection_baumgartner_CADe_prediction_test1/at_0.5_and_1_FP_per_scansensitivity_array_5000bootstrap_for_each_OP_FROC_test1.npy",),
]

figure_name = "figure_5"
root = path_model_eval / "evaluate_series"/ figure_name
root2 =  path_model_eval / "evaluate_lesions"/ figure_name
list_of_tuples_of_pairs_of_bootstrap_paths_figure_5 = [
    ("AUC_patient_model_vs_4radiolog_4_10_lad",         root / "model_prediction_4_10_mm_test3" / "AUC_array_5000_bootstrap_4_10_mm_test3.npy",       root / "4_radiologist_prediction_4_10_mm_test3" / "AUC_array_5000_bootstrap_4_10_mm_test3.npy"),
    ("AUC_patient_model_vs_4radiolog_10_20_lad",        root / "model_prediction_10_20_mm_test3" / "AUC_array_5000_bootstrap_10_20_mm_test3.npy",     root / "4_radiologist_prediction_10_20_mm_test3" / "AUC_array_5000_bootstrap_10_20_mm_test3.npy"),
    ("AUC_patient_model_vs_4radiolog_20_30_lad",        root / "model_prediction_20_30_mm_test3" / "AUC_array_5000_bootstrap_20_30_mm_test3.npy",     root / "4_radiologist_prediction_20_30_mm_test3" / "AUC_array_5000_bootstrap_20_30_mm_test3.npy"),
    ("AUC_patient_model_vs_4radiolog_stage_1a_test3",   root / "model_prediction_stage_1a_test3" / "AUC_array_5000_bootstrap_stage_1a_test3.npy",     root / "4_radiologist_prediction_stage_1a_test3" / "AUC_array_5000_bootstrap_stage_1a_test3.npy"),
    ("AUC_patient_model_vs_4radiolog_stage_1_test3",    root / "model_prediction_stage_1_test3" / "AUC_array_5000_bootstrap_stage_1_test3.npy",       root / "4_radiologist_prediction_stage_1_test3" / "AUC_array_5000_bootstrap_stage_1_test3.npy"),
    ("AUC_patient_model_vs_4radiolog_stage_late_test3", root / "model_prediction_stage_late_test3" / "AUC_array_5000_bootstrap_stage_late_test3.npy", root / "4_radiologist_prediction_stage_late_test3" / "AUC_array_5000_bootstrap_stage_late_test3.npy"),
    ("AUC_lesion_model_vs_4radiolog_4_10_lad",          root2 / "model_prediction_4_10_mm_test3" / "AUC_array_5000_bootstrap_4_10_mm_test3.npy",      root2 / "4_radiologist_prediction_4_10_mm_test3" / "AUC_array_5000_bootstrap_4_10_mm_test3.npy"),
    ("AUC_lesion_model_vs_4radiolog_10_20_lad",         root2 / "model_prediction_10_20_mm_test3" / "AUC_array_5000_bootstrap_10_20_mm_test3.npy",    root2 / "4_radiologist_prediction_10_20_mm_test3" / "AUC_array_5000_bootstrap_10_20_mm_test3.npy"),
    ("AUC_lesion_model_vs_4radiolog_20_30_lad",         root2 / "model_prediction_20_30_mm_test3" / "AUC_array_5000_bootstrap_20_30_mm_test3.npy",    root2 / "4_radiologist_prediction_20_30_mm_test3" / "AUC_array_5000_bootstrap_20_30_mm_test3.npy"),
]
figure_name = "figure_6"
root = path_model_eval / "evaluate_series"/ figure_name
root2 =  path_model_eval / "evaluate_lesions"/ figure_name
list_of_tuples_of_pairs_of_bootstrap_paths_figure_6 = [
    ("AUC_lesion_model_T-1_vs_model_evolution_test6", root2 / "model_prediction_T-1_test6" / "AUC_array_5000_bootstrap_test6.npy", root2 / "model_prediction_evolution_test6" / "AUC_array_5000_bootstrap_test6.npy"),
    ("AUC_lesion_model_T-1_vs_delta_volume_test6",    root2 / "model_prediction_T-1_test6" / "AUC_array_5000_bootstrap_test6.npy", root2 / "delta_volume_test6" / "AUC_array_5000_bootstrap_test6.npy"),
    ("AUC_lesion_model_T-1_vs_delta_lad_test6",       root2 / "model_prediction_T-1_test6" / "AUC_array_5000_bootstrap_test6.npy", root2 / "delta_lad_test6" / "AUC_array_5000_bootstrap_test6.npy"),
    ("AUC_lesion_model_T-1_vs_RDT_test6",             root2 / "model_prediction_T-1_test6" / "AUC_array_5000_bootstrap_test6.npy", root2 / "RDT_test6" / "AUC_array_5000_bootstrap_test6.npy"),
    ("AUC_lesion_delta_volume_vs_delta_lad_test6",    root2 / "delta_volume_test6" / "AUC_array_5000_bootstrap_test6.npy",         root2 / "delta_lad_test6" / "AUC_array_5000_bootstrap_test6.npy"),
    ("AUC_lesion_delta_volume_vs_RDT_test6",          root2 / "delta_volume_test6" / "AUC_array_5000_bootstrap_test6.npy",         root2 / "RDT_test6" / "AUC_array_5000_bootstrap_test6.npy"),
    ("AUC_lesion_delta_lad_vs_RDT_test6",             root2 / "delta_lad_test6" / "AUC_array_5000_bootstrap_test6.npy",            root2 / "RDT_test6" / "AUC_array_5000_bootstrap_test6.npy"),
    ("AUC_patient_model_T-1_vs_model_evolution_test6",root / "model_prediction_T-1_test6" / "AUC_array_5000_bootstrap_test6.npy",  root / "model_prediction_evolution_test6" / "AUC_array_5000_bootstrap_test6.npy"),
    ("AUC_patient_model_T-1_vs_delta_volume_test6",   root / "model_prediction_T-1_test6" / "AUC_array_5000_bootstrap_test6.npy",  root / "delta_volume_test6" / "AUC_array_5000_bootstrap_test6.npy"),
    ("AUC_patient_model_T-1_vs_delta_lad_test6",      root / "model_prediction_T-1_test6" / "AUC_array_5000_bootstrap_test6.npy",  root / "delta_lad_test6" / "AUC_array_5000_bootstrap_test6.npy"),
    ("AUC_patient_model_T-1_vs_RDT_test6",            root / "model_prediction_T-1_test6" / "AUC_array_5000_bootstrap_test6.npy",  root / "RDT_test6" / "AUC_array_5000_bootstrap_test6.npy"),
    ("AUC_patient_model__T-2_VDT_sup_400_vs_4_radiologist_VDT_sup_400_test6&3",root / "model_prediction_T-2_VDT_sup_400_test6&3" / "AUC_array_5000_bootstrap_VDT_sup_400_test6&3.npy",  root / "4_radiologist_prediction_VDT_sup_400_test6&3" / "AUC_array_5000_bootstrap_VDT_sup_400_test6&3.npy"),
    ("AUC_lesion_delta_volume_vs_delta_volume_day_norm_test6", root2 / "delta_volume_test6" / "AUC_array_5000_bootstrap_test6.npy", root2 / "delta_volume_day_norm_test6" / "AUC_array_5000_bootstrap_test6.npy"),
    ("AUC_lesion_RDT_vs_RDT_year_norm_test6",                  root2 / "RDT_test6" / "AUC_array_5000_bootstrap_test6.npy",          root2 / "RDT_year_norm_test6" / "AUC_array_5000_bootstrap_test6.npy"),
    ("AUC_lesion_delta_volume_vs_VDT_median_corrected_test6",  root2 / "delta_volume_test6" / "AUC_array_5000_bootstrap_test6.npy", root2 / "VDT_median_corrected_test6" / "AUC_array_5000_bootstrap_test6.npy"),
    ("AUC_lesion_delta_volume_vs_VDT_NELSON_criteria_test6",   root2 / "delta_volume_test6" / "AUC_array_5000_bootstrap_test6.npy", root2 / "VDT_NELSON_criteria_test6" / "AUC_array_5000_bootstrap_test6.npy"),
    ("AUC_lesion_delta_volume_vs_VDT_test6",                   root2 / "delta_volume_test6" / "AUC_array_5000_bootstrap_test6.npy", root2 / "VDT_test6" / "AUC_array_5000_bootstrap_test6.npy"),
    ("AUC_patient_model_T-1_NODCATIII_vs_model_evolution_NODCATIII_test6",root / "model_prediction_T-1_NODCATIII_test6" / "AUC_array_5000_bootstrap_NODCATIII_test6.npy",  root / "model_prediction_evolution_NODCATIII_test6" / "AUC_array_5000_bootstrap_NODCATIII_test6.npy"),
    ("AUC_patient_model_T-1_NODCATIII_vs_delta_volume_test6",             root / "model_prediction_T-1_NODCATIII_test6" / "AUC_array_5000_bootstrap_NODCATIII_test6.npy",  root / "VDT_NODCATIII_test6" / "AUC_array_5000_bootstrap_NODCATIII_test6.npy"),
]
 


list_statistical_tests_figure = [
    (list_of_tuples_of_pairs_of_bootstrap_paths_figure_1,          "figure_1",),
    (list_of_tuples_of_pairs_of_bootstrap_paths_figure_3,          "figure_3",),
    (list_of_tuples_of_pairs_of_bootstrap_paths_figure_4,          "figure_4",),
    (list_of_tuples_of_pairs_of_bootstrap_paths_figure_5,          "figure_5",),
    (list_of_tuples_of_pairs_of_bootstrap_paths_figure_6,          "figure_6",),
]