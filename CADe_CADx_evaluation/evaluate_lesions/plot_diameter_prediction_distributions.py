
import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats

from CADe_CADx_evaluation.evaluate_common.logger import logger



def compute_diameter_distributions(features, diametter_used,  prediction_used, expdir_analysis, set_name):  
 
    feature = features.loc[ features[diametter_used] >= 4 ]
    malignant_nodule_set  = feature.loc[(feature["label"] == 1) ]
    benign_nodule_set  = feature.loc[(feature["label"] == 0) ]
    
    
    benign_lad_diameter_mm =  benign_nodule_set[diametter_used].to_numpy()
    malignant_lad_diameter_mm =  malignant_nodule_set[diametter_used].to_numpy()    
    
 
    if np.amax(benign_lad_diameter_mm) > np.amax(malignant_lad_diameter_mm) :
        max_x = np.amax(benign_lad_diameter_mm)
    else:    
        max_x = np.amax(malignant_lad_diameter_mm)
    if np.amin(malignant_lad_diameter_mm) > np.amin(benign_lad_diameter_mm) :
        min_x = np.amin(benign_lad_diameter_mm)
    else:    
        min_x = np.amin(malignant_lad_diameter_mm)    
   
    plt.figure(figsize=(9.6, 7.2))
    plt.title('Joint Distribution of Malignancy prediction and diameter of Model findings')
    ax1 = plt.gca()
    ax1.scatter( malignant_nodule_set[diametter_used].to_numpy(), malignant_nodule_set[prediction_used].to_numpy(), s=10, alpha=0.5, color = "#FF0000" , marker='s', label='Malignant findings')
    ax1.set_xlim((min_x,max_x))
    ax1.set_ylim((0,1))
    ax1.scatter( benign_nodule_set[diametter_used].to_numpy(), benign_nodule_set[prediction_used].to_numpy(),  s=10, alpha=0.5, color = "#00AEEF" ,  marker='o', label='Benign findings')
    ax1.set_xlim((min_x,max_x))
    ax1.set_ylim((0,1))
    plt.xlim([min_x, max_x])
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.ylabel('Malignancy probability')
    plt.xlabel(diametter_used)
    plt.savefig(os.path.join(expdir_analysis, "Joint_Distribution_of_Malignancy_prediction_and_diameter_of_Model_findings_"  + str(set_name) + ".svg",))
    plt.savefig(os.path.join(expdir_analysis, "Joint_Distribution_of_Malignancy_prediction_and_diameter_of_Model_findings_"  + str(set_name) + ".png",))
    plt.close()
 
    bins = np.linspace(min_x, max_x, 100)
    fig, (ax) = plt.subplots(1, 1, figsize=[9.6, 7.2])
    plt.title('Distribution of nodule diameter')
    values_bins_canc, bins_canc, bars = plt.hist(malignant_nodule_set[diametter_used].to_numpy(), bins, alpha=0.5, color = "#FF0000" , label='Malignant findings')
    plt.hist(benign_nodule_set[diametter_used].to_numpy(), bins, alpha=0.5, color = "#00AEEF" , label='Benign findings')
    plt.xlim([min_x, max_x])
    plt.legend(loc='upper right')
    plt.ylabel('Number of findings')
    plt.xlabel(diametter_used)  
    
    # INSERT CARTOON ZOOM of distribution
    _ = inset_axes( ax, width="80%", height="80%",)
    plt.hist(malignant_nodule_set[diametter_used].to_numpy(), bins, alpha=0.5, color="#FF0000", label="malignant findings")
    plt.hist(benign_nodule_set[diametter_used].to_numpy(), bins, alpha=0.5, color="#00AEEF", label="benign findings")
    plt.legend(loc='upper right')
    plt.xlim([min_x, max_x])
    plt.ylim([0, values_bins_canc.max() + 10])
    plt.savefig(os.path.join(expdir_analysis, "Distribution_of_diameter_of_Model_findings_" + str(set_name) + ".svg",))
    plt.savefig(os.path.join(expdir_analysis, "Distribution_of_diameter_of_Model_findings_" + str(set_name) + ".png",))
    plt.close(fig)
    pearson_corr, p_value = stats.pearsonr(malignant_nodule_set[diametter_used].to_numpy(), malignant_nodule_set[prediction_used].to_numpy())
    logger.info(f"Pearson correlation coefficient: {pearson_corr}")
    logger.info(f"p-value: {p_value}")


#