import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def distribution_risk_malignant_benign(df: pd.DataFrame,
                                       prediction: str,
                                       expdir_analysis: str, 
                                       set_name: str) -> None:
    
    benign_df = df.loc[(df["label"] == 0)]
    cancer_df = df.loc[(df["label"] == 1)]

    benign_test_proba =  benign_df[prediction].to_numpy() 
    cancer_test_proba =  cancer_df[prediction].to_numpy()
    operating_points_scores = pd.read_csv(os.path.join(expdir_analysis, "operating_point_scores_" + str(set_name) + ".csv",))
    bins_list = operating_points_scores["OP #"].tolist()
    
    bins_list.insert(0, float(0))
    
    bins = bins_list
    values_bins_canc, bins_canc, bars = plt.hist(cancer_test_proba, bins, edgecolor='black', alpha=0.5, color = "#FF0000" )
    values_bins_beni, bins_beni, bars = plt.hist(benign_test_proba, bins, edgecolor='black', alpha=0.5, color = "#00AEEF" )
    values_bins_canc_norm = values_bins_canc*100 / (values_bins_canc+values_bins_beni)
    values_bins_beni_norm = values_bins_beni*100 / (values_bins_canc+values_bins_beni)

    cancer_per_scores = operating_points_scores[["OP #","Th."]].copy()
    cancer_per_scores["cancer % per score"] = values_bins_canc_norm
    cancer_per_scores["benign % per score"] = values_bins_beni_norm
    cancer_per_scores["cancer # per score"] = values_bins_canc
    cancer_per_scores["benign # per score"] = values_bins_beni 

    cancer_per_scores.to_csv(os.path.join(expdir_analysis, "cancer_per_scores_" + str(set_name) + ".csv",), index=False, sep=",",)
    
    label_bins = operating_points_scores["label"].tolist()
    centers = np.arange(len(values_bins_canc_norm))
    
    plt.figure(figsize=(9.6, 7.2))
    plt.title('Distribution of malignant and benign per score')
    plt.bar(centers, values_bins_canc_norm,  tick_label = label_bins, edgecolor='black', align='edge', width = 0.33, alpha=0.5, color = "#FF0000" , label='malignant patient')
    plt.bar(centers+0.3333, values_bins_beni_norm,  tick_label = label_bins, edgecolor='black', align='edge', width = 0.33, alpha=0.5, color = "#00AEEF" , label='benign patient')
    plt.xlim([0, 10]) 
    plt.legend(loc='center left')
    plt.ylabel('%'+' in each score')
    plt.xlabel('Predicted malignancy scores')
    plt.savefig(os.path.join( expdir_analysis, "distribution_malignant_benign_per_score_" + str(set_name) + ".png", ))
    plt.savefig(os.path.join(expdir_analysis, "distribution_malignant_benign_per_score_" + str(set_name) + ".svg",))
    plt.close()
    plt.figure(figsize=(9.6, 7.2))
    plt.title('Distribution of malignant per score ')
    plt.bar(centers-0.5, values_bins_canc_norm,  tick_label = label_bins, edgecolor='black', align='center', width = 0.3, color = "#FF0000" , label='malignant cases')
    plt.xlim([-1, 10]) 
    plt.ylim([0, 101]) 
    plt.legend(loc='center left')
    plt.ylabel('Cancer %'+' in each score')
    plt.xlabel('Predicted malignancy scores')
    plt.savefig(os.path.join(expdir_analysis, "distribution_malignant_per_score_"  + str(set_name) + ".png",))
    plt.savefig(os.path.join(expdir_analysis, "distribution_malignant_per_score_" + str(set_name) + ".svg",))
    plt.close()


