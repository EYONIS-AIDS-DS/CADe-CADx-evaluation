"""Common evaluation utilities shared across series- and lesion-level analyses.

Modules
-------
logger
    Package-wide logger singleton.
roc
    ROC curve computation and plotting.
roc_confidence_interval
    Bootstrap and Hanley-McNeil confidence intervals for AUC.
precision_recall
    Precision-Recall curve plotting.
sens_spec
    Sensitivity, specificity, and accuracy at an operating threshold.
sample_size_analysis
    Dataset size and class-imbalance reporting.
plot_score_distribution_benign_cancer
    Histogram / KDE of model scores stratified by label.
"""
