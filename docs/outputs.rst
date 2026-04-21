Outputs
=======

All results are written to figure-specific sub-directories under
``data/evaluate_lesions/`` or ``data/evaluate_series/``.  The exact paths
are configured in ``config_paper.py``.

CSV Files
---------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - File
     - Contents
   * - ``roc_fpr_tpr_<set>.csv``
     - ROC curve arrays (fpr, tpr, thresholds).
   * - ``roc_CI_bootstrap_hanley_<set>.csv``
     - AUC point estimate plus Hanley-McNeil and bootstrap 95 % CIs.
   * - ``operating_point_performances_<set>.csv``
     - Sensitivity, specificity, and accuracy at each operating point.
   * - ``operating_point_scores_<set>.csv``
     - Raw model thresholds at each operating point.
   * - ``sample_size_<set>.csv``
     - Total, positive, and negative sample counts plus imbalance ratio.
   * - ``froc_results_<set>.csv``
     - FROC curve (FP/scan, sensitivity) and per-OP bootstrap CIs.
   * - ``statistical_test_results_<set>.csv``
     - Bootstrap p-values for each pair of systems under comparison.

Figure Files
------------

PNG and SVG versions of every figure are saved alongside the CSV results:

* ``roc_curve_with_op_<set>.{png,svg}`` — ROC curve with operating points
* ``precision_recall_curve_model_<set>.{png,svg}`` — PR curve
* ``distribution_proba_<set>.{png,svg}`` — Score distributions
* ``froc_curve_<set>.{png,svg}`` — FROC curve with two operating points
* ``diameter_distribution_<set>.{png,svg}`` — Diameter vs. score scatter

NumPy Arrays
------------

Bootstrap sample arrays are saved as ``.npy`` files for reproducibility
and downstream analysis:

* ``AUC_array_5000_bootstrap_<set>.npy``
* ``sensitivity_array_5000_bootstrap_for_each_OP_ROC_<set>.npy``
* ``specificity_array_5000_bootstrap_for_each_OP_ROC_<set>.npy``
* ``<type>sensitivity_array_5000bootstrap_for_each_OP_FROC_<set>.npy``
