Configuration
=============

All paper-specific parameters are defined in ``config_paper.py`` at the
repository root.  The file is imported by ``run_paper_evaluation.py`` and
acts as the single source of truth for all evaluation settings.

Key Parameters
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - ``fast_computation``
     - ``false``
     - Use a reduced number of bootstrap samples for speed.
   * - ``nb_bootstrap_samples``
     - ``5000``
     - Number of bootstrap replicates for CI estimation. Reduced to
       ``100`` when ``fast_computation=true``.
   * - ``confidence_threshold``
     - ``0.95``
     - Coverage probability for all confidence intervals.
   * - ``data_dir``
     - ``data/``
     - Root directory for input CSV files.
   * - ``results_dir``
     - ``data/``
     - Root directory for output files.

Evaluation Tuples
-----------------

The ``list_series_evaluations_figure``, ``list_lesions_evaluations_figure``,
and ``list_statistical_tests_figure`` variables each map a *figure number*
to a list of named evaluation dictionaries.  Each dictionary specifies:

* ``prediction`` — column name of the model score in the input CSV
* ``set_name`` — evaluation subset identifier (e.g. ``"test1"``,
  ``"test3"``)
* ``expdir_analysis`` — output directory for results of this evaluation
* ``name_analysis`` — human-readable label used in plot titles

Overriding Parameters via the CLI
----------------------------------

``run_paper_evaluation.py`` accepts the following command-line arguments
which override the values in ``config_paper.py``:

.. code-block:: text

   --fast_computation     true/false
   --nb_bootstrap_samples INT
   --confidence_threshold FLOAT (0–1)
