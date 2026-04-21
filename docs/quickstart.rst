Quick Start
===========

Running the Paper Evaluation
-----------------------------

The repository ships with pre-processed CSV data under ``data/`` so you can
reproduce all paper figures immediately.

Fast mode (recommended for testing, ~8 minutes)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Uses 100 bootstrap samples instead of 5 000:

.. code-block:: bash

   uv run python run_paper_evaluation.py --fast_computation=true

Full mode (~24 hours)
^^^^^^^^^^^^^^^^^^^^^

Runs all 5 000 bootstrap replicates as used in the paper:

.. code-block:: bash

   uv run python run_paper_evaluation.py --fast_computation=false

Output is written to ``data/evaluate_lesions/`` and
``data/evaluate_series/`` under figure-specific sub-directories.

Running a Single Evaluation
----------------------------

Each evaluation module has its own CLI entry point.

Series-level ROC (example: model predictions, test set 1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   uv run python -m CADe_CADx_evaluation.evaluate_series.evaluate_series \
       --prediction model_prediction \
       --set_name test1 \
       --fast_computation true

Lesion-level FROC (example: nnDetection CADe, test set 1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   uv run python -m CADe_CADx_evaluation.evaluate_lesions.evaluate_lesions \
       --prediction nndetection_CADe_prediction \
       --set_name test1 \
       --fast_computation true

Logging
-------

All runs write a log file ``CADe_CADx_evaluate.log`` two levels above the
package root (i.e. at the repository root when using the standard layout).
Progress bars are displayed in the terminal using ``tqdm``.
