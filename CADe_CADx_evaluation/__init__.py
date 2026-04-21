"""CADe/CADx Evaluation Framework.

This package provides tools for evaluating the performance of
Computer-Aided Detection (CADe) and Computer-Aided Diagnosis (CADx) models.
It supports:

- **Series-level evaluation** — patient/scan-level ROC, PR curves, and
  operating-point characterisation.
- **Lesion-level evaluation** — nodule-level ROC and FROC curves with
  bootstrap confidence intervals.
- **Statistical tests** — bootstrap-based paired significance testing.

Typical usage::

    uv run python run_paper_evaluation.py --fast_computation=true
"""
