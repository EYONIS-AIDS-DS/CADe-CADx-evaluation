Changelog
=========

All notable changes to this project will be documented in this file.
The format follows `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to
`Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

Unreleased
----------

Added
^^^^^
* Sphinx/ReadTheDocs documentation with full API reference.
* NumPy-style docstrings across all public functions and modules.

Fixed
^^^^^
* ``confidence_threshold`` argument now correctly parsed as ``float``
  instead of ``int``.
* Operating-point labels type annotation corrected from ``float`` to ``str``
  in ``evaluate_series``.
* Swapped log messages for equal/unequal variance branches in
  ``statistical_tests``.
* Merged duplicate ``from typing import`` statements in
  ``roc_confidence_interval``.
* Corrected ``accuracy()`` return-type annotation from
  ``Tuple[float, float]`` to ``float``.
* Replaced ``logger.error("...", count)`` with an f-string in ``froc.py``.

1.0.0 — Initial Release
-------------------------

* Full evaluation pipeline for reproducing all paper figures.
* Series-level and lesion-level ROC / FROC / PR evaluation.
* Bootstrap and Hanley-McNeil AUC confidence intervals.
* Statistical significance testing.
