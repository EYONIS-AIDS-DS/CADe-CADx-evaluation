CADe/CADx Evaluation Documentation
=====================================

.. image:: https://img.shields.io/badge/python-3.11-blue.svg
   :alt: Python 3.11

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :alt: MIT license

.. image:: https://img.shields.io/badge/code%20style-black-black.svg
   :alt: Code style: black

.. image:: https://readthedocs.org/projects/cade-cadx-evaluation/badge/?version=latest
   :target: https://cade-cadx-evaluation.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

----

**CADe/CADx Evaluation** is a Python framework for benchmarking Computer-Aided
Detection (CADe) and Computer-Aided Diagnosis (CADx) systems on lung nodule CT
datasets.  It reproduces the full evaluation pipeline from the companion paper,
covering:

* **Series-level evaluation** — patient/scan ROC curves, PR curves, and CI estimation
* **Lesion-level evaluation** — nodule ROC, FROC curves with bootstrap CIs
* **Statistical tests** — bootstrap-based paired significance testing

.. note::

   All statistical outputs are designed to be fully reproducible: setting the
   same ``--nb_bootstrap_samples`` value and random seeds will yield identical
   results.

----

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   configuration

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   input_format
   outputs

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Project

   changelog
   contributing

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
