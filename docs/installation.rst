Installation
============

Requirements
------------

* `uv <https://docs.astral.sh/uv/>`_ (recommended — manages Python and all dependencies automatically)

.. note::

   The project requires **Python 3.11** (declared in ``pyproject.toml`` as
   ``requires-python = ">=3.11,<3.12"``).  When using ``uv``, the correct
   Python version is selected and installed automatically — no manual Python
   setup is needed.  Dependencies are pinned for reproducibility, including
   **NumPy 1.26.4**.

Installing with uv (recommended)
---------------------------------

.. code-block:: bash

   # 1. Clone the repository
   git clone git@github.com:EYONIS-AIDS-DS/CADe-CADx-evaluation.git
   cd CADe-CADx-evaluation

   # 2. Create a virtual environment and install all dependencies
   uv venv
   uv sync

Installing with pip (without cloning)
--------------------------------------

If you only need the package in your own project, install directly from GitHub:

.. code-block:: bash

   pip install git+https://github.com/EYONIS-AIDS-DS/CADe-CADx-evaluation.git

.. note::

   This installs the package only — the ``data/`` directory and
   ``config_paper.py`` needed to reproduce paper figures are not included.
   Clone the full repository for that.

Installing with pip (from a clone)
-----------------------------------

.. code-block:: bash

   git clone git@github.com:EYONIS-AIDS-DS/CADe-CADx-evaluation.git
   cd CADe-CADx-evaluation
   pip install -e .

Verifying the Installation
--------------------------

.. code-block:: bash

   uv run python run_paper_evaluation.py --help

You should see the list of command-line arguments printed to the terminal.

Building the Documentation Locally
------------------------------------

.. code-block:: bash

   pip install -r docs/requirements.txt
   sphinx-build docs docs/_build/html
   # Open docs/_build/html/index.html in a browser
