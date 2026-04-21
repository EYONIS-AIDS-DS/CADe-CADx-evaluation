Installation
============

Requirements
------------

* Python **3.11** or higher
* `uv <https://docs.astral.sh/uv/>`_ (recommended package manager)

.. note::

   The project is tested with **NumPy 1.26.4** and the exact dependencies
   pinned in ``pyproject.toml``.  Using ``uv`` ensures reproducible installs.

Installing with uv (recommended)
---------------------------------

.. code-block:: bash

   # 1. Clone the repository
   git clone git@github.com:EYONIS-AIDS-DS/CADe-CADx-evaluation.git
   cd CADe-CADx-evaluation

   # 2. Create a virtual environment and install all dependencies
   uv venv
   uv sync

Installing with pip
-------------------

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
