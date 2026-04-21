Contributing
============

Contributions are welcome! Please follow the guidelines below.

Setting Up a Development Environment
--------------------------------------

.. code-block:: bash

   git clone git@github.com:EYONIS-AIDS-DS/CADe-CADx-evaluation.git
   cd CADe-CADx-evaluation
   uv venv
   uv sync

Code Style
----------

* Format code with `black <https://black.readthedocs.io/>`_ before committing.
* Use `NumPy-style docstrings
  <https://numpydoc.readthedocs.io/en/latest/format.html>`_ for all public
  functions and classes.
* All new functions must include type annotations.

Submitting a Pull Request
--------------------------

1. Fork the repository and create a feature branch.
2. Make your changes with appropriate docstrings and type hints.
3. Verify the Sphinx build passes locally:

   .. code-block:: bash

      pip install -r docs/requirements.txt
      sphinx-build docs docs/_build/html

4. Open a pull request against ``main``.

Reporting Issues
----------------

Please open an issue on
`GitHub <https://github.com/EYONIS-AIDS-DS/CADe-CADx-evaluation/issues>`_
with a minimal reproducible example.
