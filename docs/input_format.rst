Input Data Format
=================

All input data is provided as CSV files under the ``data/`` directory.
Two types of CSV are used: *series-level* and *lesion-level*.

Series-level CSV (``data/data_series/series.csv``)
---------------------------------------------------

One row per CT scan (series).

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Column
     - Type
     - Description
   * - ``series_uid``
     - str
     - Unique identifier for the CT series.
   * - ``patient_id``
     - str
     - Unique patient identifier.
   * - ``label``
     - int
     - Ground-truth label: ``1`` = cancer, ``0`` = benign.
   * - ``<set_name>`` (e.g. ``nlst_test_2``, ``nlst_test_3`` …)
     - bool
     - One boolean column per evaluation set.  A row belongs to a given set
       when its value is ``True``.  The column name is used as the
       ``set_name`` argument when calling the evaluation functions.
   * - ``<prediction_name>``
     - float
     - Model output score in [0, 1].  One column per evaluated system.

Lesion-level CSV (``data/data_lesions/lesions.csv``)
------------------------------------------------------

One row per detected or annotated nodule.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Column
     - Type
     - Description
   * - ``detection_id``
     - str
     - Unique identifier for the detection.
   * - ``patient_id``
     - str
     - Patient identifier (used to compute FP/scan in FROC).
   * - ``series_uid``
     - str
     - Corresponding CT series identifier.
   * - ``label``
     - int
     - Ground-truth label: ``1`` = cancer nodule, ``0`` = benign.
   * - ``<set_name>`` (e.g. ``nlst_test_1``, ``nlst_test_6`` …)
     - bool
     - One boolean column per evaluation set.  A row belongs to a given set
       when its value is ``True``.  The column name is used as the
       ``set_name`` argument when calling the evaluation functions.
   * - ``detection_status``
     - str
     - ``"TP"``, ``"FP"``, or ``"FN"`` (used for FROC visualisation).
   * - ``diameter_mm``
     - float
     - Longest diameter of the nodule in mm.
   * - ``<prediction_name>``
     - float
     - Detection/classification score.  One column per evaluated system.

Radiologist Annotation CSV (``data/data_lesions/lesions_radiologists.csv``)
----------------------------------------------------------------------------

Same schema as the lesion-level CSV but with one column per radiologist
containing their binary read (``0`` or ``1``).
