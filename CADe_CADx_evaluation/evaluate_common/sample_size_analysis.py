"""Utilities for reporting dataset size and class imbalance."""
import os
from typing import Dict

import pandas as pd


def database_sample_sizes(
    df: pd.DataFrame,
    data_label: str,
    data_name: str,
    label_name: str,
    expdir_analysis: str,
    set_name: str,
) -> Dict:
    """Compute and save class sizes and imbalance ratio to a CSV file.

    Parameters
    ----------
    df:
        Input DataFrame (already filtered to the evaluation subset).
    data_label:
        Column name of the unique sample identifier
        (e.g. ``"series_uid"`` or ``"detection_id"``).
    data_name:
        Human-readable name for the sample type, used in column headers
        (e.g. ``"series"`` or ``"lesions"``).
    label_name:
        Column name of the binary ground-truth label (0/1).
    expdir_analysis:
        Directory where ``sample_size_<set_name>.csv`` is saved.
    set_name:
        Name of the evaluation subset, appended to the output filename.

    Returns
    -------
    Dict
        Dictionary with total, positive, and negative sample counts plus
        the imbalance ratio (benign : cancer).
    """
    benign_df = df.loc[(df[label_name] == 0)]
    cancer_df = df.loc[(df[label_name] == 1)]

    cancer_data_list = list(cancer_df[data_label].unique())
    benign_data_list = list(benign_df[data_label].unique())
    total_data_list = list(df[data_label].unique())
    imbalance_ratio = (
        str(round(len(benign_data_list) / len(cancer_data_list), 1))
        if len(cancer_data_list) != 0
        else "inf"
    )

    # initialize data of lists.
    data_size = {
        f"Data set {data_name}": [set_name],
        f"Total number of {data_name}": [len(total_data_list)],
        f"Number of cancer {data_name}": [len(cancer_data_list)],
        f"Number of benign {data_name}": [len(benign_data_list)],
        "Imbalance": ["1 : " + imbalance_ratio],
    }

    # Create DataFrame
    df_data = pd.DataFrame(data_size)
    df_data.to_csv(
        os.path.join(
            expdir_analysis,
            "sample_size_" + str(set_name) + ".csv",
        ),
        index=False,
    )

    return data_size
