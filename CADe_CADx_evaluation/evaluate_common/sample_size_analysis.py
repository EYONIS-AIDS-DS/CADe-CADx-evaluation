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
