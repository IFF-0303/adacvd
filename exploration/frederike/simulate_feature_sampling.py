import itertools

import datasets as hf_datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from accelerate import Accelerator
from IPython.display import display
from scripts.condor.training.submit_sweep import (
    create_combinations,
    extract_final_layers,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForTokenClassification

from exploration.frederike.visualization.visualization_utils import (
    find_evaluation_directories,
    flatten_dict,
    setup_plotting,
)
from pandora.data import ukb_features
from pandora.data.prompt import join_prompt_parts
from pandora.data.ukb_data_utils import ASSETS_PATH
from pandora.training.dataset import (
    PromptDataset,
    get_column_names,
    load_prompt_parts,
    sample_columns,
)

feature_config = {
    "feature_groups": [
        "base_risk_score_inputs",
        "additional_risk_score_inputs_aha_acc",
        "additional_risk_score_inputs_prevent",
        "diabetes",
        "polygenic_risk_scores_all",
        "medical_history_all",
        "blood_samples",
        "icd_codes",
        "family_history",
        "physical_activity",
        "sleep",
        "smoking",
        "alcohol",
        "lifestyle_and_environment",
        "physical_measures",
        "sociodemographics",
        "urine_assays",
    ]
}
# sampling_config = {
#     "prob_by_feature_group": 0.9,
#     "prob_keep_base_inputs": 0.9,
#     "prob_no_additional_feature_groups": 0.05,
#     "sampling_share_min": 0.15,
#     "sampling_share_max": 0.25,
#     "sampling_share_within_group_min": 0.5,
#     "sampling_share_within_group_max": 0.6,
# }

data_config = {
    "path": "prompt_parts/ukb_2024_02/prompt_parts.parquet",
    "split": "config/splits/split_2024_06_29_eval_100000.json",
    "subset": "subsets/ukb_2024_02/MACE_ADO_no_previous_target.json",
    "eval_subset": True,
    "shuffle": False,
    "target_name": "MACE_ADO_10y",
    "zeroshot_prompt": False,
}


dataset = load_prompt_parts(data_config=data_config, num_samples=None)

with open("config/ukb_data/feature_groups/meta/feature_groups.yaml") as f:
    feature_groups = yaml.safe_load(f)

with open("config/training/sampling_combinations.yaml") as f:
    sweep_config = yaml.safe_load(f)

sweep_config_sampling = sweep_config["data"]["sampling"]
sweep_combinations = create_combinations(sweep_config_sampling)

cols = get_column_names(feature_config)

# check if all columns are present
for feature_group in feature_groups.keys():
    for field_id in feature_groups[feature_group].get("field_ids", []):
        if str(field_id) not in cols:
            print(f"{feature_group} - {field_id} - Missing")
    for feature in feature_groups[feature_group].get("features", []):
        if feature not in cols:
            print(f"{feature_group} - {feature} - Missing")

print("Number of columns: ", len(cols))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# Create a figure with subplots for each sweep combination
fig, axes = plt.subplots(
    len(sweep_combinations), 3, figsize=(15, 5 * len(sweep_combinations))
)

cpal = setup_plotting(context="paper", n=len(sweep_combinations))

for comb_i, sampling_config in enumerate(sweep_combinations):
    print(sampling_config)
    columns_count = pd.DataFrame(columns=cols)
    for i in tqdm(range(10)):
        sampled_columns = sample_columns(
            cols=cols, feature_config=feature_config, sampling_config=sampling_config
        )
        for col in sampled_columns:
            columns_count.loc[i, col] = 1

    columns_count.fillna(0, inplace=True)

    columns_count_summary = (
        columns_count.sum(axis=0) / columns_count.shape[0]
    ).sort_values(ascending=True)

    columns_count_summary_2 = columns_count.sum(axis=1)

    n_cols = len(cols)
    print("Total number of columns: ", n_cols)
    print(
        "Share of columns never sampled: ", (columns_count_summary == 0).sum() / n_cols
    )
    print(
        "Share of columns sampled in all iterations: ",
        (columns_count_summary == 1).sum() / n_cols,
    )
    print(
        "Share of columns sampled in at least one iteration: ",
        (columns_count_summary > 0).sum() / n_cols,
    )
    print(
        "Share of columns sampled in at least 50% of iterations: ",
        (columns_count_summary > 0.5).sum() / n_cols,
    )
    print(
        "Average share of iterations a column was sampled: ",
        columns_count_summary.mean() / n_cols,
    )

    print(
        "Share of iterations with at least one column sampled: ",
        (columns_count_summary_2 > 0).sum() / columns_count_summary_2.shape[0],
    )

    print(
        "Average number of columns sampled in an iteration: ",
        columns_count_summary_2.mean(),
    )

    print(
        "Max number of columns sampled in an iteration: ",
        columns_count_summary_2.max(),
    )

    print(
        "Min number of columns sampled in an iteration: ",
        columns_count_summary_2.min(),
    )

    color = cpal[comb_i]

    # Convert sampling_config to a readable string
    config_str = ", ".join([f"{k}={v}" for k, v in sampling_config.items()])

    # Histogram of number of iterations a column was sampled
    sns.histplot(
        columns_count_summary * columns_count.shape[0], color=color, ax=axes[comb_i, 0]
    )
    axes[comb_i, 0].set_xticks(np.arange(0, columns_count.shape[0] + 1, 20))
    axes[comb_i, 0].set_title(
        f"{config_str}\nNumber of iterations a column was sampled"
    )

    # Add configuration as text on the plot
    # axes[comb_i, 0].text(
    #     0.05,
    #     0.95,
    #     config_str,
    #     transform=axes[comb_i, 0].transAxes,
    #     verticalalignment="top",
    #     fontsize=10,
    #     bbox=dict(facecolor="white", alpha=0.7),
    # )

    # Histogram of number of columns sampled in an iteration
    sns.histplot(columns_count_summary_2, color=color, ax=axes[comb_i, 1])
    axes[comb_i, 1].set_title("Number of columns sampled in an iteration")
    axes[comb_i, 1].set_xticks(np.arange(0, n_cols + 1, 20))

    # Histogram of share of columns sampled in an iteration
    sns.histplot(columns_count_summary_2 / n_cols, color=color, ax=axes[comb_i, 2])
    axes[comb_i, 2].set_title("Share of columns sampled in an iteration")
    axes[comb_i, 2].set_xlim(0, 1)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
