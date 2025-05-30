from copy import deepcopy

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

import pandora.utils.logger
from exploration.frederike.visualization.visualization_utils import setup_plotting
from pandora.data import ukb_data_utils, ukb_features
from pandora.data.prompt import join_prompt_parts
from pandora.training.dataset import (
    PromptDataset,
    get_column_names,
    get_in_text_tokenization,
    load_prompt_parts,
)

# %cd '/home/fluebeck/biobank/biobank-llm'


setup_plotting(context="talk")
df = pd.read_csv(
    f"{ukb_data_utils.ASSETS_PATH}/ukb/ukb_2024_02/ukb677731.csv",
    nrows=10,
    low_memory=False,
)

prompt_parts = pd.read_parquet(
    ukb_data_utils.ASSETS_PATH / "prompt_parts/ukb_2024_02/prompt_parts.parquet"
)

codings, data_dict, field2date = ukb_data_utils.load_ukb_meta_files()
raw2clean = ukb_data_utils.get_mapping_raw2clean(
    col_names=df.columns, only_clean_col_name=True, data_dict=data_dict
)

raw2clean_fields = ukb_data_utils.get_mapping_raw2clean(
    col_names=df.columns, only_clean_col_name=False, data_dict=data_dict
)

with open("config/ukb_data/feature_groups/meta/feature_groups.yaml", "r") as f:
    fg = yaml.safe_load(f)


adapted_feature_groups = deepcopy(fg)

# merge ["base_risk_score_inputs", "additional_risk_score_inputs_aha_acc", "additional_risk_score_inputs_prevent"] to one group
adapted_feature_groups["all_base_risk_factors"] = {
    "field_ids": [],
    "features": [],
}

all_base_risk_factors = [
    "base_risk_score_inputs",
    "additional_risk_score_inputs_aha_acc",
    "additional_risk_score_inputs_prevent",
]

for feature_group in all_base_risk_factors:
    adapted_feature_groups["all_base_risk_factors"]["field_ids"] += fg[
        feature_group
    ].get("field_ids", [])
    adapted_feature_groups["all_base_risk_factors"]["features"] += fg[
        feature_group
    ].get("features", [])

# %%

fg_info_dict = {}

data_config = {
    "path": "prompt_parts/ukb_2024_02/prompt_parts.parquet",
    "split": "config/splits/split_2025_02_17_test_100000.json",
    "subset": "subsets/ukb_2024_02/MACE_ADO_no_previous_target.json",
    "eval_subset": True,
    "target_name": "MACE_ADO_10y",
    "zeroshot_prompt": False,
}

summary_token_lenghts = pd.DataFrame()
tmp_df = {}

model_name = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

dataset = load_prompt_parts(data_config=data_config)["train"]

for feature_group in tqdm(adapted_feature_groups):
    data_config["feature_config"] = {"feature_groups": [feature_group]}

    if feature_group == "all_base_risk_factors":
        prompt_part_names = get_column_names({"feature_groups": all_base_risk_factors})
    else:
        prompt_part_names = get_column_names({"feature_groups": [feature_group]})

    prompt_parts = dataset.select_columns(prompt_part_names)
    prompts = prompt_parts.map(
        lambda x: {"prompt": join_prompt_parts([x[col] for col in prompt_part_names])},
        remove_columns=prompt_part_names,
        # batched=True,
    )
    prompts = prompts.map(lambda x: {"num_prompt_parts": len(x["prompt"].split("\n"))})
    tokenized_prompts = prompts.map(
        lambda x: tokenizer(x["prompt"]),
        batched=True,
    )
    tokenized_prompts = tokenized_prompts.map(
        lambda x: {"length": len(x["input_ids"])},
    )

    tmp_dict = {}

    # missingness

    missing_condition = lambda x: x != x or x == ""
    missing_values = prompt_parts.to_pandas().applymap(missing_condition)
    missing_values_per_col = missing_values.sum()
    tmp_dict["missing_values_per_feature"] = (
        missing_values_per_col / len(prompt_parts)
    ).to_dict()

    # more than x% complete values
    for percent in [0.9, 0.95]:
        n_ub = len(prompt_parts) * (1 - percent)
        missing_values_per_col[missing_values_per_col <= n_ub]
        tmp_dict[f"num_features_complete_{str(percent).split('.')[-1]}"] = len(
            missing_values_per_col[missing_values_per_col <= n_ub]
        ) / len(missing_values_per_col)

    missing_values_per_row = missing_values.sum(axis=1)

    tmp_dict["num_features_per_participant_mean"] = (
        len(prompt_part_names) - missing_values_per_row
    ).mean()
    tmp_dict["num_features_per_participant_median"] = (
        len(prompt_part_names) - missing_values_per_row
    ).median()
    tmp_dict["num_features_per_participant_max"] = (
        len(prompt_part_names) - missing_values_per_row
    ).max()
    tmp_dict["num_features_per_participant_min"] = (
        len(prompt_part_names) - missing_values_per_row
    ).min()

    tmp_dict["num_tokens_mean"] = np.mean(tokenized_prompts["length"])
    tmp_dict["num_tokens_median"] = np.median(tokenized_prompts["length"])
    tmp_dict["num_tokens_max"] = np.max(tokenized_prompts["length"])
    tmp_dict["num_tokens_min"] = np.min(tokenized_prompts["length"])

    fg_info_dict[feature_group] = deepcopy(tmp_dict)


fg_info = (
    pd.DataFrame(fg_info_dict)
    .T.reset_index()
    .rename(columns={"index": "feature_group"})
)

fg_info.to_csv("feature_group_info.csv", index=False)

# %%

for col in fg_info.columns[2:]:
    plt.figure(figsize=(4, 6))
    sns.barplot(data=fg_info, y="feature_group", x=col)
    plt.xticks(rotation=90)
    plt.title(col)
    plt.show()

# %%

fg_info = pd.read_csv("feature_group_info.csv")

from exploration.frederike.visualization import name_mapping

name_mapping.feature_group_names

groups = [
    "all_base_risk_factors",
    "urine_assays",
    "sociodemographics",
    "physical_measures",
    "family_history",
    "blood_samples",
    "lifestyle_and_environment",
    "polygenic_risk_scores_all",
    "icd_codes",
    "medical_history_all",
]

group_to_name = {"all_base_risk_factos": "Base Risk Factors"}

for g in groups:
    if g in name_mapping.feature_group_names:
        group_to_name[g] = name_mapping.feature_group_names[g]["long_name"]


plt.figure(figsize=(8, 10))
