from copy import deepcopy

import pandas as pd
import yaml
from IPython.display import display

from pandora.data import ukb_data_utils, ukb_features
from pandora.training.dataset import PromptDataset, load_prompt_parts

# assets/ukb/ukb_2023_12/ukb675384.csv
# assets/ukb/ukb_2024_06/ukb51068.csv
df = pd.read_csv(
    f"{ukb_data_utils.ASSETS_PATH}/ukb/ukb_2024_02/ukb677731.csv",
    nrows=10,
    low_memory=False,
    index_col=0,
)

codings, data_dict, field2date = ukb_data_utils.load_ukb_meta_files()
raw2clean = ukb_data_utils.get_mapping_raw2clean(
    col_names=df.columns, only_clean_col_name=True, data_dict=data_dict
)

raw2clean_fields = ukb_data_utils.get_mapping_raw2clean(
    col_names=df.columns, only_clean_col_name=False, data_dict=data_dict
)

with open("config/training/train_settings.yaml", "r") as f:
    config = yaml.safe_load(f)


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

num_samples = None
dataset = load_prompt_parts(data_config=config["data"], num_samples=num_samples)

for feature_group in adapted_feature_groups.keys():
    print("*" * 100)
    print(f"Feature group: {feature_group}")
    if feature_group == "all_base_risk_factors":
        for g in all_base_risk_factors:
            features = ukb_features.get_feature_names_from_feature_group(g)
    else:
        features = ukb_features.get_feature_names_from_feature_group(feature_group)
    df_train = dataset.select_columns(features)["train"].to_pandas()
    na_mask = df_train.isna()
    na = na_mask.sum()
    na = na[na > 0]
    if len(na) > 0:
        print(f"Feature group {feature_group} has missing values")
        print(na)

    empty_mask = df_train.applymap(lambda x: x == "")
    empty = empty_mask.sum()

    na_or_empty_mask = na_mask | empty_mask

    fg_summary_df = (na_or_empty_mask.sum() / len(df_train)).to_frame("missing_rate")
    fg_summary_df["feature_name"] = None

    for i, col in enumerate(na_or_empty_mask.columns):
        if col.isdigit():  # field_id
            field_name = data_dict[data_dict["FieldID"] == int(col)].iloc[0]["Field"]
        else:
            field_name = col
        fg_summary_df.loc[col, "feature_name"] = field_name

    display(fg_summary_df)

    # number of rows with at least one non-empty and non-na value
    at_least_x_features = {}
    for x in [1, 2, 3]:
        at_least_x_features[x] = ((~na_or_empty_mask).sum(axis=1) >= x).sum() / len(
            df_train
        )
    print(f"Number of features: {len(df_train.columns)}")
    print(at_least_x_features)
    print(fg_summary_df["feature_name"].values)
