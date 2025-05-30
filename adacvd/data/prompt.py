import random
from typing import List

import pandas as pd

from adacvd.data import ukb_data_utils, ukb_features


def to_prompt(feature: pd.Series) -> pd.Series:
    """Convert processed UKB feature values to prompt."""
    col = feature.name
    if col.isdigit():  # field_id
        field_id = int(col)
        prompt = to_prompt_by_field_id(feature, field_id)

    elif col.split("-")[0].isdigit():
        field_id = int(col.split("-")[0])
        prompt = to_prompt_by_field_id(feature, field_id)

    elif col in ukb_features.FEATURES.keys():
        na_mask = feature.isna()
        prompt = ukb_features.FEATURES[col]["prompt"] + ": "
        prompt += ukb_data_utils.round_on_string(feature.astype(str), decimals=2)
        replace_dict = {"False": "No", "True": "Yes"}
        for key, value in replace_dict.items():
            prompt = prompt.str.replace(key, value)
        prompt[na_mask] = ""

    else:
        raise ValueError(
            f"Feature {col} not found in features or data dictionary. "
            "Please check the feature name and ensure it is valid."
        )
    return prompt


def write_prompts(features: pd.DataFrame) -> pd.DataFrame:
    """Write prompts from processed UKB features."""
    prompt_parts = pd.DataFrame(index=features.index)
    for col in features.columns:
        prompt_parts = pd.concat([prompt_parts, to_prompt(features[col])], axis=1)

    prompts = join_prompt_parts_df(prompt_parts, features.columns)

    return prompts, prompt_parts


def join_prompt_parts_df(prompt_parts, features, shuffle=False):
    prompt_parts = (
        prompt_parts[features]
        .apply(
            lambda x: ";\n ".join(x.values.astype("str")[x.values.astype("str") != ""]),
            axis=1,
        )
        .to_frame("prompt")
    )
    return prompt_parts


def join_prompt_parts(prompt_parts: List[str], shuffle=False):
    """Glue prompt parts together and optionally shuffle them. Empty prompt parts are removed."""
    prompt_parts = [x for x in prompt_parts if x != ""]
    if shuffle:
        random.shuffle(prompt_parts)
    return ";\n".join(prompt_parts)


def to_prompt_by_field_id(
    feature: pd.Series, field_id: int, remove_unit: bool = False
) -> pd.DataFrame:
    """Convert processed UKB feature values to prompt based on the data dictionary."""
    data_field = ukb_data_utils.get_data_field(field_id)
    prompt = f"{data_field['Field']}: "

    # Specify certain names that should be replaced
    replace_dict = {
        "Age when attended assessment centre": "Age",
    }

    for key, value in replace_dict.items():
        prompt = prompt.replace(key, value)

    values = feature.copy()
    na_mask = feature.isna()

    first_valid_index = values.first_valid_index()  # Get the first non-NaN index
    first_valid_element = (
        values[first_valid_index] if first_valid_index is not None else None
    )
    list_or_set = isinstance(first_valid_element, (set, list))

    if list_or_set:
        values = values.apply(
            lambda x: ", ".join(x) if isinstance(x, (set, list)) else x
        )

    values = values.astype(str)
    values = ukb_data_utils.round_on_string(values, decimals=1)
    replace_dict = {
        ".0": "",
    }

    for key, value in replace_dict.items():
        values = values.str.replace(key, value)

    prompt += values

    if not ukb_data_utils.isNaN(data_field["Units"]) and not remove_unit:
        prompt += f" {data_field['Units']}"

    prompt[na_mask] = ""

    return prompt
