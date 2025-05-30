import logging
import math
import random
from numbers import Number
from typing import Any, Dict, List, Optional

import datasets as hf_datasets
import numpy as np
import pandas as pd
import torch
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForTokenClassification

from pandora.data import ukb_features, ukb_field_ids
from pandora.data.prompt import join_prompt_parts
from pandora.data.ukb_data_utils import ASSETS_PATH
from pandora.data.ukb_features import sort_features_by_feature_group


class PromptDataset(torch.utils.data.Dataset):

    def __init__(
        self, dataset, tokenizer, data_config, fix_seed=False, mask_labels=True
    ):

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.feature_config = data_config["feature_config"]
        self.prompt_part_names = get_column_names(self.feature_config)
        self.shuffle = data_config.get("shuffle", False)
        self.sampling = data_config.get("sampling", None)
        for col in self.prompt_part_names:
            if col not in dataset.column_names:
                raise ValueError(f"Column {col} not found in dataset.")
        self.zeroshot_prompt = data_config.get("zeroshot_prompt", False)
        self.text_generation_prompt = data_config.get("text_generation_prompt", False)
        self.fix_seed = fix_seed
        if fix_seed:
            self.random_states = np.random.randint(2**32 - 1, size=len(dataset))
        self.mask_labels = mask_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        item = self.dataset[idx]

        if self.sampling is not None:
            if self.fix_seed:
                original_state = random.getstate()
                original_state_np = np.random.get_state()
                random.seed(self.random_states[idx])
                np.random.seed(self.random_states[idx])

            selected_columns = sample_columns(
                cols=self.prompt_part_names,
                feature_config=self.feature_config,
                sampling_config=self.sampling,
            )

            selected_columns = sort_features_by_feature_group(
                selected_columns, remove_duplicates=True
            )

            if self.fix_seed:
                random.setstate(original_state)
                np.random.set_state(original_state_np)

        else:
            selected_columns = self.prompt_part_names

        base_prompt = join_prompt_parts(
            prompt_parts=[item[col] for col in selected_columns], shuffle=self.shuffle
        )

        if self.zeroshot_prompt:
            task = "Based on the provided patient description, what is the estimated 10-year risk of cardiovascular disease (CVD)? Please provide your answer solely as a numeric percentage in a machine-readable JSON format."
            prompt = f"Patient description:\n\n{base_prompt}.\n\n{task}\n\n"

        elif self.text_generation_prompt:
            prompt_start = "Here is a description of a patient:"

            prompt_end = "Based on this information, generate a brief summary of the patient with an emphasis on relevant cardiovascular-related information. Do not provide risk evaluation or any clinical judgment."

            prompt = f"{prompt_start}\n\n{base_prompt}.\n\n{prompt_end}\n\n"

        else:
            task = "Will this patient have a major adverse cardiovascular event in the next 10 years? Reply with Yes or No."
            prompt = f"Patient description:\n\n{base_prompt}.\n\n{task}\n\n"

        completion_mapping = {"Positive": "Yes", "Negative": "No"}
        if self.zeroshot_prompt:
            tokenized_output = self.tokenizer(
                prompt,
                add_special_tokens=False,
                return_tensors="pt",
            )
            tokenized = {
                "input_ids": tokenized_output["input_ids"].squeeze(),
                "attention_mask": tokenized_output["attention_mask"].squeeze(),
                "completion": {"Positive": 1, "Negative": 0}[item["completion"]],
            }
        elif self.text_generation_prompt:
            tokenized_input = self.tokenizer(
                prompt,
                add_special_tokens=False,
                return_tensors="pt",
            )
            tokenized = {
                "input_ids": tokenized_input["input_ids"].squeeze(),
                "attention_mask": tokenized_input["attention_mask"].squeeze(),
                "completion": {"Positive": 1, "Negative": 0}[item["completion"]],
            }
        else:
            tokenized = format_to_chat_template(
                {
                    "prompt": prompt,
                    "completion": completion_mapping[item["completion"]],
                },
                self.tokenizer,
                mask_labels=self.mask_labels,
            )

            # if the sequence is too long, then cut some parts of the prompt. This is a workaround for the maximum token limit of the model, without cropping the question and the completion
            # Note: in all relevant runs for the manuscript, this never happened (only needed for debugging on smaller GPUs)
            MAX_LENGHT = 6000
            logged = False
            while len(tokenized["input_ids"]) > MAX_LENGHT:
                if not logged:
                    logging.warning(
                        f"Prompt too long, cropping prompt. Length: {len(tokenized['input_ids'])}"
                    )
                    logged = True

                feature_start = prompt.lfind("Patient description:\n\n") + len(
                    "Patient description:\n\n"
                )
                task_start = prompt.rfind(f"\n{task}")
                prompt_parts = prompt[:task_start].split("\n")
                factor = min(MAX_LENGHT / len(tokenized["input_ids"]), 0.9)
                n = math.floor(len(prompt_parts) * factor)
                prompt_parts_cropped = prompt_parts[:n]
                prompt_cropped = "\n".join(prompt_parts_cropped)
                prompt_cropped = f"{prompt_cropped}.\n{task}"
                tokenized = format_to_chat_template(
                    {
                        "prompt": prompt_cropped,
                        "completion": completion_mapping[item["completion"]],
                    },
                    self.tokenizer,
                )

        return {"eid": item["eid"], **tokenized}


def load_split(data_config: dict):
    with open(data_config["split"], "r") as f:
        split = yaml.load(f, Loader=yaml.CLoader)

    # load additional subset information for training if specified (e.g., filter out patients with previous CVD)
    if data_config.get("subset") is not None:
        with open(ASSETS_PATH / data_config["subset"], "r") as f:
            subset = yaml.load(f, Loader=yaml.CLoader)
            n_train_before = len(split["train"])
            split["train"] = list(set(split["train"]) & set(subset))
            logging.info(
                f"Reduced training set from {n_train_before} to {len(split['train'])}."
            )
            # if specified, apply the subset to the test & validation sets as well
            if data_config.get("eval_subset", False):
                for eval_mode in ["test", "validation"]:
                    n_before = len(split[eval_mode])
                    split[eval_mode] = list(set(split[eval_mode]) & set(subset))
                    logging.info(
                        f"Reduced {eval_mode} set from {n_before} to {len(split[eval_mode])}."
                    )

    if data_config.get("subgroup") is not None:
        with open(ASSETS_PATH / data_config["subgroup"], "r") as f:
            subgroup = yaml.load(f, Loader=yaml.CLoader)
            if data_config.get("add_from_other_subgroups_n") is not None:
                other_subgroups = list(set(split["train"]) - set(subgroup))
                n_add = data_config["add_from_other_subgroups_n"]
                subgroup += random.sample(
                    other_subgroups, min(n_add, len(other_subgroups))
                )
                logging.info(
                    f"Added {min(n_add, len(other_subgroups))} samples from other subgroups."
                )

            n_train_before = len(split["train"])
            split["train"] = list(set(split["train"]) & set(subgroup))
            logging.info(
                f"Reduced training set from {n_train_before} to {len(split['train'])}."
            )
    return split


def load_prompt_parts(
    data_config: dict, num_samples: Optional[int] = None
) -> hf_datasets.Dataset:
    """Load the dataset from a parquet file. Select a subset of rows if specified in data config. Split into train, validation and test sets according to a pre-defined split. Oversample the positive class in the training set if specified in the data_config."""

    split = load_split(data_config)

    dataset_path = ASSETS_PATH / data_config["path"]

    # prompt parts dataset in parquet format, with columns: eid, target, and prompt parts per feature
    _dataset = hf_datasets.load_dataset(
        "parquet", data_files=str(dataset_path), split="train"
    )

    # bring target column into the format of the prompt completion
    if data_config["target_name"] in _dataset.column_names:
        _dataset = _dataset.add_column(
            "completion",
            pd.Series(_dataset[data_config["target_name"]]).map(
                lambda x: {True: "Positive", False: "Negative"}[x]
            ),
        )
    else:
        raise ValueError(
            f"Target column {data_config['target_name']} not found in prompt parts dataset."
        )

    test_mask = pd.Series(_dataset["eid"]).isin(split["test"])
    train_mask = pd.Series(_dataset["eid"]).isin(split["train"])
    validation_mask = pd.Series(_dataset["eid"]).isin(split["validation"])
    test_ids = test_mask[test_mask & ~train_mask & ~validation_mask].index.to_list()
    train_ids = test_mask[~test_mask & train_mask & ~validation_mask].index.to_list()
    validation_ids = test_mask[
        ~test_mask & ~train_mask & validation_mask
    ].index.to_list()

    if data_config.get("num_training_samples", None) is not None:
        # train only on a subset of the training set
        train_ids = random.sample(
            train_ids, min(data_config["num_training_samples"], len(train_ids))
        )
        logging.info(f"Subsampled training set to {len(train_ids)} samples.")

    if num_samples is not None:
        # for debugging purposes only
        train_ids = random.sample(train_ids, min(num_samples, len(train_ids)))
        test_ids = random.sample(test_ids, min(num_samples, len(test_ids)))
        validation_ids = random.sample(
            validation_ids, min(num_samples, len(validation_ids))
        )

    dataset = hf_datasets.DatasetDict(
        {
            "train": _dataset.select(train_ids),
            "test": _dataset.select(test_ids),
            "validation": _dataset.select(validation_ids),
        }
    )
    assert len(list(set(dataset["train"]["eid"]) & set(dataset["test"]["eid"]))) == 0
    assert (
        len(list(set(dataset["train"]["eid"]) & set(dataset["validation"]["eid"]))) == 0
    )

    # oversample the positive class in the training set if specified
    oversampled_pos_fraction = data_config.get("oversampled_pos_fraction", 0.0)
    if oversampled_pos_fraction > 0.0:
        completions = dataset["train"]["completion"]
        initial_fractions = {
            x: completions.count(x) / len(completions) for x in set(completions)
        }

        if "Positive" in initial_fractions:
            weights = [
                (
                    oversampled_pos_fraction / initial_fractions["Positive"]
                    if x == "Positive"
                    else (1 - oversampled_pos_fraction)
                    / (1 - initial_fractions["Positive"])
                )
                for x in completions
            ]
            random.seed(0)
            idx = random.choices(
                range(len(completions)), weights=weights, k=len(completions)
            )
            dataset["train"] = dataset["train"].select(idx)
        else:
            logging.warning(
                "Cannot oversample the positive class. No positive samples in the training set."
            )
    train_eids = dataset["train"]["eid"]
    test_eids = dataset["test"]["eid"]
    validation_eids = dataset["validation"]["eid"]
    assert len(set(train_eids) & set(test_eids)) == 0
    assert len(set(train_eids) & set(validation_eids)) == 0
    logging.info(
        f"Class shares (train): {pd.DataFrame(dataset['train']['completion']).value_counts(normalize=True)}"
    )
    logging.info(
        f"Class shares (test): {pd.DataFrame(dataset['test']['completion']).value_counts(normalize=True)}"
    )
    logging.info(
        f"Class shares (validation): {pd.DataFrame(dataset['validation']['completion']).value_counts(normalize=True)}"
    )
    return dataset


def get_in_text_tokenization(tokenizer, completion: str) -> Number:
    """
    Tokenizers have problems with whitespace, so this tries to handle the recognition of the answers
    e.g. gpt2 encodes "Yes" and " Yes" differently. This function returns the token id of the completion token
    when it is preceded by text.

    Arguments:
        tokenizer: the tokenizer used to encode the text
        completion: the completion to search for
    """
    # follow same approach as in preprocess_chat_template to make sure we get the same token id for the completion
    text = apply_chat_template(
        tokenizer=tokenizer,
        chat_list=[
            {"role": "user", "content": "Dummy content"},
            {"role": "assistant", "content": completion},
        ],
    )
    encoded_text = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    token_idx = get_completion_token_idx(encoded_text, text, completion, tokenizer)
    return encoded_text["input_ids"][0][token_idx]


def get_completion_token_idx(
    encoded_text: torch.Tensor, text: str, completion: str, tokenizer=None
):
    """
    Extract the single index corresponding to the completion token in the text, searching from the back.
    Make sure to encode the full prompt with completion because without it may result in a different
    token id of the completion token.

    Arguments:
        encoded_text: encoded text with input_ids of shape [1, seq_len]
        text: the original text
        completion: the completion to search for
        tokenizer: the tokenizer used to encode the text, if None, the function checks that the decoded token matches the completion
    """

    completion_start_char_idx = text.rfind(completion)
    assert (
        completion_start_char_idx != -1
    ), f"Completion not found: {completion} not in {text}"

    completion_tokens = tokenizer.encode(
        text[completion_start_char_idx:], add_special_tokens=False
    )

    # check if the token changes if we include more text, and stop if another token is added
    for i in range(10):  # 10 is a safe margin for anything added by the chat template
        completion_tokens_longer = tokenizer.encode(
            text[completion_start_char_idx - i :], add_special_tokens=False
        )
        if len(completion_tokens_longer) > len(completion_tokens):
            before_completion = tokenizer.encode(
                text[completion_start_char_idx - i : completion_start_char_idx],
                add_special_tokens=False,
            )
            if len(before_completion) < len(completion_tokens_longer):
                completion_tokens = completion_tokens_longer[len(before_completion) :]
                break
        else:
            completion_tokens = completion_tokens_longer

    completion_token = completion_tokens[0]

    assert (
        completion_token in encoded_text["input_ids"][0, -5:].tolist()
    ), f"Completion token not found in end of encoded text. Completion: {completion}. Text: {text}. Encoded text: {encoded_text['input_ids'][0, -5:]}. Encoded completion: {completion_token}"

    # find the position of the completion token
    token_idx = (
        (encoded_text["input_ids"].squeeze() == completion_token).nonzero()[-1].item()
    )
    id_token = encoded_text["input_ids"].squeeze()[token_idx]
    assert tokenizer is None or (
        completion
        == tokenizer.decode(id_token, clean_up_tokenization_spaces=True).strip()
    ), f"""Completion may map to more than one word: expected '{completion}' vs '{tokenizer.decode(id_token, clean_up_tokenization_spaces=True)}'"""
    assert (
        token_idx >= len(encoded_text["input_ids"].squeeze()) - 10
    ), "May have misrecognized answer token, not towards the end"  # 10 should be a safe margin for anything added by the chat template
    return token_idx


def apply_chat_template(tokenizer: AutoTokenizer, chat_list: List[Dict[str, str]]):
    """Apply chat template if it exists, otherwise concatenate the chat content."""
    if False:  # tokenizer.chat_template is not None:
        chat = tokenizer.apply_chat_template(chat_list, tokenize=False)
    else:
        chat = "".join([x["content"] for x in chat_list])
        chat += ". "
    return chat


def format_to_chat_template(
    item: Dict[str, Any], tokenizer: AutoTokenizer, mask_labels: bool = True
) -> dict:
    """
    Format a single item to the chat template and tokenize it.
    """

    chat_list = [
        {"role": "user", "content": item["prompt"]},
        {
            "role": "assistant",
            "content": item["completion"],
        },
    ]

    chat = apply_chat_template(tokenizer=tokenizer, chat_list=chat_list)
    tokenized_chat = tokenizer(chat, add_special_tokens=False, return_tensors="pt")

    # retrieve the completion index
    completion_idx = get_completion_token_idx(
        tokenized_chat,
        chat,
        item["completion"],
        tokenizer,
    )

    model_input = dict()
    model_input["input_ids"] = tokenized_chat["input_ids"].squeeze()
    model_input["attention_mask"] = tokenized_chat["attention_mask"].squeeze()

    # labels will be shifted to the left by 1 in the model/evaluation, so not done here
    label = tokenized_chat["input_ids"].squeeze().data.clone()

    if mask_labels:
        masked_label = torch.full_like(label, -100)
        masked_label[completion_idx] = label[completion_idx]
    else:
        masked_label = label

    model_input["labels"] = masked_label

    return model_input


def get_column_names(data_config: dict) -> List[str]:
    """
    Get all column names from the data config. This includes field_ids, features, and feature groups.
    """
    feature_field_ids = data_config.get("feature_field_ids", [])
    features = data_config.get("features", [])
    feature_groups = data_config.get("feature_groups", [])

    feature_names = feature_field_ids + features
    for feature_group in feature_groups:
        feature_names += ukb_features.get_feature_names_from_feature_group(
            feature_group
        )

    cols = [col for col in feature_names if col not in ["eid", "target"]]
    return cols


def sample_columns(
    cols: List[str], feature_config: dict, sampling_config: dict
) -> List[str]:

    cols = list(dict.fromkeys(cols))  # remove duplicates
    sampling = True

    # whether to sample by feature group
    prob_by_feature_group = sampling_config.get("prob_by_feature_group", 1.0)
    by_feature_group = random.random() < prob_by_feature_group

    # whether to keep base inputs
    prob_keep_base_inputs = sampling_config.get("prob_keep_base_inputs", 1.0)
    keep_base_inputs = random.random() < prob_keep_base_inputs

    # no additional feature groups beyond the base inputs
    if keep_base_inputs:
        prob_no_additional_feature_groups = sampling_config.get(
            "prob_no_additional_feature_groups", 0.01
        )
        if random.random() < prob_no_additional_feature_groups:
            sampling = False

    # group sampling share
    sampling_share_min = sampling_config.get("sampling_share_min", 0.5)
    sampling_share_max = sampling_config.get("sampling_share_max", 0.5)
    sampling_share = random.uniform(sampling_share_min, sampling_share_max)

    # within group sampling share
    sampling_share_within_group_min = sampling_config.get(
        "sampling_share_within_group_min", 1.0
    )
    sampling_share_within_group_max = sampling_config.get(
        "sampling_share_within_group_max", 1.0
    )
    sampling_share_within_group = random.uniform(
        sampling_share_within_group_min, sampling_share_within_group_max
    )

    if sampling:
        if not by_feature_group:
            sampled_cols = random.sample(
                cols, int(sampling_share * sampling_share_within_group * len(cols))
            )
        else:
            # randomly sample feature groups according to sampling share
            feature_groups = feature_config.get("feature_groups", [])

            # remove base risk score inputs
            feature_groups = [
                feature_group
                for feature_group in feature_groups
                if feature_group
                not in [
                    "base_risk_score_inputs",
                    "additional_risk_score_inputs_aha_acc",
                    "additional_risk_score_inputs_prevent",
                ]
            ]

            sampled_feature_groups = np.random.choice(
                feature_groups,
                max(1, math.ceil(sampling_share * len(feature_groups))),
                replace=False,
            )

            if len(sampled_feature_groups) == 1:
                sampling_share_within_group = None  # keep all columns within the group if only one group is sampled

            sampled_cols = []
            for feature_group in sampled_feature_groups:
                feature_group_cols = ukb_features.get_feature_names_from_feature_group(
                    feature_group
                )
                if len(list(set(cols) & set(feature_group_cols))) == 0:
                    logging.warning(
                        f"Feature group {feature_group} has no columns in the dataset."
                    )
                if sampling_share_within_group is not None:
                    sampled_cols += random.sample(
                        list(set(cols) & set(feature_group_cols)),
                        math.ceil(
                            sampling_share_within_group * len(feature_group_cols)
                        ),
                    )
                else:
                    sampled_cols += list(set(cols) & set(feature_group_cols))
    else:  # no additional feature groups
        sampled_cols = []

    if keep_base_inputs:
        base_inputs_feature_groups = [
            "base_risk_score_inputs",
        ]
        p_add = sampling_config.get("prob_additional_base_inputs", 0.8)
        if random.random() < p_add:
            base_inputs_feature_groups += [
                "additional_risk_score_inputs_aha_acc",
            ]
        if random.random() < p_add:
            base_inputs_feature_groups += [
                "additional_risk_score_inputs_prevent",
            ]
        base_inputs = []
        for feature_group in base_inputs_feature_groups:
            base_inputs += ukb_features.get_feature_names_from_feature_group(
                feature_group
            )
        sampled_cols = list((set(sampled_cols) | set(base_inputs)) & set(cols))

    # reorder to original order
    sampled_cols = [col for col in cols if col in sampled_cols]

    return sampled_cols


class FraminghamPromptDataset(torch.utils.data.Dataset):
    """
    Dataset for creating prompts for participants in the Framingham Heart Study.
    """

    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        base_prompt = item["prompt"]

        task = "Will this patient have a major adverse cardiovascular event in the next 10 years? Reply with Yes or No."
        prompt = f"Patient description:\n\n{base_prompt}.\n\n{task}\n\n"

        completion_mapping = {"Positive": "Yes", "Negative": "No"}
        tokenized = format_to_chat_template(
            {
                "prompt": prompt,
                "completion": completion_mapping[item["completion"]],
            },
            self.tokenizer,
        )
        return {"eid": item["RANDID"], **tokenized}


class TextDataset(torch.utils.data.Dataset):
    """
    Dataset for creating prompts for free-text patient descriptions.
    """

    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        base_prompt = item["prompt"]

        task = "Will this patient have a major adverse cardiovascular event in the next 10 years? Reply with Yes or No."
        prompt = f"Patient description:\n\n{base_prompt}.\n\n{task}\n\n"

        completion_mapping = {1: "Yes", 0: "No"}
        tokenized = format_to_chat_template(
            {
                "prompt": prompt,
                "completion": completion_mapping[item["completion"]],
            },
            self.tokenizer,
        )
        return {"eid": item["eid"], **tokenized}
