import numpy as np
import pandas as pd
import torch
import yaml
from accelerate import Accelerator
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
)

from pandora.data import ukb_features
from pandora.data.prompt import join_prompt_parts
from pandora.data.ukb_data_utils import ASSETS_PATH
from pandora.training.dataset import (
    PromptDataset,
    format_to_chat_template,
    get_column_names,
    load_prompt_parts,
)

with open("config/training/train_settings.yaml", "r") as f:
    config = yaml.safe_load(f)

tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2", padding_side="left"
)
tokenizer.pad_token_id = tokenizer.eos_token_id

data_collator = DataCollatorForTokenClassification(tokenizer, padding=True)


# train_dataset: config as specified in train_settings
# eval_dataset: config as specified in train_settings
# additional eval_datasets: pre-specified (base, all, no shuffling, etc.)
# what differs is only the data_config passed to the dataset
# even the raw dataset is the same, just split


# load raw dataset: done
# split into train, eval, test: done
# oversampling: done
# create PromptDatasets
# create DataLoaders


# _dataset = prepare_prompt_dataset_from_parts(
#     dataset=_dataset, data_config=data_config["data_config"]
# )


dataset = load_prompt_parts(data_config=config["data"], num_samples=None)

train_dataset = PromptDataset(
    dataset=dataset["train"],
    tokenizer=tokenizer,
    data_config=config["data"]["data_config"],
)
eval_dataset = PromptDataset(
    dataset=dataset["eval"],
    tokenizer=tokenizer,
    data_config=config["data"]["data_config"],
)
test_dataset = PromptDataset(
    dataset=dataset["test"],
    tokenizer=tokenizer,
    data_config=config["data"]["data_config"],
)

train_dataloader = DataLoader(
    train_dataset, collate_fn=data_collator, batch_size=32, shuffle=False
)
eval_dataloader = DataLoader(
    eval_dataset, collate_fn=data_collator, batch_size=8, shuffle=False
)
test_dataloader = DataLoader(
    test_dataset, collate_fn=data_collator, batch_size=8, shuffle=False
)

# base config: base feature set (risk score inputs)
data_config_base = config["data"]["data_config"].copy()
data_config_base["feature_groups"] = [
    "base_risk_score_inputs",
    "additional_risk_score_inputs_aha_acc",
    "additional_risk_score_inputs_prevent",
]

# full config: full feature set (risk score inputs)
data_config_full = config["data"]["data_config"].copy()
data_config_full["feature_groups"] = [
    "base_risk_score_inputs",
    "additional_risk_score_inputs_aha_acc",
    "additional_risk_score_inputs_prevent",
    "polygenic_risk_scores_subset",
    "family_history",
    "diabetes",
    "medical_history_all",
    # TODO: list is not complete yet
]

# additional evaluation datasets
additional_eval_dataset_base = PromptDataset(
    dataset=dataset["eval"], tokenizer=tokenizer, data_config=data_config_base
)
additional_eval_dataset_full = PromptDataset(
    dataset=dataset["eval"], tokenizer=tokenizer, data_config=data_config_full
)

additional_eval_dataloader_base = DataLoader(
    additional_eval_dataset_base, collate_fn=data_collator, batch_size=8, shuffle=False
)
additional_eval_dataloader_full = DataLoader(
    additional_eval_dataset_full, collate_fn=data_collator, batch_size=8, shuffle=False
)


# we only need to keep the dataloaders in the training script
for batch in train_dataloader:
    print(batch["input_ids"].shape)
    print(batch["attention_mask"])
    print(batch["labels"])
    print(batch["eid"])
    prompts = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
    for prompt in prompts:
        print(prompt)
    break
