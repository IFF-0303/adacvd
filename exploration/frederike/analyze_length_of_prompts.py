# %cd '/home/fluebeck/biobank/biobank-llm'


import logging

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
from pandora.data import ukb_features
from pandora.data.prompt import join_prompt_parts
from pandora.training.dataset import (
    PromptDataset,
    get_column_names,
    get_in_text_tokenization,
    load_prompt_parts,
)

logger = logging.getLogger()

config_path = "config/training/train_settings.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

set_seed(0)

# load the pre-trained model and tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

dataset = load_prompt_parts(data_config=config["data"])

data_collator = DataCollatorForTokenClassification(tokenizer, padding=True)
train_dataset = PromptDataset(
    dataset=dataset["train"],
    tokenizer=tokenizer,
    data_config=config["data"],
)

train_dataloader = DataLoader(
    train_dataset,
    collate_fn=data_collator,
    batch_size=256,
    shuffle=True,
)

max_so_far = 0

lens = []
for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
    prompts = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
    l = len(batch["input_ids"][0])
    lens += [l]
    if l > max_so_far:
        max_so_far = l
        logger.info(f"New max: {max_so_far}")

    # logger.info(f"Length: {len(batch['input_ids'][0])}")
    # if i == 0:
    #     for prompt in prompts:
    #         logger.info("*" * 80)
    #         logger.info(prompt)
    #         logger.info("\n")
    # if i > 100:
    #     break

logger.info(f"Mean length: {np.mean(lens)}")
logger.info(f"Max length: {np.max(lens)}")
logger.info(f"Min length: {np.min(lens)}")


logger.info(prompts[0])

# plt.hist(lens, bins=100)
# plt.show()


# # ----


# data_config = {
#     "path": "prompt_parts/ukb_2024_02/prompt_parts.parquet",
#     "split": "config/splits/split_2024_06_29_eval_100000.json",
#     "subset": "subsets/ukb_2024_02/MACE_ADO_no_previous_target.json",
#     "eval_subset": True,
#     "target_name": "MACE_ADO_10y",
#     "zeroshot_prompt": False,
# }

# with open("config/ukb_data/feature_groups/meta/feature_groups.yaml", "r") as f:
#     feature_groups = yaml.safe_load(f)

# summary_token_lenghts = pd.DataFrame()
# summary_feature_groups = {}

# dataset = load_prompt_parts(data_config=data_config)["train"]

# for feature_group in tqdm(feature_groups):
#     data_config["feature_config"] = {"feature_groups": [feature_group]}

#     prompt_part_names = get_column_names(data_config["feature_config"])
#     prompt_parts = dataset.select_columns(prompt_part_names)
#     prompts = prompt_parts.map(
#         lambda x: {"prompt": join_prompt_parts([x[col] for col in prompt_part_names])},
#         remove_columns=prompt_part_names,
#         # batched=True,
#     )
#     prompts = prompts.map(lambda x: {"num_prompt_parts": len(x["prompt"].split("\n"))})
#     tokenized_prompts = prompts.map(
#         lambda x: tokenizer(x["prompt"]),
#         batched=True,
#     )
#     tokenized_prompts = tokenized_prompts.map(
#         lambda x: {"length": len(x["input_ids"])},
#     )

#     tmp_df = pd.DataFrame(tokenized_prompts["length"], columns=["token_length"])
#     tmp_df["num_features"] = len(prompt_part_names)
#     tmp_df["num_non_null_features"] = prompts["num_prompt_parts"]
#     tmp_df["eid"] = dataset["eid"]
#     tmp_df["feature_group"] = feature_group

#     summary_token_lenghts = pd.concat([summary_token_lenghts, tmp_df], axis=0)

#     summary_feature_groups["num_features"] = len(prompt_part_names)
#     summary_feature_groups["mean_num_non_null_prompt_parts"] = np.mean(
#         prompts["num_prompt_parts"]
#     )
#     summary_feature_groups["max_num_non_null_prompt_parts"] = np.max(
#         prompts["num_prompt_parts"]
#     )
#     summary_feature_groups["min_num_non_null_prompt_parts"] = np.min(
#         prompts["num_prompt_parts"]
#     )

# # visualize the distribution of prompt lengths per feature group

# summary_token_lenghts.groupby("feature_group")["token_length"].describe()

# summary_token_lenghts.to_csv("token_lenghts_per_feature_group.csv")

# cpal = setup_plotting(context="paper")

# plt.figure(figsize=(12, 6))
# sns.displot(
#     data=summary_token_lenghts,
#     x="token_length",
#     # stat='density',
#     kind="hist",
#     fill=True,
#     col="feature_group",
#     col_wrap=3,
#     aspect=3,
#     height=2,
#     bins=80,
# )

# plt.savefig(
#     "token_length_per_feature_group_hist.pdf", format="pdf", bbox_inches="tight"
# )


# plt.figure(figsize=(8, 10))
# sns.boxplot(
#     data=summary_token_lenghts,
#     x="token_length",
#     fill=True,
#     y="feature_group",
#     fliersize=0.3,
# )

# plt.savefig("token_length_per_feature_group.pdf", format="pdf", bbox_inches="tight")

# # visualize the distribution of the number of non-null prompt parts per feature group

# plt.figure(figsize=(8, 10))
# sns.boxplot(
#     data=summary_token_lenghts,
#     x="num_non_null_features",
#     fill=True,
#     y="feature_group",
#     fliersize=0.3,
# )

# plt.savefig(
#     "num_non_null_features_per_feature_group.pdf", format="pdf", bbox_inches="tight"
# )


# # correlation num_non_null_features and token_length
# sns.FacetGrid(
#     summary_token_lenghts, col="feature_group", col_wrap=3, aspect=3, height=2
# ).map(
#     sns.scatterplot,
#     "num_non_null_features",
#     "token_length",
# )

# plt.savefig(
#     "correlation_num_non_null_features_token_length.pdf",
#     format="pdf",
#     bbox_inches="tight",
# )
