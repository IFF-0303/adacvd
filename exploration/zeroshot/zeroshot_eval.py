# load prompt data (evaluation set)
# load model from huggingface (without LoRA adapter)
# write system prompt and explain how the model should answer
# run inference (generate a longer prompt completion)
# retrieve the predicted risk score from the model output
# save the response and the predicted risk score in a csv file

import argparse
import json
import logging
import re
import socket
from datetime import datetime
from os.path import join

import pandas as pd
import torch
import wandb
import yaml
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForTokenClassification

import pandora.utils.logger
from pandora.data.ukb_data_utils import WANDB_ENTITY
from pandora.training.dataset import PromptDataset, load_prompt_parts
from pandora.training.model import HuggingfaceModel
from pandora.training.utils import RuntimeLimits, get_latest_checkpoint_dir
from pandora.utils.metrics import compute_binary_classification_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_dir",
        help="Path to a directory. Should contain a 'train_settings.yaml' file",
        default="config/zeroshot",
    )
    parser.add_argument(
        "--device",
        help="Device to train on (e.g cpu, cuda or mps)",
        default="mps",
    )
    parser.add_argument(
        "--num_samples",
        help="Number of training and evaluation samples (for debugging purposes)",
        type=int,
        default=None,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = logging.getLogger()
    logger.info(f"args: {args.__dict__}")
    logger.info(f"Node: {socket.gethostname()}")
    if args.device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name()}")

    set_seed(0)

    with open(join(args.train_dir, "train_settings.yaml"), "r") as f:
        config = yaml.safe_load(f)

    config["train_dir"] = args.train_dir
    config["num_samples"] = args.num_samples

    dataset = load_prompt_parts(
        data_config=config["data"], num_samples=args.num_samples
    )

    logger.info(
        f"Dataset sizes: Train: {len(dataset['train'])}, Test: {len(dataset['test'])}, Validation: {len(dataset['validation'])}"
    )

    checkpoint_dir = get_latest_checkpoint_dir(args.train_dir)

    if config["model"].get("resume_training", False) and checkpoint_dir is not None:
        logger.info(f"Checkpoint dir: {checkpoint_dir}")
        model = HuggingfaceModel(model_dir=checkpoint_dir, device=args.device)
    else:
        logger.info("Building model from scratch")
        model = HuggingfaceModel(config=config, device=args.device)

    data_collator = DataCollatorForTokenClassification(model.tokenizer, padding=True)

    test_dataset = PromptDataset(
        dataset=dataset["test"],
        tokenizer=model.tokenizer,
        data_config=config["data"],
    )
    validation_dataset = PromptDataset(
        dataset=dataset["validation"],
        tokenizer=model.tokenizer,
        data_config=config["data"],
        fix_seed=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=data_collator,
        batch_size=config["training"]["eval_batch_size"],
        shuffle=False,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        collate_fn=data_collator,
        batch_size=config["training"]["eval_batch_size"],
        shuffle=False,
    )

    (model.network, test_dataloader, validation_dataloader) = model.accelerator.prepare(
        model.network, test_dataloader, validation_dataloader
    )

    model.accelerator.init_trackers(
        config=config,
        project_name="UKB_LLM_zeroshot",
        init_kwargs={
            "wandb": {
                "name": datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f"),
                "entity": WANDB_ENTITY,
                # "mode": "disabled",
            }
        },
    )

    wandb_tracker = model.accelerator.get_tracker("wandb")

    def extract_json_from_completion(completion):
        # extract json from completion
        json_matches = re.findall(r"({.*?})", completion, re.DOTALL)

        if len(json_matches) > 0:
            json_output = json_matches[0]
            cleaned_json = "\n".join(
                line.split("#")[0].rstrip() for line in json_output.split("\n")
            )
            try:

                json_output = json.loads(json_output)
                return json_output
            except json.JSONDecodeError:
                logger.info(f"Could not decode json: {json_output}")
                return None
        return None

    MODEL_INPUT_VARS = ["input_ids", "attention_mask"]
    df = pd.DataFrame(columns=["eid", "risk_score", "completion"])

    model.network.eval()
    with torch.no_grad():
        for k, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):

            outputs = model.network.generate(
                **{key: batch[key] for key in MODEL_INPUT_VARS},
                max_new_tokens=100,
                pad_token_id=model.tokenizer.pad_token_id,
                do_sample=False,
            )
            completions = model.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

            risk_scores = []
            for i, completion in enumerate(completions):
                # logger.info(f"Completion: {completion}")
                risk_score = None
                json_output = extract_json_from_completion(completion)
                if json_output:
                    vals = list(json_output.values())
                    if len(vals) >= 1:
                        risk_score = vals[0]
                    try:
                        risk_score = float(risk_score)
                        if risk_score >= 1.0:  # risk score is a percentage
                            risk_score = risk_score / 100
                    except ValueError:
                        risk_score = None

                risk_scores.append(risk_score)
                # logger.info(f"Prediction: {risk_score}")

            df_batch = pd.DataFrame(
                {
                    "eid": batch["eid"].tolist(),
                    "risk_score": risk_scores,
                    "completion": batch["completion"].tolist(),
                }
            )
            df = pd.concat([df, df_batch])

            if k % 1000 == 0:

                logging.info("Logging batch summary to wandb")

                batch_summary = df_batch.copy()
                batch_summary["text_completion"] = completions

                wandb_table = wandb.Table(
                    columns=batch_summary.columns.to_list(),
                    data=batch_summary.values.tolist(),
                )

                wandb_tracker.log({f"batch_summary_{k}": wandb_table})

            na_mask = df["risk_score"].isna()
            na_share = na_mask.mean()

            df_metrics = df[~na_mask]

            metrics = compute_binary_classification_metrics(
                df_metrics["completion"].astype(bool).values,
                df_metrics["risk_score"].values,
            )
            # logger.info(f"Metrics: {metrics}")

            metrics["na_share"] = na_share

            model.accelerator.log(metrics)

    # save results to train_dir
    df.to_csv(join(args.train_dir, "zeroshot_evals.csv"), index=False)

    print("")


if __name__ == "__main__":
    main()
