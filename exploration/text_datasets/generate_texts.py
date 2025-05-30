import argparse
import logging
import os
import socket
from datetime import datetime
from os.path import join

import datasets as hf_datasets
import pandas as pd
import torch
import yaml
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForTokenClassification

import pandora.utils.logger
from pandora.data.ukb_data_utils import WANDB_ENTITY
from pandora.training.dataset import PromptDataset, load_prompt_parts
from pandora.training.model import HuggingfaceModel
from pandora.training.utils import RuntimeLimits, get_latest_checkpoint_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_dir",
        help="Path to the training directory. Should contain a 'text_generation_settings.yaml' file.",
        default="config/text_generation",
    )
    parser.add_argument(
        "--device",
        help="Device to train on (e.g cpu, cuda or mps).",
        default="mps",
    )
    parser.add_argument(
        "--num_samples",
        help="Number of training and evaluation samples (for debugging). If None, use all samples.",
        type=int,
        default=None,
    )
    return parser.parse_args()


def main():
    logger = logging.getLogger()
    args = parse_args()
    logger.info(f"args: {args.__dict__}")
    logger.info(f"Node: {socket.gethostname()}")
    if args.device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name()}")

    set_seed(0)

    with open(join(args.train_dir, "text_generation_settings.yaml"), "r") as f:
        config = yaml.safe_load(f)

    if config["training"].get("random_seed", None) is not None:
        set_seed(config["training"]["random_seed"])
        logger.info(f"Set random seed to {config['training']['random_seed']}")

    config["train_dir"] = args.train_dir
    config["num_samples"] = args.num_samples

    dataset = load_prompt_parts(
        data_config=config["data"], num_samples=args.num_samples
    )

    logger.info(
        f"Dataset sizes: Train: {len(dataset['train'])}, Test: {len(dataset['test'])}, Validation: {len(dataset['validation'])}"
    )

    if config["model"].get("resume_training", False):
        logger.info("Resuming training")
        if config["model"].get("model_checkpoint") is not None:
            checkpoint_dir = config["model"]["model_checkpoint"]
        else:
            # load from training checkpoint
            checkpoint_dir = get_latest_checkpoint_dir(args.train_dir)

    if config["model"].get("resume_training", False) and checkpoint_dir is not None:
        logger.info(f"Checkpoint dir: {checkpoint_dir}")
        model = HuggingfaceModel(
            model_dir=checkpoint_dir, device=args.device, config=config
        )
    else:
        logger.info("Building model from scratch")
        model = HuggingfaceModel(config=config, device=args.device)

    # translate parameters for multi-gpu training
    # batch_size = effective_batch_size = batch_size_per_device * num_gpus
    num_gpus = model.accelerator.num_processes
    config["training"]["batch_size_per_device"] = (
        config["training"]["batch_size"] // num_gpus
    )

    data_collator = DataCollatorForTokenClassification(model.tokenizer, padding=True)

    # concatenate train-test-validation datasets
    concatenated_dataset = hf_datasets.concatenate_datasets(
        [dataset[key] for key in dataset.keys()]
    )

    full_dataset = PromptDataset(
        dataset=concatenated_dataset,
        tokenizer=model.tokenizer,
        data_config=config["data"],
    )

    full_dataloader = DataLoader(
        full_dataset,
        collate_fn=data_collator,
        batch_size=config["training"]["batch_size_per_device"],
        shuffle=False,
    )

    (model.network, full_dataloader) = model.accelerator.prepare(
        model.network, full_dataloader
    )

    model.accelerator.init_trackers(
        config=config,
        project_name="UKB_LLM_text_generation",
        init_kwargs={
            "wandb": {
                "name": datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f"),
                "entity": WANDB_ENTITY,
                # "mode": "disabled",
            }
        },
    )

    MODEL_INPUT_VARS = ["input_ids", "attention_mask"]

    OUTPUT_CSV = join(args.train_dir, "text_generation_output.csv")
    # remove existing output file
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)

    share_eos_all = []
    max_new_tokens = config["model"]["max_new_tokens"]

    model.network.eval()
    with torch.no_grad():

        for batch_idx, batch in tqdm(
            enumerate(full_dataloader), total=len(full_dataloader)
        ):

            # breakpoint()
            outputs = model.network.generate(
                **{key: batch[key] for key in MODEL_INPUT_VARS},
                max_new_tokens=max_new_tokens,
                pad_token_id=model.tokenizer.pad_token_id,
                do_sample=False,
            )

            input_length = batch["input_ids"].shape[1]
            generated_tokens = outputs[:, input_length:]

            # len of generated tokens without eos and padding
            num_generated_tokens_without_eos = (
                (generated_tokens != model.tokenizer.pad_token_id).sum(dim=1).cpu()
            ).numpy()

            completions = model.tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )

            prompts = model.tokenizer.batch_decode(
                batch["input_ids"], skip_special_tokens=True
            )

            # completion ends with eos token as list
            share_eos = (outputs[:, -1] == model.tokenizer.eos_token_id).cpu()
            share_eos_all += share_eos.tolist()

            batch_df = pd.DataFrame(
                {
                    "eid": batch["eid"].tolist(),
                    "text_summary": completions,
                    "prompt": prompts,
                    "completion": batch["completion"].tolist(),
                    "ends_with_eos": share_eos.tolist(),
                }
            )

            batch_df.to_csv(
                OUTPUT_CSV, mode="a", header=not os.path.exists(OUTPUT_CSV), index=False
            )

            model.accelerator.log(
                {
                    "generated_tokens": generated_tokens.shape[1],
                    "num_generated_tokens_without_eos_mean": num_generated_tokens_without_eos.mean(),
                    "num_generated_tokens_without_eos_max": num_generated_tokens_without_eos.max(),
                    "share_eos": sum(share_eos_all) / len(share_eos_all),
                }
            )

    logger.info("All batches processed successfully!")


if __name__ == "__main__":
    main()
