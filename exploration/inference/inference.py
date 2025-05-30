import argparse
import logging
import socket
from datetime import datetime
from os.path import join
from pathlib import Path

import torch
import yaml
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForTokenClassification

import pandora.utils.logger
from pandora.data.ukb_data_utils import WANDB_ENTITY
from pandora.training.dataset import PromptDataset, load_prompt_parts
from pandora.training.model import HuggingfaceModel, evaluate_step
from pandora.training.utils import get_latest_checkpoint_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inference_dir",
        help="Path to inference directory. Should contain a 'inference_settings.yaml' file. Inference outcomes will be stored here.",
    )
    parser.add_argument(
        "--device",
        help="Device to run inference on (e.g cpu, cuda or mps)",
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

    with open(join(args.inference_dir, "inference_settings.yaml"), "r") as f:
        inference_config = yaml.safe_load(f)

    train_dir = inference_config["base_model_dir"]

    with open(join(train_dir, "train_settings.yaml"), "r") as f:
        train_config = yaml.safe_load(f)

    inference_config["train_settings"] = train_config.copy()

    if train_config["data"]["split"] != inference_config["data"]["split"]:
        logger.WARN(
            f"Train-Test-Validation split between training and inference config do not match."
        )

    dataset = load_prompt_parts(
        data_config=inference_config["data"], num_samples=args.num_samples
    )

    logger.info(
        f"Dataset sizes: Train: {len(dataset['train'])}, Validation: {len(dataset['validation'])}, Test: {len(dataset['test'])}"
    )

    checkpoint_dir = get_latest_checkpoint_dir(
        train_dir, fixed_epoch=inference_config.get("fixed_epoch", None)
    )
    checkpoint_dir = Path(
        "/fast/groups/hfm-users/pandora-med-box/results/2025_03_10_flexible_from_full/checkpoint_3_n128000"
    )
    logger.info(f"Checkpoint dir: {checkpoint_dir}")

    if checkpoint_dir is None:
        logger.error("No checkpoint found. Aborting.")
        raise FileNotFoundError

    model = HuggingfaceModel(
        model_dir=checkpoint_dir,
        device=args.device,
        config=inference_config["train_settings"],
    )

    data_collator = DataCollatorForTokenClassification(model.tokenizer, padding=True)

    test_dataset = PromptDataset(
        dataset=dataset["test"],
        tokenizer=model.tokenizer,
        data_config=inference_config["data"],
    )

    model.accelerator.init_trackers(
        config=inference_config,
        project_name="UKB_LLM_Inference",
        init_kwargs={
            "wandb": {
                "name": datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f"),
                "entity": WANDB_ENTITY,
                # "mode": "disabled",
            }
        },
    )

    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=data_collator,
        batch_size=inference_config["inference"]["eval_batch_size"],
        shuffle=False,
    )

    for batch in test_dataloader:
        prompts = model.tokenizer.batch_decode(batch["input_ids"])
        logging.info(prompts[0])
        break

    (
        model.network,
        test_dataloader,
    ) = model.accelerator.prepare(
        model.network,
        test_dataloader,
    )

    metrics, evals = evaluate_step(model, test_dataloader)
    model.accelerator.log(metrics)
    evals.to_csv(f"{args.inference_dir}/evals_inference.csv")


if __name__ == "__main__":
    main()
