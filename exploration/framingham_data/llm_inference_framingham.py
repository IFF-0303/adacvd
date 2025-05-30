import argparse
import logging
import random
import socket
from os.path import join

import datasets as hf_datasets
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification

from adacvd.data.ukb_data_utils import ASSETS_PATH
from adacvd.training.dataset import FraminghamPromptDataset
from adacvd.training.model import HuggingfaceModel, evaluate_step
from adacvd.training.utils import RuntimeLimits, get_latest_checkpoint_dir

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inference_dir",
        help="Path to inference directory. Should contain a 'inference_settings.yaml' file. Inference outcomes will be stored here.",
    )
    parser.add_argument(
        "--device",
        help="Device to run inference on (e.g cpu, cuda or mps)",
        default="cpu",
    )
    parser.add_argument(
        "--num_samples",
        help="Number of training and evaluation samples (for debugging purposes)",
        type=int,
        default=None,
    )
    return parser.parse_args()


def main():
    logger = logging.getLogger()
    args = parse_args()
    print(f"args: {args.__dict__}")
    print(f"Node: {socket.gethostname()}")
    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    random.seed(0)

    with open(join(args.inference_dir, "inference_settings.yaml"), "r") as f:
        train_config = yaml.safe_load(f)

    dataset = hf_datasets.load_dataset(
        "parquet",
        data_files=str(ASSETS_PATH / train_config["data"]["path"]),
        split="train",
    )

    checkpoint_dir = None

    if train_config["model"].get("resume_training", False):
        logger.info("Resuming training")
        if train_config["model"].get("model_checkpoint") is not None:
            checkpoint_dir = train_config["model"]["model_checkpoint"]
        else:
            # load from training checkpoint
            checkpoint_dir = get_latest_checkpoint_dir(args.train_dir)

    if (
        train_config["model"].get("resume_training", False)
        and checkpoint_dir is not None
    ):
        logger.info(f"Checkpoint dir: {checkpoint_dir}")
        model = HuggingfaceModel(
            model_dir=checkpoint_dir, device=args.device, config=train_config
        )
    else:
        logger.info("Building model from scratch")
        model = HuggingfaceModel(config=train_config, device=args.device)

    model = HuggingfaceModel(model_dir=checkpoint_dir, config=train_config)

    data_collator = DataCollatorForTokenClassification(model.tokenizer, padding=True)

    ids = dataset["RANDID"]
    train_ids, eval_ids = train_test_split(
        ids,
        test_size=0.5,
        random_state=train_config["training"]["random_seed"],
    )
    train_dataset = FraminghamPromptDataset(
        dataset=dataset.filter(lambda x: x["RANDID"] in train_ids),
        tokenizer=model.tokenizer,
    )

    eval_dataset = FraminghamPromptDataset(
        dataset=dataset.filter(lambda x: x["RANDID"] in eval_ids),
        tokenizer=model.tokenizer,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=train_config["training"]["eval_batch_size"],
        shuffle=False,
    )

    (
        model.network,
        eval_dataloader,
    ) = model.accelerator.prepare(
        model.network,
        eval_dataloader,
    )

    metrics, evals = evaluate_step(model, eval_dataloader)
    model.accelerator.log(metrics)
    evals.to_csv(f"{args.inference_dir}/evals_inference.csv")
    print(metrics)

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=data_collator,
        batch_size=train_config["training"]["batch_size"],
        shuffle=True,
    )

    RUNTIME_LIMITS = {
        "max_time_per_run": 360000,
        "max_epochs_per_run": 1000,
    }
    runtime_limits = RuntimeLimits(
        epoch_start=model.epoch,
        max_epochs_total=train_config["training"]["epochs"],
        **RUNTIME_LIMITS,
    )

    train_dataloader = model.accelerator.prepare(train_dataloader)
    model.train(
        train_dir=args.inference_dir,
        train_loader=train_dataloader,
        validation_loader=eval_dataloader,
        test_loader=eval_dataloader,
        runtime_limits=runtime_limits,
        trackers_kwargs={"project_name": "UKB_adaptation_framingham"},
        extended_dataloaders={},
        **train_config["training"],
    )


if __name__ == "__main__":
    main()
