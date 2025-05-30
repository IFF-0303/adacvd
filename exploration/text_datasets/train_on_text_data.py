import argparse
import logging
import random
import socket
from os.path import join

import datasets as hf_datasets
import torch
import yaml
from accelerate.utils import set_seed
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification

from adacvd.data.ukb_data_utils import ASSETS_PATH
from adacvd.training.dataset import TextDataset, load_split
from adacvd.training.model import HuggingfaceModel, evaluate_step
from adacvd.training.utils import RuntimeLimits, get_latest_checkpoint_dir

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_dir",
        default="config/text_generation",
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
    logger = logging.getLogger()
    args = parse_args()
    print(f"args: {args.__dict__}")
    print(f"Node: {socket.gethostname()}")
    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    random.seed(0)

    with open(join(args.train_dir, "text_train_settings.yaml"), "r") as f:
        train_config = yaml.safe_load(f)

    dataset = hf_datasets.load_dataset(
        "csv",
        data_files=train_config["data"]["path"],
        split="train",
    )

    # filter eos if specified
    # this is used to filter out cropped samples that do not end with the end-of-sequence token
    filter_eos = train_config["data"].get("filter_eos", False)
    if filter_eos:
        logging.info(f"Dataset size: {len(dataset)}")
        dataset = dataset.filter(lambda x: x["ends_with_eos"])
        logging.info(f"Dataset size after filtering: {len(dataset)}")

    # rename text_summary to prompt and remove the prompt field
    dataset = dataset.remove_columns(["prompt"])
    dataset = dataset.rename_column("text_summary", "prompt")

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

    num_gpus = model.accelerator.num_processes
    train_config["training"]["batch_size_per_device"] = (
        train_config["training"]["batch_size"] // num_gpus
    )
    # step_size in config = "step size after x batches of effective batch size"
    # need to translate it for multi-gpu setting with accelerate
    if train_config["training"]["scheduler"].get("type") == "step":
        step_size = train_config["training"]["scheduler"]["step_size"]
        step_size_multi_gpu = step_size * num_gpus
        train_config["training"]["scheduler"]["step_size"] = step_size_multi_gpu
        logger.info(
            f"Step size adjusted for multi-gpu training: {step_size} -> {step_size_multi_gpu}"
        )
    elif train_config["training"]["scheduler"].get("type") == "cosine_warmup":
        T_0 = train_config["training"]["scheduler"]["T_0"]
        T_0_multi_gpu = T_0 * num_gpus
        train_config["training"]["scheduler"]["T_0"] = T_0_multi_gpu
        logger.info(f"T_0 adjusted for multi-gpu training: {T_0} -> {T_0_multi_gpu}")
    else:
        logger.info(
            f"Scheduler type {train_config['training']['scheduler'].get('type')} may need to be adjusted for multi-gpu training."
        )

    data_collator = DataCollatorForTokenClassification(model.tokenizer, padding=True)

    # load split
    split = load_split(train_config["data"])
    eids = dataset["eid"]

    train_eids = list(set(eids) & set(split["train"]))
    test_eids = list(set(eids) & set(split["test"]))
    validation_eids = list(set(eids) & set(split["validation"]))

    if train_config["data"].get("num_training_samples") is not None:
        # training is done on test data (not seen during pre-training)
        set_seed(train_config["training"].get("random_seed", 0))
        test_eids = random.sample(
            test_eids,
            min(train_config["data"]["num_training_samples"], len(test_eids)),
        )
        logger.info(f"Number of training samples: {len(test_eids)}")

    train_dataset = TextDataset(
        dataset=dataset.filter(lambda x: x["eid"] in train_eids),
        tokenizer=model.tokenizer,
    )

    test_dataset = TextDataset(
        dataset=dataset.filter(lambda x: x["eid"] in test_eids),
        tokenizer=model.tokenizer,
    )

    validation_dataset = TextDataset(
        dataset=dataset.filter(lambda x: x["eid"] in validation_eids),
        tokenizer=model.tokenizer,
    )

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=data_collator,
        batch_size=train_config["training"]["batch_size_per_device"],
        shuffle=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=data_collator,
        batch_size=train_config["training"]["batch_size_per_device"],
        shuffle=True,
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        collate_fn=data_collator,
        batch_size=train_config["training"]["batch_size_per_device"],
        shuffle=False,
    )
    # train on test data loader, evaluate on validation

    (model.network, train_dataloader, test_dataloader, validation_dataloader) = (
        model.accelerator.prepare(
            model.network, train_dataloader, test_dataloader, validation_dataloader
        )
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

    x = next(iter(train_dataloader))

    examples = model.tokenizer.batch_decode(x["input_ids"], skip_special_tokens=False)
    for example in examples:
        logger.info(example)

    model.train(
        train_dir=args.train_dir,
        train_loader=test_dataloader,  # train on test data (not seen during pre training)
        validation_loader=validation_dataloader,
        test_loader=validation_dataloader,
        runtime_limits=runtime_limits,
        trackers_kwargs={"project_name": "UKB_LLM_text_adaptation"},
        extended_dataloaders={},
        **train_config["training"],
    )


if __name__ == "__main__":
    main()
