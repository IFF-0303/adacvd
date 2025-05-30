import argparse
import json
import logging
import os
import socket
from copy import deepcopy
from os.path import join

import torch
import yaml
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification

import pandora.utils.logger
from pandora.training.dataset import PromptDataset, load_prompt_parts
from pandora.training.model import HuggingfaceModel
from pandora.training.utils import RuntimeLimits, get_latest_checkpoint_dir

RUNTIME_LIMITS = {
    "max_time_per_run": 360000,
    "max_epochs_per_run": 1000,
}


def parse_args():
    """
    Parse command-line arguments for training.

    Returns:
        argparse.Namespace: Parsed arguments with train_dir, device, and num_samples.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_dir",
        help="Path to the training directory. Should contain a 'train_settings.yaml' file.",
        default="config/training",
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
    """
    Main training routine:
        - Loads configuration and dataset.
        - Initializes or resumes the model.
        - Prepares dataloaders and collators.
        - Optionally sets up extended evaluation.
        - Trains the model and saves checkpoints.
    """
    logger = logging.getLogger()
    args = parse_args()
    logger.info(f"args: {args.__dict__}")
    logger.info(f"Node: {socket.gethostname()}")
    if args.device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name()}")

    set_seed(0)

    # Load configuration
    with open(join(args.train_dir, "train_settings.yaml"), "r") as f:
        config = yaml.safe_load(f)

    # Set random seed if specified in config
    if config["training"].get("random_seed", None) is not None:
        set_seed(config["training"]["random_seed"])
        logger.info(f"Set random seed to {config['training']['random_seed']}")

    config["train_dir"] = args.train_dir
    config["num_samples"] = args.num_samples

    # Load dataset
    dataset = load_prompt_parts(
        data_config=config["data"], num_samples=args.num_samples
    )

    logger.info(
        f"Dataset sizes: Train: {len(dataset['train'])}, Test: {len(dataset['test'])}, Validation: {len(dataset['validation'])}"
    )

    # Resume from checkpoint if specified
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

    # Adjust batch size and scheduler for multi-GPU training
    # batch_size = effective_batch_size = batch_size_per_device * num_gpus
    num_gpus = model.accelerator.num_processes
    config["training"]["batch_size_per_device"] = (
        config["training"]["batch_size"] // num_gpus
    )
    # step_size in config = "step size after x batches of effective batch size"
    # need to translate it for multi-gpu setting with accelerate
    if config["training"]["scheduler"].get("type") == "step":
        step_size = config["training"]["scheduler"]["step_size"]
        step_size_multi_gpu = step_size * num_gpus
        config["training"]["scheduler"]["step_size"] = step_size_multi_gpu
        logger.info(
            f"Step size adjusted for multi-gpu training: {step_size} -> {step_size_multi_gpu}"
        )
    elif config["training"]["scheduler"].get("type") == "cosine_warmup":
        T_0 = config["training"]["scheduler"]["T_0"]
        T_0_multi_gpu = T_0 * num_gpus
        config["training"]["scheduler"]["T_0"] = T_0_multi_gpu
        logger.info(f"T_0 adjusted for multi-gpu training: {T_0} -> {T_0_multi_gpu}")
    else:
        logger.info(
            f"Scheduler type {config['training']['scheduler'].get('type')} may need to be adjusted for multi-gpu training."
        )

    model.initialize_optimizer_and_scheduler(
        config["training"]["optimizer"],
        config["training"]["scheduler"],
    )

    # Optionally save train eids for reproducibility
    if config["data"].get("num_training_samples") is not None:
        eids = {"train": dataset["train"]["eid"]}
        with open(join(args.train_dir, "train_eids.json"), "w") as f:
            json.dump(eids, f)

    # Prepare data collator and datasets
    data_collator = DataCollatorForTokenClassification(
        model.tokenizer, padding=True, pad_to_multiple_of=8
    )
    train_dataset = PromptDataset(
        dataset=dataset["train"],
        tokenizer=model.tokenizer,
        data_config=config["data"],
        mask_labels=config["training"].get("mask_labels", False),
    )
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

    # Extended evaluation: evaluate on different feature groups during training
    extended_evaluation = config["training"].get("extended_evaluation", False)
    extended_datalaoders = {}
    if extended_evaluation:
        data_config = deepcopy(config["data"])
        data_config.pop("sampling", None)  # no sampling

        # different feature groups
        extended_datasets = {}

        base_config = deepcopy(data_config)
        base_groups = [
            "base_risk_score_inputs",
            "additional_risk_score_inputs_aha_acc",
            "additional_risk_score_inputs_prevent",
        ]

        add_groups = [
            [],
            ["polygenic_risk_scores_all"],
            ["medical_history_all"],
            ["blood_samples"],
            ["lifestyle_and_environment"],
            ["icd_codes"],
        ]
        for add_group in add_groups:
            base_config["feature_config"]["feature_groups"] = base_groups + add_group
            config_name = "extended_" + "_".join(add_group)
            extended_datasets[config_name] = PromptDataset(
                dataset=dataset["validation"],
                tokenizer=model.tokenizer,
                data_config=base_config,
            )

        for config_name, dataset in extended_datasets.items():
            extended_datalaoders[config_name] = DataLoader(
                dataset,
                collate_fn=data_collator,
                batch_size=config["training"]["eval_batch_size"],
                shuffle=False,
            )

    # Prepare main dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=data_collator,
        batch_size=config["training"]["batch_size_per_device"],
        shuffle=True,
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

    # Set up runtime limits for training
    runtime_limits = RuntimeLimits(
        epoch_start=model.epoch,
        max_epochs_total=config["training"]["epochs"],
        **RUNTIME_LIMITS,
    )

    # Start training
    model.train(
        train_dir=args.train_dir,
        train_loader=train_dataloader,
        test_loader=test_dataloader,
        validation_loader=validation_dataloader,
        runtime_limits=runtime_limits,
        trackers_kwargs={"project_name": "UKB_LLM_Models"},
        extended_dataloaders=extended_datalaoders,
        **config["training"],
    )


if __name__ == "__main__":
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
    main()
