import logging
import os
import re
import time
from typing import Iterable

import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import (
    AdaptionPromptConfig,
    LoraConfig,
    PeftConfig,
    PrefixTuningConfig,
    TaskType,
)
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification

from exploration.evaluation.evaluation_utils import find_highest_epoch_step
from pandora.data import prompt, ukb_features

# Default configuration dictionaries for finetuning approaches
FINETUNING_DEFAULT_DICT = {
    "lora": {
        "task_type": TaskType.CAUSAL_LM,
        "inference_mode": False,
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "target_modules": [
            "q_proj",
            "k_proj",
        ],
    },
    "prefix-tuning": {
        "task_type": TaskType.CAUSAL_LM,
        "num_virtual_tokens": 30,
    },
}

# Mapping of scheduler names to PyTorch scheduler classes
SCHEDULER_DICT = {
    "linear": torch.optim.lr_scheduler.LambdaLR,
    "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
    "cosine_warmup": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    "constant": torch.optim.lr_scheduler.StepLR,
}


class RuntimeLimits:
    """
    Keeps track of the runtime limits (time limit, epoch limit, max. number
    of epochs for model). Used to control training duration and early stopping.
    """

    def __init__(
        self,
        max_time_per_run: float = None,
        max_epochs_per_run: int = None,
        max_epochs_total: int = None,
        epoch_start: int = None,
    ):
        """
        Parameters
        ----------
        max_time_per_run: float = None
            Maximum time for run, in seconds (soft limit, break only after full epoch).
        max_epochs_per_run: int = None
            Maximum number of epochs for run.
        max_epochs_total: int = None
            Maximum total number of epochs for model.
        epoch_start: int = None
            Start epoch of run.
        """
        self.max_time_per_run = max_time_per_run
        self.max_epochs_per_run = max_epochs_per_run
        self.max_epochs_total = max_epochs_total
        self.epoch_start = epoch_start
        self.time_start = time.time()
        if max_epochs_per_run is not None and epoch_start is None:
            raise ValueError("epoch_start required to check max_epochs_per_run.")

    def limits_exceeded(self, epoch: int = None):
        """
        Check whether any of the runtime limits are exceeded.

        Parameters
        ----------
        epoch: int = None

        Returns
        -------
        limits_exceeded: bool
            Flag whether runtime limits are exceeded and run should be stopped.
            If True, prints a message for the reason.
        """
        # check time limit for run
        if self.max_time_per_run is not None:
            if time.time() - self.time_start >= self.max_time_per_run:
                logging.info(
                    f"Stop run: Time limit of {self.max_time_per_run}s exceeded."
                )
                return True
        # check epoch limit for run
        if self.max_epochs_per_run is not None:
            if epoch is None:
                raise ValueError("epoch required")
            if epoch - self.epoch_start >= self.max_epochs_per_run:
                logging.info(
                    f"Stop run: Epoch limit of {self.max_epochs_per_run} per run reached."
                )
                return True
        # check total epoch limit
        if self.max_epochs_total is not None:
            if epoch >= self.max_epochs_total:
                logging.info(
                    f"Stop run: Total epoch limit of {self.max_epochs_total} reached."
                )
                return True
        # return False if none of the limits is exceeded
        return False

    def local_limits_exceeded(self, epoch: int = None):
        """
        Check whether any of the local runtime limits are exceeded. Local runtime
        limits include max_epochs_per_run and max_time_per_run, but not max_epochs_total.

        Parameters
        ----------
        epoch: int = None

        Returns
        -------
        limits_exceeded: bool
            Flag whether local runtime limits are exceeded.
        """
        # check time limit for run
        if self.max_time_per_run is not None:
            if time.time() - self.time_start >= self.max_time_per_run:
                return True
        # check epoch limit for run
        if self.max_epochs_per_run is not None:
            if epoch is None:
                raise ValueError("epoch required")
            if epoch - self.epoch_start >= self.max_epochs_per_run:
                return True
        # return False if none of the limits is exceeded
        return False


def get_optimizer_from_kwargs(
    model_parameters: Iterable,
    **optimizer_kwargs,
):
    """
    Select and instantiate a PyTorch optimizer from keyword arguments.

    Parameters
    ----------
    model_parameters : Iterable
        Model parameters to optimize.
    **optimizer_kwargs
        Must include 'type' (e.g. 'adamw') and other optimizer-specific kwargs.

    Returns
    -------
    optimizer : torch.optim.Optimizer
    """
    optimizers_dict = {
        "adagrad": torch.optim.Adagrad,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "lbfgs": torch.optim.LBFGS,
        "RMSprop": torch.optim.RMSprop,
        "sgd": torch.optim.SGD,
    }

    optimizer = optimizers_dict[optimizer_kwargs.pop("type")]
    return optimizer(model_parameters, **optimizer_kwargs)


def get_scheduler_from_kwargs(
    optimizer: torch.optim.Optimizer,
    **scheduler_kwargs,
):
    """
    Select and instantiate a PyTorch learning rate scheduler from keyword arguments.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer to schedule.
    **scheduler_kwargs
        Must include 'type' (e.g. 'step') and other scheduler-specific kwargs.

    Returns
    -------
    scheduler : torch.optim.lr_scheduler._LRScheduler
    """
    schedulers_dict = {
        "step": torch.optim.lr_scheduler.StepLR,
        "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
        "cosine_warmup": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
        "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    }

    scheduler = schedulers_dict[scheduler_kwargs.pop("type")]
    return scheduler(optimizer, **scheduler_kwargs)


def get_finetuning_config(type, **kwargs) -> PeftConfig:
    """
    Get the finetuning config for a given finetuning approach.

    Parameters
    ----------
    type : str
        The finetuning approach to use ('lora', 'prefix-tuning', or 'prompt-adaption').
    **kwargs
        Additional arguments to pass to the finetuning config.

    Returns
    -------
    config : PeftConfig
    """
    assert (
        type in FINETUNING_DEFAULT_DICT.keys()
    ), f"Finetuning approach {type} not supported. Choose from {FINETUNING_DEFAULT_DICT.keys()}"

    if type == "lora":
        settings = {**FINETUNING_DEFAULT_DICT["lora"], **kwargs}
        return LoraConfig(**settings)
    elif type == "prefix-tuning":
        settings = {**FINETUNING_DEFAULT_DICT["prefix-tuning"], **kwargs}
        return PrefixTuningConfig(**settings)
    elif type == "prompt-adaption":
        # llama-adapter
        settings = {**FINETUNING_DEFAULT_DICT["prompt-adaption"], **kwargs}
        return AdaptionPromptConfig(**settings)
    raise ValueError(f"Finetuning approach {type} not supported.")


def get_latest_checkpoint_dir(train_dir, fixed_epoch=None):
    """
    Find the latest checkpoint directory in a training directory.

    Parameters
    ----------
    train_dir : str
        Path to the training directory.
    fixed_epoch : int or None
        If provided, only consider checkpoints from this epoch.

    Returns
    -------
    checkpoint_path : str or None
        Path to the latest checkpoint directory, or None if not found.
    """
    try:
        max_filename = find_highest_epoch_step(
            os.listdir(train_dir),
            regex="checkpoint_(\d+)_(\d+)",
            fixed_epoch=fixed_epoch,
        )
        return os.path.join(train_dir, max_filename)
    except:
        try:
            max_filename = find_highest_epoch_step(
                os.listdir(train_dir),
                regex="checkpoint_(\d+)_n(\d+)",
                fixed_epoch=fixed_epoch,
            )
            return os.path.join(train_dir, max_filename)
        except:
            return None


def get_module_names(model, pattern=r"\((\w+)\): Linear"):
    """
    Get the names of the modules in the model that match the pattern.

    Parameters
    ----------
    model : torch.nn.Module
        The model to inspect.
    pattern : str
        Regex pattern to match module names.

    Returns
    -------
    target_modules : list of str
        List of unique module names matching the pattern.
    """
    model_modules = str(model.modules)
    linear_layer_names = re.findall(pattern, model_modules)

    names = []
    for name in linear_layer_names:
        names.append(name)
    target_modules = list(set(names))
    return target_modules


def create_id(name: str) -> str:
    """
    Create a unique identifier composed of the model name and the initialization time.

    Parameters
    ----------
    name : str
        The name of the pre-trained model.

    Returns
    -------
    id_str : str
        Unique identifier string.
    """
    current_time = str(time.time()).split(".")[0]
    return f"{name}_{current_time}"


def as_dict(**kwargs):
    """
    Return the keyword arguments as a dictionary.
    """
    return kwargs


def get_nb_trainable_parameters(self) -> tuple[int, int]:
    """
    Returns the number of trainable parameters and the number of all parameters in the model.

    Parameters
    ----------
    self : nn.Module
        The model instance.

    Returns
    -------
    trainable_params : int
        Number of trainable parameters.
    all_param : int
        Total number of parameters.
    """
    # get_nb_trainable_parameters is only defined for peft models, but not for normal transformers models
    # taken from https://github.com/huggingface/peft/blob/v0.11.0/src/peft/peft_model.py#L569
    trainable_params = 0
    all_param = 0
    for _, param in self.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            if hasattr(param, "element_size"):
                num_bytes = param.element_size()
            elif not hasattr(param, "quant_storage"):
                num_bytes = 1
            else:
                num_bytes = param.quant_storage.itemsize
            num_params = num_params * 2 * num_bytes

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def set_default_chat_template(tokenizer):
    """
    Explicitly sets the default chat template for the tokenizer to avoid a Hugging Face warning.

    Parameters
    ----------
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer to set the chat template for.
    """
    # tokenizer.chat_template = tokenizer.default_chat_template
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{ message.content }}{{ eos_token }}{% endfor %}"
        logging.info(f"Setting default chat template: {tokenizer.chat_template}")
    else:
        logging.info(
            f"Tokenizer already has a chat template: {tokenizer.chat_template}"
        )
