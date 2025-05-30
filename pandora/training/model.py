import logging
import os
import pickle
import shutil
from copy import deepcopy
from datetime import datetime, timedelta
from os.path import join
from pathlib import Path
from typing import List, Union

import evaluate
import pandas as pd
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed
from peft import AutoPeftModelForCausalLM, get_peft_model
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from pandora.data.ukb_data_utils import WANDB_ENTITY
from pandora.training.dataset import get_in_text_tokenization
from pandora.training.utils import (
    RuntimeLimits,
    as_dict,
    create_id,
    get_finetuning_config,
    get_nb_trainable_parameters,
    get_optimizer_from_kwargs,
    get_scheduler_from_kwargs,
    set_default_chat_template,
)


class HuggingfaceModel:
    """
    Wrapper for Hugging Face Causal Language Models with training and evaluation utilities.

    Handles model initialization, fine-tuning (with PEFT adapters), optimizer and scheduler setup,
    checkpointing, and evaluation. Uses the Accelerate library for distributed training and logging.

    Attributes:
        MODEL_INPUT_VARS (list): List of input variable names for the model.
        config (dict): Model and training configuration.
        device (str): Device to use ('cuda' or 'cpu').
        model_kwargs (dict): Model-specific keyword arguments.
        optimizer_kwargs (dict): Optimizer configuration.
        scheduler_kwargs (dict): Scheduler configuration.
        network (torch.nn.Module): The underlying Hugging Face model.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the model.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        scheduler (torch.optim.lr_scheduler): Scheduler instance.
        epoch (int): Current epoch.
        model_dir (str): Directory for saving/loading the model.
        torch_dtype (torch.dtype): Precision for model weights.
        accelerator (Accelerator): Accelerate instance for distributed training.
        id (str): Unique identifier for the model instance.
    """

    MODEL_INPUT_VARS = ["input_ids", "attention_mask", "labels"]

    def __init__(
        self,
        model_dir: str = None,
        config: dict = None,
        device: str = "cuda",
        log_with="wandb",
        **kwargs,
    ):
        """
        Initialize the HuggingfaceModel.

        Args:
            model_dir (str, optional): Directory to load the model from.
            config (dict, optional): Model and training configuration.
            device (str): Device to use ('cuda' or 'cpu').
            log_with (str): Logging backend for Accelerate.
            **kwargs: Additional keyword arguments.
        """
        self.config = deepcopy(config)
        self.device = device
        if self.config is not None:
            self.model_kwargs = self.config["model"]

        self.optimizer_kwargs = None
        self.scheduler_kwargs = None

        self.network = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.epoch = 0
        self.model_dir = model_dir

        precision = self.model_kwargs.get("precision", "auto")
        precision_dtype_dict = {
            "16": torch.float16,
            "16b": torch.bfloat16,
            "32": torch.float32,
            "auto": "auto",
        }
        if precision not in precision_dtype_dict.keys() and precision is not None:
            raise NotImplementedError(
                f"Precision {precision} is not supported. Supported values: {precision_dtype_dict.keys()}"
            )
        self.torch_dtype = precision_dtype_dict[str(precision)] if precision else None

        kwargs = [InitProcessGroupKwargs(timeout=timedelta(seconds=2400))]
        self.accelerator = Accelerator(
            log_with=log_with,
            cpu=self.device == "cpu",
            kwargs_handlers=kwargs,
        )
        self.accelerator.free_memory()

        if model_dir is not None:
            self.load_model(self.model_dir)
        else:
            self._initialize_network()
            self.id = create_id(self.model_kwargs["name"])

        self.initialize_optimizer_and_scheduler(
            config["training"]["optimizer"],
            config["training"]["scheduler"],
        )

        dtypes_trainable_params = set(
            [p.dtype for p in self.network.parameters() if p.requires_grad]
        )
        dtypes_non_trainable_params = set(
            [p.dtype for p in self.network.parameters() if not p.requires_grad]
        )

        logging.info(
            f"Trainable parameters: {dtypes_trainable_params}. Non-trainable parameters: {dtypes_non_trainable_params}"
        )

    def __call__(self, prompt: Union[str, List[str]], *args, **kwargs):
        """
        Run inference on the model.

        Args:
            prompt (str or List[str]): Prompt(s) in natural language.

        Returns:
            List[str]: Completions of the prompt(s) in natural language.
        """
        self.network.eval()
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        with torch.no_grad():
            network = self.accelerator.unwrap_model(self.network)  # needed!
            inputs = inputs.to(network.device)  # device changes after unwrapping
            outputs = network.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=5,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            return self.tokenizer.batch_decode(
                outputs.detach().cpu().numpy(), skip_special_tokens=True
            )

    def _initialize_network(self):
        """
        Initialize the Hugging Face model and tokenizer, optionally with PEFT adapters for fine-tuning.
        """
        model_name = self.model_kwargs["name"]
        is_finetuning = self.model_kwargs.get("finetuning", False)
        if is_finetuning:
            extra_args = {}
            if model_name in ["google/gemma-2-9b-it", "google/gemma-2-2b-it"]:
                extra_args["attn_implementation"] = "eager"
        else:
            extra_args = {}

        if self.model_dir is not None:
            logging.warning(
                "Loading model from scratch even though model_dir is provided."
            )

        self.network = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            **extra_args,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if model_name == "gpt2":
            set_default_chat_template(self.tokenizer)

        if is_finetuning:

            if model_name == "microsoft/Phi-3.5-mini-instruct":
                self.model_kwargs["finetuning"]["target_modules"] = ["qkv_proj"]

            autocast_adapter_dtype = self.model_kwargs["autocast_adapter_dtype"]

            peft_config = get_finetuning_config(**self.model_kwargs["finetuning"])
            self.network = get_peft_model(
                self.network,
                peft_config,
                autocast_adapter_dtype=autocast_adapter_dtype,
            )
            self.network.print_trainable_parameters()

    def initialize_optimizer_and_scheduler(self, optimizer_kwargs, scheduler_kwargs):
        """
        Initializes the optimizer and scheduler with optimizer_kwargs and scheduler_kwargs, respectively.
        The kwargs are stored in attributes as metadata for saving/loading purposes.

        Args:
            optimizer_kwargs (dict): Optimizer configuration.
            scheduler_kwargs (dict): Scheduler configuration.
        """
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer = get_optimizer_from_kwargs(
            self.network.parameters(), **optimizer_kwargs
        )
        self.scheduler_kwargs = scheduler_kwargs
        self.scheduler = get_scheduler_from_kwargs(self.optimizer, **scheduler_kwargs)

    def save_model(
        self,
        model_dir: str,
    ):
        """
        Save the model, tokenizer, optimizer, scheduler, and metadata to the specified directory.

        Args:
            model_dir (str): Directory to save the model and related files.
        """
        model_dir = Path(model_dir)
        metadata = {
            "model_kwargs": self.model_kwargs,
            "epoch": self.epoch,
            "id": self.id,
        }

        if self.optimizer_kwargs is not None:
            metadata["optimizer_kwargs"] = self.optimizer_kwargs
        if self.scheduler_kwargs is not None:
            metadata["scheduler_kwargs"] = self.scheduler_kwargs

        self.accelerator.save_state(model_dir)
        with open(join(model_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)

        # save model and tokenizer for inference
        self.accelerator.unwrap_model(self.network).save_pretrained(
            model_dir / "network"
        )
        self.tokenizer.save_pretrained(model_dir / "tokenizer")
        logging.info(f"Model and tokenizer saved to '{model_dir}'")

    def load_model(self, model_dir: str):
        """
        Load an existing model, optimizer, and scheduler from the given directory.

        Args:
            model_dir (str): Path to the saved model.
        """
        logging.info(f"Loading model from '{model_dir}'")
        logging.info(f"Torch Dtype: {self.torch_dtype}")
        self.network = AutoPeftModelForCausalLM.from_pretrained(
            os.path.join(model_dir, "network"),
            torch_dtype=self.torch_dtype,
            is_trainable=True,
        )
        logging.info(f"Loaded.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(model_dir, "tokenizer"), padding_side="left"
        )

        with open(join(model_dir, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)
        self.id = metadata["id"]
        self.model_kwargs = metadata["model_kwargs"]
        self.epoch = metadata["epoch"]

        self.accelerator.load_state(model_dir)

    def train(
        self,
        train_dir,
        train_loader,
        test_loader,
        validation_loader,
        runtime_limits: RuntimeLimits,
        trackers_kwargs=None,
        extended_dataloaders={},
        **kwargs,
    ):
        """
        Train the model using the provided dataloaders and runtime limits.

        Args:
            train_dir (str): Directory to save checkpoints and logs.
            train_loader (DataLoader): Training data loader.
            test_loader (DataLoader): Test data loader.
            validation_loader (DataLoader): Validation data loader.
            runtime_limits (RuntimeLimits): Object to control training duration.
            trackers_kwargs (dict, optional): Additional tracker configuration.
            extended_dataloaders (dict, optional): Additional dataloaders for evaluation.
            **kwargs: Additional training parameters.
        """
        # prepare model components for training
        (
            self.network,
            train_loader,
            test_loader,
            validation_loader,
            self.optimizer,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.network,
            train_loader,
            test_loader,
            validation_loader,
            self.optimizer,
            self.scheduler,
        )

        for key, dataloader in extended_dataloaders.items():
            extended_dataloaders[key] = self.accelerator.prepare(dataloader)

        self.accelerator.print(self.network)

        if trackers_kwargs is not None:
            self.accelerator.init_trackers(
                config=self.config,
                **trackers_kwargs,
                init_kwargs={
                    "wandb": {
                        "name": datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f"),
                        "entity": WANDB_ENTITY,
                        # "mode": "disabled",
                    }
                },
            )

        nb_trainable_params = get_nb_trainable_parameters(
            self.accelerator.unwrap_model(self.network)
        )
        self.accelerator.log(
            {
                "trainable_n_parameters": nb_trainable_params[0],
                "total_n_parameters": nb_trainable_params[1],
                "trainable_n_parameters_percent": nb_trainable_params[0]
                / nb_trainable_params[1],
                "train_dataset_size": len(train_loader.dataset),
            }
        )

        set_seed(kwargs.get("random_seed", 0))  # necessary for reproducibility

        n = 0
        full_eval_step_n = kwargs.get("full_eval_step_n", 400_000)  # TODO
        max_length = 0
        best_roc_auc_validation = 0
        best_roc_auc_test = 0
        previous_checkpoint_dir = None
        previous_checkpoint_dir_val = None

        while not runtime_limits.limits_exceeded(self.epoch):
            self.epoch += 1
            self.network.train()
            total_loss = 0  # summed loss in epoch
            epoch_losses = (
                []
            )  # losses in current epoch, each loss is averaged over the batch

            kwargs.get("checkpoint_steps", 10000)
            for step, batch in enumerate(tqdm(train_loader)):
                # batch to cpu
                batch = {key: val.to("cpu") for key, val in batch.items()}
                if step % kwargs.get("eval_steps", 50) == 0:
                    logging.info(
                        f"Evaluate model at step {step} in epoch {self.epoch}. Number of training points so far: {n}."
                    )

                    loader = validation_loader
                    base_name = "validation"

                    if len(extended_dataloaders) > 0:
                        loaders = {base_name: loader, **extended_dataloaders}
                    else:
                        loaders = {base_name: loader}

                    for name, loader in loaders.items():
                        metrics, evals = evaluate_step(self, loader)
                        log_dict = {
                            **metrics,
                        }

                        log_dict = {
                            f"{key}_{name}": val for key, val in log_dict.items()
                        }
                        evals.to_csv(f"{train_dir}/evals_valid_{self.epoch}_n{n}.csv")

                        self.accelerator.log(log_dict)

                        if name == base_name:
                            roc_auc = "roc_auc_validation"
                            if (log_dict[roc_auc] > best_roc_auc_validation) and (
                                kwargs.get("save_model", True)
                            ):
                                checkpoint_name = (
                                    f"{train_dir}/checkpoint_{self.epoch}_n{n}"
                                )
                                logging.info(
                                    f"New best roc_auc: {log_dict[roc_auc]}, saving model to {checkpoint_name}"
                                )
                                self.accelerator.wait_for_everyone()
                                if self.accelerator.is_main_process:
                                    self.save_model(checkpoint_name)
                                    if (
                                        previous_checkpoint_dir is not None
                                        and os.path.isdir(previous_checkpoint_dir)
                                    ):
                                        shutil.rmtree(previous_checkpoint_dir)
                                previous_checkpoint_dir = checkpoint_name
                                best_roc_auc_validation = log_dict[roc_auc]

                                evals.to_csv(f"{train_dir}/evals_best_valid.csv")

                                self.accelerator.wait_for_everyone()

                # batch to GPU
                batch = {key: val.to(self.device) for key, val in batch.items()}

                # training step
                self.network.train()

                outputs = self.network(
                    **{key: batch[key] for key in self.MODEL_INPUT_VARS}
                )
                loss = outputs.loss
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                log_dict = {
                    "lr": self.scheduler.get_last_lr()[0],
                }
                # gather_for_metrics for loss
                loss_all_gpus = self.accelerator.gather_for_metrics(loss)

                total_loss += loss_all_gpus.detach().float().sum()
                epoch_losses.append(loss_all_gpus.detach().float().mean())

                if step != 0:
                    log_dict["training_loss_avg"] = sum(epoch_losses[-50:]) / min(
                        len(epoch_losses), 50
                    )
                log_dict["training_loss_step"] = epoch_losses[-1]

                self.accelerator.log(log_dict)

                n += (
                    batch["input_ids"].shape[0] * self.accelerator.num_processes
                )  # only approximate due to multi-gpu

                if batch["input_ids"].shape[1] > max_length:
                    max_length = batch["input_ids"].shape[1]

                if step % 50 == 0:
                    self.accelerator.log(
                        {
                            "step": step,
                            "epoch": self.epoch,
                            "n": n,
                            "max_length": max_length,
                        }
                    )

                del (
                    batch,
                    outputs,
                )
                # evaluate model only after at least x data points
                if n >= full_eval_step_n:
                    logging.info(
                        f"Evaluate model in {self.epoch}. Number of training points so far: {n}."
                    )
                    metrics, evals = evaluate_step(self, test_loader)
                    log_dict = {
                        **metrics,
                    }
                    logging.info(log_dict)
                    self.accelerator.log(log_dict)
                    evals.to_csv(f"{train_dir}/evals_{self.epoch}_n{n}.csv")
                    full_eval_step_n += kwargs.get("full_eval_step_n", 100_000)

                    if (log_dict["roc_auc"] > best_roc_auc_test) and (
                        kwargs.get("save_model", True)
                    ):
                        checkpoint_name = f"{train_dir}/checkpoint_{self.epoch}_n{n}"
                        logging.info(
                            f"New best roc_auc: {log_dict['roc_auc']}, saving model to {checkpoint_name}"
                        )
                        self.accelerator.wait_for_everyone()
                        if self.accelerator.is_main_process:
                            self.save_model(checkpoint_name)
                            if (
                                previous_checkpoint_dir_val is not None
                                and os.path.isdir(previous_checkpoint_dir_val)
                            ):
                                shutil.rmtree(previous_checkpoint_dir_val)
                        previous_checkpoint_dir_val = checkpoint_name
                        best_roc_auc_test = log_dict["roc_auc"]

                        evals.to_csv(f"{train_dir}/evals_best_test.csv")

                        self.accelerator.wait_for_everyone()

            logging.info(f"Epoch {self.epoch} finished.")
            epoch_loss = total_loss / (
                len(train_loader) * self.accelerator.num_processes
            )
            logging.info(f"Epoch loss: {epoch_loss}")
            self.accelerator.log({"epoch_loss": epoch_loss})

        if kwargs.get("save_model", True):
            self.save_model(f"{train_dir}/checkpoint_{self.epoch}_n{n}")
            logging.info(f"Model saved.")

        logging.info(f"Evaluate model after final epoch {self.epoch}.")
        metrics, evals = evaluate_step(self, test_loader)
        log_dict = {
            **metrics,
        }
        logging.info(log_dict)
        self.accelerator.log(log_dict)
        evals.to_csv(f"{train_dir}/evals_{self.epoch}_n{n}.csv")

        logging.info(f"Training finished.")


def evaluate_step(model: HuggingfaceModel, validation_loader):
    """
    Evaluate the model on the provided validation loader and compute metrics.

    Args:
        model (HuggingfaceModel): The model to evaluate.
        validation_loader (DataLoader): DataLoader for validation data.

    Returns:
        tuple: (metric_dict, evals)
            metric_dict (dict): Dictionary of computed metrics.
            evals (pd.DataFrame): DataFrame with predictions and true labels.
    """
    model.network.eval()
    glue_metric = evaluate.load("glue", "mrpc")
    roc_auc_metric = evaluate.load("roc_auc")

    n_pos = 0
    n_pos_pred = 0
    y_true = []
    y_pred = []
    y_pred_scores = []
    eids = []
    eval_loss_sum = 0

    for step, batch in enumerate(tqdm(validation_loader)):

        # tensors need to be of the same size when using `gather_for_metrics`
        # need to pad them across processes due to dynamic padding
        padded_batch = {}
        for key in batch.keys():
            if key in model.MODEL_INPUT_VARS:
                pad_index = (
                    model.tokenizer.pad_token_id
                    if key in ["input_ids", "labels"]
                    else 0  # attention mask
                )
                padded_batch[key] = model.accelerator.pad_across_processes(
                    batch[key],
                    dim=1,
                    pad_index=pad_index,
                    pad_first=True,  # left padding
                )
                padded_batch[key].to(model.device)
            else:
                padded_batch[key] = batch[key]
        del batch

        with torch.no_grad():
            outputs = model.network(
                **{key: padded_batch[key] for key in model.MODEL_INPUT_VARS},
                use_cache=True,
            )
        preds = model.accelerator.gather_for_metrics(outputs)
        batch_ = model.accelerator.gather_for_metrics(padded_batch)

        # loss
        eval_loss = preds.loss.detach().cpu().numpy()
        eval_loss_sum += (
            eval_loss.sum()
        )  # sum over all processes, each loss is already averaged over the batch

        del padded_batch
        del outputs

        pos_token_id = model.tokenizer.convert_tokens_to_ids("Yes")
        neg_token_id = model.tokenizer.convert_tokens_to_ids("No")

        completion_positions = torch.logical_and(
            batch_["labels"] != -100,
            batch_["labels"] != model.tokenizer.pad_token_id,
        )  # shape (batch_size, seq_len)
        assert torch.all(completion_positions.sum(1) == 1)

        # -1 because the logits are for the next position
        completion_positions_logits = torch.roll(
            completion_positions, shifts=-1, dims=-1
        )
        bs = batch_["labels"].shape[0]  # batch size

        logits = preds.logits[
            :, :, torch.tensor([neg_token_id, pos_token_id])
        ]  # position 0 = neg, 1 = pos --> argmax corresponds to label

        logits_at_completion = logits[completion_positions_logits].float()

        probs = torch.softmax(logits_at_completion, dim=-1)  # shape (batch_size, 2)
        assert probs.shape == (bs, 2)

        completion_labels = batch_["labels"][completion_positions]
        assert completion_labels.shape == (bs,)  # shape (batch_size, 1)
        pos_labels = (completion_labels == pos_token_id).long()

        glue_metric.add_batch(
            predictions=probs.argmax(-1).cpu(), references=pos_labels.cpu()
        )
        roc_auc_metric.add_batch(
            references=pos_labels.cpu(), prediction_scores=probs[:, 1].cpu()
        )

        n_pos += pos_labels.tolist().count(1)
        n_pos_pred += probs.argmax(-1).tolist().count(1)

        y_true.extend(pos_labels.cpu().numpy().tolist())
        y_pred.extend(probs.argmax(-1).cpu().numpy().tolist())
        y_pred_scores.extend(probs[:, 1].to(torch.float32).cpu().numpy().tolist())
        eids.extend(batch_["eid"].cpu().numpy().tolist())

        del (
            completion_positions,
            completion_positions_logits,
            logits_at_completion,
            logits,
            probs,
            completion_labels,
            pos_labels,
            batch_,
        )

    model.accelerator.wait_for_everyone()

    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    balanced_acc = balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    prec = precision_score(y_true=y_true, y_pred=y_pred)
    rec = recall_score(y_true=y_true, y_pred=y_pred)

    try:
        roc_auc_s = roc_auc_score(y_true=y_true, y_score=y_pred_scores)
    except:
        print("Could not compute roc_auc_score.")
        roc_auc_s = None

    glue = glue_metric.compute()
    try:
        roc_auc = roc_auc_metric.compute()
    except:
        print("Could not compute roc_auc_metric.")
        roc_auc = {"roc_auc": None}
    share_pos = n_pos / len(validation_loader.dataset)
    share_pos_pred = n_pos_pred / len(validation_loader.dataset)
    avg_eval_loss = torch.tensor(eval_loss_sum) / (
        len(validation_loader) * model.accelerator.num_processes
    )

    metric_dict = {
        **glue,
        **roc_auc,
        "share_pos": share_pos,
        "share_pos_pred": share_pos_pred,
        "s_acc": acc,
        "s_balanced_acc": balanced_acc,
        "s_f1": f1,
        "s_roc_auc": roc_auc_s,
        "precision_score": prec,
        "recall_score": rec,
        "eval_loss_avg": avg_eval_loss,
    }

    evals = pd.DataFrame(
        {"eid": eids, "y_pred": y_pred, "y_pred_score": y_pred_scores, "y_true": y_true}
    )

    return metric_dict, evals
