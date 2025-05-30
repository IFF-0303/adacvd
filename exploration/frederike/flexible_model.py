import argparse
import logging
import socket
from os.path import join

import datasets
import torch
import yaml
from accelerate.utils import set_seed

from pandora.training.dataset import build_train_test_loader, load_and_prepare_dataset
from pandora.training.model import HuggingfaceModel, build_model_from_kwargs
from pandora.training.utils import RuntimeLimits, get_latest_checkpoint_dir

logging.basicConfig(format="%(asctime)s – %(levelname)s: %(message)s")
logging.getLogger().setLevel(logging.INFO)

MAX_LENGTH = 256

RUNTIME_LIMITS = {
    "max_time_per_run": 36000,
    "max_epochs_per_run": 20,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_dir",
        help="Path to training directory. Should contain a 'train_settings.yaml' file",
        default="config/training",
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
    logging.info(f"args: {args.__dict__}")
    logging.info(f"Node: {socket.gethostname()}")
    if args.device == "cuda":
        logging.info(f"GPU: {torch.cuda.get_device_name()}")

    set_seed(0)

    with open(join(args.train_dir, "train_settings.yaml"), "r") as f:
        config = yaml.safe_load(f)

    dataset = load_and_prepare_dataset(
        data_config=config["data"], num_samples=args.num_samples
    )

    print("")
    # steps
    # data selection (filter certain subgroups, ...)
    # load data (features, prompt parts)


if __name__ == "__main__":
    main()
