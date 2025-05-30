import argparse
import itertools
import os
import sys
from os.path import join

import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        required=False,
        help="Base directory with config files",
        default="config/ukb_data/all_feature_groups.yaml",
    )
    parser.add_argument(
        "--num_samples",
        required=False,
        help="Number of samples to process",
        default=None,
    )

    args = parser.parse_args()
    return args


def create_submission_file(train_dir, condor_settings, filename="submission_file.sub"):
    lines = []
    lines.append(f'executable = {condor_settings["executable"]}\n')
    lines.append(f"getenv = True\n")
    lines.append(f'request_cpus = {condor_settings["request_cpus"]}\n')
    lines.append(f'request_memory = {condor_settings["request_memory"]}\n')
    lines.append(f'arguments = {condor_settings["arguments"]}\n')
    lines.append(f"error = /home/fluebeck/biobank/logs/$(ClusterId).err\n")
    lines.append(f"output = /home/fluebeck/biobank/logs/$(ClusterId).out\n")
    lines.append(f"log = /home/fluebeck/biobank/logs/$(ClusterId).log\n")
    lines.append("queue")

    with open(join(train_dir, filename), "w") as f:
        for line in lines:
            f.write(line)


if __name__ == "__main__":
    args = parse_args()
    config_file = args.config_file
    num_samples = args.num_samples
    config_dir = os.path.dirname(config_file)

    condor_arguments = f"pandora/data/preprocess.py --config_path={config_file}"
    if num_samples is not None:
        condor_arguments += f" --num_samples={num_samples}"

    submission_file = f"submission_file.sub"

    condor_settings = dict(request_memory=500000, request_cpus=64, getenv=True, bid=35)
    condor_settings["arguments"] = condor_arguments
    condor_settings["executable"] = sys.executable

    create_submission_file(config_dir, condor_settings, filename=submission_file)
    bid = condor_settings["bid"]
    os.system(f"condor_submit_bid {bid} " f"{join(config_dir, submission_file)}")
