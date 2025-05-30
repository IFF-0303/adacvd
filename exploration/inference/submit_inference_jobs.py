import itertools
import os
import random
import sys
from os.path import join

import yaml
from scripts.condor.training.submit_sweep import create_submission_file

from exploration.evaluation import evaluation_utils
from exploration.evaluation.evaluation_utils import find_three_digit_directories
from pandora.data.ukb_data_utils import RESULTS_PATH

if __name__ == "__main__":

    ports = range(29500, 29500 + 1000)

    base_inference_dir_root = "/fast/groups/hfm-users/pandora-med-box/results/2025_03_10_flexible_model_inference/2025_03_10_flexible_from_full/"

    base_inference_dirs = [
        os.path.join(base_inference_dir_root, x)
        for x in [  #  [f"model_{str(x).zfill(3)}" for x in range(48)]
            # "model_000",
            "model_",
            # "model_002",
            # "model_003",
            # "model_002",
            # "model_032",
        ]  # find_three_digit_directories(base_inference_dir_root)
    ]

    for base_inference_dir in base_inference_dirs:
        print(base_inference_dir)

        # create submission file, and submit condor job
        for inference_dir in os.listdir(base_inference_dir):
            print(inference_dir)
            inference_dir = join(base_inference_dir, inference_dir)

            if os.path.isdir(inference_dir):
                port = random.choice(ports)
                condor_arguments = f"launch --main_process_port {port} exploration/frederike/inference/inference.py --inference_dir={inference_dir} --device=cuda"
                submission_file = f"submission_file.sub"

                condor_settings = dict(
                    request_memory=200000,
                    request_cpus=8,
                    getenv=True,
                    bid=100,
                    request_gpus=2,
                    memory_gpus=51000,
                )
                condor_settings["arguments"] = condor_arguments
                condor_settings["executable"] = sys.executable.replace(
                    "python", "accelerate"
                )

                create_submission_file(
                    inference_dir, condor_settings, filename=submission_file
                )
                bid = condor_settings["bid"]
                os.system(
                    f"condor_submit_bid {bid} "
                    f"{join(inference_dir, submission_file)}"
                )
