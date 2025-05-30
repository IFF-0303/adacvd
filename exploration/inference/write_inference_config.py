import os

import yaml

from exploration.evaluation.evaluation_utils import find_three_digit_directories
from scripts.training.submit_sweep import (
    create_combinations,
    extract_final_layers,
    replace_values,
)

# model to use for inference
base_model_dir_root = (
    "/fast/groups/hfm-users/pandora-med-box/results/2025_03_10_flexible_from_full/"
)

# where to save the results
results_base_dir = (
    "/fast/groups/hfm-users/pandora-med-box/results/2025_03_10_flexible_model_inference"
)

# which configs to use for inference
# config_base_dir = None  # "/fast/groups/hfm-users/pandora-med-box/results/2024_11_30_data_efficiency_fixed_features"
sweep_settings_file = "config/training/feature_combinations.yaml"
train_settings_file = "config/training/train_settings.yaml"

# base_model_dirs = [
#     os.path.join(base_model_dir_root, x)
#     for x in [  # [  #  [f"model_{str(x).zfill(3)}" for x in range(48)]
#         # "000",
#         # "001",
#         # "002",
#         # "003",
#         # "008",
#         "018",
#     ]
#     # find_three_digit_directories(base_model_dir_root)
# ]

base_model_dirs = [base_model_dir_root]

for base_model_dir in base_model_dirs:

    inference_config = {
        "base_model_dir": str(base_model_dir),
        "data": {
            "path": "prompt_parts/ukb_2024_02/prompt_parts.parquet",
            "split": "config/splits/split_2025_02_17_test_100000.json",
            "subset": "subsets/ukb_2024_02/MACE_ADO_EXTENDED_no_previous_target.json",
            "eval_subset": True,
            "target_name": "MACE_ADO_EXTENDED_10y",
            "zeroshot_prompt": False,
            "shuffle": False,
            # "feature_config": {},
        },
        "inference": {"eval_batch_size": 128},  # "fixed_epoch": 1},
    }

    ending = base_model_dir.split(base_model_dir_root)[-1].replace("/", "")
    base_run_name = os.path.basename(os.path.normpath(base_model_dir_root))
    inference_dir = os.path.join(results_base_dir, base_run_name, "model_" + ending)

    # new ----
    with open(train_settings_file, "r") as f:
        train_settings = yaml.safe_load(f)

    with open(sweep_settings_file, "r") as f:
        sweep_settings = yaml.safe_load(f)

    final_layer_dict = extract_final_layers(sweep_settings)
    combination_dicts = create_combinations(final_layer_dict)

    for i, settings in enumerate(combination_dicts):

        folder_name = str(i).zfill(3)

        new_settings = replace_values(train_settings, settings)

        # specify which settings to use for inference
        # currently: only the feature config
        inference_config["data"]["feature_config"] = new_settings["data"][
            "feature_config"
        ].copy()

        # inference batch size dependent on feature config
        if len(inference_config["data"]["feature_config"]["feature_groups"]) <= 3:
            inference_config["inference"]["eval_batch_size"] = 64
        if len(inference_config["data"]["feature_config"]["feature_groups"]) == 4:
            inference_config["inference"]["eval_batch_size"] = 8
        else:
            inference_config["inference"]["eval_batch_size"] = 4

        # # add the training settings to inference_config
        inference_config["train_settings"] = {}
        inference_config["train_settings"]["model"] = train_settings["model"].copy()
        inference_config["train_settings"]["training"] = train_settings[
            "training"
        ].copy()

        dir_to_save = os.path.join(inference_dir, folder_name)
        os.makedirs(dir_to_save, exist_ok=True)
        with open(os.path.join(dir_to_save, "inference_settings.yaml"), "w") as f:
            yaml.dump(inference_config, f)
        print(f"Saved inference settings for {dir_to_save}")
