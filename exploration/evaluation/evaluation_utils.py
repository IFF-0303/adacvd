import glob
import os
import re

import pandas as pd

from pandora.data import ukb_data_utils


def find_highest_epoch_step(
    filenames, regex=r"evals_(\d+)_(\d+)\.csv", fixed_epoch=None
):
    max_epoch = -1
    max_step = -1
    max_filename = ""
    for filename in filenames:
        if isinstance(filename, str):
            match = re.match(regex, filename)
        else:
            match = re.match(regex, str(filename.name))
        if match:
            epoch = int(match.group(1))
            if fixed_epoch and epoch != fixed_epoch:
                continue
            step = int(match.group(2))
            if epoch > max_epoch or (epoch == max_epoch and step > max_step):
                max_epoch = epoch
                max_step = step
                max_filename = filename
    if max_epoch == -1:
        raise ValueError(f"No valid filename found in.")
    return max_filename


def find_latest_evals(results_path):
    csv_paths = glob.glob(os.path.join(results_path, "evals_*.csv"))
    csv_files = [os.path.basename(path) for path in csv_paths]
    if len(csv_files) == 0:
        raise ValueError("No evaluation files found in the specified path.")
    highest_file, highest_epoch, highest_step = find_highest_epoch_step(csv_files)
    print(f"Path '{results_path}': Latest evaluation file: {highest_file}")
    return os.path.join(results_path, highest_file)


def load_medical_risk_scores(
    framingham: bool = True,
    pcrs: bool = True,
    prevent: bool = True,
    qrisk: bool = True,
    score: bool = True,
):
    risk_scores = pd.DataFrame()

    if not (framingham or pcrs or prevent or qrisk or score):
        raise ValueError("At least one risk score must be loaded.")

    if framingham:
        fram = (
            pd.read_csv(
                ukb_data_utils.ASSETS_PATH / "risk_scores/framingham_risk_scores.csv"
            )
            .set_index("eid")
            .rename(columns={"0": "Framingham Risk Score"})
        )
        risk_scores = pd.concat([risk_scores, fram], axis=1)
    if pcrs:
        pcrs = (
            pd.read_csv(
                ukb_data_utils.ASSETS_PATH / "risk_scores/pooled_cohort_risk_scores.csv"
            )
            .set_index("eid")
            .rename(columns={"0": "ACC/AHA Risk Score"})
        )
        risk_scores = pd.concat([risk_scores, pcrs], axis=1)
    if prevent:
        prevent = (
            pd.read_csv(
                ukb_data_utils.ASSETS_PATH / "risk_scores/prevent_risk_scores.csv"
            )
            .set_index("eid")
            .rename(columns={"0": "PREVENT Risk Score"})
        )
        risk_scores = pd.concat([risk_scores, prevent], axis=1)
    if qrisk:
        qrisk = (
            pd.read_csv(ukb_data_utils.ASSETS_PATH / "risk_scores/qrisk_scores.csv")
            .set_index("eid")
            .rename(columns={"0": "QRISK Risk Score"})
        )
        risk_scores = pd.concat([risk_scores, qrisk], axis=1)

    if score:
        score = (
            pd.read_csv(ukb_data_utils.ASSETS_PATH / "risk_scores/score_scores.csv")
            .set_index("eid")
            .rename(columns={"0": "SCORE Risk Score"})
        )
        risk_scores = pd.concat([risk_scores, score], axis=1)

    return risk_scores


def find_three_digit_directories(base_dir):
    three_digit_directories = []
    if os.path.isdir(base_dir):
        for dir_name in os.listdir(base_dir):
            if re.match(r"\d{3}", dir_name) and os.path.isdir(
                os.path.join(base_dir, dir_name)
            ):
                three_digit_directories.append(dir_name)
    else:
        print(f"Base directory '{base_dir}' not found.")
    three_digit_directories.sort(key=lambda x: int(x))
    return three_digit_directories
