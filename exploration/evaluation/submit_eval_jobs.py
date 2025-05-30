import argparse
import itertools
import os
import sys
from os.path import join
from pathlib import Path

import orjson
import yaml

from exploration.evaluation import evaluation, evaluation_utils
from pandora.data.ukb_data_utils import ASSETS_PATH, RESULTS_PATH


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_dir",
        help="Path to evaluation directory. This is where all evaluation results are saved.",
    )
    parser.add_argument(
        "--base_prediction_dir",
        help="Path to a base prediction directory. Should contain directories with predictions.",
        default=None,
    )
    parser.add_argument(
        "--subgroup",
        help="Value of the subgroup to evaluate on. If None, evaluate on the whole dataset.",
        action="store_true",
    )

    parser.add_argument(
        "--evaluation_subset",
        help="Path to the evaluation subset.",
        default="evaluation_subsets/ukb_2024_02/test_subset_MACE_ADO_EXTENDED_no_previous_target.json",
    )
    parser.add_argument(
        "--target",
        help="Name of the target column",
        default="MACE_ADO_EXTENDED_10y",
    )
    parser.add_argument(
        "--add_risk_scores",
        help="Whether to add the risk scores or not",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()
    return args


def create_submission_file(dir, condor_settings, filename="submission_file.sub"):
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

    with open(join(dir, filename), "w") as f:
        for line in lines:
            f.write(line)


def submit_evaluation_job(
    prediction_dir: Path,
    eval_dir: Path,
    evaluation_subset: Path,
    target: str,
    subgroup: str = None,
):

    condor_arguments = f"exploration/frederike/evaluation/draft_evaluation.py --prediction_dir={prediction_dir} --eval_dir={eval_dir} --evaluation_subset={Path(evaluation_subset)} --target={target}"

    if subgroup is not None:
        condor_arguments += f" --subgroup={subgroup}"

    submission_file = f"submission_file_{i}.sub"
    condor_settings = dict(request_memory=100000, request_cpus=32, getenv=True, bid=16)
    condor_settings["arguments"] = condor_arguments
    condor_settings["executable"] = sys.executable
    create_submission_file(eval_dir, condor_settings, filename=submission_file)
    bid = condor_settings["bid"]
    os.system(f"condor_submit_bid {bid} " f"{join(eval_dir, submission_file)}")


if __name__ == "__main__":
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    base_prediction_dir = (
        Path(args.base_prediction_dir) if args.base_prediction_dir is not None else None
    )  # TODO: add to a config file
    if args.add_risk_scores:
        prediction_dirs = ["framingham", "pcrs", "prevent", "qrisk", "score"]
    else:
        prediction_dirs = []

    if base_prediction_dir is not None and base_prediction_dir.is_dir():
        try:
            evaluation.get_evals_path(base_prediction_dir)
            evaluation.get_settings(base_prediction_dir)
            prediction_dirs.append(base_prediction_dir)
            print(f"Found prediction directory: {base_prediction_dir}")
        except ValueError:
            print(f"Skipping directory {base_prediction_dir}")

    # find all prediction directories (contain at least one evals.csv file, and contains train_settings.yaml or inference_settings.yaml)
    if base_prediction_dir is not None:
        for dir in base_prediction_dir.rglob("*"):
            if dir.is_dir():
                try:
                    evaluation.get_evals_path(dir)
                    evaluation.get_settings(dir)
                    prediction_dirs.append(dir)
                    print(f"Found prediction directory: {dir}")
                except ValueError:
                    print(f"Skipping directory {dir}")

    # create submission file, and submit condor job
    for i, prediction_dir in enumerate(prediction_dirs):
        if True:
            # if "018" in str(prediction_dir):
            if not args.subgroup:
                # prediction dir relative to base_prediction_dir
                if base_prediction_dir is not None:
                    prediction_dir_i = prediction_dir.relative_to(base_prediction_dir)
                    eval_dir_i = eval_dir / prediction_dir_i
                    eval_dir_i.mkdir(parents=True, exist_ok=True)
                else:
                    eval_dir_i = eval_dir

                submit_evaluation_job(
                    prediction_dir=prediction_dir,
                    eval_dir=eval_dir_i,
                    evaluation_subset=args.evaluation_subset,
                    target=args.target,
                )
            else:
                with open(ASSETS_PATH / "subgroups/subgroups.json", "rb") as f:
                    subgroups = orjson.loads(f.read())  # TODO: add to a config file

                # settings = draft_evaluation.get_settings(prediction_dir)

                for subgroup in subgroups.keys():
                    skip = [
                        # "Ethnic_Background",
                        "Previous_MACE",
                        "Number_in_Household",
                        "Time_Employed_in_Current_Main_Job",
                        "Length_of_Working_Week",
                        "Job_Involves_Heavy_Manual_or_Physical_Work",
                        "Job_Involves_Night_Shift",
                        "Country_of_Birth_UK_Elsewhere",
                        "Home_Area_Population_Density",
                        "Year_Immigrated_to_UK",
                    ]
                    if subgroup in skip:
                        continue
                    # if subgroup not in ["Age_Group"]:
                    #     continue

                    print(subgroup)

                    # subgroup_trained_on = settings["data"].get("subgroup", None)
                    # if subgroup_trained_on is not None:
                    #     if subgroup not in subgroup_trained_on:
                    #         continue

                    submit_evaluation_job(
                        prediction_dir=prediction_dir,
                        eval_dir=eval_dir,
                        evaluation_subset=args.evaluation_subset,
                        target=args.target,
                        subgroup=subgroup,
                    )
