import argparse
import logging
from pathlib import Path

import numpy as np
import orjson
import pandas as pd
import yaml
from sklearn.utils import resample
from tqdm import tqdm

import pandora.utils.logger
from exploration.evaluation.evaluation_utils import (
    find_highest_epoch_step,
    load_medical_risk_scores,
)
from pandora.data.ukb_data_utils import ASSETS_PATH, RESULTS_PATH
from pandora.utils.metrics import compute_bootstrapped_metrics, compute_roc_curve_values


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction_dir",
        help="Path to the directory containing the evals_{*}.csv files and a {train/inference}_settings.yaml",
    )
    parser.add_argument(
        "--eval_dir",
        help="Path to evaluation directory. This is where all evaluation results are saved.",
    )
    parser.add_argument(
        "--subgroup",
        help="Value of the subgroup to evaluate on. If None, evaluate on the whole dataset.",
        default=None,
    )
    parser.add_argument(
        "--evaluation_subset",
        help="Path to the evaluation subset.",
        default="/fast/groups/hfm-users/pandora-med-box/assets/evaluation_subsets/ukb_2024_02/test_subset_MACE_ADO_EXTENDED_no_previous_target.json",
    )
    parser.add_argument(
        "--target",
        help="Name of the target column",
        default="MACE_ADO_EXTENDED_10y",
    )
    return parser.parse_args()


def get_settings(dir: Path) -> dict:
    """
    Return settings dictionary from directory
    (either train_settings.yaml or inference_settings.yaml).
    """
    if (dir / "train_settings.yaml").is_file():
        with open(dir / "train_settings.yaml", "r") as f:
            settings = yaml.load(f, Loader=yaml.CLoader)
    elif (dir / "inference_settings.yaml").is_file():
        with open(dir / "inference_settings.yaml", "r") as f:
            settings = yaml.load(f, Loader=yaml.CLoader)
    elif (dir / "config.yaml").is_file():
        with open(dir / "config.yaml", "r") as f:
            settings = yaml.load(f, Loader=yaml.CLoader)
    else:
        raise ValueError(f"No settings file found in directory {dir}.")
    return settings


def create_name_from_dir(dir: Path) -> str:
    """
    Create a name from the directory.
    """
    if RESULTS_PATH in dir.parents:
        name = dir.relative_to(RESULTS_PATH)  # TODO: replace / by _? or keep nested?
    else:
        name = dir.name
    return name


def get_evals_path(dir: Path) -> Path:
    # find latest evals.csv files
    files = list(dir.glob("*evals*.csv"))
    # logger.info(files)
    if len(files) == 0:
        raise ValueError(f"No evals*.csv files found in directory {dir}.")
    elif len(files) == 1:
        file = files[0]
    elif len(files) > 1:
        # check if "evals_best_test.csv" exists
        if (dir / "evals_best_test.csv").is_file():
            file = dir / "evals_best_test.csv"
        else:
            try:
                file = find_highest_epoch_step(files, regex=r"evals_(\d+)_(\d+)\.csv")
            except:
                file = find_highest_epoch_step(files, regex=r"evals_(\d+)_n(\d+)\.csv")
    return file


def evaluate(
    evals_subset: pd.DataFrame,
    evaluation_dir: Path,
    settings: dict,
    n_bootstrap_rounds: int = 5000,
):
    metrics = compute_bootstrapped_metrics(
        y_true=evals_subset["y_true"],
        y_pred=evals_subset["y_pred_score"],
        event_times=evals_subset["event_times"],
        n_bootstrap_rounds=n_bootstrap_rounds,
    )

    roc_curve_values = compute_roc_curve_values(
        y_true=evals_subset["y_true"], y_pred=evals_subset["y_pred_score"]
    )

    # save results
    metrics.to_csv(evaluation_dir / "metrics.csv")
    with open(evaluation_dir / "roc_curve_values.yaml", "w") as f:
        yaml.dump(roc_curve_values, f)

    # save settings
    with open(evaluation_dir / "settings.yaml", "w") as f:
        yaml.dump(settings, f)

    # save evals subset
    evals_subset.to_csv(evaluation_dir / "evals_subset.csv")
    logger.info(f"Saved evaluation results to {evaluation_dir}")


if __name__ == "__main__":
    logger = logging.getLogger()
    n_bootstrap_rounds = 5000

    args = parse_args()
    evaluation_base_dir = Path(args.eval_dir)
    prediction_dir = Path(args.prediction_dir)

    prediction_name = create_name_from_dir(prediction_dir)

    risk_scores = {
        "framingham": "Framingham Risk Score",
        "pcrs": "ACC/AHA Risk Score",
        "prevent": "PREVENT Risk Score",
        "qrisk": "QRISK Risk Score",
        "score": "SCORE Risk Score",
    }

    # load true outcomes
    true_outcomes = pd.read_parquet(
        ASSETS_PATH / "targets" / "ukb_2024_02" / "targets.parquet"
    )
    target_name = args.target
    target_name_days = target_name.split("_10y")[0] + "_days_from_baseline"
    true_outcomes["event_times"] = true_outcomes[target_name_days].fillna(365 * 10)

    if str(prediction_dir) in risk_scores.keys():
        settings = {"model": risk_scores[str(prediction_dir)]}
        prediction_name = prediction_dir
        risk_score_values = load_medical_risk_scores()
        prompt_parts = pd.read_parquet(
            ASSETS_PATH / "prompt_parts" / "ukb_2024_02" / "prompt_parts.parquet",
        )
        idx = risk_score_values.index.values.tolist()
        risk_score_values["y_true"] = prompt_parts.loc[idx, args.target]
        evals = risk_score_values[[risk_scores[str(prediction_dir)], "y_true"]].rename(
            columns={risk_scores[str(prediction_dir)]: "y_pred_score"}
        )
    else:

        # load and copy settings
        settings = get_settings(prediction_dir)

        # load evals
        evals_path = get_evals_path(prediction_dir)
        evals = pd.read_csv(evals_path, index_col=0)

        if evals.index.name != "eid":
            evals.set_index("eid", inplace=True)

    evals = evals.merge(
        right=true_outcomes, left_index=True, right_index=True, how="left"
    )
    # breakpoint()
    if "y_true" not in evals.columns:  # zeroshot
        evals["y_true"] = evals["completion"]
    if "y_pred_score" not in evals.columns and "risk_score" in evals.columns:
        evals["y_pred_score"] = evals["risk_score"]
    elif "y_pred_score" not in evals.columns and "y_pred" in evals.columns:
        evals["y_pred_score"] = evals["y_pred"]
    assert (evals["y_true"] == evals[target_name]).all()

    evaluation_dir = evaluation_base_dir / prediction_name
    evaluation_dir.mkdir(exist_ok=True, parents=True)

    # specify evaluation subset
    with open(ASSETS_PATH / args.evaluation_subset, "r") as f:
        evaluation_subset = yaml.load(f, Loader=yaml.CLoader)

    is_subset = all(item in evals.index.tolist() for item in evaluation_subset)

    if not is_subset:
        logger.info("Warning: Evaluation subset is not a subset of the evaluation set.")
        evaluation_subset = [x for x in evaluation_subset if x in evals.index.values]

    evals_subset = evals.loc[evaluation_subset]

    if args.subgroup is not None:
        with open(ASSETS_PATH / "subgroups/subgroups.json", "rb") as f:
            subgroups = orjson.loads(f.read())

        for subgroup_value in subgroups[args.subgroup].keys():
            try:
                subgroup_indices = subgroups[args.subgroup][subgroup_value]
                idx = list(set(evals_subset.index.values) & set(subgroup_indices))
                settings["subgroup"] = {
                    "name": args.subgroup,
                    "value": subgroup_value,
                    "n": len(idx),
                }
                logger.info(
                    f"Subgroup {args.subgroup}={subgroup_value} with {len(idx)} individuals in the evaluation set."
                )
                evals_subset_subgroup = evals_subset.loc[idx]
                assert len(idx) == len(evals_subset_subgroup)
                subgroup_evaluation_dir = (
                    evaluation_dir / "subgroups" / args.subgroup / subgroup_value
                )
                subgroup_evaluation_dir.mkdir(exist_ok=True, parents=True)
                evaluate(
                    evals_subset=evals_subset_subgroup,
                    evaluation_dir=subgroup_evaluation_dir,
                    settings=settings,
                    n_bootstrap_rounds=n_bootstrap_rounds,
                )
            except:
                logger.info(f"Subgroup {args.subgroup}={subgroup_value} failed.")
    else:
        evaluation_dir = evaluation_dir / "full"
        evaluation_dir.mkdir(exist_ok=True, parents=True)
        evaluate(
            evals_subset=evals_subset,
            evaluation_dir=evaluation_dir,
            settings=settings,
            n_bootstrap_rounds=n_bootstrap_rounds,
        )
