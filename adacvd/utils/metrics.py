import multiprocessing as mp
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from accelerate.utils import set_seed
from lifelines.utils import concordance_index
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils import resample
from sksurv.metrics import concordance_index_censored
from tqdm import tqdm


def compute_binary_classification_metrics(y_true, y_pred):
    """
    Compute binary classification metrics.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    dict
        Dictionary containing the computed metrics.
    """

    metrics = {}
    try:
        metrics["share_pos"] = y_true.mean()
    except:
        metrics["share_pos"] = None
    try:
        metrics["share_pos_pred"] = (y_pred > 0.5).mean()
    except:
        metrics["share_pos_pred"] = None
    try:
        metrics["acc"] = accuracy_score(y_true, y_pred > 0.5)
    except:
        metrics["acc"] = None
    try:
        metrics["balanced_acc"] = balanced_accuracy_score(y_true, y_pred > 0.5)
    except:
        metrics["balanced_acc"] = None
    try:
        metrics["f1"] = f1_score(y_true, y_pred > 0.5)
    except:
        metrics["f1"] = None
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred)
    except:
        metrics["roc_auc"] = None
    try:
        metrics["precision_score"] = precision_score(
            y_true, y_pred > 0.5, zero_division=0
        )
    except:
        metrics["precision_score"] = None
    try:
        metrics["recall_score"] = recall_score(y_true, y_pred > 0.5)
    except:
        metrics["recall_score"] = None
    try:
        metrics["average_precision_score"] = average_precision_score(y_true, y_pred)
        metrics["brier_score_loss"] = brier_score_loss(y_true, y_pred)
    except:
        metrics["average_precision_score"] = None
    try:
        metrics["brier_score_loss"] = brier_score_loss(y_true, y_pred)
    except:
        metrics["brier_score_loss"] = None

    return metrics


def plot_roc_auc_curve(y_true: pd.Series, y_pred: pd.Series, name=""):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve {name}")
    plt.show()


def compute_roc_curve_values(y_true: pd.Series, y_pred: pd.Series) -> dict:
    fpr, tpr, thresholds = roc_curve(
        y_true=y_true,
        y_score=y_pred,
    )
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()}


def compute_survival_analysis_metrics(
    event_indicator: pd.Series, event_time: pd.Series, estimate: pd.Series
) -> dict:
    c_index = concordance_index(
        event_times=event_time,
        predicted_scores=estimate * -1,
        event_observed=event_indicator,
    )

    return {"c_index": c_index}


def bootstrap_iteration(y_true, y_pred, event_times, random_state):
    y_true.index.values.tolist()

    pos_indices = y_true[y_true == 1].index.values
    neg_indices = y_true[y_true == 0].index.values

    # stratified bootstrap
    fraction = 0.8

    pos_indices_bootstrap = resample(
        pos_indices,
        n_samples=int(len(pos_indices) * fraction),
        replace=True,
        random_state=random_state,
    )
    neg_indices_bootstrap = resample(
        neg_indices,
        n_samples=int(len(neg_indices) * fraction),
        replace=True,
        random_state=random_state,
    )

    bootstrap_indices = np.concatenate([pos_indices_bootstrap, neg_indices_bootstrap])

    # bootstrap_indices = resample(
    #     indices, n_samples=len(indices), replace=True, random_state=random_state
    # )

    metrics_tmp = compute_binary_classification_metrics(
        y_true=y_true.loc[bootstrap_indices],
        y_pred=y_pred.loc[bootstrap_indices],
    )
    if event_times is not None:
        metrics_tmp.update(
            compute_survival_analysis_metrics(
                event_indicator=y_true.loc[bootstrap_indices],
                event_time=event_times.loc[bootstrap_indices],
                estimate=y_pred.loc[bootstrap_indices],
            )
        )
    return metrics_tmp


def compute_bootstrapped_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    event_times: pd.Series = None,
    n_bootstrap_rounds: int = 1000,
) -> pd.DataFrame:
    metrics_dict = {}
    metrics_dict["all"] = compute_binary_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
    )

    if event_times is not None:
        metrics_dict["all"].update(
            compute_survival_analysis_metrics(
                event_indicator=y_true,
                event_time=event_times,
                estimate=y_pred,
            )
        )

    set_seed(0)

    assert (
        y_true.index == y_pred.index
    ).all(), "Indices must match between y_true and y_pred"

    print("Starting parallel bootstrap")
    n_jobs = mp.cpu_count() - 1

    bootstrap_iteration_partial = partial(
        bootstrap_iteration, y_true, y_pred, event_times
    )
    random_states = np.random.randint(2**32 - 1, size=n_bootstrap_rounds)
    pool = mp.Pool(processes=n_jobs)
    results = []
    for result in tqdm(
        pool.imap_unordered(bootstrap_iteration_partial, random_states),
        total=len(random_states),
    ):
        results.append(result)

    metrics_dict = {i: result for i, result in enumerate(results)}

    print("End parallel bootstrap")

    metrics_df = pd.DataFrame(metrics_dict).T
    return metrics_df
