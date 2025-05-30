import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from pandora.data import ukb_data_utils


def split_data(
    ids: pd.DataFrame,
    train_set_fraction: float = 0.8,
    test_length: int = None,
    seed: int = 0,
    dir: Union[str, Path] = None,
    save: bool = False,
) -> dict:
    """Split data into train and test set and save the split to a json file."""
    np.random.seed(seed)

    validation_length = 20_000

    if test_length is not None:
        n = len(ids) - test_length - validation_length
    else:
        n = int(train_set_fraction * len(ids))
        test_length = len(ids) - n - validation_length

    if test_length <= 0:
        raise ValueError("test_length must be greater than 0")

    train_ids = np.random.choice(
        ids,
        size=n,
        replace=False,
    )
    test_validation_ids = np.setdiff1d(ids, train_ids)
    assert len(train_ids) + len(test_validation_ids) == len(ids)

    test_ids = np.random.choice(
        test_validation_ids,
        size=test_length,
        replace=False,
    )

    validation_ids = np.setdiff1d(test_validation_ids, test_ids)
    assert len(test_ids) + len(validation_ids) == len(test_validation_ids)

    split = {
        "train": train_ids.tolist(),
        "test": test_ids.tolist(),
        "validation": validation_ids.tolist(),
    }

    if save:
        # save split
        split_id = pd.Timestamp.now().strftime("%Y_%m_%d")
        path = Path(dir) / f"split_{split_id}_test_{test_length}.json"
        path.parent.mkdir(exist_ok=True, parents=True)

        with open(path, "w") as file:
            json.dump(split, file)

    print(
        f"Split: \nTrain: {len(split['train'])}, Test: {len(split['test'])}, Validation: {len(split['validation'])}"
    )
    print(f"Saved split to {path}")

    return split


if __name__ == "__main__":

    df = pd.read_csv(
        ukb_data_utils.ASSETS_PATH / "ukb/ukb_2024_02/ukb677731.csv",
        usecols=["eid"],
        low_memory=False,
    )

    split = split_data(df["eid"], test_length=100_000, dir="config/splits/", save=True)
