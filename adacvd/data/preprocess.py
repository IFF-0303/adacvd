import argparse
import logging
import multiprocessing as mp

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from adacvd.data import prompt, ukb_data_utils, ukb_features

logging.basicConfig(format="%(asctime)s – %(levelname)s: %(message)s")
logging.getLogger().setLevel(logging.INFO)


def parse_args():
    """
    Parse command-line arguments for the preprocessing script.

    Returns:
        argparse.Namespace: Parsed arguments containing:
            - config_path (str): Path to the configuration file.
            - num_samples (int): Number of samples to process (optional).
            - num_workers (int): Number of parallel workers to use.
    """
    parser = argparse.ArgumentParser(description="Preprocess UKB data.")
    parser.add_argument(
        "--config_path",
        help="Path to config for generating the dataset",
        default="config/ukb_data/all_feature_groups.yaml",
    )
    parser.add_argument(
        "--num_samples", help="Number of samples to generate", type=int, default=None
    )
    parser.add_argument(
        "--num_workers",
        help="Number of parallel workers",
        type=int,
        default=mp.cpu_count() - 1,
    )
    return parser.parse_args()


def preprocess_ukb_data(df, config):
    """
    Preprocess the UKB dataset to generate features, targets, and prompts.

    Args:
        df (pd.DataFrame): Input UKB dataset.
        config (dict): Configuration dictionary specifying preprocessing parameters.

    Returns:
        tuple: A tuple containing:
            - features (pd.DataFrame): Processed features.
            - target (pd.DataFrame): Target variables.
            - prompts (pd.DataFrame): Generated prompts.
            - prompt_parts (pd.DataFrame): Parts of the prompts.
            - subset_no_previous_target (pd.DataFrame): Subset of participants with no previous incident according to the target definition.
    """
    period = ukb_features.get_period_from_baseline_assessment(
        df, years=config["follow_up_years"]
    )
    features = get_features(df, config, period)
    target = get_target(df, config, period)
    prompts, prompt_parts = prompt.write_prompts(features)

    subset_no_previous_target = ukb_features.retrieve_subset_no_previous_target(
        df, period, target=config.get("target")
    ).rename(columns={"previous_MACE": "no_previous_target"})

    return features, target, prompts, prompt_parts, subset_no_previous_target


def get_features(df: pd.DataFrame, config: dict, period: pd.DataFrame):
    """
    Extract and preprocess features from the UKB dataset.

    Args:
        df (pd.DataFrame): Input UKB dataset.
        config (dict): Configuration dictionary specifying feature extraction parameters.
        period (pd.DataFrame): DataFrame containing start and end dates for the considered period.

    Returns:
        pd.DataFrame: Processed features.

    Feature Types:
        - Features specified via field IDs.
        - Features specified via custom functions.
    """
    feature_field_ids = config.get("feature_field_ids")

    features = pd.DataFrame(index=df.eid)
    logging.info("Extracting features from field IDs.")
    for field_id in tqdm(feature_field_ids, desc="Field ID Features"):
        feature_values = ukb_features.get_feature_values_from_field_id(
            df.set_index("eid"), field_id
        )
        features = pd.concat([features, feature_values], axis=1)

    logging.info("Extracting features from custom functions.")
    if config.get("features") is not None:
        for feature in tqdm(config.get("features"), desc="Custom Features"):
            feature_func = ukb_features.FEATURES.get(feature).get("func")
            if feature in ["ICD_10", "ICD_9"]:
                feature_values = feature_func(
                    df.set_index("eid"), start_dates=period["start_date"]
                )
            elif feature == "previous_MACE":
                feature_values = feature_func(
                    df.set_index("eid"),
                    start_dates=period["start_date"],
                    target=config.get("target"),
                )
            else:
                feature_values = feature_func(df.set_index("eid"))
            features = pd.concat([features, feature_values], axis=1)

    assert (
        features.index == df.eid
    ).all(), "Feature indices do not match dataset indices."
    return features


def get_target(df, config, period):
    """
    Generate target variables based on the configuration.

    Args:
        df (pd.DataFrame): Input UKB dataset.
        config (dict): Configuration dictionary specifying target parameters.
        period (pd.DataFrame): DataFrame containing start and end dates for the considered period.

    Returns:
        pd.DataFrame: Target variables with incident dates and days from baseline.

    Target Options:
        - `MACE_ADO`: UKB-provided definition of outcomes ("algorithmically defined outcomes") for MI and Stroke.
        - `MACE_ADO_EXTENDED`: Extended outcome definition including custom ICD codes.
    """
    target = config.get("target")
    target_name = f"{target}_{config.get('follow_up_years')}y"

    if target == "MACE_ADO":
        extended_def = False
    elif target == "MACE_ADO_EXTENDED":
        extended_def = True
    else:
        raise NotImplementedError(f"Target {target} is not implemented.")

    target_values = ukb_features.ADO_MACE_target(
        df,
        period["start_date"],
        period["end_date"],
        extended_def=extended_def,
        return_date=True,
    )

    outputs = pd.DataFrame(index=df.eid)
    outputs[target_name] = target_values["target"]
    outputs[f"{target}_days_from_baseline"] = (
        target_values["incident_date"] - period["start_date"]
    ).dt.days

    return outputs


def process_chunk(args):
    """
    Process a single chunk of the dataset in parallel.

    Args:
        args (tuple): A tuple containing:
            - chunk_number (int): Chunk index.
            - df (pd.DataFrame): Chunk of the dataset.
            - config (dict): Configuration dictionary.

    Returns:
        dict: Processed datasets including tabular, text, and prompt parts.
    """
    chunk_number, df, config = args
    logging.info(f"Processing chunk {chunk_number}")

    features, target, prompts, prompt_parts, subset_no_previous_target = (
        preprocess_ukb_data(df, config)
    )

    tabular_dataset = features.merge(target, how="left", on="eid")
    prompt_parts_dataset = prompt_parts.merge(target, how="left", on="eid")
    text_dataset = prompts.reset_index().to_dict(orient="records")

    return {
        "tabular": tabular_dataset,
        "prompt_parts": prompt_parts_dataset,
        "text": text_dataset,
        "target": target,
        "subset_no_previous_target": subset_no_previous_target,
    }


def main():
    """
    Main function to preprocess the UKB dataset.

    Steps:
        1. Parse command-line arguments.
        2. Load configuration file.
        3. Read and process data in chunks using parallel workers.
        4. Combine and save processed datasets.
    """
    args = parse_args()

    # Load configuration
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load data
    logging.info("Loading data...")
    data_config = config.get("data")
    DATA_PATH = (
        ukb_data_utils.ASSETS_PATH
        / "ukb"
        / data_config.get("data_dir")
        / data_config.get("data_file")
    )
    df_chunks = pd.read_csv(
        DATA_PATH,
        nrows=args.num_samples,
        chunksize=10000,
        low_memory=False,
    )
    dataset_name = data_config.get("dataset_name")

    tabular_dataset_concatenated = pd.DataFrame()
    PATH_TABULAR_DATA = (
        ukb_data_utils.ASSETS_PATH
        / "tab_datasets"
        / data_config.get("data_dir")
        / f"{dataset_name}.parquet"
    )
    PATH_TABULAR_DATA.parent.mkdir(parents=True, exist_ok=True)

    prompt_parts_dataset_concatenated = pd.DataFrame()
    PATH_PROMPT_PARTS = (
        ukb_data_utils.ASSETS_PATH
        / "prompt_parts"
        / data_config.get("data_dir")
        / f"{dataset_name}.parquet"
    )
    PATH_PROMPT_PARTS.unlink(missing_ok=True)
    PATH_PROMPT_PARTS.parent.mkdir(parents=True, exist_ok=True)

    PATH_TEXT_DATA = (
        ukb_data_utils.ASSETS_PATH
        / "text_datasets"
        / data_config.get("data_dir")
        / f"{dataset_name}.jsonl"
    )
    PATH_TEXT_DATA.unlink(missing_ok=True)
    PATH_TEXT_DATA.parent.mkdir(parents=True, exist_ok=True)

    target_concatenated = pd.DataFrame()
    PATH_TARGET_DATA = (
        ukb_data_utils.ASSETS_PATH
        / "targets"
        / data_config.get("data_dir")
        / "targets.parquet"
    )
    PATH_TARGET_DATA.parent.mkdir(parents=True, exist_ok=True)

    subset_no_previous_target_concatenated = pd.DataFrame()
    target_name = str(data_config.get("target"))
    PATH_SUBSET_NO_PREVIOUS_TARGET = (
        ukb_data_utils.ASSETS_PATH
        / "subsets"
        / data_config.get("data_dir")
        / f"{target_name}_no_previous_target.json"
    )
    PATH_SUBSET_NO_PREVIOUS_TARGET.parent.mkdir(parents=True, exist_ok=True)

    # get feature IDs for feature groups
    feature_field_ids = data_config.get("feature_field_ids", [])
    if data_config.get("feature_groups") is not None:
        for fg in data_config.get("feature_groups"):
            feature_field_ids += ukb_features.get_field_ids_from_feature_group(fg)
    feature_field_ids = list(dict.fromkeys(feature_field_ids))
    data_config["feature_field_ids"] = feature_field_ids

    data_config = config.get("data")
    # Process chunks in parallel
    num_workers = args.num_workers
    logging.info(f"Processing chunks in parallel with {num_workers} workers...")
    with mp.Pool(num_workers) as pool:
        results = list(
            pool.imap(
                process_chunk,
                [(i, df, data_config) for i, df in enumerate(df_chunks)],
            )
        )

    # Combine and save results
    logging.info("Combining processed chunks...")

    # Combine results
    logging.info("Combining processed chunks...")

    tabular_dataset_concatenated = pd.concat([r["tabular"] for r in results])
    prompt_parts_dataset_concatenated = pd.concat([r["prompt_parts"] for r in results])
    target_concatenated = pd.concat([r["target"] for r in results])
    subset_no_previous_target_concatenated = pd.concat(
        [r["subset_no_previous_target"] for r in results]
    )

    # Replace "Do not know" and "Prefer not to answer" with NaN in tabular dataset
    tabular_dataset_concatenated = tabular_dataset_concatenated.replace(
        {"Do not know": pd.NA, "Prefer not to answer": pd.NA}
    )

    # save dataset; only if full dataset was processed
    if args.num_samples is None:
        if PATH_TABULAR_DATA.exists():
            full_tab_df = pd.read_parquet(PATH_TABULAR_DATA)
            if len(full_tab_df) == len(tabular_dataset_concatenated):
                full_tab_df = pd.merge(
                    left=full_tab_df,
                    right=tabular_dataset_concatenated,
                    left_index=True,
                    right_index=True,
                    suffixes=["_OLD", ""],
                    how="inner",
                )

                # compare duplicate columns
                OLD_cols = full_tab_df.filter(regex="_OLD").columns.tolist()
                equal_cols = []
                for col in OLD_cols:
                    new_col = col.split("_OLD")[0]
                    is_equal = full_tab_df[col].equals(full_tab_df[new_col])
                    if not is_equal:

                        def compare(x, y):
                            if isinstance(x, list | set | np.ndarray) and isinstance(
                                y, list | set | np.ndarray
                            ):
                                return set(x) == set(y)
                            if not isinstance(
                                x, list | set | np.ndarray
                            ) and not isinstance(y, list | set | np.ndarray):
                                if pd.isna(x) & pd.isna(y):
                                    return True
                                elif (
                                    x == y
                                ) is True:  # fixes "boolean value of NA is ambiguous"
                                    return True
                                else:
                                    return False
                            else:
                                return False

                        df_equal = full_tab_df.apply(
                            lambda x: compare(x[col], x[new_col]), axis=1
                        )

                        logging.info(
                            f"Updated column {col}. {(df_equal == False).sum()} rows were unequal."
                        )
                    else:
                        equal_cols.append(col)

                # remove duplicate columns
                full_tab_df = full_tab_df.drop(columns=OLD_cols)

                full_tab_df.to_parquet(
                    PATH_TABULAR_DATA, index=True, engine="auto", compression=None
                )
                logging.info(
                    f"Updated tabular dataset. {len(tabular_dataset_concatenated.columns)} new columns, of which {len(equal_cols)} existed and were equal, and {len(OLD_cols) - len(equal_cols)} existed but were updated, and {len(tabular_dataset_concatenated.columns)-len(OLD_cols)} are new. Total columns: {len(full_tab_df.columns)}."
                )
            else:
                logging.error(
                    "Number of rows in the new tabular dataset does not match the number of rows in the base dataset."
                )
                tabular_dataset_concatenated.to_parquet(
                    PATH_TABULAR_DATA, index=True, engine="auto", compression=None
                )
                logging.info(f"Saved base tabular dataset.")
        else:
            tabular_dataset_concatenated.to_parquet(
                PATH_TABULAR_DATA, index=True, engine="auto", compression=None
            )
            logging.info(f"Saved base tabular dataset.")

    # save concatenated prompt parts dataset
    prompt_parts_dataset_concatenated.to_parquet(
        PATH_PROMPT_PARTS, index=True, engine="auto", compression=None
    )

    # add concatenated target dataset to targets
    if PATH_TARGET_DATA.exists():
        target_all = pd.read_parquet(PATH_TARGET_DATA)
        if len(target_all) == len(target_concatenated):
            target_all = pd.merge(
                left=target_all,
                right=target_concatenated,
                left_index=True,
                right_index=True,
                suffixes=["_OLD", ""],
                how="inner",
            )

            # compare duplicate columns
            OLD_cols = target_all.filter(regex="_OLD").columns.tolist()
            equal_cols = []
            for col in OLD_cols:
                new_col = col.split("_OLD")[0]
                is_equal = target_all[col].equals(target_all[new_col])
                if not is_equal:
                    df_equal = target_all[col] == target_all[new_col]
                    logging.info(
                        f"Updated column {col}. {(df_equal == False).sum()} rows were unequal."
                    )
                else:
                    equal_cols.append(col)

            # remove duplicate columns
            target_all = target_all.drop(columns=OLD_cols)

            target_all.to_parquet(
                PATH_TARGET_DATA, index=True, engine="auto", compression=None
            )
            logging.info(
                f"Updated target dataset. {len(target_concatenated.columns)} new target columns, of which {len(equal_cols)} existed and were equal, and {len(OLD_cols) - len(equal_cols)} existed but were updated, and {len(target_concatenated.columns)-len(OLD_cols)} are new. Total columns: {len(target_all.columns)}."
            )

        else:
            logging.error(
                "Number of rows in the new target dataset does not match the number of rows in the base dataset."
            )
    else:
        target_concatenated.to_parquet(
            PATH_TARGET_DATA, index=True, engine="auto", compression=None
        )
        logging.info(f"Saved target dataset.")

    # add prompt parts to base prompt parts dataset (containing all prompt parts so far) if specified
    if (
        data_config.get("base_prompt_parts_file", None) is not None
        and args.num_samples is None
    ):
        PATH_ALL_PROMPT_PARTS = (
            ukb_data_utils.ASSETS_PATH
            / "prompt_parts"
            / data_config["data_dir"]
            / data_config["base_prompt_parts_file"]
        )

        if PATH_ALL_PROMPT_PARTS.exists():
            all_prompt_parts = pd.read_parquet(PATH_ALL_PROMPT_PARTS)
            if len(all_prompt_parts) == len(prompt_parts_dataset_concatenated):
                all_prompt_parts = pd.merge(
                    left=all_prompt_parts,
                    right=prompt_parts_dataset_concatenated,
                    left_index=True,
                    right_index=True,
                    suffixes=["_OLD", ""],
                    how="inner",
                )
                # compare duplicate columns
                OLD_cols = all_prompt_parts.filter(regex="_OLD").columns.tolist()
                equal_cols = []
                for col in OLD_cols:
                    new_col = col.split("_OLD")[0]
                    is_equal = all_prompt_parts[col].equals(all_prompt_parts[new_col])
                    if not is_equal:
                        df_equal = all_prompt_parts[col] == all_prompt_parts[new_col]
                        logging.info(
                            f"Updated column {col}. {(df_equal == False).sum()} rows were unequal."
                        )
                    else:
                        equal_cols.append(col)

                # remove duplicate columns
                all_prompt_parts = all_prompt_parts.drop(columns=OLD_cols)

                all_prompt_parts.to_parquet(
                    PATH_ALL_PROMPT_PARTS, index=True, engine="auto", compression=None
                )
                logging.info(
                    f"Updated base prompt parts dataset. {len(prompt_parts_dataset_concatenated.columns)} new columns, of which {len(equal_cols)} existed and were equal, and {len(OLD_cols) - len(equal_cols)} existed but were updated, and {len(prompt_parts_dataset_concatenated.columns)-len(OLD_cols)} are new. Total columns: {len(all_prompt_parts.columns)}."
                )
            else:
                logging.error(
                    "Number of rows in the new prompt parts dataset does not match the number of rows in the base dataset."
                )
                prompt_parts_dataset_concatenated.to_parquet(
                    PATH_ALL_PROMPT_PARTS, index=True, engine="auto", compression=None
                )
                logging.info(f"Saved base prompt parts dataset.")
        else:
            prompt_parts_dataset_concatenated.to_parquet(
                PATH_ALL_PROMPT_PARTS, index=True, engine="auto", compression=None
            )
            logging.info(f"Saved base prompt parts dataset.")

    # save subset_no_previous_target
    eids_no_previous_target = subset_no_previous_target_concatenated[
        subset_no_previous_target_concatenated["no_previous_target"]
    ].index.tolist()
    if PATH_SUBSET_NO_PREVIOUS_TARGET.exists():
        with open(PATH_SUBSET_NO_PREVIOUS_TARGET, "r") as f:
            subset_old = yaml.load(f, Loader=yaml.CLoader)
        if set(eids_no_previous_target) == set(subset_old):
            logging.info(
                f"Subset of eids with no previous target is the same as the previous subset."
            )
        else:
            intersection = list(set(eids_no_previous_target) & set(subset_old))
            logging.info(
                f"Subset of eids with no previous target is different from the previous subset. Intersection: {len(intersection)}. Old: {len(set(subset_old))}. New: {len(set(eids_no_previous_target))}."
            )
    with open(PATH_SUBSET_NO_PREVIOUS_TARGET, "w") as f:
        yaml.dump(eids_no_previous_target, f)
    logging.info(f"Saved subset of eids with no previous target.")


if __name__ == "__main__":
    main()
