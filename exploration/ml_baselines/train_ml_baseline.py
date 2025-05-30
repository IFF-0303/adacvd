import argparse
import logging
import multiprocessing
import os
import random
import re

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
import yaml
from accelerate.utils import set_seed
from lifelines import CoxPHFitter
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

import adacvd.utils.logger
from adacvd.data import ukb_features, ukb_field_ids
from adacvd.data.ukb_data_utils import ASSETS_PATH, WANDB_ENTITY, load_ukb_meta_files
from adacvd.training.dataset import (
    get_column_names,
    load_prompt_parts,
    load_split,
    sample_columns,
)
from adacvd.utils.metrics import compute_binary_classification_metrics


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sweep",
        help="Whether to run a sweep or not",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--random",
        help="Whether to use different random seeds for each run",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--model",
        help="What model to use",
        default="lgbm",
        type=str,
    )

    parser.add_argument(
        "--num_samples",
        help="Number of subsamples to use",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--config",
        help="Path to config file",
        default="config/training/train_settings.yaml",
        type=str,
    )
    parser.add_argument(
        "--train_dir",
        help="Path to the training directory. If provided, evaluation results will be saved there.",
        default=None,
    )
    return parser.parse_args()


def prepare_data(config, model, num_samples=None):

    # share this across runs from one sweep

    df_ = pd.read_parquet(
        ASSETS_PATH / "tab_datasets/ukb_2024_02/all_feature_groups.parquet"
    )

    df_target = pd.read_parquet(ASSETS_PATH / "targets/ukb_2024_02/targets.parquet")

    set_seed(config["training"].get("random_seed", 0))

    dataset = load_prompt_parts(data_config=config["data"], num_samples=num_samples)
    train_eids = dataset["train"]["eid"]
    test_eids = dataset["test"]["eid"]
    validation_eids = dataset["validation"]["eid"]

    del dataset

    col_names = get_column_names(config["data"]["feature_config"])

    for col in col_names:
        if col not in df_.columns:
            raise ValueError(f"Column {col} not found in dataset")

    df = df_[list(set(col_names))]

    df = df.loc[list(set(train_eids + test_eids + validation_eids))]

    # special cases
    fill_dict = {
        str(ukb_field_ids.SYSTOLIC_BLOOD_PRESSURE): str(
            ukb_field_ids.SYSTOLIC_BLOOD_PRESSURE_MANUAL
        )
    }

    # PREPROCESSING

    # fill missing values
    for k, v in fill_dict.items():
        if k in df.columns:
            df[k] = df[k].fillna(df[v])
            df = df.drop(columns=v)
            col_names.remove(v)

    # special case: ICD codes
    for icd_col in tqdm(["ICD_10", "ICD_9"]):
        if icd_col in df.columns:
            if icd_col == "ICD_10":
                # regex = r"\b[A-Z]\d{2}"  # three digits = \d{2}
                regex = r"\b[A-Z]\d{2}.\d{1}"  #  four digits = \d{2}.\d{1}
            elif icd_col == "ICD_9":
                regex = r"\b\d{4}"

            icd_codes_series = (
                df[icd_col].astype(str).apply(lambda x: re.findall(regex, x))
            )
            if icd_col == "ICD_9":
                icd_codes_series = icd_codes_series.apply(
                    lambda x: [code[:4] for code in x]  # three of four digits
                )
            df = df.drop(columns=icd_col)
            df = pd.concat([df, icd_codes_series], axis=1)

    # process list/set columns
    cols_type_list = []
    for col in tqdm(df.columns):
        mask = df[col].map(type).isin([list, set, np.ndarray])
        if mask.any():
            cols_type_list.append(col)
            # Convert only the relevant rows to sets
            df.loc[mask, col] = df.loc[mask, col].map(set)
            # Fill other rows with empty sets directly without applying a function row-wise
            df.loc[~mask, col] = [set()] * (~mask).sum()

    logging.info(f"Columns with list/set type: {cols_type_list}")

    mlb = MultiLabelBinarizer()
    transformed_dfs = []

    for col in tqdm(cols_type_list):
        # Fit and transform the column using MultiLabelBinarizer
        one_hot_encoded_array = mlb.fit_transform(df[col])

        if "ICD" in col:
            logging.info(f"{col}: Number of classes: {len(mlb.classes_)}")

        # Create a DataFrame from the transformed array
        one_hot_encoded_df = pd.DataFrame(
            one_hot_encoded_array,
            columns=[f"{col}_{label}" for label in mlb.classes_],
            index=df.index,
        )
        one_hot_encoded_df = one_hot_encoded_df.astype(bool)

        transformed_dfs.append(one_hot_encoded_df)

    # Drop original columns and concatenate new one-hot-encoded columns
    df = df.drop(columns=cols_type_list)
    df = pd.concat([df] + transformed_dfs, axis=1)

    for col in df.columns:
        df[col] = df[col].replace({"nan": pd.NA, "<NA>": pd.NA})

    df = df.loc[:, ~df.columns.duplicated()].copy()

    df_train = df.loc[train_eids]
    df_test = df.loc[test_eids]
    df_validation = df.loc[validation_eids]
    # target = df_[config["data"]["target_name"]]
    target = df_target[config["data"]["target_name"]]

    assert (target.index == df_.index).all(), "Index mismatch"

    logging.info(f"Training dataset: {df_train.shape}")

    if config["train_dir"] is not None:
        logging.info(f"Saving training data to {config['train_dir']}")
        df[0:10].to_parquet(os.path.join(config["train_dir"], "df_sample.parquet"))

    numerical_features = df.select_dtypes(
        include=["float64", "int64", "float32"]
    ).columns
    categorical_features = [col for col in df.columns if col not in numerical_features]

    # round numerical columns to 1 digit
    for col in numerical_features:
        df_train[col] = df_train[col].round(1)
        df_test[col] = df_test[col].round(1)
        df_validation[col] = df_validation[col].round(1)

    # check if type is correct
    check_columns = True
    if check_columns:
        codings, data_dict, field2date = load_ukb_meta_files()

        for col in col_names:
            # logging.info(f"Checking column {col}")
            if col.split("_")[0].isdigit():
                entry = data_dict[data_dict["FieldID"] == int(col)].iloc[0]
                if entry["ValueType"] in ["Integer", "Continuous"]:
                    if col.split("_")[0] in numerical_features:
                        continue
                    else:
                        logging.info(f"Column {col} should be numerical")
                elif entry["ValueType"] in [
                    "Categorical sinlge",
                    "Categorial multiple",
                ]:
                    if col.split("_")[0] in categorical_features:
                        continue
                    else:
                        logging.info(f"Column {col} should be categorical")
            else:
                ftype = "unknown"
                if col.split("_")[0] in numerical_features:
                    ftype = "numerical"
                elif col.split("_")[0] in categorical_features:
                    ftype = "categorical"
                logging.info(f"Column {col} not found in data_dict. Type: {ftype}")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop="first",
                ),
                categorical_features,
            ),
            ("num", StandardScaler(), numerical_features),
        ]
    )

    if model in ["logreg", "svc", "decisiontree", "rf", "mlp", "cox"]:
        # fillna for numerical features
        for col in numerical_features:
            df_train[col] = df_train[col].fillna(df_train[col].mean())
            df_test[col] = df_test[col].fillna(df_train[col].mean())
            df_validation[col] = df_validation[col].fillna(df_train[col].mean())

    logging.info("Preprocessing data")
    preprocessor.fit(df_train)

    for col in categorical_features:
        train_values = set(df_train[col].unique())
        test_values = set(df_test[col].unique())
        validation_values = set(df_validation[col].unique())
        eval_values = set(test_values | validation_values)
        missing_in_train = [x for x in eval_values if x not in train_values]
        if len(missing_in_train) > 0:
            logging.info(f"Missing values in train for {col}: {missing_in_train}")

    df_train = pd.DataFrame(preprocessor.transform(df_train), index=df_train.index)
    df_test = pd.DataFrame(preprocessor.transform(df_test), index=df_test.index)
    df_validation = pd.DataFrame(
        preprocessor.transform(df_validation), index=df_validation.index
    )

    return (
        df_train,
        df_test,
        df_validation,
        target,
        df_target,
        preprocessor,
    )


def run_model(
    df_train,
    df_test,
    df_validation,
    target,
    df_target,
    preprocessor,
    config,
    model,
    use_random_state=False,
):

    if use_random_state:
        random_state = random.randint(0, 1000)
    else:
        random_state = 0

    logging.info(config["sweep_params"])
    cpus = multiprocessing.cpu_count()

    wandb.log({"df_train_len": len(df_train)})

    if model in ["logreg", "svc", "decisiontree", "rf", "mlp"]:

        if model == "logreg":
            classifier = LogisticRegression(
                solver="saga",
                n_jobs=-1,
                random_state=random_state,
                **config["sweep_params"],
            )
        elif model == "svc":
            classifier = SVC(probability=True, **config["sweep_params"])
        elif model == "decisiontree":
            classifier = DecisionTreeClassifier(
                random_state=random_state, **config["sweep_params"]
            )
        elif model == "rf":
            classifier = RandomForestClassifier(
                random_state=random_state,
                n_jobs=-1,
                verbose=1,
                **config["sweep_params"],
            )
        elif model == "mlp":
            classifier = MLPClassifier(
                random_state=random_state, **config["sweep_params"]
            )

        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=False)),
                ("classifier", classifier),
            ]
        )

        pipeline.fit(df_train, target.loc[df_train.index])
        y_preds = pipeline.predict_proba(df_test)
        y_pred = y_preds[:, 1]
        y_pred_binary = y_pred >= 0.5

    elif model == "lgbm":

        feature_names = preprocessor.get_feature_names_out()

        lgb_train_data = lgb.Dataset(
            df_train,
            label=target.loc[df_train.index],
        )
        lgb_test_data = lgb.Dataset(
            df_test,
            reference=lgb_train_data,
            label=target.loc[df_test.index],
        )
        lgb_valid_data = lgb.Dataset(
            df_validation,
            reference=lgb_train_data,
            label=target.loc[df_validation.index],
        )
        base_param = {
            "metric": "auc,average_precision",
            "num_threads": (cpus - 1),
            "verbosity": 1,
            "verbose_eval": 1,
            "feature_fraction": 1.0,
            "learning_rate": 0.01,
            "objective": "binary",
            "max_depth": -1,  # -1
            "min_data_in_leaf": 20,
            "num_leaves": 31,
        }

        param = {**base_param, **config["sweep_params"]}
        eval_results = {}
        bst = lgb.train(
            params=param,
            train_set=lgb_train_data,
            num_boost_round=10000,
            valid_sets=[lgb_valid_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=1),
                lgb.record_evaluation(eval_result=eval_results),
            ],
            keep_training_booster=True,
        )

        wandb.log({"best_iteration": bst.best_iteration})
        # best auc
        wandb.log(
            {"auc_validation": eval_results["valid_0"]["auc"][bst.best_iteration - 1]}
        )

        y_pred = bst.predict(
            df_test,
            num_iteration=bst.best_iteration,
        )
        y_pred_binary = y_pred >= 0.5

        # Feature importance
        feature_importance_split = bst.feature_importance(importance_type="split")
        feature_importance_gain = bst.feature_importance(importance_type="gain")

        feature_imp = pd.DataFrame(
            {
                "feature": feature_names,
                "importance_split": feature_importance_split
                / feature_importance_split.sum(),
                "importance_gain": feature_importance_gain
                / feature_importance_gain.sum(),
            }
        ).sort_values("importance_split", ascending=False)

        # Plot feature importances
        feature_imp_subset = feature_imp.head(min(20, len(feature_imp)))
        plt.figure(figsize=(10, 6))
        sns.barplot(x="importance_split", y="feature", data=feature_imp_subset)
        plt.title("Feature Importances Split")
        plt.tight_layout()

        wandb.log({"feature_importance_split": wandb.Image(plt)})

        feature_imp_subset = feature_imp.sort_values(
            "importance_gain", ascending=False
        ).head(min(20, len(feature_imp)))
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x="importance_gain", y="feature", data=feature_imp_subset, color="navy"
        )
        plt.title("Feature Importances Gain")
        plt.tight_layout()

        wandb.log({"feature_importance_gain": wandb.Image(plt)})
        # log table to wandb feature_imp
        # wandb.log({"feature_importance_table": wandb.Table(data=feature_imp)})

        # write feature importance to csv
        if config["train_dir"] is not None:
            feature_imp.to_csv(
                os.path.join(config["train_dir"], "feature_importance.csv")
            )

        plt.figure(figsize=(10, 6))
        sns.lineplot(eval_results["valid_0"]["auc"], label="AUC")
        wandb.log({"learning_curve_AUC": wandb.Image(plt)})

        plt.figure(figsize=(10, 6))
        sns.lineplot(
            eval_results["valid_0"]["average_precision"], label="Average Precision"
        )
        wandb.log({"learning_curve_Average_Precision": wandb.Image(plt)})

        # save model and preprocessor
        if config["train_dir"] is not None:
            joblib.dump(
                preprocessor, open(f"{config['train_dir']}/preprocessor.pkl", "wb")
            )
            joblib.dump(bst, open(f"{config['train_dir']}/model.pkl", "wb"))

    elif model == "cox":

        # drop columns with low variance (df_train.std() <= 0.001)
        drop_columns_without_variance = df_train.columns[
            df_train.var() <= 0.0001
        ].tolist()

        logging.info(
            f"Dropping columns without variance: {drop_columns_without_variance}"
        )

        df_train = df_train.drop(columns=drop_columns_without_variance)
        df_test = df_test.drop(columns=drop_columns_without_variance)

        # drop duplicated columns
        duplicated_cols = getDuplicateColumns(df_train)
        logging.info(f"Dropping duplicated columns: {duplicated_cols}")
        df_train = df_train.drop(columns=duplicated_cols)
        df_test = df_test.drop(columns=duplicated_cols)

        penalizer = config["sweep_params"].get("penalizer", 0.01)
        cph = CoxPHFitter(penalizer=penalizer)
        event_times = df_target[
            f"{config['data']['target_name'].split('_10y')[0]}_days_from_baseline"
        ].rename("event_times")
        event_times = event_times.fillna(365 * 10)
        cph_data = pd.concat(
            [
                df_train,
                target.loc[df_train.index].rename("target"),
                event_times.loc[df_train.index],
            ],
            axis=1,
        )
        cph.fit(cph_data, duration_col="event_times", event_col="target")
        cph.print_summary()

        predictions = 1 - cph.predict_survival_function(df_test).T[365 * 10]
        y_pred = predictions.values
        y_pred_binary = y_pred >= 0.5

        if config["train_dir"] is not None:
            joblib.dump(
                preprocessor, open(f"{config['train_dir']}/preprocessor.pkl", "wb")
            )
            joblib.dump(cph, open(f"{config['train_dir']}/model.pkl", "wb"))

    logging.info(classification_report(target.loc[df_test.index], y_pred_binary))
    metrics = compute_binary_classification_metrics(target.loc[df_test.index], y_pred)
    for k, v in metrics.items():
        logging.info(f"{k}: {v}")
    wandb.log(metrics)

    # plot distribution
    plt.figure(figsize=(6, 6), constrained_layout=False)
    sns.displot(
        data=pd.DataFrame({"Predictions": y_pred, "Target": target.loc[df_test.index]}),
        x="Predictions",
        hue="Target",
        kind="kde",
        fill=True,
        common_norm=False,
    )
    plt.xlim(0, 1)
    plt.title("Score distribution")
    plt.show()
    wandb.log({"score_distribution": wandb.Image(plt)})

    # save predictions to csv
    if config["train_dir"] is not None:
        predictions = pd.DataFrame(
            {
                "eid": df_test.index,
                "y_true": target.loc[df_test.index],
                "y_pred_score": y_pred,
                "y_pred": y_pred_binary,
            }
        )

        predictions.to_csv(os.path.join(config["train_dir"], "evals.csv"), index=False)
        # config to yaml
        with open(os.path.join(config["train_dir"], "config.yaml"), "w") as outfile:
            yaml.dump(config, outfile, default_flow_style=False)


def getDuplicateColumns(df):
    duplicateColumnNames = set()
    for x in range(df.shape[1]):
        col = df.iloc[:, x]
        for y in range(x + 1, df.shape[1]):
            otherCol = df.iloc[:, y]
            if col.equals(otherCol):
                duplicateColumnNames.add(df.columns.values[y])
    return list(duplicateColumnNames)


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config["train_dir"] = args.train_dir

    # create dir if not None and not exists
    if args.train_dir is not None and not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)

    # copy settings to train_dir
    if args.train_dir is not None:
        with open(os.path.join(args.train_dir, "config.yaml"), "w") as outfile:
            yaml.dump(config, outfile, default_flow_style=False)

    # preprocess data, then run model
    (
        df_train,
        df_test,
        df_validation,
        target,
        df_target,
        preprocessor,
    ) = prepare_data(config, args.model.lower(), num_samples=args.num_samples)

    if args.sweep:
        # log reg
        if args.model.lower() == "logreg":
            sweep_configuration = {
                "method": "bayes",
                "metric": {"goal": "maximize", "name": "roc_auc"},
                "parameters": {
                    "penalty": {"values": ["l1", "l2", "elasticnet"]},
                    "C": {"min": 0.01, "max": 10.0},
                    "max_iter": {"values": [100, 1000, 5000]},
                    "class_weight": {"values": ["balanced", None]},
                },
            }

        elif args.model.lower() == "svc":
            sweep_configuration = {
                "method": "bayes",
                "metric": {"goal": "maximize", "name": "roc_auc"},
                "parameters": {
                    "kernel": {"values": ["linear"]},
                    "C": {"min": 0.01, "max": 10.0},
                },
            }

        elif args.model.lower() == "decisiontree":
            sweep_configuration = {
                "method": "bayes",
                "metric": {"goal": "maximize", "name": "roc_auc"},
                "parameters": {
                    "criterion": {"values": ["gini", "entropy", "log_loss"]},
                    "splitter": {"values": ["best", "random"]},
                    "max_depth": {"values": [3, 4, 5, 6, 7, 8, 9, 10]},
                    "min_samples_split": {"values": [2, 10, 50]},
                    "max_features": {
                        "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                    },
                    "class_weight": {"values": ["balanced", None]},
                },
            }
        elif args.model.lower() == "rf":
            sweep_configuration = {
                "method": "bayes",
                "metric": {"goal": "maximize", "name": "roc_auc"},
                "parameters": {
                    "n_estimators": {"values": [10, 50, 100, 200, 500, 1000]},
                    "criterion": {"values": ["gini", "entropy", "log_loss"]},
                    "max_depth": {"values": [3, 4, 5, 6, 7, 8, 9, 10]},
                    "min_samples_split": {"values": [2, 10, 50]},
                    "class_weight": {"values": ["balanced", None]},
                },
            }
        elif args.model.lower() == "mlp":
            sweep_configuration = {
                "method": "bayes",
                "metric": {"goal": "maximize", "name": "roc_auc"},
                "parameters": {
                    "hidden_layer_sizes": {
                        "values": [
                            (10,),
                            (50,),
                            (10, 10),
                        ]
                    },
                    "activation": {"values": ["relu", "tanh", "logistic"]},
                    "alpha": {"min": 0.0001, "max": 0.01},
                    "learning_rate": {"values": ["constant", "invscaling", "adaptive"]},
                    "learning_rate_init": {"min": 0.0001, "max": 0.01},
                },
            }
        elif args.model.lower() == "lgbm":
            sweep_configuration = {
                "method": "grid",
                "metric": {"goal": "maximize", "name": "roc_auc"},
                "parameters": {
                    "objective": {"values": ["binary"]},
                    "max_depth": {"values": [3, 4, 5, -1]},
                    "num_leaves": {"values": [5, 15, 31, 127, 255, 511]},
                    "min_data_in_leaf": {"values": [20, 50, 100]},
                    "feature_fraction": {"values": [0.5, 1.0]},
                },
            }
        elif args.model.lower() == "cox":
            sweep_configuration = {
                "method": "grid",
                "metric": {"goal": "maximize", "name": "roc_auc"},
                "parameters": {
                    "penalizer": {"values": [0.0, 0.1, 0.01, 0.001]},
                },
            }
        else:
            raise ValueError(f"Model {args.model} not supported")

        def sweep_run():
            wandb.init(
                entity=WANDB_ENTITY,
                project="UKB_ML_baselines",
            )

            config["sweep_params"] = {}
            config["model"] = args.model
            for k, v in dict(wandb.config).items():
                config["sweep_params"][k] = v

            if args.train_dir is not None:
                config["train_dir"] = args.train_dir + f"/{wandb.run.id}"

                if not os.path.exists(config["train_dir"]):
                    os.makedirs(config["train_dir"])

            wandb.config.update(config)
            return run_model(
                df_train=df_train,
                df_test=df_test,
                df_validation=df_validation,
                target=target,
                df_target=df_target,
                preprocessor=preprocessor,
                config=config,
                model=args.model.lower(),
                use_random_state=args.random,
            )

        sweep_id = wandb.sweep(sweep=sweep_configuration, project="UKB_ML_baselines")
        wandb.agent(sweep_id, function=sweep_run)

    else:
        config["sweep_params"] = {}
        config["model"] = args.model
        wandb.init(
            entity=WANDB_ENTITY,
            project="UKB_ML_baselines",
            config=config,
            # mode="disabled",
        )
        run_model(
            df_train=df_train,
            df_test=df_test,
            df_validation=df_validation,
            target=target,
            df_target=df_target,
            preprocessor=preprocessor,
            config=config,
            model=args.model.lower(),
            use_random_state=args.random,
        )
