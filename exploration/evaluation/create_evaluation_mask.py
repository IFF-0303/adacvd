import yaml

from exploration.evaluation import evaluation_utils
from adacvd.data.ukb_data_utils import ASSETS_PATH

medical_risk_scores = evaluation_utils.load_medical_risk_scores()
medical_risk_scores.isna().sum(axis=0) / len(medical_risk_scores)

# evaluation mask:
# medical risk scores not na (i.e. no missing values in input data)
# evaluation set for training the models
# additional mask, e.g. not previous MACE

data_config = {
    "split": "config/splits/split_2025_02_17_test_100000.json",
    "subset": "subsets/ukb_2024_02/MACE_ADO_EXTENDED_no_previous_target.json",
    "eval_subset": True,
}

# load the eval split
with open(data_config["split"], "r") as f:
    split = yaml.load(f, Loader=yaml.CLoader)

# load additional subset information
if data_config.get("subset") is not None:
    with open(ASSETS_PATH / data_config["subset"], "r") as f:
        subset = yaml.load(f, Loader=yaml.CLoader)

medical_risk_scores_not_na = medical_risk_scores.index[
    medical_risk_scores.notna().all(axis=1)
].values.tolist()

eids_test = list(set(split["test"]) & set(subset) & set(medical_risk_scores_not_na))

print(f"Test Set: {len(eids_test)}")

evaluation_subset_path = ASSETS_PATH / "evaluation_subsets" / "ukb_2024_02"
evaluation_subset_path.mkdir(exist_ok=True, parents=True)
with open(
    evaluation_subset_path / "test_subset_MACE_ADO_EXTENDED_no_previous_target.json",
    "w",
) as f:
    yaml.dump(eids_test, f)
