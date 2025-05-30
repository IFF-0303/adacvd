import datetime
import json
import logging
import os
import pickle
import re
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import wget
from tqdm import tqdm

import pandora.data.outcome_definition as outcome_definition
import pandora.data.ukb_field_ids as ukb_field_ids

# PATHS
ASSETS_PATH = Path(os.environ.get("ASSETS_PATH", "assets"))
RESULTS_PATH = Path(os.environ.get("RESULTS_PATH", "results"))
CODING_PATH = ASSETS_PATH / "ukb/codings_raw.tsv"
DATA_DICT_PATH = ASSETS_PATH / "ukb/data_dictionary_raw.tsv"
DTYPE_DICT_PATH = ASSETS_PATH / "ukb/ukb_data_noisy_hashed_dtypes.pickle"
FIELD2DATE_PATH = ASSETS_PATH / "ukb/field2date.json"
DATA_PATH = ASSETS_PATH / "ukb" / "ukb_2024_02" / "ukb677731.csv"
ALGORITHMICALLY_DEFINED_OUTCOMES_PATH = (
    ASSETS_PATH / "ukb" / "algorithm_outcome_codes.xlsx"
)
WANDB_ENTITY = "biobank"


def isNaN(x):
    return x != x


def get_mapping_raw2clean(
    col_names: Optional[List[str]] = None,
    only_clean_col_name: Optional[bool] = True,
    data_dict: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Construct mapping from all raw column names to clean column names based on the data dict.

    If only_clean_col_name is True, the mapping will only contain the clean column name, otherwise it will be a dictionary containing the entire data field in the data dictionary.
    """
    if col_names is None:
        col_names = pd.read_csv(DATA_PATH, nrows=0).columns.to_list()
    if data_dict is None:
        data_dict = pd.read_csv(DATA_DICT_PATH, sep="\t")

    raw2clean = {}
    for col_name in col_names:
        try:
            field_id, instance_array = col_name.split("-")
            instance, array = instance_array.split(".")
            instance = int(instance)
            array = int(array)
            field_id = int(field_id)
        except:
            logging.error(f"Could not parse column name: {col_name}.")
            continue

        # Check if field_id exists in data dictionary
        if field_id not in data_dict["FieldID"].values:
            raise ValueError(f"FieldID {field_id} not found in data dictionary.")

        # Get data field from data dictionary and raise error if multiple fields are found
        data_field = data_dict[data_dict["FieldID"] == field_id]

        if len(data_field) > 1:
            logging.error(f"Multiple data fields found for FieldID {field_id}.")
            continue

        data_field = data_field.iloc[0]

        if only_clean_col_name:
            raw2clean[col_name] = data_field["Field"] + f"...{instance}_{array}"
        else:
            raw2clean[col_name] = data_field
    return raw2clean


def get_instance_array_from_raw_col_name(col_name: str):
    return col_name.split("-")[1]


def get_raw_prefixes(col_names: List[str], raw2clean: dict) -> List[str]:
    """Clean raw prefixes to clean column names"""
    raw_col_prefixes = []
    for prefix in col_names:
        d = {k: v for k, v in raw2clean.items() if v.split("...")[0] == prefix}
        raw_prefixes = set([k.split("-")[0] for k in d.keys()])
        raw_col_prefixes.extend(raw_prefixes)
    return raw_col_prefixes


def get_raw_values_from_clean_codings(
    clean_values: List[str], codings: Optional[pd.DataFrame] = None
) -> List[str]:
    """
    Get raw values from clean values based on the codings file.
    """
    if codings is None:
        codings = pd.read_csv(CODING_PATH, sep="\t")
    raw_values = []
    for clean_value in clean_values:
        try:
            raw_value = codings[codings["Meaning"] == clean_value]["Value"].iloc[0]
        except:
            logging.error(f"Could not find raw value for clean value {clean_value}.")
            continue
        raw_values.append(raw_value)
    return raw_values


def get_data_field(
    raw_column_name: str, data_dict: Optional[pd.DataFrame] = None
) -> pd.Series:
    """Get data field from data dictionary based on raw column name (either field_id or full raw column name)"""
    if data_dict is None:
        data_dict = pd.read_csv(DATA_DICT_PATH, sep="\t")
    try:
        # raw_col_name is prefix
        field_id = int(raw_column_name)
    except ValueError:
        # raw_col_name is <prefix>-<instance>.<array>
        field_id, instance_array = raw_column_name.split("-")
        field_id = int(field_id)

    return data_dict[data_dict["FieldID"] == field_id].iloc[0]


def download_latest_ukb_metadata_files():
    """
    Download latest versions of original UKB metadata files (Codings and data dictionary).
    """
    # path to local file and url for download
    files = {
        "codings": {
            "local_path": CODING_PATH,
            "url": "https://biobank.ctsu.ox.ac.uk/~bbdatan/Codings.tsv",
        },
        "data_dict": {
            "local_path": DATA_DICT_PATH,
            "url": "https://biobank.ctsu.ox.ac.uk/~bbdatan/Data_Dictionary_Showcase.tsv",
        },
    }
    for file in files.keys():
        logging.info(f"Downloading {file} file.")
        local_path = files[file]["local_path"]
        url = files[file]["url"]
        wget.download(url, str(local_path))


def load_ukb_meta_files():
    """
    Load UKB metadata files (Codings, data dictionary, field2date) into pandas dataframes.
    """
    codings = pd.read_csv(CODING_PATH, sep="\t")
    data_dict = pd.read_csv(DATA_DICT_PATH, sep="\t")
    field2date = json.load(open(FIELD2DATE_PATH, "r"))[0]

    return codings, data_dict, field2date


def get_all_raw_col_names_from_field_id(
    df_columns: List[str], field_id: int
) -> List[str]:
    """
    Get all full raw column names corresponding to a raw column prefix.
    """
    return [col for col in df_columns if col.split("-")[0] == str(field_id)]


def get_long_format(
    df: pd.DataFrame,
    raw_col_prefix: str,
    date_col_name: Optional[str] = None,
    id_var: Optional[str] = "eid",
) -> pd.DataFrame:
    """Bring the dataframe into long format of all columns belonging to the raw_col_prefix. If `date_col_name` is not None, the corresponding date is included for each entry."""
    if df.index.name == "eid":
        df = df.reset_index()

    # subselect all target columns
    raw_col_names = get_all_raw_col_names_from_field_id(df.columns, raw_col_prefix)

    # put the dataframe from wide format into long format, such that each row contains one target value
    value_df_long = df.melt(
        id_vars=id_var,
        value_vars=raw_col_names,
        var_name="raw_col_name",
        value_name=raw_col_prefix,
    )
    value_df_long["instance_array"] = value_df_long["raw_col_name"].apply(
        get_instance_array_from_raw_col_name
    )

    if date_col_name is not None:
        # get dates corresponding to the target columns
        raw_col_names_date = get_all_raw_col_names_from_field_id(
            df.columns, date_col_name
        )
        date_df_long = df.melt(
            id_vars=id_var,
            value_vars=raw_col_names_date,
            var_name="raw_col_name",
            value_name=date_col_name,
        )
        date_df_long["instance_array"] = date_df_long["raw_col_name"].apply(
            get_instance_array_from_raw_col_name
        )

        # merge the value data frame with the date data frame based on instance_array and id_var (eid)
        df_long = value_df_long.merge(date_df_long, on=["eid", "instance_array"])[
            [id_var, raw_col_prefix, date_col_name, "instance_array"]
        ].dropna()
    else:
        df_long = value_df_long.dropna()

    return df_long


def filter_target_values(
    df: pd.DataFrame,
    raw_col_prefix: str,
    target_values: List[str],
    add_min_date: bool = True,
    field2date: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Search in all columns starting with `raw_col_prefix` for the target values.

    raw_col_prefix: column prefix of all columns containing target items
    target_values: list of raw target values
    add_min_date: whether to include the minimal date of the target values. The corresponding date column is determined by the field2date dictionary.

    """
    # TODO: clean up and make it a bit more generic

    if add_min_date:
        if field2date is None:
            field2date = json.load(open(FIELD2DATE_PATH, "r"))[0]
        date_col_name = field2date[raw_col_prefix]["date_column"]
    else:
        date_col_name = None

    # bring dataframe into long format
    df_long = get_long_format(
        df=df, raw_col_prefix=raw_col_prefix, date_col_name=date_col_name
    )

    # filter for target values
    # df_long["target_value"] = df_long[raw_col_prefix].isin(target_values)
    df_long["target_value"] = df_long[raw_col_prefix].apply(
        lambda x: max([x.startswith(target_value) for target_value in target_values])
    )

    aggs = {"target_value": "max"}

    if add_min_date:
        aggs[date_col_name] = "min"

    results = df_long.groupby("eid").agg(aggs)

    if add_min_date:
        results = results.rename(columns={date_col_name: "target_min_date"})
        # set min date to NaN if no target value was found
        results.loc[~results["target_value"], "target_min_date"] = None

    results_df = pd.DataFrame(index=df["eid"])
    results_df[results.columns] = results

    results_df["target_value"].fillna(False, inplace=True)

    assert (results_df.index == df["eid"]).all()
    return results_df.reset_index()


def get_strategy_for_data_field(
    data_field: pd.Series, strategy_dict: Optional[dict] = None
) -> str:
    """Strategy to map from data_field to prompt type"""

    if strategy_dict is not None:
        if data_field["FieldID"] in strategy_dict.keys():
            return strategy_dict[data_field["FieldID"]]

    if data_field["ValueType"] == "Continuous":
        strategy = "first"  # "mean_array"
    else:
        strategy = "first"
    return strategy


def round_on_string(s: pd.Series, decimals: int = 0) -> pd.Series:
    """Round on string"""

    if decimals == 0:
        pattern = r"(\d+)\.\d+"
    elif decimals == 1:
        pattern = r"(\d+\.\d)\d*"
    elif decimals > 1:
        pattern = r"(\d+\.\d{1,2})\d*"
    # Use a lambda function with regex to replace the part after dot
    replaced_series = s.apply(lambda x: re.sub(pattern, r"\1", x) if pd.notna(x) else x)

    # always replace ".0" and ".00" by ""
    replaced_series = replaced_series.str.replace(r"\.0{1,2}(?!\d)", "", regex=True)

    return replaced_series


def prefix_to_prompt(
    df: pd.DataFrame,
    raw_col_prefix: str | int,
    data_dict: Optional[pd.DataFrame] = None,
    codings: Optional[pd.DataFrame] = None,
    tabular: bool = False,
) -> pd.Series:
    """Write prompt to prefix"""

    raw_col_names = get_all_raw_col_names_from_field_id(df.columns, raw_col_prefix)
    data_field = get_data_field(raw_col_prefix, data_dict)

    _df = df[raw_col_names]

    prompt = f"""{data_field["Field"]}: """

    # decode values if coding exists
    if ~isNaN(
        data_field["Coding"]
    ):  #  TODO: and tabular is False (maybe coding is not necessary?)
        if codings is None:
            codings = pd.read_csv(CODING_PATH, sep="\t")
        coding = codings[codings["Coding"] == data_field["Coding"]]

        dtype = _df.dtypes.iloc[0]
        coding.loc[:, "Value"] = coding["Value"].astype(dtype)

        coding_dict = dict(zip(coding["Value"], coding["Meaning"]))

        def replace_value(value):
            return coding_dict.get(value, value)

        _df = _df.map(replace_value)

    strategy = get_strategy_for_data_field(data_field)

    # Option 1: take first value (Instance 0, Array 0)
    if strategy == "first":
        raw_col_name = f"{raw_col_prefix}-0.0"

        values = _df[raw_col_name].astype(str)
        na_mask = _df[raw_col_name].isna()

        prompt += values

    # Option 2: mean of all instances and arrays
    elif strategy == "mean_all":
        values = _df.mean(axis=1)
        na_mask = values.isna()
        prompt += values.astype(str)

    # Option 3: mean of all arrays (Instance 0)
    elif strategy == "mean_array":
        _df = _df.filter(regex=f"{raw_col_prefix}-0\.\d+")
        values = _df.mean(axis=1)
        na_mask = values.isna()
        prompt += values.astype(str)

    # Option 4: all arrays as list (Instance 0)
    elif strategy == "all_arrays":
        _df = _df.filter(regex=f"{raw_col_prefix}-0\.\d+")
        na_mask = _df.isna().all(axis=1)
        values = _df.apply(lambda x: ", ".join(x.dropna().astype(str)), axis=1)
        prompt += values.astype(str)

    # add units
    if isNaN(data_field["Units"]) is False:
        prompt += f" {data_field['Units']}"

    # round
    if data_field["ValueType"] == "Integer":
        digits = 0
    else:
        digits = 1

    # text data format
    prompt = round_on_string(prompt, decimals=digits)
    prompt[na_mask] = ""

    # tabular data format
    tab = values.copy()
    tab[na_mask] = np.nan

    return prompt, tab


def create_dtype_dict():
    _, data_dict, _ = load_ukb_meta_files()
    ukb_sample = pd.read_csv(DATA_PATH, nrows=1000)

    data_types = pd.DataFrame(ukb_sample.dtypes.T, columns=["dtype"]).reset_index(
        names=["raw_col_name"]
    )
    data_types["raw_col_prefix"] = data_types["raw_col_name"].apply(
        lambda x: x.split("-")[0]
    )

    expected_data_types = data_dict[["FieldID", "ValueType"]].copy()
    expected_data_types["FieldID"] = expected_data_types["FieldID"].astype(str)

    data_types = data_types.merge(
        expected_data_types, right_on="FieldID", left_on="raw_col_prefix", how="left"
    )

    map_expected_dtype = {
        "Continuous": np.dtype("float64"),
        "Categorical single": np.dtype("object"),
        "Date": np.dtype("object"),
        "Integer": np.dtype("float64"),
        "Text": np.dtype("object"),
        "Categorical multiple": np.dtype("object"),
        "Time": np.dtype("object"),
        "Compound": np.dtype("object"),
    }

    data_types["ValueType"] = data_types["ValueType"].fillna("Other")

    data_types["safe_dtype"] = (
        data_types["ValueType"].map(map_expected_dtype).fillna(np.dtype("object"))
    )

    data_types["safe_dtype"].value_counts()

    dict_dtypes = (
        data_types[["raw_col_name", "safe_dtype"]]
        .set_index("raw_col_name")
        .T.to_dict(orient="records")
    )[0]

    file_path = DATA_PATH.replace(".csv", "_dtypes.pickle")
    with open(file_path, "wb") as f:
        pickle.dump(dict_dtypes, f)


def get_ado_incidents(
    df: pd.DataFrame, types: List[str] = ["MI", "Stroke"], extended_def: bool = False
):
    """
    Get all MI + Stroke incidents according to the criteria used for the algorithmically defined outcomes (ADO).
    """

    ado_sources = {
        "ICD10": {
            "field_id": ukb_field_ids.ICD_10_FIELD_ID,
            "condition_field_id": ukb_field_ids.ICD_10_FIELD_ID,
            "filter_startswith": True,
        },
        "ICD9": {
            "field_id": ukb_field_ids.ICD_9_FIELD_ID,
            "condition_field_id": ukb_field_ids.ICD_9_FIELD_ID,
            "filter_startswith": True,
        },
        "Death_Primary": {
            "field_id": ukb_field_ids.DEATH_PRIMARY_CAUSE_FIELD_ID,
            "condition_field_id": ukb_field_ids.ICD_10_FIELD_ID,
            "filter_startswith": True,
        },
        "Death_Contributory": {
            "field_id": ukb_field_ids.DEATH_CONTRIBUTORY_CAUSE_FIELD_ID,
            "condition_field_id": ukb_field_ids.ICD_10_FIELD_ID,
            "filter_startswith": True,
        },
        "Self_Reported": {
            "field_id": ukb_field_ids.SELF_REPORTED_ILLNESS_FIELD_ID,
            "condition_field_id": ukb_field_ids.SELF_REPORTED_ILLNESS_FIELD_ID,
            "filter_startswith": True,  # .0 because of float values
        },
    }

    ado_definitions = {}
    for type in types:
        if type not in ["MI", "Stroke"]:
            raise ValueError("Only MI and Stroke are currently implemented.")
        ado_definitions[type] = get_ado_definition(type)

    custom_def_icd_10 = pd.DataFrame(
        {
            "field_id": ukb_field_ids.ICD_10_FIELD_ID,
            "code": outcome_definition.OUTCOME_ICD_10_CODES,
        }
    )
    custom_def_icd_9 = pd.DataFrame(
        {
            "field_id": ukb_field_ids.ICD_9_FIELD_ID,
            "code": outcome_definition.OUTCOME_ICD_9_CODES,
        }
    )
    custom_def = pd.concat([custom_def_icd_9, custom_def_icd_10])
    if extended_def:
        ado_definitions["custom"] = custom_def

    ado_definitions_summary = (
        pd.concat(ado_definitions.values(), axis=0)
        .groupby("field_id")
        .agg({"code": "unique"})
    )

    field2date = json.load(open(FIELD2DATE_PATH, "r"))[0]

    # Retrieve all incidents from the different ADO sources, based on the ADO definition
    incidents = pd.DataFrame()
    for source in ado_sources.keys():
        if ado_sources[source]["condition_field_id"] in ado_definitions_summary.index:
            incidents_source = get_icd_incidents(
                df=df,
                field_id=ado_sources[source]["field_id"],
                codes=ado_definitions_summary.loc[
                    ado_sources[source]["condition_field_id"], "code"
                ]
                .astype(str)
                .tolist(),
                filter_startswith=(
                    ado_sources[source]["filter_startswith"]
                    if "filter_startswith" in ado_sources[source]
                    else False
                ),
                field2date=field2date,
                rename=True,
            )
            if source == "Self_Reported":
                if len(incidents_source) > 0:
                    incidents_source["date"] = incidents_source["date"].apply(
                        parse_date
                    )
            incidents = pd.concat([incidents, incidents_source], axis=0)
        else:
            raise ValueError(f"No ADO definition found for source {source}.")

    return incidents


def parse_date(input_value: Union[float, str, int]):
    """Parse the date from the UKB interpolated format (e.g. 2019.1) to a standard date format. Necessary for the self-reported illness field."""
    try:
        year = int(input_value)
        days_fraction = input_value - year
        total_days_in_year = int(datetime.datetime(year, 12, 31).strftime("%-j"))
        days = int(total_days_in_year * days_fraction)
        date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=days)
        return date.strftime("%Y-%m-%d")
    except Exception:
        # logging.error(f"Could not parse date {input_value}.")
        return None


def get_icd_incidents(
    df: pd.DataFrame,
    field_id: int,
    codes: List[str],
    field2date: Optional[dict] = None,
    filter_startswith: Optional[bool] = False,
    rename: Optional[dict] = False,
):
    """
    In the UKB, ICD codes are stored in wide format (e.g. 41270-0.* for all codes that occurred).
    Their corresponding date is the date each diagnosis was first recorded, and the field_id of that field can be found in the field2date mapping.
    Only the first occurence of the ICD code is recorded. Hence, if a participant has the same diagnosis more than once, we only see the first date.

    Args:
        df (pd.DataFrame): UKB dataset
        field_id (int): field_id (only the prefix, e.g. 41270)
        codes (List[str]): list of target codes, in the correct value format (TODO)
        filter_startswith (bool): if True, filter for all codes that start with the target code, not only the exact match
    """
    if field2date is None:
        field2date = json.load(open(FIELD2DATE_PATH, "r"))[0]
    date_col_name = field2date[str(field_id)]["date_column"]

    if str(date_col_name) not in set(col.split("-")[0] for col in df.columns).union(
        set(df.columns)
    ):
        raise ValueError(f"Date column {date_col_name} not found in dataframe.")
    if str(field_id) not in set(col.split("-")[0] for col in df.columns).union(
        set(df.columns)
    ):
        raise ValueError(f"Field ID {field_id} not found in dataframe.")

    # codes are stored without "." (A001 instead of A00.1) in df
    codes = [code.replace(".X", "") for code in codes]
    codes = [code.replace(".", "") for code in codes]

    # bring dataframe into long format
    df_long = get_long_format(
        df=df,
        raw_col_prefix=field_id,
        date_col_name=date_col_name if type(date_col_name) == int else None,
    )

    # in some cases, there is only one date to multiple instances/arrays. In this case, the date col is "fieldid-0.0".
    if type(date_col_name) == str:
        df_long = df_long.merge(df[["eid", date_col_name]], on="eid", how="left")

    # filter for target values
    if filter_startswith:
        df_long["target_value"] = (
            df_long[field_id]
            .astype(str)
            .apply(
                lambda x: max([x.startswith(target_value) for target_value in codes])
            )
        )
    else:
        df_long["target_value"] = df_long[field_id].astype(str).isin(codes)

    df_long_filtered = df_long[df_long["target_value"]]

    if rename:
        df_long_filtered = df_long_filtered.rename(
            columns={field_id: "code", date_col_name: "date"}
        )
        df_long_filtered["source"] = field2date[str(field_id)]["desc_data"]

    return df_long_filtered


# ADO functions


def read_ado_definition(sheet: str, header: int = 5) -> pd.DataFrame:
    """Read the ADO definition from a particular sheet in the ADO Excel file.

    Args:
        sheet (str): Name of the Sheet of the specific outcome
        header (int, optional): Number of row in which the columns start. Defaults to 5.

    Returns:
        pd.DataFrame: DataFrame containing the ADO definition
    """
    ado_xls = pd.ExcelFile(ALGORITHMICALLY_DEFINED_OUTCOMES_PATH)
    ado_df = pd.read_excel(ado_xls, sheet_name=sheet, header=header)
    return ado_df


def get_ado_field_id(code_type: str, code: str) -> Union[str, int]:
    """Map ADO Code Type to Field ID, or find Field ID in the code itself."""
    map = {
        "ICD 9": ukb_field_ids.ICD_9_FIELD_ID,
        "ICD 10": ukb_field_ids.ICD_10_FIELD_ID,
    }
    if code_type in map.keys():
        return map[code_type]
    elif code_type == "UK Biobank Self Report":
        matches = re.findall(r"Field (\d+) Code \d+", code)[0]
        return int(matches)
    else:
        raise ValueError(f"Unknown Code Type {code_type}.")


def get_ado_code_value(code_type: str, code: str) -> Union[str, int]:
    """Clean up ADO code value."""
    if code_type == "UK Biobank Self Report":
        matches = re.findall(r"Field \d+ Code (\d+)", code)[0]
        return int(matches)
    else:
        return code


def get_ado_definition(outcome: str) -> pd.DataFrame:
    """
    Get the ADO definition for a specific outcome. An ADO definition lists all the codes used to define the outcome in particular fields.

    Args:
        outcome (str): Outcome of interest. Currently, only implemented for "MI" and "Stroke".
    """
    if outcome == "MI":
        sheet = "Myocardial infarction - P1"
        header = 5
    elif outcome == "Stroke":
        sheet = "Stroke - P1"
        header = 5
    else:
        raise ValueError(
            "No valid ADO outcome provided. Currently, only implemented for MI and Stroke."
        )

    ado_def = read_ado_definition(sheet=sheet, header=header)
    ado_def["field_id"] = ado_def.apply(
        lambda row: get_ado_field_id(code_type=row["Code Type"], code=row["Code"]),
        axis=1,
    )
    ado_def["code"] = ado_def.apply(
        lambda row: get_ado_code_value(code_type=row["Code Type"], code=row["Code"]),
        axis=1,
    )
    return ado_def


# run download_latest_ukb_metadata_files() on main
if __name__ == "__main__":
    download_latest_ukb_metadata_files()
