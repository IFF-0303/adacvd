import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from datasets import Dataset, load_dataset
from IPython.display import display

from adacvd.data import ukb_data_utils, ukb_features, ukb_field_ids
from adacvd.utils import metrics

# load raw csv file and meta data
df_ = pd.read_csv(
    f"{ukb_data_utils.ASSETS_PATH}/ukb/ukb_2024_02/ukb677731.csv",
    nrows=2,
    low_memory=False,
)

codings, data_dict, field2date = ukb_data_utils.load_ukb_meta_files()
raw2clean = ukb_data_utils.get_mapping_raw2clean(
    col_names=df_.columns, only_clean_col_name=True, data_dict=data_dict
)

raw2clean_fields = ukb_data_utils.get_mapping_raw2clean(
    col_names=df_.columns, only_clean_col_name=False, data_dict=data_dict
)

# specify subgroup columns to load full data on subgroups from raw csv file

FIELD_IDS = [
    # potential subgroup variables
    ukb_field_ids.AGE_AT_ASSESSMENT_CENTER,
    ukb_field_ids.SEX,
    ukb_field_ids.ETHNIC_BACKGROUND,
    ukb_field_ids.DIABETES,
    ukb_field_ids.SMOKING_STATUS,
    ukb_field_ids.BMI,
    ukb_field_ids.QUALIFICATIONS,
    20115,  # country of birth
    1647,  # country of birth UK elsewhere
    20118,  # home area population density - urban/rural
    # sociodemographic variables
    ukb_field_ids.NUMBER_IN_HOUSEHOLD,
    ukb_field_ids.TIME_EMPLOYED_IN_CURRENT_MAIN_JOB,
    ukb_field_ids.LENGTH_OF_WORKING_WEEK,
    ukb_field_ids.JOB_INVOLVES_HEAVY_MANUAL_OR_PHYSICAL_WORK,
    ukb_field_ids.JOB_INVOLVES_NIGHT_SHIFT,
    ukb_field_ids.YEAR_IMMIGRATED_TO_UK,
    # socio new
    ukb_field_ids.AVG_TOTAL_HOUSEHOLD_INCOME,
    ukb_field_ids.NUM_VEHICLES_IN_HOUSEHOLD,
    ukb_field_ids.JOBS_INVOLVES_SHIFT_WORK,
    ukb_field_ids.QUALIFICATIONS,
    ukb_field_ids.PRIVATE_HEALTHCARE,
    # variables needed for identification of previous MACE
    ukb_field_ids.YEAR_OF_BIRTH,
    ukb_field_ids.MONTH_OF_BIRTH,
    ukb_field_ids.ICD_10_FIELD_ID,
    ukb_field_ids.ICD_10_DATE,
    ukb_field_ids.ICD_9_FIELD_ID,
    ukb_field_ids.ICD_9_DATE,
    ukb_field_ids.DEATH_PRIMARY_CAUSE_FIELD_ID,
    ukb_field_ids.DEATH_CONTRIBUTORY_CAUSE_FIELD_ID,
    ukb_field_ids.DATE_OF_DEATH,
    ukb_field_ids.SELF_REPORTED_ILLNESS_FIELD_ID,
    ukb_field_ids.SELF_REPORTED_ILLNESS_DATE,
]

cols = ["eid"]
for field_id in FIELD_IDS:
    cols.extend(
        ukb_data_utils.get_all_raw_col_names_from_field_id(
            df_.columns, field_id=field_id
        )
    )

print("Load full dataset")
df = pd.read_csv(
    f"{ukb_data_utils.ASSETS_PATH}/ukb/ukb_2024_02/ukb677731.csv",
    usecols=cols,
    low_memory=False,
    # nrows=1000,
)
print("Loaded full dataset")
df = df.set_index("eid")

# create features from subgroup variables
subgroups = pd.DataFrame(index=df.index)
subgroups["Sex"] = ukb_features.get_feature_values_from_field_id(df, ukb_field_ids.SEX)
subgroups["Age Group"] = pd.cut(
    ukb_features.get_feature_values_from_field_id(
        df, ukb_field_ids.AGE_AT_ASSESSMENT_CENTER
    )[str(ukb_field_ids.AGE_AT_ASSESSMENT_CENTER)],
    bins=[38, 50, 60, 70],
).astype(str)
subgroups["Ethnic Background"] = ukb_features.get_feature_values_from_field_id(
    df, ukb_field_ids.ETHNIC_BACKGROUND
)

subgroups["Diabetes"] = ukb_features.get_feature_values_from_field_id(
    df, ukb_field_ids.DIABETES
)
subgroups["Smoking Status"] = ukb_features.get_feature_values_from_field_id(
    df, ukb_field_ids.SMOKING_STATUS
)

subgroups["BMI Group"] = ukb_features.get_feature_values_from_field_id(
    df, ukb_field_ids.BMI
)
subgroups["BMI Group"] = pd.cut(
    subgroups["BMI Group"],
    bins=[0, 18.5, 25, 30, 100],
    labels=["Underweight", "Normal", "Overweight", "Obese"],
).astype(str)

# previous MACE
period = ukb_features.get_period_from_baseline_assessment(df, years=10)
subgroups["Previous MACE"] = ukb_features.previous_MACE(
    df, start_dates=period["start_date"]
)

subgroups["Highest Qualification"] = ukb_features.highest_qualification(
    df.filter(regex=str(ukb_field_ids.QUALIFICATIONS))
)

subgroups["Average Household Income"] = ukb_features.get_feature_values_from_field_id(
    df, ukb_field_ids.AVG_TOTAL_HOUSEHOLD_INCOME
)

subgroups["Number of Vehicles in Household"] = (
    ukb_features.get_feature_values_from_field_id(
        df, ukb_field_ids.NUM_VEHICLES_IN_HOUSEHOLD
    )
)

subgroups["Job Involves Shift Work"] = ukb_features.get_feature_values_from_field_id(
    df, ukb_field_ids.JOBS_INVOLVES_SHIFT_WORK
)


def categorize_shift_work(response):
    if response in {"Usually", "Always", "Sometimes"}:
        return "Yes"
    elif response == "Never/rarely":
        return "No"
    else:
        return pd.NA


subgroups["Shift Work Category"] = subgroups["Job Involves Shift Work"].map(
    categorize_shift_work
)

subgroups["Private Healthcare"] = ukb_features.get_feature_values_from_field_id(
    df, ukb_field_ids.PRIVATE_HEALTHCARE
)


def categorize_private_healthcare(response):
    if pd.isna(response):
        return pd.NA
    if "Yes" in response:
        return "Yes"
    elif "No, never" in response:
        return "No"
    else:
        return pd.NA  # Assign missing values for other cases


subgroups["Private Healthcare Grouped"] = subgroups["Private Healthcare"].map(
    categorize_private_healthcare
)


qualifications = ukb_features.get_feature_values_from_field_id(
    df, ukb_field_ids.QUALIFICATIONS
)[str(ukb_field_ids.QUALIFICATIONS)]

mask = qualifications.map(type).isin([list, set])
qualifications.loc[mask] = qualifications.loc[mask].map(set)
qualifications.loc[~mask] = [set()] * (~mask).sum()

unique_qualifications = set([x for y in qualifications.dropna() for x in y])

qualifications_one_hot = pd.DataFrame(
    [[1 if q in qs else 0 for q in unique_qualifications] for qs in qualifications],
    columns=["Qualification " + x for x in unique_qualifications],
    index=df.index,
)

higher_education = {
    "College or University degree",
    "NVQ or HND or HNC or equivalent",
    "Other professional qualifications eg: nursing, teaching",
}

secondary_education = {
    "A levels/AS levels or equivalent",
    "O levels/GCSEs or equivalent",
    "CSEs or equivalent",
}

no_formal_education = {
    "None of the above",
    "Prefer not to answer",
}


# Function to categorize qualifications
def categorize_qualifications(qs):
    if higher_education & qs:
        return "Higher Education"
    elif secondary_education & qs:
        return "Secondary Education"
    elif no_formal_education & qs:
        return "No Formal Education"
    else:
        return pd.NA


qualifications_grouped = qualifications.map(categorize_qualifications)
subgroups["Qualifications Grouped"] = qualifications_grouped

# sociodemographics

replace_str = lambda x: None if type(x) == str else x

subgroups["Number in Household"] = pd.cut(
    ukb_features.get_feature_values_from_field_id(df, ukb_field_ids.NUMBER_IN_HOUSEHOLD)
    .iloc[:, 0]
    .apply(replace_str),
    bins=[0, 1, 2, 4, 100],
    labels=["1", "2", "3-4", "5+"],
)

subgroups["Time Employed in Current Main Job"] = pd.cut(
    ukb_features.get_feature_values_from_field_id(
        df, ukb_field_ids.TIME_EMPLOYED_IN_CURRENT_MAIN_JOB
    )
    .iloc[:, 0]
    .apply(replace_str),
    bins=[0, 5, 10, 20, 100],
    labels=["0-5", "5-10", "10-20", "20+"],
)

subgroups["Length of Working Week"] = pd.cut(
    ukb_features.get_feature_values_from_field_id(
        df, ukb_field_ids.LENGTH_OF_WORKING_WEEK
    )
    .iloc[:, 0]
    .apply(replace_str),
    bins=[0, 10, 20, 30, 40, 50, 100],
    labels=["0-10", "10-20", "20-30", "30-40", "40-50", "50+"],
)

subgroups["Job Involves Heavy Manual or Physical Work"] = (
    ukb_features.get_feature_values_from_field_id(
        df, ukb_field_ids.JOB_INVOLVES_HEAVY_MANUAL_OR_PHYSICAL_WORK
    )
)

subgroups["Job Involves Night Shift"] = ukb_features.get_feature_values_from_field_id(
    df, ukb_field_ids.JOB_INVOLVES_NIGHT_SHIFT
)

subgroups["Year Immigrated to UK"] = pd.cut(
    ukb_features.get_feature_values_from_field_id(
        df, ukb_field_ids.YEAR_IMMIGRATED_TO_UK
    )
    .iloc[:, 0]
    .apply(replace_str),
    bins=[1945, 1960, 1975, 1990, 2024],
    labels=["1945-1960", "1960-1975", "1975-1990", "1990+"],
)

subgroups["County of Birth"] = ukb_features.get_feature_values_from_field_id(df, 20115)

subgroups["Country of Birth UK Elsewhere"] = (
    ukb_features.get_feature_values_from_field_id(df, 1647)
)

subgroups["Home Area Population Density"] = (
    ukb_features.get_feature_values_from_field_id(df, 20118)
)

# Define grouping categories
urban = {
    "England/Wales - Urban - less sparse",
    "Scotland - Large Urban Area",
    "Scotland - Other Urban Area",
}

suburban = {
    "England/Wales - Town and Fringe - less sparse",
    "Scotland - Accessible Small Town",
    "Scotland - Remote Small Town",
    "Scotland - Very Remote Small Town",
}

rural = {
    "England/Wales - Village - less sparse",
    "England/Wales - Hamlet and Isolated Dwelling - less sparse",
    "England/Wales - Village - sparse",
    "England/Wales - Hamlet and Isolated dwelling - sparse",
    "Scotland - Accessible Rural",
    "Scotland - Remote Rural",
    "Scotland - Very Remote Rural",
}

sparse_urban_fringe = {
    "England/Wales - Urban - sparse",
    "England/Wales - Town and Fringe - sparse",
}

unknown = {"Postcode not linkable"}


def categorize_area(area):
    if pd.isna(area):
        return pd.NA
    if area in urban:
        return "Urban"
    elif area in suburban:
        return "Suburban & Small Town"
    elif area in rural:
        return "Rural"
    elif area in sparse_urban_fringe:
        return "Sparse Urban & Fringe"
    elif area in unknown:
        return "Unknown"
    else:
        return pd.NA  # Assign missing value if not categorized


subgroups["Home Area Category"] = subgroups["Home Area Population Density"].map(
    categorize_area
)


# check size per subgroup on train and eval data
# for small subgroups, group into "other" category
for col in subgroups.columns:
    subgroup_stats = (
        subgroups.groupby(col, dropna=False, observed=True).size().to_frame("n (all)")
    )

    display(subgroup_stats.sort_values(by="n (all)", ascending=False))

import json

subgroups_file = ukb_data_utils.ASSETS_PATH / "subgroups/subgroups.json"

all_eids = df.index.values.tolist()
subgroups_dict = {}

for subgroup in subgroups.columns:
    subgroups_dict[subgroup.replace(" ", "_")] = {}
    for subgroup_value in subgroups[subgroup].unique().tolist():
        if str(subgroup_value) == "nan":
            continue
        print(subgroup_value)

        subset = subgroups[subgroups[subgroup] == subgroup_value].index.values.tolist()
        subgroups_dict[subgroup.replace(" ", "_")][
            str(subgroup_value).replace(" ", "_")
        ] = subset


# qualifications
subgroups_dict["Qualifications"] = {}
for col in qualifications_one_hot.columns:
    subgroup_value = col
    subset = qualifications_one_hot[
        qualifications_one_hot[col] == 1
    ].index.values.tolist()
    subgroups_dict["Qualifications"][str(subgroup_value).replace(" ", "_")] = subset


with open(subgroups_file, "w") as f:
    json.dump(subgroups_dict, f, indent=4)
