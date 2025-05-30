# UK Biobank data preparation

Here, we describe how to process the UK Biobank (UKB) dataset.

## Raw UKB data

The raw UKB dataset is stored in `assets/ukb/...` and consists of the following files:

- Data
    - `ukbxxxxxx.csv`: Raw CSV data. This file contains 1 row per participant and 18864 columns. Each participant has a unique `eid`. For information on the columns, check out the [UKB Showcase](https://biobank.ndph.ox.ac.uk/showcase/search.cgi)
- Meta Information
    - `algorithm_outcome_codes.xlsx`: Information on how the UKB team derived the outcome variables. Downloaded from [UKB website](https://biobank.ndph.ox.ac.uk/ukb/docs.cgi?id=0).
    - `codings_raw.tsv`: Table containing the coding information of UKB variables. Downloaded from [UKB website](https://biobank.ndph.ox.ac.uk/ukb/docs.cgi?id=0).
    - `data_dictionary_raw.tsv`: Table containing the data dictionary, i.e., information on each column. Downloaded from [UKB website](https://biobank.ndph.ox.ac.uk/ukb/docs.cgi?id=0).


## Processing

The preprocessing of the UKB dataset is handled by the script [pandora/data/preprocess.py](pandora/data/preprocess.py). This script performs the following steps:

1. **Feature Extraction**: Extracts features from the raw UKB dataset based on field IDs and custom feature definitions.
2. **Target Generation**: Creates target variables using predefined algorithms or extended definitions.
3. **Prompt Creation**: Generates text-based prompt parts for each feature.
4. **Data Splitting**: Processes the dataset in chunks for efficient handling of large data.

### Outputs

The script generates the following files:

- **Prompt Parts**: Intermediate components used to construct prompts.
- **Tabular Inputs**: Processed tabular data for machine learning models.
- **Targets**: Outcome variables for CVD risk prediction.
- **Subset with No Previous Target**: A subset of participants with no prior incidents based on the target definition.

These outputs are saved in the `assets/` directory under their respective subfolders (e.g., `tab_datasets`, `prompt_parts`, `targets`).

To run the preprocessing, run the following command. Make sure the desired features are listed in the specified config file.

```
python scripts/preprocessing/submit_preprocessing_jobs.py --config_file config/ukb_data/all_feature_groups.yaml
```
