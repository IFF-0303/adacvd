# UK Biobank data preparation

## Raw UKB data

The raw UKB dataset is stored in `assets/ukb/...` and consists of the following files:

- Data
    - `ukbxxxxxx.csv`: Raw CSV data. This file contains 1 row per participant and many columns with features. Each participant has a unique `eid`. For information on the columns, check out the [UKB Showcase](https://biobank.ndph.ox.ac.uk/showcase/search.cgi)
- Meta Information
    - `algorithm_outcome_codes.xlsx`: Information on how the UKB team derived the outcome variables. Downloaded from [UKB website](https://biobank.ndph.ox.ac.uk/ukb/docs.cgi?id=0).
    - `codings_raw.tsv`: Table containing the coding information of UKB variables. Downloaded from [UKB website](https://biobank.ndph.ox.ac.uk/ukb/docs.cgi?id=0).
    - `data_dictionary_raw.tsv`: Table containing the data dictionary, i.e., information on each column. Downloaded from [UKB website](https://biobank.ndph.ox.ac.uk/ukb/docs.cgi?id=0).


## Processing

The preprocessing of the UKB dataset is handled by the [`pandora/data/preprocess.py`](../pandora/data/preprocess.py) script. This script performs the following steps:

1. **Feature Extraction**: Extracts meaningful features from the raw UKB dataset based on field IDs and custom feature definitions.
2. **Target Generation**: Creates target variables based on the definition of the outcome.
3. **Prompt Creation**: Generates text-based prompt parts for each feature.
4. **Chunked processing**: Processes the dataset in chunks for efficient handling.

### Outputs

The script generates the following output files, stored in `assets/`:

- **Prompt Parts**: Text-based prompt parts for each feature used to construct the structured prompts.
- **Tabular Datasets**: Processed tabular data.
- **Targets**: Outcome variables for CVD risk prediction.
- **Subset with No Previous Target**: A subset of participants with no prior incidents based on the target definition.

These outputs are saved in the `assets/` directory under their respective subfolders (e.g., `tab_datasets`, `prompt_parts`, `targets`).

To run the preprocessing, run the following command. Make sure the desired features are listed in the specified config file, as for example in this file [`config/ukb_data/all_feature_groups.yaml`](../config/ukb_data/all_feature_groups.yaml).


```bash
python scripts/preprocessing/submit_preprocessing_jobs.py --config_file config/ukb_data/all_feature_groups.yaml
```

## Clinical Notes

To generate clinical notes based on the structured inputs, run the following command:

```bash
python exploration/text_datasets/generate_texts.py --train_dir={train_dir} --device cuda
```

The notes are saved as a CSV file.