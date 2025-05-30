# Cluster Data Files

Shared directory: `/fast/groups/hfm-users/pandora-med-box/`

## Assets (`/fast/groups/hfm-users/pandora-med-box/assets`)

This directory contains our dataset in different formats, explained in the following.

- `ukb`
    - Contains the *raw* CSV data files that we receive from Sergios. Currently, there are three different versions: `ukb_2023_07`, `ukb_2023_12`, `ukb_2024_02`. Use the most recent one whenever possible.
    - Contains files with meta information on the UKB dataset. Paths are stored in [ukb_data_utils](pandora/data/ukb_data_utils.py).
        - `algorithm_outcome_codes.xlsx`: Information on how the UKB team derived the outcome variables. Downloaded from [UKB website](https://biobank.ndph.ox.ac.uk/ukb/docs.cgi?id=0).
        - `codings_raw.tsv`: Table containing the coding information of UKB variables. Downloaded from [UKB website](https://biobank.ndph.ox.ac.uk/ukb/docs.cgi?id=0).
        - `data_dictionary_raw.tsv`: Table containing the data dictionary, i.e., information on each column. Downloaded from [UKB website](https://biobank.ndph.ox.ac.uk/ukb/docs.cgi?id=0).
        - `field2date.json`: A dictionary that we have created ourselves to map certain columns to their respective date column.

- `risk_scores`
    - Contains the pre-computed medical risk scores as CSV files.

- `text_datasets`
    - Contains pre-computed prompt datasets in `jsonl` format. Each dataset should contain the prompt, the label and some meta information.
    - The datasets are stored in subdirectories based on different data versions. Use the most recent one.
    - The datasets are created using [this script](pandora/data/preprocess.py) with their respective configuration file.
    - Important files:
        - `ukb_2024_02/Risk_Score_Inputs.jsonl`: Text dataset containing the input variables used for the medical risk scores. This can be seen as the most basic feature set.

- `tab_datasets`
    - Contains pre-computed tabular datasets in `parquet` format. These can be used for e.g. training lightgbm models.

- `prompt_parts` (prefered approach)
    - Contains pre-computed datasets of prompt parts per feature group in `parquet` format. These prompt parts can be glued together in during training by specifying the feature groups to use.
    - The file `prompt_parts.parquet` contains all pre-computed parts.


## Results (`/fast/groups/hfm-users/pandora-med-box/results`)

Shared directory for storing model runs and evaluation results.
