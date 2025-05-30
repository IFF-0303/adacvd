import yaml


# Function to read the YAML file and process feature groups
def process_feature_groups(file_path):
    with open(file_path, "r") as file:
        # Load the YAML content
        feature_groups = yaml.safe_load(file)

    # Initialize lists to store all field ids and features
    all_field_ids = []
    all_features = []

    # Iterate through each feature group
    for feature_group in feature_groups.keys():
        # if feature_group != "blood_samples":
        #     continue
        field_ids = feature_groups[feature_group].get("field_ids", [])
        features = feature_groups[feature_group].get("features", [])

        # Add the field ids and features to the respective lists
        all_field_ids.extend(field_ids)
        all_features.extend(features)

    return list(set(all_field_ids)), list(set(all_features))


if __name__ == "__main__":
    file_path = "config/ukb_data/feature_groups/meta/feature_groups.yaml"
    field_ids, features = process_feature_groups(file_path)

    print("All Field IDs:")
    print(field_ids)

    print("\nAll Features:")
    print(features)

    # save to config file
    with open("config/ukb_data/Risk_Score_Inputs.yaml", "r") as file:
        template = yaml.safe_load(file)

    template["data"]["feature_field_ids"] = [str(x) for x in field_ids]
    template["data"]["features"] = features
    template["data"]["dataset_name"] = "all_feature_groups"

    with open("config/ukb_data/all_feature_groups.yaml", "w") as file:
        yaml.dump(template, file)
