
import yaml

def load_config(file_path):
    with open(file_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config

global_params = load_config("config.yml")

def print_features(features):
    for k,v in features.copy().items():

        if "denoised_data" in v:
            v.pop("denoised_data")
        print(f"{k}: {v}")

