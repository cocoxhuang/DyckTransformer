from pathlib import Path
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_config(config, config_path):
    with open(config_path, 'w') as file:
        yaml.dump(config, file)