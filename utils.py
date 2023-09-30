import os.path as op
import json


def load_config(config_path):
    config_file = op.join(config_path)
    with open(config_file) as f:
        config_data = json.load(f)
    return config_data