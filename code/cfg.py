import json
import munch


def process_config(config_file):
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
    config = munch.Munch(config_dict)
    return config
