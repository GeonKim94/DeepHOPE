import yaml
from argparse import ArgumentParser
from pathlib import Path
from os.path import basename, splitext

def load_yaml_config(path_config):
    with open(path_config, "r") as file:
        return yaml.safe_load(file)

def parse_args():
    parser = ArgumentParser()
    
    parser.add_argument("--config", type=str, default="/data4/tutorial/config/preset1.yaml", help="Path to config file")
    args = parser.parse_args()
    
    # Load YAML config
    config = load_yaml_config(args.config)

    return config, splitext(basename(args.config))[0]
