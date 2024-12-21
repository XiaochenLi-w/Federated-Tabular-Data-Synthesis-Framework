# this script is used to train the synthesizer
import os
import sys

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

import argparse
from lib.commons import load_config
from lib.config import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="great")
    parser.add_argument("--dataset", "-d", type=str, default="adult")
    parser.add_argument("--cuda", "-c", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    # load template config
    # Check if config file exists, if not create necessary directories
    config_dir = os.path.join(config.root_dir, "exp", args.dataset, args.model)
    config_path = os.path.join(config_dir, "config.toml")

    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
        print(f"Created directory: {config_dir}")

    if not os.path.exists(config_path):
        print(f"Warning: Config file not found at {config_path}")
        print("Please ensure the config file exists before running training")
        sys.exit(1)

    model_config = load_config(config_path)

    # dynamically import model interface
    synthesizer = __import__("synthesizer." + args.model, fromlist=[args.model])
    print("Training {0} on {1}".format(args.model, args.dataset))
    synthesizer.train(model_config, args.cuda, args.seed)


def train():
    for dataset in config.ml.datasets:
        # Your training code
        params_path = config.tuned_params_path / dataset
        # ...


if __name__ == "__main__":
    main()
