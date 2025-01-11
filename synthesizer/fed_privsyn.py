import os
import sys

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

import pickle
import pandas as pd

from lib.config import config
from lib.commons import improve_reproducibility, read_csv

from synthesizer.privsyn_lib.data_loader import DataLoader
from synthesizer.privsyn_lib.data_trasnformer import DataTransformer
from synthesizer.privsyn_lib.core_fedsyn import FedprivSyn

def train(args, cuda, seed=0):
    improve_reproducibility(seed)

    path_params = args["path_params"]
    model_params = args["model_params"]

    epsilon = model_params["epsilon"]
    delta = model_params["delta"]
    max_bins = model_params["max_bins"]
    update_iterations = model_params["update_iterations"]

    budget_split = {"noise_to_one_way_marginal": model_params["noise_to_one_way_marginal"] * epsilon, "noise_to_two_way_marginal": model_params["noise_to_two_way_marginal"] * epsilon, "two-way-publish": model_params["two-way-publish"] * epsilon}

    # prepare data
    train_data_pd, meta_data, discrete_columns = read_csv(
        path_params["train_data"], path_params["meta_data"]
    )
    val_data_pd, _, _ = read_csv(path_params["val_data"], path_params["meta_data"])
   
    # combine train and val data
    data_pd = pd.concat([train_data_pd, val_data_pd], ignore_index=True, sort=False)

    data_transformer = DataTransformer(max_bins)

    transformed_data = data_transformer.fit_transform(data_pd, discrete_columns)
    encode_mapping = data_transformer.get_mapping()

    # dataloader initialization
    data_loader = DataLoader()
    data_loader.load_data(private_data=transformed_data, encode_mapping=encode_mapping)

    synthesizer = FedprivSyn(
        data_loader, update_iterations, epsilon, delta, sensitivity=1, budget_split_method=budget_split
    )
    synthesizer.train()

    model = {}
    model["learned_privsyn"] = synthesizer
    model["data_transformer"] = data_transformer
    model["data_loader"] = data_loader

    # save training record and model
    path_params = args["path_params"]
    os.makedirs(os.path.dirname(path_params["out_model"]), exist_ok=True)
    pickle.dump(model, open(path_params["out_model"], "wb"))
