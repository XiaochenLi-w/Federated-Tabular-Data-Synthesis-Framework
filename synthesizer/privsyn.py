import os
import sys

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

import pickle
import optuna
import pandas as pd

from lib.tune_helper import fidelity_tuner, utility_tuner
from lib.config import config
from lib.commons import improve_reproducibility, read_csv

from synthesizer.privsyn_lib.postprocessor import RecordPostprocessor
from synthesizer.privsyn_lib.data_loader import DataLoader
from synthesizer.privsyn_lib.data_trasnformer import DataTransformer
from synthesizer.privsyn_lib.core import PrivSyn


def train(args, cuda, seed=0):
    improve_reproducibility(seed)

    path_params = args["path_params"]
    model_params = args["model_params"]

    epsilon = model_params["epsilon"]
    delta = model_params["delta"]
    max_bins = model_params["max_bins"]
    update_iterations = model_params["update_iterations"]

    # budget allocation for DP
    ratio = model_params["ratio"] if "ratio" in model_params else None

    budget_split = {"one-way-publish": model_params["one-way-publish"] * epsilon, "two-way-select": model_params["two-way-select"] * epsilon,
    "two-way-publish": model_params["two-way-publish"] * epsilon, "combine": model_params["combine"] * epsilon, "client_num": model_params["client_num"], "delta": model_params["delta"]}

    # prepare data
    train_data_pd, meta_data, discrete_columns = read_csv(
        path_params["train_data"], path_params["meta_data"]
    )
    val_data_pd, _, _ = read_csv(path_params["val_data"], path_params["meta_data"])
    if tune:
        data_pd = train_data_pd
    else:
        # combine train and val data
        data_pd = pd.concat([train_data_pd, val_data_pd], ignore_index=True, sort=False)

    data_transformer = DataTransformer(max_bins)

    transformed_data = data_transformer.fit_transform(data_pd, discrete_columns)
    encode_mapping = data_transformer.get_mapping()

    # dataloader initialization
    data_loader = DataLoader()
    data_loader.load_data(private_data=transformed_data, encode_mapping=encode_mapping)

    synthesizer = PrivSyn(
        data_loader, update_iterations, epsilon, delta, sensitivity=1, budget_split_method=budget_split, ratio=ratio
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


def sample(args, n_samples=0, seed=0):
    improve_reproducibility(seed)

    path_params = args["path_params"]

    train_data_pd, meta_data, discrete_columns = read_csv(
        path_params["train_data"], path_params["meta_data"]
    )
    val_data_pd, _, _ = read_csv(path_params["val_data"], path_params["meta_data"])
    # combine train and val data
    data_pd = pd.concat([train_data_pd, val_data_pd], ignore_index=True, sort=False)

    model = pickle.load(open(path_params["out_model"], "rb"))
    learned_privsyn = model["learned_privsyn"]
    data_transformer = model["data_transformer"]
    data_loader = model["data_loader"]

    # sample the same number of data as the real data
    n_samples = n_samples if n_samples > 0 else len(data_pd)
    syn_data = learned_privsyn.synthesize(num_records=n_samples)

    syn_data = data_transformer.inverse_transform(syn_data)

    # # post-processing generated data, map records with grouped/binned attribute back to original attributes
    # print("********************* START POSTPROCESSING ***********************")
    # postprocessor = RecordPostprocessor()
    # syn_data = postprocessor.post_process(syn_data, data_loader.decode_mapping)
    # syn_data = syn_data[data_pd.columns]
    # print("------------------------>synthetic data post-processed:")
    # print(syn_data)

    os.makedirs(os.path.dirname(path_params["out_data"]), exist_ok=True)
    # save synthetic data to csv
    syn_data.to_csv(path_params["out_data"], index=False)


def tune(config, cuda, dataset, seed=0):
    """
    tune privsyn
    """

    def privsyn_objective(trial):
        # configure the model for this trial
        model_params = {}
        model_params["epsilon"] = 100000000.0
        model_params["delta"] = 3.4498908254380166e-11
        model_params["max_bins"] = trial.suggest_int("max_bins", 10, 50)
        model_params["update_iterations"] = trial.suggest_int(
            "update_iterations", 10, 100
        )

        # store configures
        trial.set_user_attr("config", model_params)
        config["model_params"] = model_params

        try:
            # train the model
            model = train_wrapper(config, tune=True)
            learned_privsyn = model["learned_privsyn"]
            data_transformer = model["data_transformer"]

            # sample synthetic data
            n_samples = meta_data["train_size"] + meta_data["val_size"]
            syn_data = learned_privsyn.synthesize(num_records=n_samples)
            sampled = data_transformer.inverse_transform(syn_data)
            os.makedirs(os.path.dirname(path_params["out_data"]), exist_ok=True)
            sampled.to_csv(path_params["out_data"], index=False)

            # evaluate the temporary synthetic data
            fidelity = fidelity_tuner(config, seed)
            affinity, query_error = utility_tuner(config, dataset, cuda, seed)
            print(
                "fidelity: {0}, affinity: {1}, query error: {2}".format(
                    fidelity, affinity, query_error
                )
            )
            error = fidelity + affinity + query_error
        except Exception as e:
            print("*" * 20 + "Error when tuning" + "*" * 20)
            print(e)
            error = 1e10

        return error

    path_params = config["path_params"]

    # load real data
    real_train_data_pd, meta_data, discrete_columns = read_csv(
        path_params["train_data"], path_params["meta_data"]
    )

    study_name = "tune_privsyn_{0}".format(dataset)
    try:
        optuna.delete_study(study_name=study_name, storage=STORAGE)
    except:
        pass
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=0),
        # storage=STORAGE,
        study_name=study_name,
    )

    study.optimize(privsyn_objective, n_trials=10, show_progress_bar=True)

    # update the best params
    config["model_params"] = study.best_trial.user_attrs["config"]
    config["sample_params"]["num_samples"] = (
        meta_data["train_size"] + meta_data["val_size"]
    )
    config["sample_params"]["num_train_samples"] = meta_data["train_size"]
    config["sample_params"]["num_val_samples"] = meta_data["val_size"]

    print("best score for privsyn {0}: {1}".format(dataset, study.best_value))

    return config
