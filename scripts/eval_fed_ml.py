# This script is used to evaluate the application performance of the synthetic data
# machine learning efficiency: train ml model with synthetic data and evaluate the performance
# range query: evaluate the error of range query

import os
import sys

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

import argparse
import numpy as np
from lib.commons import load_config, improve_reproducibility
from lib.config import config
from evaluator.utility.eval_helper import save_utility_results
from evaluator.utility.eval_helper import ml_evaluation, query_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="great")
    parser.add_argument("--dataset", "-d", type=str, default="adult")
    parser.add_argument("--cuda", "-c", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    print(
        "Evalute utility performance for dataset {0} with algorithm {1}".format(
            args.dataset, args.model
        )
    )

    # load template config
    model_config = "exp/{0}/{1}/config_fed.toml".format(args.dataset, args.model)
    model_config = load_config(os.path.join(ROOT, model_config))

    seed = args.seed
    n_samples = model_config["sample_params"]["num_samples"]

    # dynamically import model interface
    synthesizer = __import__("synthesizer." + args.model, fromlist=[args.model])

    model_path = model_config["path_params"]["out_model"]
    if not os.path.exists(model_path):
        raise ValueError(
            "Please train the synthesizer first (script/train_synthesizer.py)"
        )

    ml_results = {}  # ml resutls for synthetic data and original data
    query_results = {}
    for i in range(config.n_exps):
        print("Evaluate experiment {0}/{1}".format(i + 1, config.n_exps))
        seed = i
        synthesizer.sample(model_config, n_samples, seed)
        # evaluate with ml model
        ml_results = ml_evaluation(model_config, args.dataset, args.model, args.cuda, seed, ml_results)
        # evalute with range query
        # query_results = query_evaluation(
        #     model_config, query_results, n_samples=1000, seed=seed
        # )
    
    result_coun_roc_auc = 0
    result_coun_f1 = 0
    for models, metrics in ml_results.items():
        for metric, value in metrics.items():
            ml_results[models][metric] = {}
            ml_results[models][metric]["mean"] = sum(value) / len(value)
            ml_results[models][metric]["std"] = float(np.std(value))
            if metric == 'roc_auc':
            #if metric == 'r2':
                result_coun_roc_auc += ml_results[models][metric]["mean"]
            else:
                result_coun_f1 += ml_results[models][metric]["mean"]

    print("final:", ml_results)
    print("avg_mean_auc:", result_coun_roc_auc / config.n_exps)
    print("avg_mean_f1:", result_coun_f1 / config.n_exps)
    


    # # save the result
    # save_utility_results(
    #     ml_results, query_results, model_config["path_params"]["utility_result"]
    # )


if __name__ == "__main__":
    main()
