import os
import sys

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

from typing import Dict, Tuple, Any
import numpy as np
from loguru import logger
import copy
import math
from lib import advanced_composition
import synthesizer.privsyn_lib.compute_indiff
import synthesizer.privsyn_lib.marginal_selection


def get_noisy_marginals(
    data_loader: Any,
    marginal_config: Dict,
    split_method: Dict,
    eps: float,
    delta: float,
    sensitivity: int,
) -> Dict[Tuple[str], np.array]:
    """
    Generate noisy marginals based on configuration.

    Args:
        data_loader: DataLoader instance
        marginal_config: Configuration for marginal generation
        split_method: Method for splitting privacy budget
        eps: Epsilon parameter for differential privacy
        delta: Delta parameter for differential privacy
        sensitivity: Sensitivity parameter for differential privacy

    Returns:
        Dict mapping attribute tuples to noisy marginals
    """
    # Generate marginals
    marginal_sets = data_loader.generate_marginal_by_config(
        data_loader.private_data, marginal_config
    )
    
    args_sel = {}
    args_sel['indif_rho'] = split_method["two-way-select"]
    args_sel['two-way-publish'] = split_method["two-way-publish"]
    args_sel['one-way-publish'] = split_method["one-way-publish"]
    args_sel['combined_marginal_rho'] = split_method["combine"] # don't used in this phase, just as a penalty term
    args_sel['client_num'] = split_method["client_num"]
    args_sel['delta'] = split_method["delta"]
    args_sel['marg_sel_threshold'] = 0.1

    # Get any marginal from one_way_marginals
    one_way_marginals = marginal_sets.get("priv_all_one_way", {})
    any_key = next(iter(one_way_marginals))  # Get a random key
    sample_num = int(np.sum(one_way_marginals[any_key]))  # Compute sum of values

    # Calculate diff_score for all marginals
    diff_scores = synthesizer.privsyn_lib.compute_indiff.calculate_indif(marginal_sets, args_sel)

    # 2-way marginals selection
    
    selected_marginal_sets = synthesizer.privsyn_lib.marginal_selection.marginal_selection_with_diff_score(marginal_sets, diff_scores, args_sel, sample_num)

    print("???", len(selected_marginal_sets.keys()))

    # Add unselected 1-way marginals

    completed_marginals = synthesizer.privsyn_lib.marginal_selection.handle_isolated_attrs(marginal_sets, selected_marginal_sets, method="isolate")

    converted_marginal_sets = convert_selected_marginals(completed_marginals)

    # Add noise
    noisy_marginals = anonymize(
        copy.deepcopy(converted_marginal_sets), args_sel, delta, sensitivity, sample_num
    )

    # # Calculate difference scores
    # diff_scores = []
    # for key in noisy_marginals:
    #     try:
    #         diff = noisy_marginals[key] - completed_marginals[key]
    #     except:
    #         diff = noisy_marginals[key] - completed_marginals[key]
    #     diff_scores.append(diff.sum().sum())

    # logger.info(f"Average difference score: {np.mean(diff_scores)}")

    del marginal_sets  # Clean up original marginals
    return noisy_marginals


def anonymize(
    marginal_sets: Dict, split_method: Dict, delta: float, sensitivity: int, sample_num: int
) -> Dict[Tuple[str], np.array]:
    """
    Add noise to marginals for differential privacy.

    Args:
        marginal_sets: Dict[set_key, marginals] where set_key is key for eps and noise_type
        epss: Dict mapping set_key to epsilon values
        split_method: Dict mapping set_key to noise type
        delta: Delta parameter for differential privacy
        sensitivity: Sensitivity parameter for differential privacy

    Returns:
        Dict mapping attribute tuples to noisy marginals
    """
    noisy_marginals = {}

    for set_key, marginals in marginal_sets.items():
        # # Calculate average record count before noise
        # avg_count = np.mean(
        #     [np.sum(marginal.values) for marginal in marginals.values()]
        # )
        # logger.debug(f"Average record count before noise: {avg_count}")

        #eps = epss[set_key]
        if set_key == "priv_all_one_way":

            eps = split_method['one-way-publish']

            # Add noise
            noise_param = advanced_composition.gauss_zcdp(
                eps, delta, sensitivity, len(marginals)
            )
            for marginal_att, marginal in marginals.items():
                noisy_marginals[marginal_att] = marginal + np.random.normal(
                    scale=noise_param, size=np.shape(marginal)
                )
                noisy_marginals[marginal_att] = noisy_marginals[marginal_att] / sample_num

                # Ensure all values in noisy_marginals[key] are non-negative
                noisy_marginals[marginal_att] = np.maximum(noisy_marginals[marginal_att], 0)

        else:
            eps = split_method['two-way-publish']

            # Add noise
            noise_param = advanced_composition.gauss_zcdp(
                eps, delta, sensitivity, len(marginals)
            )
            for marginal_att, marginal in marginals.items():
                noisy_marginals[marginal_att] = marginal + np.random.normal(
                    scale=noise_param, size=np.shape(marginal)
                )
                noisy_marginals[marginal_att] = noisy_marginals[marginal_att] / sample_num

                # Ensure all values in noisy_marginals[key] are non-negative
                noisy_marginals[marginal_att] = np.maximum(noisy_marginals[marginal_att], 0)

        logger.info(
            f"Marginal {set_key}: eps={eps}, delta: {delta}, Gaussian Noise, "
            f"param={noise_param}, sensitivity={sensitivity}"
        )

    return noisy_marginals


def convert_selected_marginals(selected_marginal_sets):
    converted_marginal_sets = {"priv_all_one_way": {}, "priv_all_two_way": {}}

    marginal_tmp_one = {}
    marginal_tmp_two = {}

    for marginal_key, marginals in selected_marginal_sets.items():
        for key, val in marginals.items():
            if hasattr(val, "fillna"):
                marginals[key] = val.fillna(0.0)

        # Determine the category based on the number of attributes in the key
        if isinstance(marginal_key, str):
            attrs = marginal_key.split(",")  # Assuming attributes are comma-separated
        else:
            attrs = list(marginal_key)  # If marginal_key is a tuple

        # One-way or two-way classification
        if len(attrs) == 1:
            marginal_tmp_one[marginal_key] = marginals
        elif len(attrs) == 2:
            marginal_tmp_two[marginal_key] = marginals
        else:
            raise ValueError(f"Unsupported key format: {marginal_key}")
        
    converted_marginal_sets["priv_all_one_way"] = marginal_tmp_one
    converted_marginal_sets["priv_all_two_way"] = marginal_tmp_two
 
    return converted_marginal_sets