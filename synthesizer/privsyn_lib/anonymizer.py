import os
import sys

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

from typing import Dict, Tuple, Any
import numpy as np
from loguru import logger
import copy
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
    marginal_sets, epss = data_loader.generate_marginal_by_config(
        data_loader.private_data, marginal_config
    )
    
    args_sel = {}
    args_sel['indif_rho'] = split_method["two-way-select"]
    args_sel['two-way-publish'] = split_method["two-way-publish"]
    args_sel['combined_marginal_rho'] = split_method["combine"] # don't used in this phase, just as a penalty term
    args_sel['marg_sel_threshold'] = 500

    # Calculate diff_score for all marginals
    diff_scores = synthesizer.privsyn_lib.compute_indiff.calculate_indif(marginal_sets, args_sel)

    # Marginal Selection
    
    selected_marginal_sets = synthesizer.privsyn_lib.marginal_selection.marginal_selection_with_diff_score(marginal_sets, diff_scores, args_sel)

    completed_marginals = synthesizer.privsyn_lib.marginal_selection.handle_isolated_attrs(marginal_sets, selected_marginal_sets, method="isolate")

    # Add noise
    noisy_marginals = anonymize(
        copy.deepcopy(marginal_sets), epss, split_method, delta, sensitivity
    )

    # Calculate difference scores
    diff_scores = []
    for key in noisy_marginals:
        try:
            diff = noisy_marginals[key] - marginal_sets["priv_all_two_way"][key]
        except:
            diff = noisy_marginals[key] - marginal_sets["priv_all_one_way"][key]
        diff_scores.append(diff.sum().sum())

    logger.info(f"Average difference score: {np.mean(diff_scores)}")

    del marginal_sets  # Clean up original marginals
    return noisy_marginals


def anonymize(
    marginal_sets: Dict, epss: Dict, split_method: Dict, delta: float, sensitivity: int
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
        # Calculate average record count before noise
        avg_count = np.mean(
            [np.sum(marginal.values) for marginal in marginals.values()]
        )
        logger.debug(f"Average record count before noise: {avg_count}")

        eps = epss[set_key]
        logger.info(
            f"Noise parameters - eps: {eps}, delta: {delta}, "
            f"sensitivity: {sensitivity}, marginals: {len(marginals)}"
        )

        # Determine noise type and parameters
        noise_type, noise_param = advanced_composition.get_noise(
            eps, delta, sensitivity, len(marginals)
        )
        logger.info(f"Using {noise_type} noise with parameter {noise_param}")

        # Add noise based on type
        if noise_type == "lap":
            noise_param = 1 / advanced_composition.lap_comp(
                eps, delta, sensitivity, len(marginals)
            )
            for marginal_att, marginal in marginals.items():
                noisy_marginals[marginal_att] = marginal + np.random.laplace(
                    scale=noise_param, size=marginal.shape
                )
        else:
            noise_param = advanced_composition.gauss_zcdp(
                eps, delta, sensitivity, len(marginals)
            )
            for marginal_att, marginal in marginals.items():
                noisy_marginals[marginal_att] = marginal + np.random.normal(
                    scale=noise_param, size=marginal.shape
                )

        logger.info(
            f"Marginal {set_key}: eps={eps}, noise={noise_type}, "
            f"param={noise_param}, sensitivity={sensitivity}"
        )

    return noisy_marginals
