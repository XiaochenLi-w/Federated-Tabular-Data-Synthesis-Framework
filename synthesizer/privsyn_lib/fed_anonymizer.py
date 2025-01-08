import os
import sys

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

from typing import Dict, Tuple, Any
import numpy as np
from loguru import logger
import copy
from lib import advanced_composition


def get_distributed_noisy_marginals(
    data_loader: Any,
    marginal_config: Dict,
    split_method: Dict,
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
    args_sel['noise_to_one_way_marginal'] = split_method["noise_to_one_way_marginal"]
    args_sel['noise_to_two_way_marginal'] = split_method["noise_to_two_way_marginal"] # don't used in this phase, just as a penalty term
    args_sel['marg_sel_threshold'] = 500

    #----------------Simulate the distributed collaboration process-------------------#

    # Compute all one-way marginals and two-way marginals on User side. (In practice, these are distributed across multiple clients)
    one_way_marginals = marginal_sets.get("priv_all_one_way", {})
    two_way_marginals = marginal_sets.get("priv_all_two_way", {})

    # Add noise to all one-way marginals
    noisy_one_way_marginals = anonymize_one_way_marginals(copy.deepcopy(one_way_marginals), args_sel, delta, sensitivity)
    print(noisy_one_way_marginals)


    

    # del marginal_sets  # Clean up original marginals
    # return noisy_marginals

def anonymize_one_way_marginals(
    marginal_sets: Dict, split_method: Dict, delta: float, sensitivity: int
) -> Dict[Tuple[str], np.array]:

    noisy_marginals = {}

    for key, marginal in marginal_sets.items():
        # # Calculate average record count before noise
        # avg_count = np.mean(
        #     [np.sum(marginal.values) for marginal in marginals.values()]
        # )
        # logger.debug(f"Average record count before noise: {avg_count}")

        #eps = epss[set_key]
        eps = split_method['noise_to_one_way_marginal']
        logger.info(
            f"Noise parameters - eps: {eps}, delta: {delta}, "
            f"sensitivity: {sensitivity}, marginals: {key}"
        )

        # Determine noise type and parameters
        noise_type, noise_param = advanced_composition.get_noise(
            eps, delta, sensitivity, len(marginal)
        )
        logger.info(f"Using {noise_type} noise with parameter {noise_param}")

        # Add noise based on type
        if noise_type == "lap":
            noise_param = 1 / advanced_composition.lap_comp(
                eps, delta, sensitivity, len(marginal)
            )
           
            noisy_marginals[key] = marginal + np.random.laplace(
                    scale=noise_param, size=np.shape(marginal)
                )
        else:
            noise_param = advanced_composition.gauss_zcdp(
                eps, delta, sensitivity, len(marginal)
            )
         
            noisy_marginals[key] = marginal + np.random.normal(
                scale=noise_param, size=np.shape(marginal)
            )

        logger.info(
            f"Marginal {key}: eps={eps}, noise={noise_type}, "
            f"param={noise_param}, sensitivity={sensitivity}"
        )

    return noisy_marginals