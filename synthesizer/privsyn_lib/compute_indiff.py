import os
import sys

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

import numpy as np

def calculate_indif(marginal_sets, arg_sel):
    """
    Calculate Indif_score for all two-way marginals.

    Args:
        marginal_sets (dict): The structure containing all two-way marginals.
        encode_mapping (dict): The encoding mapping for the dataset.

    Returns:
        dict: A dictionary storing the Indif_score for each two-way marginal pair.
    """
    indif_scores = {}

    # Extract one-way and two-way marginals
    one_way_marginals = marginal_sets.get("priv_all_one_way", {})
    two_way_marginals = marginal_sets.get("priv_all_two_way", {})

    for pair, real_marginal in two_way_marginals.items():
        # Extract attributes
        attr1, attr2 = list(pair)

        # Get the one-way marginals for each attribute
        one_way_marginal_attr1 = one_way_marginals.get(frozenset([attr1]))
        one_way_marginal_attr2 = one_way_marginals.get(frozenset([attr2]))

        # Normalize the one-way marginals
        norm_one_way_attr1 = one_way_marginal_attr1 / np.sum(one_way_marginal_attr1.values)
        norm_one_way_attr2 = one_way_marginal_attr2 / np.sum(one_way_marginal_attr2.values)

        # Get the domain sizes for the attributes
        domain_size_attr1 = len(norm_one_way_attr1)
        domain_size_attr2 = len(norm_one_way_attr2)

        # Create the independent distribution with shape [m, n]
        independent_distribution = np.zeros((domain_size_attr1, domain_size_attr2))
        for i, prob_attr1 in enumerate(norm_one_way_attr1.values.flatten()):
            for j, prob_attr2 in enumerate(norm_one_way_attr2.values.flatten()):
                independent_distribution[i, j] = prob_attr1 * prob_attr2

        # Normalize the two-way marginal
        norm_real_marginal = real_marginal / np.sum(real_marginal.values)

        if independent_distribution.shape != norm_real_marginal.values.shape:
            independent_distribution = np.transpose(independent_distribution)

        # Compute the Indif_score as the sum of absolute differences
        indif_score = np.sum(np.abs(norm_real_marginal.values - independent_distribution))

        # Store the result using the attribute pair as the key
        indif_scores[pair] = indif_score

    # Add noise to Indif_scores
    if arg_sel['indif_rho'] != 0.0:
        keys = list(indif_scores.keys())
        values = np.array(list(indif_scores.values()))
        noise = np.random.normal(scale=8 * len(values) / arg_sel['indif_rho'], size=len(values))
        noisy_values = values + noise

    # Update indif_scores with noisy values
    indif_scores = dict(zip(keys, noisy_values))

    #print(indif_scores)

    return indif_scores