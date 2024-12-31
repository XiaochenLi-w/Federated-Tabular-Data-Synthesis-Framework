import os
import sys

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

import numpy as np
import math

def marginal_selection_with_diff_score(marginal_sets, Indiff_scores, select_args):
    """
    Selects marginals using diff_score.
    """

    # Initialize variables
    gap = 1e10
    selected_marginals = []
    selected_attrs = set()

    overall_score = sum(Indiff_scores.values())

    one_way_marginals = marginal_sets.get("priv_all_one_way", {})
    two_way_marginals = marginal_sets.get("priv_all_two_way", {})
    domain_sizes = {attr: len(values) for attr, values in one_way_marginals.items()}

    num_cells = np.array([
        domain_sizes[frozenset([attr1])] * domain_sizes[frozenset([attr2])]
        for attr1, attr2 in two_way_marginals
    ])

    selected = set()
    unselected = set(range(len(two_way_marginals)))

    two_way_keys = list(two_way_marginals.keys())  # List of keys to access marginals by index

    gauss_error_normalizer = 1.0

    while gap > select_args['marg_sel_threshold']:
        current_score = np.sum(Indiff_scores)
        selected_index = None

        for j in unselected:
            select_candidate = selected.union({j})

            # Convert indices to keys for calculation
            candidate_keys = [two_way_keys[i] for i in select_candidate]

            cells_square_sum = np.sum(
                np.power(num_cells[list(select_candidate)], 2.0 / 3.0))
            gauss_constant = np.sqrt(cells_square_sum / (math.pi * select_args['combined_marginal_rho']))
            gauss_error = np.sum(
                gauss_constant * np.power(num_cells[list(select_candidate)], 2.0 / 3.0))

            # Calculate the new score with the candidate marginal added
            candidate_score = sum(Indiff_scores[key] for key in candidate_keys)

            gauss_error *= gauss_error_normalizer

            pairwise_error = sum(
                Indiff_scores[two_way_keys[i]]
                for i in unselected.difference(select_candidate)
            )
            current_score = gauss_error + pairwise_error

            if candidate_score < current_score:
                selected_index = j
                current_score = candidate_score

        # Update selection and calculate the gap
        gap = overall_score - current_score
        overall_score = current_score

        if selected_index is not None:
            selected.add(selected_index)
            unselected.remove(selected_index)

            # Retrieve the corresponding attributes
            first_attr, second_attr = two_way_keys[selected_index]
            selected_marginals.append(frozenset([first_attr, second_attr]))
            selected_attrs.update((first_attr, second_attr))

    # Handle isolated attributes
    # selected_marginals = handle_isolated_attrs(
    #     dataset_domain, selected_attrs, marginal_sets, selected_marginals, method="connect", sort=True)

    # Convert selected marginals to the same format as marginal_sets
    selected_marginal_sets = {}
    for selected_key in selected_marginals:
        # Ensure that the marginal exists in two_way_marginals
        selected_marginal_sets[selected_key] = two_way_marginals[selected_key]


    print(selected_marginal_sets.keys())

    return selected_marginal_sets