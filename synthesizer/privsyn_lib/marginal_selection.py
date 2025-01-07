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
        current_score = sum(Indiff_scores.values())
        selected_index = None

        for j in unselected:
            select_candidate = selected.union({j})

            cells_square_sum = np.sum(
                np.power(num_cells[list(select_candidate)], 2.0 / 3.0))
            gauss_constant = np.sqrt(cells_square_sum / (math.pi * select_args['two-way-publish']))
            gauss_error = np.sum(
                gauss_constant * np.power(num_cells[list(select_candidate)], 2.0 / 3.0))

            gauss_error *= gauss_error_normalizer

            pairwise_error = sum(
                Indiff_scores[two_way_keys[i]]
                for i in unselected.difference(select_candidate)
            )
            score_temp = gauss_error + pairwise_error

            # print("score?", current_score, score_temp)

            if score_temp < current_score:
                selected_index = j
                current_score = score_temp

        # Update selection and calculate the gap
        gap = overall_score - current_score
        overall_score = current_score
        # print("gap", gap)

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


    #print(selected_marginal_sets.keys())

    return selected_marginal_sets

def handle_isolated_attrs(marginal_sets, selected_marginal_sets, method="isolate"):
    """
    Handle isolated attributes that are not included in selected 2-way marginals.

    Args:
        marginal_sets (dict): The structure containing all one-way and two-way marginals.
        selected_marginal_sets (dict): The selected 2-way marginals.
        method (str): The method to handle isolated attributes ("isolate" or "connect").

    Returns:
        dict: Updated selected marginal sets including missing 1-way marginals.
    """
    # Extract one-way and two-way marginals
    one_way_marginals = marginal_sets.get("priv_all_one_way", {})
    two_way_marginals = marginal_sets.get("priv_all_two_way", {})
    
    # Extract all selected attributes from the 2-way marginals
    selected_attrs = set()
    for pair in selected_marginal_sets.keys():
        selected_attrs.update(pair)
    
    # Find attributes that are missing in the selected 2-way marginals
    all_attrs = set(attr for key in one_way_marginals.keys() for attr in key)
    missing_attrs = all_attrs - selected_attrs

    # Add missing 1-way marginals based on the chosen method
    for attr in missing_attrs:
        if method == "isolate":
            # Add the one-way marginal for the missing attribute
            one_way_key = frozenset([attr])
            if one_way_key in one_way_marginals:
                selected_marginal_sets[one_way_key] = one_way_marginals[one_way_key]

        elif method == "connect":
            # Find the best two-way marginal to connect the isolated attribute
            best_pair = None
            best_score = float("inf")

            for pair, marginal in two_way_marginals.items():
                if attr in pair and any(a in selected_attrs for a in pair):
                    # Compute a score (e.g., Indif_score or size) to choose the best connection
                    score = np.sum(marginal.values)  # Replace with an actual scoring metric if available
                    if score < best_score:
                        best_pair = pair
                        best_score = score

            if best_pair:
                selected_marginal_sets[best_pair] = two_way_marginals[best_pair]

    # print("Missing", selected_marginal_sets.keys())

    return selected_marginal_sets