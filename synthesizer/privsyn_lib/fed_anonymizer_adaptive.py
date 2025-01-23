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
import pandas as pd
import synthesizer.privsyn_lib.marginal_selection as marginal_selection
import synthesizer.privsyn_lib.anonymizer as anonymizer
import synthesizer.privsyn_lib.fed_anonymizer as fed_anonymizer
from .consistenter import Consistenter


def get_distributed_noisy_marginals(
    synthesizer: object,
    data_transformer: object,
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
    args_sel['noise_to_two_way_marginal'] = split_method["noise_to_two_way_marginal"]
    args_sel['two-way-publish'] = split_method["two-way-publish"]
    args_sel['client_num'] = split_method["client_num"]
    args_sel['marg_sel_threshold'] = 50

    #----------------Simulate the distributed collaboration process-------------------#

    # Compute all one-way marginals and two-way marginals on User side. (In practice, these are distributed across multiple clients)
    one_way_marginals = marginal_sets.get("priv_all_one_way", {})
    two_way_marginals = marginal_sets.get("priv_all_two_way", {})

    # Add noise to all one-way marginals
    noisy_one_way_marginals, sigma_1 = fed_anonymizer.anonymize_marginals(copy.deepcopy(one_way_marginals), args_sel, delta, sensitivity, Flag_ = 1)
    
    # Map 2-way marginals onto a lower-dimensional space
    k = 10
    projected_two_way_marginals, projection_matrix = fed_anonymizer.project_marginals(two_way_marginals, k)

    max_norms = -1000000

    for _, P_ab in projection_matrix.items():
        # Compute row-wise Euclidean norms
        row_norms = np.sqrt(np.sum(P_ab**2, axis=1))
        # Take the maximum of these norms
        tmp = np.max(row_norms)
        if  tmp > max_norms:
            max_norms = tmp

    noisy_two_way_marginals, sigma_2 = fed_anonymizer.anonymize_marginals(copy.deepcopy(projected_two_way_marginals), args_sel, delta, max_norms, Flag_ = 2)

    # Calculate Indif score
    indif_scores = fed_anonymizer.calculate_indif_fed(noisy_one_way_marginals, noisy_two_way_marginals, projection_matrix, sigma_1, sigma_2, args_sel['client_num'])

    # Select the marginals
    selected_marginal_sets = marginal_selection_with_dynamic_sampling(synthesizer, data_transformer, data_loader, marginal_config, len(data_loader.private_data), marginal_sets,  noisy_two_way_marginals, indif_scores, projection_matrix, args_sel)

    # Add unselected 1-way marginals

    completed_marginals = marginal_selection.handle_isolated_attrs(marginal_sets, selected_marginal_sets, method="isolate")

    converted_marginal_sets = anonymizer.convert_selected_marginals(completed_marginals)

    added_one_way_marginals = converted_marginal_sets.get("priv_all_one_way", {})
    added_one_way_marginals, _ = fed_anonymizer.anonymize_marginals(copy.deepcopy(added_one_way_marginals), args_sel, delta, sensitivity, Flag_ = 3)
    converted_marginal_sets["priv_all_one_way"] = added_one_way_marginals


    noisy_marginals = {}
    for _, marginals in converted_marginal_sets.items():
        for marginal_att, marginal in marginals.items():
            noisy_marginals[marginal_att] = marginal

    # print(noisy_marginals)

    del marginal_sets  # Clean up original marginals
    return noisy_marginals


def marginal_selection_with_dynamic_sampling(synthesizer, data_transformer, data_loader, marginal_config, sample_num, marginal_sets,  noisy_two_way_marginals, Indiff_scores, projection_matrix, select_args):
    """
    Selects marginals dynamically using diff_score and updates Indiff_scores with a synthesized dataset.

    Args:
        marginal_sets (dict): Structure containing all one-way and two-way marginals.
        Indiff_scores (dict): Initial Indiff_scores for the marginals.
        select_args (dict): Arguments for selection, including thresholds.

    Returns:
        dict: Selected marginal sets after the process.
    """
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

            if score_temp < current_score:
                selected_index = j
                current_score = score_temp

        gap = overall_score - current_score
        overall_score = current_score

        if selected_index is not None:
            selected.add(selected_index)
            unselected.remove(selected_index)

            # Retrieve the corresponding attributes
            first_attr, second_attr = two_way_keys[selected_index]
            selected_marginals.append(frozenset([first_attr, second_attr]))
            selected_attrs.update((first_attr, second_attr))

            # Process isolated attributes and recalculate Indiff_scores
            completed_marginals = marginal_selection.handle_isolated_attrs(marginal_sets, {
                frozenset([first_attr, second_attr]): two_way_marginals[frozenset([first_attr, second_attr])]
            })
            
            #------------------------------------------------------------------------------------------------#
            #--------Some necessary steps to synthesize the data using the current selected marginals--------#
            num_synthesize_records = (
                np.mean([np.sum(x.values) for _, x in completed_marginals.items()])
                .round()
                .astype(int)
            )
            print(
                "------------------------> now we get the estimate of records' num by averaging from noisy marginals:",
                num_synthesize_records,
            )

            # the list of all attributes' names (strings) except the identifier attribute
            synthesizer.attr_list = synthesizer.data.obtain_attrs()
            # domain_size_list is an array recording how many distinct values each attribute has
            synthesizer.domain_size_list = np.array(
                [len(synthesizer.data.encode_schema[att]) for att in synthesizer.attr_list]
            )
            # map from attribute string to its index in attr_list
            synthesizer.attr_index_map = dict(zip(synthesizer.attr_list, range(len(synthesizer.attr_list))))

            # Build marginal dictionaries from noisy data
            noisy_onehot_marginal_dict, noisy_attr_marginal_dict = synthesizer.construct_marginals(
                completed_marginals
            )

            # By default, we will not rely on any "public" data in this example,
            # so set the "pub" dictionaries to the same as the "noisy" ones.
            pub_onehot_marginal_dict = noisy_onehot_marginal_dict
            pub_attr_marginal_dict = noisy_attr_marginal_dict

            synthesizer.onehot_marginal_dict, synthesizer.attrs_marginal_dict = synthesizer.normalize_marginals(
                pub_onehot_marginal_dict,
                pub_attr_marginal_dict,
                noisy_attr_marginal_dict,
                synthesizer.attr_index_map,
                num_synthesize_records,
            )

            # Next, ensure that the marginals in onehot_marginal_dict are consistent
            consistenter = Consistenter(synthesizer.onehot_marginal_dict, synthesizer.domain_size_list)
            consistenter.consist_marginals()

            # After consistency, optionally normalize each marginal
            for _, marginal_dict in synthesizer.onehot_marginal_dict.items():
                total = sum(marginal_dict["count"])
                if total > 0:
                    marginal_dict["count"] /= total

            # Rebuild marginals from the consistent dictionaries
            remapped_marginals = {}
            for attrs, marginal_dict in synthesizer.attrs_marginal_dict.items():
                # 'attrs' might be a frozenset or a tuple. Convert it to a sorted tuple.
                canonical_attrs = canonical_key(attrs)
                remapped_marginals[canonical_attrs] = marginal_dict["count"]
            #------------------------------------------------------------------------------------------------#

            # Dynamically sample a new dataset
            # sample the same number of data as the real data
            syn_data = synthesizer.synthesize(num_records=sample_num)
            
            # Update Indiff score
            marginal_sets = data_loader.generate_marginal_by_config(syn_data, marginal_config)
            temp_two_way_marginals = marginal_sets.get("priv_all_two_way", {})

            # Recalculate Indiff_scores based on the new dataset
            Indiff_scores = calculate_temp_indif_fed(temp_two_way_marginals, noisy_two_way_marginals, projection_matrix)
            # print(Indiff_scores)

    # Convert selected marginals to the same format as marginal_sets
    selected_marginal_sets = {}
    for selected_key in selected_marginals:
        selected_marginal_sets[selected_key] = two_way_marginals[selected_key]

    return selected_marginal_sets


def calculate_temp_indif_fed(temp_two_way_marginals, noisy_two_way_marginals, projection_matrix):
    """
    Calculate Indif_score for all two-way marginals.

    Args:
        marginal_sets (dict): The structure containing all two-way marginals.
        encode_mapping (dict): The encoding mapping for the dataset.

    Returns:
        dict: A dictionary storing the Indif_score for each two-way marginal pair.
    """
    indif_scores = {}
    
    # Get the same projection_matrix as two-way-marginals
    projected_marginals = {}

    for key, df in temp_two_way_marginals.items():
        P_ab = projection_matrix[key]

        marginals_array = df.values.flatten().reshape(1, -1)

        marginals_array = marginals_array / np.sum(marginals_array)

        # Perform the projection
        projected_array = marginals_array @ P_ab

        # Reshape back into the original DataFrame shape
        projected_df = pd.DataFrame(
            projected_array
        )

        # Store the projected DataFrame in the result dictionary
        projected_marginals[key] = projected_df

    for pair, real_marginal in noisy_two_way_marginals.items():

        # Normalize the two-way marginal
        norm_real_marginal = real_marginal / np.sum(real_marginal.values)

        # Compute the Indif_score as the sum of absolute differences
        # print(pair)
        indif_score = np.sum(np.abs(norm_real_marginal.values - projected_marginals[pair].values))

        # Store the result using the attribute pair as the key
        indif_scores[pair] = indif_score

    return indif_scores


def canonical_key(attrs):
    """
    Convert any iterable (list, set, frozenset) of attribute names
    into a sorted tuple for consistent usage as a dictionary or Pandas index key.
    """
    if isinstance(attrs, (list, set, frozenset)):
        return tuple(sorted(attrs))
    return attrs  # if already a tuple or string, just return as is