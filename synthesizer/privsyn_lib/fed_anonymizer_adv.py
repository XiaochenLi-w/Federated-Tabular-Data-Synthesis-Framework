import os
import sys

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

from typing import Dict, Tuple, Any
import numpy as np
from loguru import logger
import copy
from lib import advanced_composition
import pandas as pd
import synthesizer.privsyn_lib.marginal_selection as marginal_selection
import synthesizer.privsyn_lib.anonymizer as anonymizer
from .consistenter import Consistenter

import math


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
    args_sel['delta'] = split_method["delta"]
    args_sel['marg_sel_threshold'] = 0.1

    #----------------Simulate the distributed collaboration process-------------------#
    c = split_method["client_num"]
    dist_method = split_method["dist_method"]
    # obtain full data
    full_data = data_loader.private_data

    data_splits, sizes, sample_num_total = _split_records(
    df=full_data,
    c=c,
    dist_method=dist_method,  # support 'uniform' and 'random'
    )

    # use n_i and sample_num_total as weight
    aggregated_one_way = {}
    aggregated_two_way = {}

    for user_data, n_i in zip(data_splits, sizes):
        # generate marginals on the client
        marginal_sets = data_loader.generate_marginal_by_config(user_data, marginal_config)

        args_sel = {
            'noise_to_one_way_marginal': split_method["noise_to_one_way_marginal"],
            'noise_to_two_way_marginal': split_method["noise_to_two_way_marginal"],
            'two-way-publish': split_method["two-way-publish"],
            'client_num': split_method["client_num"],
            'delta': split_method["delta"],
            'marg_sel_threshold': 0.1,
        }

        one_way_marginals = marginal_sets.get("priv_all_one_way", {})
        two_way_marginals = marginal_sets.get("priv_all_two_way", {})

        # number of clients
        sample_num_client = n_i

        # add noise
        noisy_one_way_marginals, sigma_1 = anonymize_marginals(
            copy.deepcopy(one_way_marginals),
            args_sel, delta, sensitivity, sample_num_client, 0, Flag_=1
        )

        k = 10
        projected_two_way_marginals, projection_matrix = project_marginals(two_way_marginals, k)

        max_norms = -1e9
        for _, P_ab in projection_matrix.items():
            row_norms = np.sqrt(np.sum(P_ab**2, axis=1))
            max_norms = max(max_norms, float(np.max(row_norms)))

        noisy_two_way_marginals, sigma_2 = anonymize_marginals(
            copy.deepcopy(projected_two_way_marginals),
            args_sel, delta, max_norms, sample_num_client, 0, Flag_=2
        )

        # aggregate with n_i / sample_num_total
        w = n_i / float(sample_num_total)

        for k_attr, arr in noisy_one_way_marginals.items():
            if k_attr not in aggregated_one_way:
                aggregated_one_way[k_attr] = arr * w
            else:
                aggregated_one_way[k_attr] += arr * w

        for k_attr, arr in noisy_two_way_marginals.items():
            if k_attr not in aggregated_two_way:
                aggregated_two_way[k_attr] = arr * w
            else:
                aggregated_two_way[k_attr] += arr * w


    noisy_one_way_marginals = aggregated_one_way
    noisy_two_way_marginals = aggregated_two_way

    # ---------------- compute alpha ---------------- #
    #alpha = sum([n_i ** 2 for n_i in sizes]) / (sample_num_total ** 2)
    alpha = 1 / (sample_num_total ** 2) # we add noise to count instead of frequency

    # # Compute all one-way marginals and two-way marginals on User side. (In practice, these are distributed across multiple clients)
    # one_way_marginals = marginal_sets.get("priv_all_one_way", {})
    # two_way_marginals = marginal_sets.get("priv_all_two_way", {})

    # # Get any marginal from one_way_marginals
    # any_key = next(iter(one_way_marginals))  # Get a random key
    # sample_num = int(np.sum(one_way_marginals[any_key]))  # Compute sum of values
    
    # # indif_scores_list = []
    
    # # icoun = 0
    # # for _ in range(20):

    # # Add noise to all one-way marginals
    # noisy_one_way_marginals, sigma_1 = anonymize_marginals(copy.deepcopy(one_way_marginals), args_sel, delta, sensitivity, sample_num, Flag_ = 1)
    
    # # Map 2-way marginals onto a lower-dimensional space
    # k = 10
    # projected_two_way_marginals, projection_matrix = project_marginals(two_way_marginals, k)
    
    # max_norms = -1000000

    # for _, P_ab in projection_matrix.items():
    #     # Compute row-wise Euclidean norms
    #     row_norms = np.sqrt(np.sum(P_ab**2, axis=1))
    #     # Take the maximum of these norms
    #     tmp = np.max(row_norms)
    #     if  tmp > max_norms:
    #         max_norms = tmp

    # noisy_two_way_marginals, sigma_2 = anonymize_marginals(copy.deepcopy(projected_two_way_marginals), args_sel, delta, max_norms, sample_num, Flag_ = 2)
    marginal_sets = data_loader.generate_marginal_by_config(
        data_loader.private_data, marginal_config
    )


    # Calculate Indif score
    indif_scores = calculate_indif_fed(noisy_one_way_marginals, noisy_two_way_marginals, projection_matrix, sigma_1, sigma_2, args_sel['client_num'], sample_num_total, alpha)
    
    max_select_num = int(len(two_way_marginals) / 3)
    # Select the marginals
    #selected_marginal_sets = marginal_selection.marginal_selection_with_diff_score(marginal_sets, indif_scores, args_sel, sample_num, Flag_ = 1)
    selected_marginal_sets = marginal_selection_with_dynamic_sampling(synthesizer, data_transformer, data_loader, marginal_config, sample_num_total, marginal_sets,  noisy_two_way_marginals, indif_scores, projection_matrix, args_sel, delta, sensitivity, max_select_num)

    
    print("???", len(selected_marginal_sets.keys()))
    # Add noise to the selected marginals
    selected_marginal_sets, _ = anonymize_marginals(copy.deepcopy(selected_marginal_sets), args_sel, delta, sensitivity, sample_num_total, max_select_num, Flag_ = 4)

    # Add unselected 1-way marginals
    completed_marginals = marginal_selection.handle_isolated_attrs(marginal_sets, selected_marginal_sets, method="isolate")

    converted_marginal_sets = anonymizer.convert_selected_marginals(completed_marginals)
    
    added_one_way_marginals = converted_marginal_sets.get("priv_all_one_way", {})
    added_one_way_marginals, _ = anonymize_marginals(copy.deepcopy(added_one_way_marginals), args_sel, delta, sensitivity, sample_num_total, 0, Flag_ = 1)
    converted_marginal_sets["priv_all_one_way"] = added_one_way_marginals

    noisy_marginals = {}
    for _, marginals in converted_marginal_sets.items():
        for marginal_att, marginal in marginals.items():
            noisy_marginals[marginal_att] = marginal

    #print(noisy_marginals)
    del marginal_sets  # Clean up original marginals
    return noisy_marginals

def anonymize_marginals(
    marginal_sets: Dict, split_method: Dict, delta: float, sensitivity: int, sample_num: int, max_num: int, Flag_: int
) -> Dict[Tuple[str], np.array]:

    noisy_marginals = {}

    if Flag_ == 1:
        eps = split_method['noise_to_one_way_marginal']

        noise_param = advanced_composition.gauss_zcdp(
            eps, delta, sensitivity, len(marginal_sets)
        )
    elif Flag_ == 2:
        eps = split_method['noise_to_two_way_marginal']

        noise_param = advanced_composition.gauss_zcdp(
            eps, delta, sensitivity, len(marginal_sets)
        )
    elif Flag_ == 3:
        eps = split_method['two-way-publish'] / split_method['client_num']

        noise_param = advanced_composition.gauss_zcdp(
            eps, delta, sensitivity, len(marginal_sets)
        )
    else:
        eps = split_method['two-way-publish'] / split_method['client_num']

        noise_param = advanced_composition.gauss_zcdp(
            eps, delta, sensitivity, max_num
        )


    

    for key, marginal in marginal_sets.items():

        noisy_marginals[key] = marginal + np.random.normal(
            scale=noise_param, size=np.shape(marginal)
        )

        noisy_marginals[key] = noisy_marginals[key] / sample_num

        # Ensure all values in noisy_marginals[key] are non-negative
        noisy_marginals[key] = np.maximum(noisy_marginals[key], 0)
        
        logger.info(
            f"Marginal {key}: eps={eps}, delta: {delta}, Gaussian Noise, "
            f"param={noise_param}, sensitivity={sensitivity}"
        )

    return noisy_marginals, noise_param**2

def project_marginals(marginals, k):
    """
    Project two-way marginals onto a lower-dimensional space using a random matrix.

    Parameters:
    marginals (dict): A dictionary where keys are frozensets representing categorical pairs and 
                      values are pandas DataFrames representing the counts.
    k (int): The target dimensionality for the projection.

    Returns:
    dict: A dictionary with the same structure as `marginals` but with projected values.
    """
    projected_marginals = {}
    random_matrices = {}

    for key, df in marginals.items():
        # Here, we ignore a fact that the lengths of some 2-way-marginals are smaller than k
        # if len(df) <= k:
        #     projected_marginals[key] = df
        #     continue
        # Extract marginals as a NumPy array and flatten it
        marginals_array = df.values.flatten().reshape(1, -1)

        # Dimensions of the original data
        sa, sb = df.shape

        # Generate the random projection matrix P_ab
        P_ab = np.random.normal(0, 1 / np.sqrt(k), size=(sa * sb, k))
        random_matrices[key] = P_ab

        # Perform the projection
        projected_array = marginals_array @ P_ab

        # Reshape back into the original DataFrame shape
        projected_df = pd.DataFrame(
            projected_array
        )

        # Store the projected DataFrame in the result dictionary
        projected_marginals[key] = projected_df

    return projected_marginals, random_matrices

def calculate_indif_fed(noisy_one_way_marginals, noisy_two_way_marginals, projection_matrix, sigma_1, sigma_2, c, sample_num, alpha):
    """
    Calculate Indif_score for all two-way marginals.

    Args:
        marginal_sets (dict): The structure containing all two-way marginals.
        encode_mapping (dict): The encoding mapping for the dataset.

    Returns:
        dict: A dictionary storing the Indif_score for each two-way marginal pair.
    """
    indif_scores = {}

    for pair, real_marginal in noisy_two_way_marginals.items():
        
        # Extract attributes
        attr1, attr2 = list(pair)

        # Get the one-way marginals for each attribute
        one_way_marginal_attr1 = noisy_one_way_marginals.get(frozenset([attr1]))
        one_way_marginal_attr2 = noisy_one_way_marginals.get(frozenset([attr2]))

        # Normalize the one-way marginals
        norm_one_way_attr1 = one_way_marginal_attr1 #/ np.sum(one_way_marginal_attr1.values)
        norm_one_way_attr2 = one_way_marginal_attr2 #/ np.sum(one_way_marginal_attr2.values)

        # Get the domain sizes for the attributes
        domain_size_attr1 = len(norm_one_way_attr1)
        domain_size_attr2 = len(norm_one_way_attr2)

        # Create the independent distribution with shape [m, n]
        independent_distribution = np.zeros((domain_size_attr1, domain_size_attr2))
        for i, prob_attr1 in enumerate(norm_one_way_attr1.values.flatten()):
            for j, prob_attr2 in enumerate(norm_one_way_attr2.values.flatten()):
                independent_distribution[i, j] = prob_attr1 * prob_attr2

        # Flatten and project the independent distribution
        independent_distribution_flat = independent_distribution.flatten().reshape(1, -1)
        
        # Get the same projection_matrix as two-way-marginals
        P_ab = projection_matrix[pair]

        row_norms = np.sqrt(np.sum(P_ab**2, axis=1))
        sensitivity_of_P_ab = np.max(row_norms)

        #projected_independent_distribution = independent_distribution_flat @ P_ab - sigma_2 * sensitivity_of_P_ab ** 2 / sample_num ** 2
        projected_independent_distribution = independent_distribution_flat @ P_ab

        # Debias the Indif_score
        s_a = domain_size_attr1
        s_b = domain_size_attr2

        Ep = s_a * s_b

        #indif_score = np.sqrt((np.linalg.norm((real_marginal.values - projected_independent_distribution)) ** 2 - sigma_1 / sample_num ** 2) / Ep)
        
        cols = P_ab.shape[1]

        # indif_score = np.sqrt((np.linalg.norm((real_marginal.values - projected_independent_distribution)) ** 2 
        # - c ** 2 * sigma_1 * (s_b * np.linalg.norm(norm_one_way_attr1) ** 2 + s_a * np.linalg.norm(norm_one_way_attr2) ** 2) * np.linalg.norm(P_ab) ** 2 / (sample_num ** 2)
        # - c ** 2 * sigma_2 * cols ** 2 / sample_num ** 2 + np.power(c, 4) * sigma_1 ** 2 * s_b ** 2 * s_a ** 2 / (np.power(sample_num, 4))) / Ep)
        
        indif_score = np.sqrt(np.linalg.norm((real_marginal.values - projected_independent_distribution)) ** 2 
        - cols * alpha * sigma_2 - alpha * sigma_1 * (s_b * np.linalg.norm(norm_one_way_attr1) ** 2 + s_a * np.linalg.norm(norm_one_way_attr2) ** 2) + s_a * s_b * alpha ** 2 * sigma_1 ** 2)
        
        # Store the result using the attribute pair as the key
        indif_scores[pair] = indif_score
    
    print(indif_scores)

    return indif_scores


def calculate_real_indif(one_way_marginals, two_way_marginals, sample_num, projection_matrix):
    """
    Calculate Indif_score for all two-way marginals.

    Args:
        marginal_sets (dict): The structure containing all two-way marginals.
        encode_mapping (dict): The encoding mapping for the dataset.

    Returns:
        dict: A dictionary storing the Indif_score for each two-way marginal pair.
    """
    indif_scores = {}

    for pair, real_marginal in two_way_marginals.items():
        
        # Extract attributes
        attr1, attr2 = list(pair)

        # Get the one-way marginals for each attribute
        norm_one_way_attr1 = one_way_marginals.get(frozenset([attr1]))
        norm_one_way_attr2 = one_way_marginals.get(frozenset([attr2]))

        # Normalize the one-way marginals
        norm_one_way_attr1 = norm_one_way_attr1 / sample_num
        norm_one_way_attr2 = norm_one_way_attr2 / sample_num

        real_marginal = real_marginal / sample_num

        # Get the domain sizes for the attributes
        domain_size_attr1 = len(norm_one_way_attr1)
        domain_size_attr2 = len(norm_one_way_attr2)

        # Create the independent distribution with shape [m, n]
        independent_distribution = np.zeros((domain_size_attr1, domain_size_attr2))
        for i, prob_attr1 in enumerate(norm_one_way_attr1.values.flatten()):
            for j, prob_attr2 in enumerate(norm_one_way_attr2.values.flatten()):
                independent_distribution[i, j] = prob_attr1 * prob_attr2

        # Flatten and project the independent distribution
        independent_distribution_flat = independent_distribution.flatten().reshape(1, -1)

        # Get the same projection_matrix as two-way-marginals
        P_ab = projection_matrix[pair]
        projected_independent_distribution = independent_distribution_flat @ P_ab

        # # # Flatten and project the independent distribution
        # if independent_distribution.shape != real_marginal.values.shape:
        #     independent_distribution = independent_distribution.T

        # Compute the Indif_score 
        # indif_score = np.sqrt(np.sum(np.linalg.norm(real_marginal.values - independent_distribution) ** 2))
        indif_score = np.sqrt(np.sum(np.linalg.norm(real_marginal.values - projected_independent_distribution) ** 2))

        indif_scores[pair] = indif_score

    print("True Scores:", indif_scores)

    return indif_scores

def marginal_selection_with_dynamic_sampling(synthesizer, data_transformer, data_loader, marginal_config, sample_num, marginal_sets,  noisy_two_way_marginals, Indiff_scores, projection_matrix, select_args, delta, sensitivity, max_select_num):
    """
    Selects marginals dynamically using diff_score and updates Indiff_scores with a synthesized dataset.

    Args:
        marginal_sets (dict): Structure containing all one-way and two-way marginals.
        Indiff_scores (dict): Initial Indiff_scores for the marginals.
        select_args (dict): Arguments for selection, including thresholds.

    Returns:
        dict: Selected marginal sets after the process.
    """
    update_control = 0
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

    # gauss_error_normalizer = 1.0
    current_score = sum(Indiff_scores.values())

    while gap > select_args['marg_sel_threshold']:
        
        selected_index = None
            
        for j in unselected:
            select_candidate = selected.union({j})

            # cells_square_sum = np.sum(
            #     np.power(num_cells[list(select_candidate)], 2.0 / 3.0))
            # gauss_constant = np.sqrt(cells_square_sum / (math.pi * select_args['two-way-publish']))
            # gauss_error = np.sum(
            #     gauss_constant * np.power(num_cells[list(select_candidate)], 2.0 / 3.0))

            # gauss_error *= gauss_error_normalizer
            gauss_error = 0
            for select_idx in select_candidate:
                tmp_var = 2 * math.log(1 / select_args['delta'])
                sigma_ = math.sqrt(len(two_way_keys)) / (math.sqrt(tmp_var + 2 * select_args['two-way-publish'] / select_args['client_num']) - math.sqrt(tmp_var))
                
                gauss_error += sigma_ * np.sqrt(len(two_way_marginals[two_way_keys[select_idx]])) / sample_num


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
            update_control += 1

##-------------------------Synthesize Data according to the update------------------------##
            # Process isolated attributes and recalculate Indiff_scores
            if update_control % 10 == 0:
                select_marginal_sets = {}
                for selected_key in selected_marginals:
                    select_marginal_sets[selected_key] = two_way_marginals[selected_key]
                
                select_marginal_sets, _ = anonymize_marginals(copy.deepcopy(select_marginal_sets), select_args, delta, sensitivity, sample_num, max_select_num, Flag_ = 4)

                completed_marginals = marginal_selection.handle_isolated_attrs(marginal_sets, select_marginal_sets, method="isolate")

                converted_marginal_sets = anonymizer.convert_selected_marginals(completed_marginals)
        
                added_one_way_marginals = converted_marginal_sets.get("priv_all_one_way", {})
                added_one_way_marginals, _ = anonymize_marginals(copy.deepcopy(added_one_way_marginals), select_args, delta, sensitivity, sample_num, 0, Flag_ = 1)
                converted_marginal_sets["priv_all_one_way"] = added_one_way_marginals

                noisy_marginals = {}
                for _, marginals in converted_marginal_sets.items():
                    for marginal_att, marginal in marginals.items():
                        noisy_marginals[marginal_att] = marginal
                
                #------------------------------------------------------------------------------------------------#
                #--------Some necessary steps to synthesize the data using the current selected marginals--------#
                num_synthesize_records = (
                    np.mean([np.sum(x.values) for _, x in noisy_marginals.items()])
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
                    noisy_marginals
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
                syn_marginal_sets = data_loader.generate_marginal_by_config(syn_data, marginal_config)
                temp_two_way_marginals = syn_marginal_sets.get("priv_all_two_way", {})
               
                # Recalculate Indiff_scores based on the new dataset
                Indiff_scores_up = calculate_temp_indif_fed(temp_two_way_marginals, noisy_two_way_marginals, projection_matrix, sample_num)

                for pair, _ in Indiff_scores.items():
                    if first_attr in pair or second_attr in pair:
                        if Indiff_scores[pair] < Indiff_scores_up[pair]:
                            Indiff_scores[pair] = Indiff_scores_up[pair]
        
        if update_control == max_select_num:
            break
        
    # Convert selected marginals to the same format as marginal_sets
    selected_marginal_sets = {}
    for selected_key in selected_marginals:
        selected_marginal_sets[selected_key] = two_way_marginals[selected_key]
    
    return selected_marginal_sets


def calculate_temp_indif_fed(temp_two_way_marginals, noisy_two_way_marginals, projection_matrix, sample_num):
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

        marginals_array = marginals_array / sample_num

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
        # norm_real_marginal = real_marginal / np.sum(real_marginal.values)

        # Compute the Indif_score as the sum of absolute differences
        # print(pair)
        indif_score = np.sqrt(np.linalg.norm(real_marginal.values - projected_marginals[pair].values) ** 2)

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

# def _split_records(df: pd.DataFrame, c: int, dist_method: str):
#     """
#     将 DataFrame 随机打乱后切分为 c 份，返回 (splits, sizes, total_num)
#     支持:
#       - 均匀随机分布: 每份尽量相等 (array_split)
#       - 随机分配: 每份长度不同, 但总和为 total_num (随机切分点)
#     """
#     if not isinstance(df, pd.DataFrame):
#         raise TypeError("data_loader.private_data must be pandas.DataFrame")

#     total_num = len(df)
#     if c <= 0:
#         raise ValueError("c must be integer")
#     if c > total_num:
#         raise ValueError("c must be smaller than totall num")

#     # 兼容中英文写法
#     dm = dist_method.lower()
#     if dm == "uniform":
#         # 先打乱行位置，再尽量等分
#         perm = np.random.permutation(total_num)
#         idx_chunks = np.array_split(perm, c)          # 近似等分 (大小相差不超过 1)
#         splits = [df.iloc[idx_chunk] for idx_chunk in idx_chunks]
#         sizes = [len(s) for s in splits]

#     elif dm == "random":
#         # 生成 c-1 个不重复切分点，保证每份 >= 1
#         cuts = sorted(np.random.choice(np.arange(1, total_num), size=c-1, replace=False))
#         bounds = [0] + cuts + [total_num]
#         sizes = [bounds[i+1] - bounds[i] for i in range(c)]

#         # 再打乱整体索引，按 sizes 逐段切
#         perm = np.random.permutation(total_num)
#         splits = []
#         start = 0
#         for sz in sizes:
#             splits.append(df.iloc[perm[start:start+sz]])
#             start += sz
#     else:
#         raise ValueError("dist_method must be 'uniform' or 'random'")

#     # 防御式检查
#     assert sum(sizes) == total_num, "切分总数与原数据不一致。"
#     return splits, sizes, total_num


def _split_records(
    df: pd.DataFrame,
    c: int,
    dist_method: str,
    label_col: str = "label",
):
    """
    将 DataFrame 切分为 c 份，返回 (splits, sizes, total_num)

    支持:
      - uniform: 尽量等分
      - random: 随机大小
      - label: 按 label 切分，每个 client 只包含单一 label
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("data_loader.private_data must be pandas.DataFrame")

    total_num = len(df)
    if c <= 0:
        raise ValueError("c must be positive integer")
    if c > total_num:
        raise ValueError("c must be smaller than total num")

    dm = dist_method.lower()

    if dm == "uniform":
        perm = np.random.permutation(total_num)
        idx_chunks = np.array_split(perm, c)
        splits = [df.iloc[idx] for idx in idx_chunks]
        sizes = [len(s) for s in splits]

    elif dm == "random":
        cuts = sorted(np.random.choice(np.arange(1, total_num), size=c-1, replace=False))
        bounds = [0] + cuts + [total_num]
        sizes = [bounds[i+1] - bounds[i] for i in range(c)]

        perm = np.random.permutation(total_num)
        splits = []
        start = 0
        for sz in sizes:
            splits.append(df.iloc[perm[start:start+sz]])
            start += sz

    elif dm == "label":
        if label_col not in df.columns:
            raise ValueError(f"label column `{label_col}` not found in DataFrame")

        # 按 label 分组
        label_groups = {
            k: v.sample(frac=1).reset_index(drop=True)  # shuffle
            for k, v in df.groupby(label_col)
        }

        labels = list(label_groups.keys())
        num_labels = len(labels)

        if num_labels > c:
            raise ValueError("number of labels cannot exceed client number")

        # client 按照 label 分配
        client_alloc = []
        base = c // num_labels
        rem = c % num_labels
        for i in range(num_labels):
            cnt = base + (1 if i < rem else 0)
            client_alloc.append(cnt)

        splits = []
        sizes = []

        for label, num_clients in zip(labels, client_alloc):
            group_df = label_groups[label]
            n = len(group_df)

            if num_clients > n:
                raise ValueError(
                    f"Label `{label}` has only {n} samples, "
                    f"cannot split into {num_clients} clients"
                )

            perm = np.random.permutation(n)
            idx_chunks = np.array_split(perm, num_clients)

            for idx in idx_chunks:
                part = group_df.iloc[idx]
                splits.append(part)
                sizes.append(len(part))

        assert len(splits) == c, "Number of splits != client number"

    else:
        raise ValueError("dist_method must be 'uniform', 'random', or 'label'")

    assert sum(sizes) == total_num, "切分总数与原数据不一致"
    return splits, sizes, total_num