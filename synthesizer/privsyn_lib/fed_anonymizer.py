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
    args_sel['noise_to_two_way_marginal'] = split_method["noise_to_two_way_marginal"]
    args_sel['two-way-publish'] = split_method["two-way-publish"]
    args_sel['client_num'] = split_method["client_num"]
    args_sel['delta'] = split_method["delta"]
    args_sel['marg_sel_threshold'] = 0.1

    #----------------Simulate the distributed collaboration process-------------------#

    # Compute all one-way marginals and two-way marginals on User side. (In practice, these are distributed across multiple clients)
    one_way_marginals = marginal_sets.get("priv_all_one_way", {})
    two_way_marginals = marginal_sets.get("priv_all_two_way", {})

    # Get any marginal from one_way_marginals
    any_key = next(iter(one_way_marginals))  # Get a random key
    sample_num = int(np.sum(one_way_marginals[any_key]))  # Compute sum of values
    
    # indif_scores_list = []
    
    # icoun = 0
    # for _ in range(20):

    # Add noise to all one-way marginals
    noisy_one_way_marginals, sigma_1 = anonymize_marginals(copy.deepcopy(one_way_marginals), args_sel, delta, sensitivity, sample_num, Flag_ = 1)
    
    # Map 2-way marginals onto a lower-dimensional space
    k = 10
    projected_two_way_marginals, projection_matrix = project_marginals(two_way_marginals, k)
    
    max_norms = -1000000

    for _, P_ab in projection_matrix.items():
        # Compute row-wise Euclidean norms
        row_norms = np.sqrt(np.sum(P_ab**2, axis=1))
        # Take the maximum of these norms
        tmp = np.max(row_norms)
        if  tmp > max_norms:
            max_norms = tmp

    noisy_two_way_marginals, sigma_2 = anonymize_marginals(copy.deepcopy(projected_two_way_marginals), args_sel, delta, max_norms, sample_num, Flag_ = 2)

    # Calculate Indif score
    indif_scores = calculate_indif_fed(noisy_one_way_marginals, noisy_two_way_marginals, projection_matrix, sigma_1, sigma_2, args_sel['client_num'], sample_num)

    # #-------------For test---------------------#
    #     icoun += 1
    #     print("It's", icoun, ":", indif_scores)
    #     indif_scores_list.append(indif_scores)
    
    # # Initialize a dictionary to store the values for each key
    # key_values = {}

    # true_indif_score = calculate_real_indif(one_way_marginals, projected_two_way_marginals, sample_num, projection_matrix)

    # # Collect the values for each key across all indif_scores
    # for indif_score in indif_scores_list:
    #     for key, value in indif_score.items():
    #         if key not in key_values:
    #             key_values[key] = []  # Initialize the list for the key
    #         key_values[key].append(value - true_indif_score[key])

    # # Now calculate the variance for each key
    # variances = {}

    # for key, values in key_values.items():
    #     # Calculate variance for the values of each key
    #     variances[key] = np.var(values, ddof=1)  # ddof=1 for sample variance

    # # Print the variances for each key
    # for key, variance in variances.items():
    #     print(f"Variance for {key}: {variance}")

    # Select the marginals
    selected_marginal_sets = marginal_selection.marginal_selection_with_diff_score(marginal_sets, indif_scores, args_sel, sample_num, Flag_ = 1)
    
    print("???", len(selected_marginal_sets.keys()))
    # Add noise to the selected marginals
    selected_marginal_sets, _ = anonymize_marginals(copy.deepcopy(selected_marginal_sets), args_sel, delta, sensitivity, sample_num, Flag_ = 3)

    # Add unselected 1-way marginals
    completed_marginals = marginal_selection.handle_isolated_attrs(marginal_sets, selected_marginal_sets, method="isolate")

    converted_marginal_sets = anonymizer.convert_selected_marginals(completed_marginals)
    
    added_one_way_marginals = converted_marginal_sets.get("priv_all_one_way", {})
    added_one_way_marginals, _ = anonymize_marginals(copy.deepcopy(added_one_way_marginals), args_sel, delta, sensitivity, sample_num, Flag_ = 3)
    converted_marginal_sets["priv_all_one_way"] = added_one_way_marginals

    noisy_marginals = {}
    for _, marginals in converted_marginal_sets.items():
        for marginal_att, marginal in marginals.items():
            noisy_marginals[marginal_att] = marginal

    #print(noisy_marginals)
    del marginal_sets  # Clean up original marginals
    return noisy_marginals

def anonymize_marginals(
    marginal_sets: Dict, split_method: Dict, delta: float, sensitivity: int, sample_num: int, Flag_: int
) -> Dict[Tuple[str], np.array]:

    noisy_marginals = {}

    if Flag_ == 1:
        eps = split_method['noise_to_one_way_marginal'] / split_method['client_num']
    elif Flag_ == 2:
        eps = split_method['noise_to_two_way_marginal'] / split_method['client_num']
    else:
        eps = split_method['two-way-publish'] / split_method['client_num']

    noise_param = advanced_composition.gauss_zcdp(
            eps, delta, sensitivity, len(marginal_sets)
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

def calculate_indif_fed(noisy_one_way_marginals, noisy_two_way_marginals, projection_matrix, sigma_1, sigma_2, c, sample_num):
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

        indif_score = np.sqrt((np.linalg.norm((real_marginal.values - projected_independent_distribution)) ** 2 
        - c ** 2 * sigma_1 * (s_b * np.linalg.norm(norm_one_way_attr1) ** 2 + s_a * np.linalg.norm(norm_one_way_attr2) ** 2) * np.linalg.norm(P_ab) ** 2 / (sample_num ** 2)
        - c ** 2 * sigma_2 * cols ** 2 / sample_num ** 2 + np.power(c, 4) * sigma_1 ** 2 * s_b ** 2 * s_a ** 2 / (np.power(sample_num, 4))) / Ep)
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