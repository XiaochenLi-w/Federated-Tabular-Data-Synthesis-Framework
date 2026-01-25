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
    sensitivity: int
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
    c = split_method["client_num"]
    dist_method = split_method["dist_method"]
    # obtain full data
    full_data = data_loader.private_data

    data_splits, sizes, sample_num_total = _split_records(
    df=full_data,
    c=c,
    dist_method=dist_method,  # support 'uniform', 'random', 'label'
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
            args_sel, delta, sensitivity, sample_num_client, Flag_=1
        )

        k = 10
        projected_two_way_marginals, projection_matrix = project_marginals(two_way_marginals, k)

        max_norms = -1e9
        for _, P_ab in projection_matrix.items():
            row_norms = np.sqrt(np.sum(P_ab**2, axis=1))
            max_norms = max(max_norms, float(np.max(row_norms)))

        noisy_two_way_marginals, sigma_2 = anonymize_marginals(
            copy.deepcopy(projected_two_way_marginals),
            args_sel, delta, max_norms, sample_num_client, Flag_=2
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
    
    alpha = 1 / (sample_num_total ** 2) # we add noise to count instead of frequency

    marginal_sets = data_loader.generate_marginal_by_config(
        data_loader.private_data, marginal_config
    )

    # Calculate Indif score
    indif_scores = calculate_indif_fed(noisy_one_way_marginals, noisy_two_way_marginals, projection_matrix, sigma_1, sigma_2, c, sample_num_total, alpha)

    # Select the marginals
    selected_marginal_sets = marginal_selection.marginal_selection_with_diff_score(marginal_sets, indif_scores, args_sel, sample_num_total, Flag_ = 1)
    
    #print("???", len(selected_marginal_sets.keys()))
    # Add noise to the selected marginals
    selected_marginal_sets, _ = anonymize_marginals(copy.deepcopy(selected_marginal_sets), args_sel, delta, sensitivity, sample_num_total, Flag_ = 3)

    # Add unselected 1-way marginals
    completed_marginals = marginal_selection.handle_isolated_attrs(marginal_sets, selected_marginal_sets, method="isolate")

    converted_marginal_sets = anonymizer.convert_selected_marginals(completed_marginals)
    
    added_one_way_marginals = converted_marginal_sets.get("priv_all_one_way", {})
    added_one_way_marginals, _ = anonymize_marginals(copy.deepcopy(added_one_way_marginals), args_sel, delta, sensitivity, sample_num_total, Flag_ = 1)
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
        eps = split_method['noise_to_one_way_marginal']
    elif Flag_ == 2:
        eps = split_method['noise_to_two_way_marginal']
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

        projected_independent_distribution = independent_distribution_flat @ P_ab

        # Debias the Indif_score
        s_a = domain_size_attr1
        s_b = domain_size_attr2

        Ep = s_a * s_b

        cols = P_ab.shape[1]
        
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