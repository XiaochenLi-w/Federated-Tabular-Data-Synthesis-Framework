import numpy as np

def create_marginal(attr_one_hot: np.array, domain_size_list: np.array):
    marginal_dict = {}
    marginal_dict["attr_one_hot"] = attr_one_hot
    marginal_dict["domain_size_list"] = domain_size_list

    marginal_dict["domain_size"] = np.prod(domain_size_list[np.nonzero(attr_one_hot)[0]])
    marginal_dict["total_num_attr"] = len(attr_one_hot)
    marginal_dict["marginal_num_attr"] = np.count_nonzero(attr_one_hot)

    marginal_dict["encode_num"] = np.zeros(marginal_dict["marginal_num_attr"], dtype=np.uint32)
    marginal_dict["cum_mul"] = np.zeros(marginal_dict["marginal_num_attr"], dtype=np.uint32)
    marginal_dict["attributes_index"] = np.nonzero(attr_one_hot)[0]

    marginal_dict["count"] = np.zeros(marginal_dict["domain_size"])
    marginal_dict["sum"] = 0
    marginal_dict["attributes_set"] = set()
    marginal_dict["tuple_key"] = np.array([0], dtype=np.uint32)
    marginal_dict["count_matrix"] = None
    marginal_dict["summations"] = None
    marginal_dict["weights"] = np.array([])  

    calculate_encode_num(marginal_dict)
    return marginal_dict

def calculate_encode_num(marginal_dict):
    if marginal_dict["marginal_num_attr"] != 0:
        categories_index = marginal_dict["attributes_index"]
        domain_size_list = marginal_dict["domain_size_list"]

        # Method 1
        categories_num = domain_size_list[categories_index].copy()
        categories_num = np.roll(categories_num, 1)
        categories_num[0] = 1
        marginal_dict["cum_mul"] = np.cumprod(categories_num)

        # Method 2
        categories_num = domain_size_list[categories_index].copy()
        categories_num = np.roll(categories_num, marginal_dict["marginal_num_attr"] - 1)
        categories_num[-1] = 1
        categories_num = np.flip(categories_num)
        marginal_dict["encode_num"] = np.flip(np.cumprod(categories_num))

def calculate_tuple_key(marginal_dict):
    marginal_num_attr = marginal_dict["marginal_num_attr"]
    marginal_dict["tuple_key"] = np.zeros((marginal_dict["domain_size"], marginal_num_attr), dtype=np.uint32)

    if marginal_num_attr != 0:
        for i, index in enumerate(marginal_dict["attributes_index"]):
            categories = np.arange(marginal_dict["domain_size_list"][index])
            column_key = np.tile(
                np.repeat(categories, marginal_dict["encode_num"][i]), 
                marginal_dict["cum_mul"][i]
            )
            marginal_dict["tuple_key"][:, i] = column_key
    else:
        marginal_dict["tuple_key"] = np.array([0], dtype=np.uint32)
        marginal_dict["domain_size"] = 1

def count_records(marginal_dict, records):
    encode_records = np.matmul(records[:, marginal_dict["attributes_index"]], marginal_dict["encode_num"])
    encode_key, count = np.unique(encode_records, return_counts=True)
    indices = np.where(np.isin(np.arange(marginal_dict["domain_size"]), encode_key))[0]
    marginal_dict["count"][indices] = count

def calculate_count_matrix(marginal_dict):
    shape = []
    for attri in marginal_dict["attributes_index"]:
        shape.append(marginal_dict["domain_size_list"][attri])
    marginal_dict["count_matrix"] = np.copy(marginal_dict["count"]).reshape(tuple(shape))
    return marginal_dict["count_matrix"]

def generate_attributes_index_set(marginal_dict):
    marginal_dict["attributes_set"] = set(marginal_dict["attributes_index"])

def calculate_encode_num_general(marginal_dict, attributes_index):
    domain_size_list = marginal_dict["domain_size_list"]
    categories_num = domain_size_list[attributes_index].copy()
    categories_num = np.roll(categories_num, attributes_index.size - 1)
    categories_num[-1] = 1
    categories_num = np.flip(categories_num)
    encode_num = np.flip(np.cumprod(categories_num))
    return encode_num

def count_records_general(marginal_dict, records):
    count = np.zeros(marginal_dict["domain_size"])
    encode_records = np.matmul(records[:, marginal_dict["attributes_index"]], marginal_dict["encode_num"])
    encode_key, value_count = np.unique(encode_records, return_counts=True)
    indices = np.where(np.isin(np.arange(marginal_dict["domain_size"]), encode_key))[0]
    count[indices] = value_count
    return count

def calculate_count_matrix_general(marginal_dict, count):
    shape = []
    for attri in marginal_dict["attributes_index"]:
        shape.append(marginal_dict["domain_size_list"][attri])
    return np.copy(count).reshape(tuple(shape))

def calculate_tuple_key_general(marginal_dict, unique_value_list):
    marginal_num_attr = marginal_dict["marginal_num_attr"]
    marginal_dict["tuple_key"] = np.zeros((marginal_dict["domain_size"], marginal_num_attr), dtype=np.uint32)

    if marginal_num_attr != 0:
        for i, index in enumerate(marginal_dict["attributes_index"]):
            categories = unique_value_list[i]
            column_key = np.tile(
                np.repeat(categories, marginal_dict["encode_num"][i]), 
                marginal_dict["cum_mul"][i]
            )
            marginal_dict["tuple_key"][:, i] = column_key
    else:
        marginal_dict["tuple_key"] = np.array([0], dtype=np.uint32)
        marginal_dict["domain_size"] = 1

def project_from_bigger_marginal_general(marginal_dict, bigger_marginal_dict):
    encode_num = np.zeros(marginal_dict["total_num_attr"], dtype=np.uint32)
    encode_num[marginal_dict["attributes_index"]] = marginal_dict["encode_num"]
    encode_num = encode_num[bigger_marginal_dict["attributes_index"]]
    encode_records = np.matmul(bigger_marginal_dict["tuple_key"], encode_num)

    for i in range(marginal_dict["domain_size"]):
        key_index = np.where(encode_records == i)[0]
        marginal_dict["count"][i] = np.sum(bigger_marginal_dict["count"][key_index])

def initialize_consist_parameters(marginal_dict, num_target_views):
    marginal_dict["summations"] = np.zeros((marginal_dict["domain_size"], num_target_views))
    marginal_dict["weights"] = np.zeros(num_target_views)

def calculate_delta(marginal_dict):
    """
    Compute the 'delta' array and return it instead of storing it in the view dictionary.
    """
    target = np.matmul(marginal_dict["summations"], marginal_dict["weights"]) / np.sum(marginal_dict["weights"])
    delta = -(marginal_dict["summations"] - target.reshape(len(target), 1))
    return delta

def update_marginal(marginal_dict, common_marginal_dict, delta, index):
    """
    Update 'marginal_dict["count"]' using an external 'delta' instead of reading it from the dictionary.
    """
    encode_num = np.zeros(marginal_dict["total_num_attr"], dtype=np.uint32)
    encode_num[common_marginal_dict["attributes_index"]] = common_marginal_dict["encode_num"]
    encode_num = encode_num[marginal_dict["attributes_index"]]
    encode_records = np.matmul(marginal_dict["tuple_key"], encode_num)

    for i in range(common_marginal_dict["domain_size"]):
        key_index = np.where(encode_records == i)[0]
        if len(key_index) > 0:
            marginal_dict["count"][key_index] += delta[i, index] / len(key_index)

def non_negativity(marginal_dict):
    count_copy = np.copy(marginal_dict["count"])
    norm_cut(count_copy)
    marginal_dict["count"] = count_copy
    
def project_from_bigger_marginal(marginal_dict, bigger_marginal_dict, index):
    encode_num = np.zeros(marginal_dict["total_num_attr"], dtype=np.uint32)
    encode_num[marginal_dict["attributes_index"]] = marginal_dict["encode_num"]
    encode_num = encode_num[bigger_marginal_dict["attributes_index"]]

    encode_records = np.matmul(bigger_marginal_dict["tuple_key"], encode_num)
    marginal_dict["weights"][index] = bigger_marginal_dict["weight_coeff"] / np.prod(
        marginal_dict["domain_size_list"][
            np.setdiff1d(bigger_marginal_dict["attributes_index"], marginal_dict["attributes_index"])
        ]
    )
    for i in range(marginal_dict["domain_size"]):
        key_index = np.where(encode_records == i)[0]
        marginal_dict["summations"][i, index] = np.sum(bigger_marginal_dict["count"][key_index])

def norm_sub(count):
    while (abs(sum(count) - 1) > 1e-6) or (count < 0).any():
        count[count < 0] = 0
        total = sum(count)
        mask = count > 0
        if sum(mask) == 0:
            count[:] = 1.0 / len(count)
            break
        diff = (1 - total) / sum(mask)
        count[mask] += diff
    return count

def norm_cut(count):
    negative_indices = np.where(count < 0.0)[0]
    negative_total = abs(np.sum(count[negative_indices]))
    count[negative_indices] = 0.0

    positive_indices = np.where(count > 0.0)[0]
    if positive_indices.size != 0:
        positive_sort_indices = np.argsort(count[positive_indices])
        sort_cumsum = np.cumsum(count[positive_indices[positive_sort_indices]])

        threshold_indices = np.where(sort_cumsum <= negative_total)[0]
        if threshold_indices.size == 0:
            count[positive_indices[positive_sort_indices[0]]] = (
                sort_cumsum[0] - negative_total
            )
        else:
            count[positive_indices[positive_sort_indices[threshold_indices]]] = 0.0
            next_index = threshold_indices[-1] + 1
            if next_index < positive_sort_indices.size:
                count[positive_indices[positive_sort_indices[next_index]]] = (
                    sort_cumsum[next_index] - negative_total
                )
    else:
        count[:] = 0.0
    return count


# Testing function left in place to illustrate usage
def test():
    marginal_dict = create_marginal(np.array([1, 1, 0, 0]), np.array([3, 3, 0, 0]))
    # This is just an example of creating a view dictionary
    # Additional tests would go here