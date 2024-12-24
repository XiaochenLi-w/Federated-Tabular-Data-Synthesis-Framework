from loguru import logger
from numpy import linalg as LA
import copy
import numpy as np
import pandas as pd

from . import marginal_utils as marginal

class GUM:
    records = None
    df = None
    error_tracker = None

    rounding_method = 'deterministic'

    under_cell_indices = None
    zero_cell_indices = None
    over_cell_indices = None
    records_throw_indices = pd.DataFrame()

    add_amount = 0
    add_amount_zero = 0
    reduce_amount = 0

    actual_marginal = None
    synthesize_marginal = None
    alpha = 1.0

    encode_records = None
    encode_records_sort_index = None

    def __init__(self, attrs, domains, num_records):
        self.attrs = attrs
        self.domains = domains
        self.num_records = num_records

    def update_alpha(self, iteration):
        self.alpha = 1.0 * 0.84 ** (iteration // 20)

    def initialize_records(self, iterate_keys, method="random", singleton_marginals=None):
        self.records = np.empty([self.num_records, len(self.attrs)], dtype=np.uint32)
        for attr_i, attr in enumerate(self.attrs):
            if method == "random":
                self.records[:, attr_i] = np.random.randint(
                    0, self.domains[attr_i], size=self.num_records
                )
            elif method == "singleton":
                self.records[:, attr_i] = self.generate_singleton_records(
                    singleton_marginals[attr]
                )
        self.df = pd.DataFrame(self.records, columns=self.attrs)
        
        iterate_keys = [self.canonical_key(x) for x in iterate_keys]
        self.error_tracker = pd.DataFrame(index=iterate_keys)

    def generate_singleton_records(self, singleton_marginal_dict):
        record = np.empty(self.num_records, dtype=np.uint32)
        dist_cumsum = np.cumsum(singleton_marginal_dict["count"])
        start = 0

        for index, value in enumerate(dist_cumsum):
            end = int(round(value * self.num_records))
            record[start:end] = index
            start = end

        np.random.shuffle(record)
        return record

    def update_order(self, iteration, marginals, iterate_keys):
        self.error_tracker.insert(loc=0, column=f"{iteration}-before", value=0)

        # Update records before sorting
        for key_i, raw_key in enumerate(iterate_keys):
            # unify it
            key = self.canonical_key(raw_key)
            
            # If 'marginals' was also stored under a canonical key, we can do:
            #   marginals[key]
            # But if 'marginals' was stored under frozenset(key), you must convert again:
            #   marginals[frozenset(key)]
            # The important part is to do exactly the same transformation you used 
            # when creating 'marginals' in the first place.
            print(key)
            print(iterate_keys, key, marginals[frozenset(key)])
            self.update_records_before(marginals[frozenset(key)], key, iteration, mute=True)

        # print("error tracker before sorting: ")
        # print(self.error_tracker)

        sort_error_tracker = self.error_tracker.sort_values(
            by=f"{iteration}-before", ascending=False
        )
        # print("error tracker after sorting: ")
        # print(sort_error_tracker)

        self.error_tracker.insert(loc=0, column=f"{iteration}-after", value=0)
        return list(sort_error_tracker.index)

    @staticmethod
    def canonical_key(attrs):
        # Convert anything to a sorted tuple
        # e.g., if attrs = frozenset({'workclass','age'}), it becomes ('age','workclass')
        if not isinstance(attrs, (list, tuple, set, frozenset)):
            raise ValueError(f"Unexpected type for attrs: {type(attrs)} -> {attrs}")
        return tuple(sorted(attrs))

    def update_records(self, original_marginal_dict, iteration, attrs):
        # Copy the dictionary rather than copying an object
        marginal_dict = copy.deepcopy(original_marginal_dict)

        self.update_records_before(marginal_dict, attrs, iteration)
        self.update_records_main(marginal_dict)
        self.determine_throw_indices()
        self.handle_zero_cells(marginal_dict)

        if iteration % 2 == 0:
            self.complete_partial_ratio(marginal_dict, 0.5)
        else:
            self.complete_partial_ratio(marginal_dict, 1.0)

        self.update_records_before(marginal_dict, attrs, iteration)

    def update_records_main(self, marginal_dict):
        alpha = self.alpha

        # Indices where the synthetic marginal is below the actual marginal
        self.under_cell_indices = np.where(
            (self.synthesize_marginal < self.actual_marginal)
            & (self.synthesize_marginal != 0)
        )[0]

        under_rate = (
            self.actual_marginal[self.under_cell_indices]
            - self.synthesize_marginal[self.under_cell_indices]
        ) / self.synthesize_marginal[self.under_cell_indices]
        ratio_add = np.minimum(
            under_rate,
            np.full(self.under_cell_indices.shape[0], alpha)
        )
        self.add_amount = self._rounding(
            ratio_add * self.synthesize_marginal[self.under_cell_indices] * self.num_records
        )

        # Indices where the synthetic marginal is zero but the actual marginal is not
        self.zero_cell_indices = np.where(
            (self.synthesize_marginal == 0) & (self.actual_marginal != 0)
        )[0]
        self.add_amount_zero = self._rounding(
            alpha * self.actual_marginal[self.zero_cell_indices] * self.num_records
        )

        # Indices where the synthetic marginal is above the actual marginal
        self.over_cell_indices = np.where(
            self.synthesize_marginal > self.actual_marginal
        )[0]
        num_add_total = np.sum(self.add_amount) + np.sum(self.add_amount_zero)
        beta = self.find_optimal_beta(num_add_total, self.over_cell_indices)

        over_rate = (
            self.synthesize_marginal[self.over_cell_indices]
            - self.actual_marginal[self.over_cell_indices]
        ) / self.synthesize_marginal[self.over_cell_indices]
        ratio_reduce = np.minimum(
            over_rate,
            np.full(self.over_cell_indices.shape[0], beta)
        )
        self.reduce_amount = self._rounding(
            ratio_reduce * self.synthesize_marginal[self.over_cell_indices] * self.num_records
        ).astype(int)

        logger.debug("alpha: %s | beta: %s" % (alpha, beta))
        logger.debug(
            "num_boost: %s | num_reduce: %s"
            % (num_add_total, np.sum(self.reduce_amount))
        )

        # Recompute encode records using the marginal dictionary
        selected_record = self.records[:, marginal_dict["attributes_index"]]
        self.encode_records = np.matmul(selected_record, marginal_dict["encode_num"])
        self.encode_records_sort_index = np.argsort(self.encode_records)
        self.encode_records = self.encode_records[self.encode_records_sort_index]

    def determine_throw_indices(self):
        """
        Identify which records to remove or update because their current cell is too large.
        """
        valid_indices = np.nonzero(self.reduce_amount)[0]
        valid_cell_over_indices = self.over_cell_indices[valid_indices]
        valid_cell_num_reduce = self.reduce_amount[valid_indices]
        valid_data_over_index_left = np.searchsorted(
            self.encode_records, valid_cell_over_indices, side="left"
        )
        valid_data_over_index_right = np.searchsorted(
            self.encode_records, valid_cell_over_indices, side="right"
        )

        valid_num_reduce = np.sum(valid_cell_num_reduce)
        self.records_throw_indices = np.zeros(valid_num_reduce, dtype=np.uint32)
        throw_pointer = 0

        for i, cell_index in enumerate(valid_cell_over_indices):
            match_records_indices = self.encode_records_sort_index[
                valid_data_over_index_left[i] : valid_data_over_index_right[i]
            ]
            throw_indices = np.random.choice(
                match_records_indices, valid_cell_num_reduce[i], replace=False
            )
            self.records_throw_indices[
                throw_pointer : throw_pointer + throw_indices.size
            ] = throw_indices
            throw_pointer += throw_indices.size

        np.random.shuffle(self.records_throw_indices)

    def handle_zero_cells(self, marginal_dict):
        """
        For cells that are zero in the synthetic marginal but non-zero in the actual marginal,
        move enough records to fill in the gap.
        """
        if self.zero_cell_indices.size != 0:
            for index, cell_index in enumerate(self.zero_cell_indices):
                num_partial = int(self.add_amount_zero[index])
                if num_partial != 0:
                    for i in range(marginal_dict["marginal_num_attr"]):
                        self.records[
                            self.records_throw_indices[:num_partial],
                            marginal_dict["attributes_index"][i]
                        ] = marginal_dict["tuple_key"][cell_index, i]
                self.records_throw_indices = self.records_throw_indices[num_partial:]

    def complete_partial_ratio(self, marginal_dict, complete_ratio):
        num_complete = np.rint(complete_ratio * self.add_amount).astype(int)
        num_partial = np.rint((1 - complete_ratio) * self.add_amount).astype(int)

        valid_indices = np.nonzero(num_complete + num_partial)
        num_complete = num_complete[valid_indices]
        num_partial = num_partial[valid_indices]

        valid_cell_under_indices = self.under_cell_indices[valid_indices]
        valid_data_under_index_left = np.searchsorted(
            self.encode_records, valid_cell_under_indices, side="left"
        )
        valid_data_under_index_right = np.searchsorted(
            self.encode_records, valid_cell_under_indices, side="right"
        )

        for valid_index, cell_index in enumerate(valid_cell_under_indices):
            match_records_indices = self.encode_records_sort_index[
                valid_data_under_index_left[valid_index] : valid_data_under_index_right[valid_index]
            ]
            np.random.shuffle(match_records_indices)

            needed = num_complete[valid_index] + num_partial[valid_index]
            if self.records_throw_indices.shape[0] >= needed:
                if num_complete[valid_index] != 0:
                    self.records[
                        self.records_throw_indices[: num_complete[valid_index]]
                    ] = self.records[
                        match_records_indices[: num_complete[valid_index]]
                    ]
                if num_partial[valid_index] != 0:
                    self.records[
                        np.ix_(
                            self.records_throw_indices[
                                num_complete[valid_index] : (num_complete[valid_index] + num_partial[valid_index])
                            ],
                            marginal_dict["attributes_index"]
                        )
                    ] = marginal_dict["tuple_key"][cell_index]
                self.records_throw_indices = self.records_throw_indices[needed:]
            else:
                # Simple fallback if we do not have enough records to adjust
                needed_count = self.records_throw_indices.size
                self.records[
                    self.records_throw_indices
                ] = self.records[match_records_indices[: needed_count]]

    def find_optimal_beta(self, num_add_total, cell_over_indices):
        actual_marginal_under = self.actual_marginal[cell_over_indices]
        synthesize_marginal_under = self.synthesize_marginal[cell_over_indices]

        lower_bound = 0.0
        upper_bound = 1.0
        beta = 0.0
        current_num = 0.0
        iteration = 0

        while abs(num_add_total - current_num) >= 1.0:
            beta = (upper_bound + lower_bound) / 2.0
            current_num = np.sum(
                np.minimum(
                    (synthesize_marginal_under - actual_marginal_under) / synthesize_marginal_under,
                    np.full(cell_over_indices.shape[0], beta)
                )
                * synthesize_marginal_under
                * self.records.shape[0]
            )
            if current_num < num_add_total:
                lower_bound = beta
            elif current_num > num_add_total:
                upper_bound = beta
            else:
                return beta

            iteration += 1
            if iteration > 50:
                break
        return beta

    def update_records_before(self, marginal_dict, marginal_key, iteration, mute=False):
        """
        Compute the synthetic marginal from the current records,
        then compute the L1 error relative to the actual marginal.
        """
        self.actual_marginal = marginal_dict["count"]
        count = marginal.count_records_general(marginal_dict, self.records)
        self.synthesize_marginal = count / np.sum(count)
        l1_error = LA.norm(self.actual_marginal - self.synthesize_marginal, 1)
        if not mute:
            logger.info("the L1 error before updating is %s" % (l1_error,))

        marginal_key = self.canonical_key(marginal_key)
        
        # Record the error in the error_tracker table if it exists
        if self.error_tracker is not None and f"{iteration}-before" in self.error_tracker.columns:
            print(marginal_key)
            self.error_tracker.loc[marginal_key, f"{iteration}-before"] = l1_error

    def update_records_after(self, marginal_dict, marginal_key, iteration):
        """
        Called after an update step to log the new L1 error.
        """
        self.actual_marginal = marginal_dict["count"]
        count = marginal.count_records_general(marginal_dict, self.records)
        self.synthesize_marginal = count / np.sum(count)
        l1_error = LA.norm(self.actual_marginal - self.synthesize_marginal, 1)
        logger.info("the L1 error after updating is %s" % (l1_error,))

        if self.error_tracker is not None and f"{iteration}-after" in self.error_tracker.columns:
            self.error_tracker.loc[marginal_key, f"{iteration}-after"] = l1_error

    def _rounding(self, vector):
        """
        Helper function to round numeric values based on the chosen rounding method.
        """
        if self.rounding_method == 'stochastic':
            ret_vector = np.zeros(vector.size)
            rand = np.random.rand(vector.size)
            integer = np.floor(vector)
            decimal = vector - integer
            ret_vector[rand > decimal] = np.floor(decimal[rand > decimal])
            ret_vector[rand < decimal] = np.ceil(decimal[rand < decimal])
            ret_vector += integer
            return ret_vector
        elif self.rounding_method == 'deterministic':
            return np.round(vector)
        else:
            raise NotImplementedError(self.rounding_method)