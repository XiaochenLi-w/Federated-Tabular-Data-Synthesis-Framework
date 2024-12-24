import os
import sys

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

import copy
from loguru import logger
import numpy as np

# Import the function-based module under the name 'marginal'
from . import marginal_utils as marginal

"""
This file ensures that the noisy marginals are consistent (non-negative counts,
matching dependencies, etc.).
"""

class Consistenter:
    class SubsetWithDependency:
        def __init__(self, attributes_set):
            self.attributes_set = attributes_set
            # This is a set of subsets (tuples) on which this object depends
            self.dependency = set()

    def __init__(self, marginals_dict, domain_size_list):
        """
        marginals_dict: a dictionary {key -> marginal_dict}, each marginal_dict is created by marginal.create_marginal(...)
        domain_size_list: np.array describing domain sizes for all attributes
        """
        self.marginals_dict = marginals_dict
        self.domain_size_list = domain_size_list
        self.iterations = 10

    def compute_dependency(self):
        """
        Build an internal structure mapping each subset of attributes to the subsets
        on which it depends. This is used when performing consistency checks.
        """
        subsets_with_dependency = {}
        ret_subsets = {}

        for key, marginal_dict in self.marginals_dict.items():
            # Each marginal_dict["attributes_set"] is a set of attribute indices
            new_subset = self.SubsetWithDependency(marginal_dict["attributes_set"])
            subsets_temp = copy.deepcopy(subsets_with_dependency)

            for subset_key, subset_value in subsets_temp.items():
                attributes_intersection = subset_value.attributes_set & marginal_dict["attributes_set"]

                if attributes_intersection:
                    if tuple(attributes_intersection) not in subsets_with_dependency:
                        intersection_subset = self.SubsetWithDependency(attributes_intersection)
                        subsets_with_dependency[tuple(attributes_intersection)] = intersection_subset

                    if tuple(attributes_intersection) != subset_key:
                        subsets_with_dependency[subset_key].dependency.add(tuple(attributes_intersection))
                    new_subset.dependency.add(tuple(attributes_intersection))

            subsets_with_dependency[tuple(marginal_dict["attributes_set"])] = new_subset

        for subset_key, subset_value in subsets_with_dependency.items():
            # If the subset is a single attribute, we remove all dependencies from it
            if len(subset_key) == 1:
                subset_value.dependency = set()
            ret_subsets[subset_key] = subset_value

        return subsets_with_dependency

    def consist_marginals(self):
        """
        Enforce consistency constraints across all marginals using dependency relationships
        and repeated attempts to fix negative or incorrect counts.
        """

        def find_subset_without_dependency():
            # Return any subset that does not depend on anything else
            for k, s in subsets_with_dependency_temp.items():
                if not s.dependency:
                    return k, s
            return None, None

        def find_marginals_containing_target(target):
            # Return all marginal_dicts whose attribute sets include 'target'
            result = []
            for k, mdict in self.marginals_dict.items():
                if target <= mdict["attributes_set"]:
                    result.append(mdict)
            return result

        def consist_on_subset(target):
            """
            Perform consistency steps on the specified subset of attributes.
            This is done by creating a 'common_marginal_dict' for that subset
            and reconciling it with any marginals that contain it.
            """
            target_marginals = find_marginals_containing_target(target)

            # Prepare an attr_one_hot array for the subset
            common_marginal_indicator = np.zeros(self.domain_size_list.shape[0], dtype=int)
            for index in target:
                common_marginal_indicator[index] = 1

            # Build the subset-level marginal dictionary
            common_marginal_dict = marginal.create_marginal(common_marginal_indicator, self.domain_size_list)
            # Prepare consistency parameters for as many target marginals as we have
            marginal.initialize_consist_parameters(common_marginal_dict, len(target_marginals))

            # Project from each bigger marginal into this subset-level dictionary
            for idx, bigger_marginal_dict in enumerate(target_marginals):
                marginal.project_from_bigger_marginal(common_marginal_dict, bigger_marginal_dict, idx)

            # Calculate 'delta' for the new subset dictionary
            c_delta = marginal.calculate_delta(common_marginal_dict)

            # If delta is large enough, push adjustments back into the bigger marginals
            if np.sum(np.abs(c_delta)) > 1e-3:
                for idx, bigger_marginal_dict in enumerate(target_marginals):
                    marginal.update_marginal(bigger_marginal_dict, common_marginal_dict, c_delta, idx)

        def remove_subset_from_dependency(subset_obj):
            """
            Remove references to this subset from the dependencies of other subsets.
            """
            for _, s in subsets_with_dependency_temp.items():
                if tuple(subset_obj.attributes_set) in s.dependency:
                    s.dependency.remove(tuple(subset_obj.attributes_set))

        # Pre-calculate needed data for each marginal_dict (e.g., tuple keys, attribute sets, sum of counts)
        for key, mdict in self.marginals_dict.items():
            marginal.calculate_tuple_key(mdict)
            marginal.generate_attributes_index_set(mdict)
            mdict["sum"] = np.sum(mdict["count"])

        # Build the dependency relationship among subsets
        subsets_with_dependency = self.compute_dependency()
        logger.debug("dependency computed")

        # We repeat consistency steps up to self.iterations times or until stable
        non_negativity = True
        iterations = 0

        while non_negativity and iterations < self.iterations:
            # Example step for the empty set (the original code did 'consist_on_subset(set())')
            consist_on_subset(set())

            # Re-calculate sums
            for _, mdict in self.marginals_dict.items():
                mdict["sum"] = np.sum(mdict["count"])

            # Copy the dependency structure each round
            subsets_with_dependency_temp = copy.deepcopy(subsets_with_dependency)

            # Reconcile subsets that have no further dependencies
            while len(subsets_with_dependency_temp) > 0:
                key, subset = find_subset_without_dependency()
                if not subset:
                    break

                consist_on_subset(subset.attributes_set)
                remove_subset_from_dependency(subset)
                subsets_with_dependency_temp.pop(key, None)

            logger.debug("consistency step complete")

            nonneg_marginal_count = 0

            # Enforce non-negativity on each marginal, track progress
            for _, mdict in self.marginals_dict.items():
                if (mdict["count"] < 0.0).any():
                    marginal.non_negativity(mdict)
                    mdict["sum"] = np.sum(mdict["count"])
                else:
                    nonneg_marginal_count += 1

            # If all marginals are non-negative, we can stop
            if nonneg_marginal_count == len(self.marginals_dict):
                logger.info(f"All marginals non-negative after round {iterations}")
                non_negativity = False

            iterations += 1
            logger.debug("non-negativity pass complete")

        # Normalize each marginal's count if desired
        for _, mdict in self.marginals_dict.items():
            mdict["sum"] = np.sum(mdict["count"])
            if mdict["sum"] > 0:
                mdict["normalize_count"] = mdict["count"] / mdict["sum"]
            else:
                mdict["normalize_count"] = mdict["count"]  # or zeros, depending on your logic