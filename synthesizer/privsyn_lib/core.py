import copy
import multiprocessing as mp
from typing import List, Tuple, Dict, KeysView

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from .consistenter import Consistenter
from .gum import GUM
from . import marginal_utils as marginal
from ..abc_synthesizer import Synthesizer
import synthesizer.privsyn_lib.anonymizer as anonymizer

def canonical_key(attrs):
    """
    Convert any iterable (list, set, frozenset) of attribute names
    into a sorted tuple for consistent usage as a dictionary or Pandas index key.
    """
    if isinstance(attrs, (list, set, frozenset)):
        return tuple(sorted(attrs))
    return attrs  # if already a tuple or string, just return as is

class PrivSyn(Synthesizer):
    """
    Note that it inherits the class Synthesizer,
    which already has the following attributes:
    (data: DataLoader, eps, delta, sensitivity) initialized
    """

    synthesized_df = None

    # The magic value is set empirically and users may change it in command lines
    update_iterations = 30

    attrs_marginal_dict = {}
    onehot_marginal_dict = {}

    attr_list = []
    domain_size_list = []
    attr_index_map = {}

    # despite python variables can be used without specifying type, 
    # we import typing to ensure clarity
    Attrs = List[str]
    Domains = np.ndarray
    Marginals = Dict[Tuple[str], np.array]
    Clusters = Dict[Tuple[str], List[Tuple[str]]]

    def obtain_consistent_marginals(
        self, priv_marginal_config, priv_split_method
    ) -> Tuple[Marginals, int]:
        """
        Marginals are specified by a dict from attribute tuples to frequency (pandas) tables.
        First obtain noisy marginals and make them consistent.
        """

        # Step 1: generate noisy marginals
        noisy_marginals = anonymizer.get_noisy_marginals(
            self.data,
            priv_marginal_config,
            priv_split_method,
            self.eps,
            self.delta,
            self.sensitivity,
        )

        # Step 2: get an estimate of the number of records
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
        self.attr_list = self.data.obtain_attrs()
        # domain_size_list is an array recording how many distinct values each attribute has
        self.domain_size_list = np.array(
            [len(self.data.encode_schema[att]) for att in self.attr_list]
        )
        # map from attribute string to its index in attr_list
        self.attr_index_map = dict(zip(self.attr_list, range(len(self.attr_list))))

        # Build marginal dictionaries from noisy data
        noisy_onehot_marginal_dict, noisy_attr_marginal_dict = self.construct_marginals(
            noisy_marginals
        )

        # By default, we will not rely on any "public" data in this example,
        # so set the "pub" dictionaries to the same as the "noisy" ones.
        pub_onehot_marginal_dict = noisy_onehot_marginal_dict
        pub_attr_marginal_dict = noisy_attr_marginal_dict

        self.onehot_marginal_dict, self.attrs_marginal_dict = self.normalize_marginals(
            pub_onehot_marginal_dict,
            pub_attr_marginal_dict,
            noisy_attr_marginal_dict,
            self.attr_index_map,
            num_synthesize_records,
        )

        # Next, ensure that the marginals in onehot_marginal_dict are consistent
        consistenter = Consistenter(self.onehot_marginal_dict, self.domain_size_list)
        consistenter.consist_marginals()

        # After consistency, optionally normalize each marginal
        for _, marginal_dict in self.onehot_marginal_dict.items():
            total = sum(marginal_dict["count"])
            if total > 0:
                marginal_dict["count"] /= total

        # Rebuild marginals from the consistent dictionaries
        remapped_marginals = {}
        for attrs, marginal_dict in self.attrs_marginal_dict.items():
            # 'attrs' might be a frozenset or a tuple. Convert it to a sorted tuple.
            canonical_attrs = canonical_key(attrs)
            remapped_marginals[canonical_attrs] = marginal_dict["count"]

        return remapped_marginals, num_synthesize_records

    def train(self):
        """
        privsyn is a non-parametric differentially private synthesizer.
        The training process obtains noisy marginals and makes them consistent.
        """
        if self.ratio is not None:
            # Divide eps into two parts
            one_way_eps = self.eps * self.ratio
            two_way_eps = self.eps * (1 - self.ratio)
            priv_marginal_config = {
                "priv_all_two_way": {"total_eps": two_way_eps},
                "priv_all_one_way": {"total_eps": one_way_eps},
            }
        else:
            priv_marginal_config = {
                "priv_all_two_way": {"total_eps": self.eps},
                "priv_all_one_way": {"total_eps": self.eps},
            }

        priv_split_method = self.budget_split_method

        # Step 1: get noisy marginals and make sure they are consistent
        _, num_records = self.obtain_consistent_marginals(
            priv_marginal_config, priv_split_method
        )
        
        self.num_records = num_records

    def synthesize(self, num_records=0) -> pd.DataFrame:
        """
        Produce a DataFrame in size num_records if specified.
        """
        if num_records != 0:
            self.num_records = num_records

        # Simple "clustering" that groups everything together in one cluster
        clusters = self.cluster(self.attrs_marginal_dict)
        attr_list = self.attr_list
        domain_size_list = self.domain_size_list

        print("------------------------> attributes: ")
        print(attr_list)
        print("------------------------> domains: ")
        print(domain_size_list)
        print("------------------------> clusters: ")
        print(clusters)
        print("********************* START SYNTHESIZING RECORDS ********************")

        self.synthesize_records(attr_list, domain_size_list, clusters, self.num_records)
        print("------------------------> synthetic dataframe before postprocessing: ")
        print(self.synthesized_df)
        return self.synthesized_df

    def synthesize_records(
        self,
        attr_list: List[str],
        domain_size_list: np.ndarray,
        clusters: Dict[Tuple[str], List[Tuple[str]]],
        num_synthesize_records: int,
    ):
        print("------------------------> num of synthesized records: ")
        print(num_synthesize_records)

        # For each cluster
        for cluster_attrs, list_marginal_attrs in clusters.items():
            logger.info("synthesizing for %s" % (cluster_attrs,))

            # Collect singleton marginals for each attribute
            singleton_marginals = {}
            for cur_attrs, marginal_dict in self.attrs_marginal_dict.items():
                if len(cur_attrs) == 1:
                    # cur_attrs might be a frozenset; unify it
                    canonical_single = canonical_key(cur_attrs)
                    single_attr = list(canonical_single)[0]  # e.g., "age"
                    singleton_marginals[single_attr] = marginal_dict

            synthesizer = GUM(attr_list, domain_size_list, num_synthesize_records)
            synthesizer.initialize_records(
                list_marginal_attrs, method="random", singleton_marginals=singleton_marginals
            )
            
            for update_iteration in range(self.update_iterations):
                logger.info(f"Update round: {update_iteration}")

                synthesizer.update_alpha(update_iteration)
                sorted_error_attrs = synthesizer.update_order(
                    update_iteration, self.attrs_marginal_dict, list_marginal_attrs
                )

                for attrs in sorted_error_attrs:
                    # unify
                    canonical_attrs = canonical_key(attrs)
                    synthesizer.update_records(
                        self.attrs_marginal_dict[canonical_attrs],
                        update_iteration,
                        canonical_attrs,
                    )

            if self.synthesized_df is None:
                self.synthesized_df = synthesizer.df
            else:
                self.synthesized_df.loc[:, cluster_attrs] = synthesizer.df.loc[
                    :, cluster_attrs
                ]

    @staticmethod
    def normalize_marginals(
        pub_onehot_marginal_dict: Dict,
        pub_attr_marginal_dict: Dict,
        noisy_marginal_dict: Dict,
        attr_index_map: Dict[str, int],
        num_synthesize_records: int,
    ) -> Tuple[Dict, Dict]:
        """
        Optionally combine 'public' marginals with 'noisy' marginals,
        then return updated onehot_marginal_dict and attrs_marginal_dict.
        """
        pub_weight = 0.00
        noisy_weight = 1 - pub_weight

        marginals_dict = pub_attr_marginal_dict
        onehot_marginal_dict = pub_onehot_marginal_dict

        for marginal_att, new_marginal_dict in noisy_marginal_dict.items():
            # unify 'marginal_att' as a sorted tuple
            canonical_att = canonical_key(marginal_att)

            # In case the same marginal already exists in pub_attr_marginal_dict
            if canonical_att in marginals_dict:
                old_marginal_dict = marginals_dict[canonical_att]

                # Blend the counts
                blended_count = (
                    pub_weight * old_marginal_dict["count"]
                    + noisy_weight * new_marginal_dict["count"]
                )
                old_marginal_dict["count"] = blended_count

                # If your logic needs a weight_coeff, define it here
                if "weight_coeff" not in old_marginal_dict:
                    old_marginal_dict["weight_coeff"] = 1
                if "weight_coeff" not in new_marginal_dict:
                    new_marginal_dict["weight_coeff"] = 1

                old_marginal_dict["weight_coeff"] = (
                    pub_weight * old_marginal_dict["weight_coeff"]
                    + noisy_weight * new_marginal_dict["weight_coeff"]
                )
            else:
                marginals_dict[canonical_att] = new_marginal_dict

                # If needed, define a default weight_coeff
                if "weight_coeff" not in new_marginal_dict:
                    new_marginal_dict["weight_coeff"] = 1

                # Build the one-hot array for this attribute set
                marginal_onehot = PrivSyn.one_hot(canonical_att, attr_index_map)
                onehot_marginal_dict[tuple(marginal_onehot)] = new_marginal_dict

        return onehot_marginal_dict, marginals_dict

    def construct_marginals(
        self, marginals: Dict[Tuple[str], pd.DataFrame]
    ) -> Tuple[Dict, Dict]:
        """
        Construct dictionary-based marginal objects for each given pandas table.
        Return (onehot_marginal_dict, attr_marginal_dict).
        """
        onehot_marginal_dict = {}
        attr_marginal_dict = {}

        for marginal_att, marginal_value in marginals.items():
            # unify
            canonical_att = canonical_key(marginal_att)

            # Build the one-hot array for the attributes in 'canonical_att'
            marginal_onehot = self.one_hot(canonical_att, self.attr_index_map)
            marginal_onehot_array = np.array(marginal_onehot, dtype=int)

            # Create a marginal_dict using marginal_utils
            marginal_dict = marginal.create_marginal(marginal_onehot_array, self.domain_size_list)
            marginal_dict["count"] = marginal_value.values.flatten()  # fill in the counts
            marginal_dict["weight_coeff"] = 1  # default

            # Map one-hot -> marginal_dict, and attributes -> marginal_dict
            onehot_marginal_dict[tuple(marginal_onehot)] = marginal_dict
            attr_marginal_dict[canonical_att] = marginal_dict

            # Consistency check
            if len(marginal_dict["count"]) != marginal_dict["domain_size"]:
                msg = (
                    f"Length of marginal_dict['count'] ({len(marginal_dict['count'])}) "
                    f"does not match marginal_dict['domain_size'] ({marginal_dict['domain_size']})."
                )
                raise ValueError(msg)

        return onehot_marginal_dict, attr_marginal_dict

    @staticmethod
    def build_attr_set(attrs: KeysView[Tuple[str]]) -> Tuple[str]:
        """
        Convert a list of attribute tuples into a single tuple containing all attributes.
        """
        attrs_set = set()
        for attr in attrs:
            attrs_set.update(attr)
        return tuple(sorted(attrs_set))

    def cluster(self, marginals: Dict[Tuple[str], np.ndarray]) -> Dict[Tuple[str], List[Tuple[str]]]:
        """
        A simple "clustering" approach that just puts all marginals together in one group.
        """
        clusters = {}
        # unify the keys
        keys = [canonical_key(k) for k in marginals.keys()]
        # build a single big cluster
        clusters[self.build_attr_set(keys)] = keys
        return clusters

    @staticmethod
    def one_hot(cur_att: Tuple[str], attr_index_map: Dict[str, int]) -> List[int]:
        """
        Return a list of 0/1 flags of length len(attr_index_map) 
        indicating which attributes appear in cur_att.
        """
        cur_marginal_key = [0] * len(attr_index_map)
        for attr in cur_att:
            cur_marginal_key[attr_index_map[attr]] = 1
        return cur_marginal_key