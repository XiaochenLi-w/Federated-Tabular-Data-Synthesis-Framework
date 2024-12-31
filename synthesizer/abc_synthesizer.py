import abc
import numpy as np
import pandas as pd
from typing import Dict, Tuple


class Synthesizer(object):
    """Base class for data synthesizers with differential privacy."""
    
    __metaclass__ = abc.ABCMeta
    Marginals = Dict[Tuple[str], np.array]

    def __init__(
        self, 
        data, 
        update_iterations: int, 
        eps: float, 
        delta: float, 
        sensitivity: int, 
        budget_split_method: dict,
        ratio: float = None
    ):
        self.data = data
        self.eps = eps
        self.delta = delta
        self.sensitivity = sensitivity
        self.update_iterations = update_iterations
        self.budget_split_method = budget_split_method
        self.ratio = ratio

    @abc.abstractmethod
    def synthesize(self, fixed_n: int) -> pd.DataFrame:
        """Generate synthetic data."""
        pass

    def synthesize_cutoff(self, submit_data: pd.DataFrame) -> pd.DataFrame:
        """Ensure synthetic data size doesn't exceed maximum."""
        if submit_data.shape > 0:
            submit_data.sample()
        return submit_data

    @abc.abstractmethod
    def get_noisy_marginals(self, marginal_config, split_method) -> Marginals:
        """Get noisy marginals using noise_utils."""
        pass
