import math
import numpy as np
from collections import Counter
import Node

class DecisionTree:
    # TODO: Cleanup docstrings and typehints
    # A binary decision tree structure
    def __init__(self, min_samples_split: int = 2, max_depth: int = 100, n_features: int = None, root: Node = None) -> None:
        # Stopping criteria
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features # May only want to use a subset of features
        # self.data = data # Same logic as in node, don't need dataset stored in here past training, waste of memory.
        self.root = root
    
    def fit(self, X, y): # Hey look, a better idea is to have the labels split off from the beginning, like every fucking lesson you've looked at!
        # TODO: better error check/exception raise
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)
    
    def predict(self):
        pass
    
    def _grow_tree(self, X, y, depth: int = 0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))
        
        # Check stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            # if n_labels is 1, trivial, else, go for most common
            leaf_value = self._most_common_label(y)
            return Node(value = leaf_value)
        
        # Find the best split
        
        # Create child nodes
        
        # Recursively call this method on the children
        pass
    
    def _most_common_label(y):
        # TODO: Much easier with 1/0 or T/F without the collections import, maybe data clean to get to that stage. Then again, who gaf?
        counter = Counter(y)
        return counter.most_common(1)[0][0]
        
    # TODO: Does this fit here, from a design perspective? Not really a dec. tree thing uniquely. Maybe a helper function file instead.
    def _calculate_entropy(self, p: float) -> float:
        """
        Calculates the entropy of some probability, p. The negative case is implicit.
        By convention, log(0) is treated as 0.
    
        Args:
            p (float): In range [0, 1]. The ratio of positive cases.
            
        Returns:
            (float): The calculated entropy.
            
        Raises:
            ValueError: When the ratio of positive cases does not fall into the range of [0, 1]
        """
        if p < 0 or p > 1:
            raise ValueError("The probability must be in range [0, 1]")
        
        try:
            H = -p * math.log(p, 2) - (1 - p) * math.log(1 - p, 2)
        except ValueError:
            # p is 0 or 1, by convention this is treated as 0 entropy
            H = 0
            
        return H
    