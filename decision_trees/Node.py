from typing import Any

class Node:
    # A node for a binary decision tree
    def __init__(self, feature: Any = None, threshold: Any = None, left: 'Node' = None, right: 'Node' = None, *, value=None) -> None:
        # self.data = data # Storing subsets of data saves minor compute at cost of relatively large memory, and is only useful during training, so.
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        # If it's a leaf node, replace with most common class.
        self.value = value
        
    def is_leaf_node(self) -> None:
        return self.value is not None
        