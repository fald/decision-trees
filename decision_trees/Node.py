from pandas import DataFrame

class Node:
    # A node for a binary decision tree
    def __init__(self, feature = None, threshold = None, left: 'Node' = None, right: 'Node' = None, *, value=None) -> None:
        # self.data = data # Storing subsets of data saves minor compute at cost of relatively large memory, and is only useful during training, so.
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        # If it's a leaf node, replace with most common class.
        self.value = None
        
        def is_leaf_node(self):
            return self.value is not None
        
        