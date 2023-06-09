import math
from  typing import Tuple
import pandas as pd


# "Scratch" - no ML modules, at least!
# TODO: Bunch of inefficient recalcs
# TODO: Started with docstrings and unittests and then just kinda stopped
# TODO: Refactor for n-ary trees instead of just binary
# TODO: How about actually making a tree structure, dummy?

sample_data = pd.read_csv("data/sample_data.csv")
cols = sample_data.columns

def calculate_entropy(p1: float) -> float:
    """
    Calculates the entropy of some probability, p1. The negative case is implicit.
    By convention, log(0) is treated as 0.

    Args:
        p1 (float): In range [0, 1]. The ratio of positive cases.
        
    Returns:
        (float): The calculated entropy.
        
    Raises:
        ValueError: When the ratio of positive cases does not fall into the range of [0, 1]
    """
    if p1 < 0 or p1 > 1:
        raise ValueError("The probability must be between 0% and 100%")

    # Basic entropy
    try:
        H = -p1 * math.log(p1, 2) - (1 - p1) * math.log(1 - p1, 2)
    except ValueError:
        # Convention, if p1 or 1-p1 are 0
        H = 0
        
    return H
    
def count_positives(data: pd.DataFrame, label_index: int = -1) -> int:
    """
    Counts the number of examples in the data that have a positive label - assumed boolean

    Args:
        data (pd.DataFrame): The dataset to count positive cases of
        label_index (int, optional): The column index of the boolean variable. Defaults to -1.

    Returns:
        int: The number of positive examples in the dataset.
    """
    # TODO: refactor lol
    cols = data.columns
    return sum(data[cols[label_index]])

def split_data(data: pd.DataFrame, split_index: int) -> Tuple[pd.DataFrame]:
    """
    Splits a given dataframe into two parts based on the given column.
    
    Args:
        data (pd.DataFrame): The set of data to split
        split_index (int): The index of the column we wish to split on.
        
    Returns:
        Tuple(pd.DataFrame): The n splits of the original data.
    
    Raises:
        IndexError: If split_index does not correspond to an existing column.
    """
    subtrees = []
    data_col = data[data.columns[split_index]] 
    values = data_col.unique()
    for value in values:
        # Also ew, wtf
        subtrees.append(data[data_col == value])
    return tuple(subtrees)

def information_gain(data: pd.DataFrame, split_index: int, label_index: int =- 1) -> float:
    """
    Calculates the information gain from splitting a dataset on some feature.
    Currently assumes the split only has a binary outcome, yes or no.
    
    Args:
        data (pd.DataFrame): The subset of data we are testing information gain on
        split_index (int): The index of the column on which we are splitting
        label_index (int): The column in the dataset that holds the goal label, assumed to be -1 in most cases
    
    Returns:
        float: The information gain from splitting the dataset on the given feature.
    
    Raises:
    """
    # Little overhead, but still no need to recalc so many times
    len_data = len(data)
    
    # p_root (float): The fraction of examples in the root node with a positive label
    # ...I don't like how this looks
    p_root = count_positives(data, label_index) / len_data
    
    # pl (float): The fraction of examples in the left subtree that have a positive label
    # pr (float): The fraction of examples in the right subtree that have a positive label
    # wl (float): The fraction of examples from the root node that end up in the left subtree
    # wr (float): The fraction of examples from the root node that end up in the right subtree
    l, r = split_data(data, split_index)
    len_l, len_r = len(l), len(r)
    pl = count_positives(l, label_index) / len_l
    pr = count_positives(r, label_index) / len_r
    wl = len_l / len_data
    wr = len_r / len_data
    
    return calculate_entropy(p_root) - (wl * calculate_entropy(pl) + (wr * calculate_entropy(pr)))

def choose_feature(data: pd.DataFrame, label_index: int = -1) -> int:
    """
    Return the index of the best feature to split a dataset on.
    Goes over each feature in the dataset, calculating information gain for
    splitting on each feature and then selecting the one with the highest gain.

    Args:
        data (pd.DataFrame): The dataset you wish to split
        label_index (int, optional): The index of the label column. Defaults to -1.

    Returns:
        int: The index of the feature to split on that results in the largest information gain
    """
    # No need for cleverness, I am a dumbass and it's not like this is a proper library anyway!
    cols = data.columns
    num_cols = len(cols)
    label = cols[label_index]
    
    if label_index == -1:
        label_index = num_cols - 1
    
    # gain, index, data
    curr_best = (0, None)
    for i in range(num_cols):
        if i != label_index:
            ig = information_gain(data, i, label_index)
            print(i, ig)
            if ig > curr_best[0]:
                curr_best = (ig, i)
                
    return curr_best[1]


if __name__ == "__main__":
    print(choose_feature(sample_data))
