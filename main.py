import math


def calculate_entropy(p1):
    """Calculates the entropy of some probability, p1. The negative case is implicit.
    By convention, log(0) is treated as 0.

    Args:
        p1 (float): The ratio of positive cases.
        
    Returns:
        H (float): The calculated entropy.
    """
    
    if p1 == 1 or p1 == 0:
        return 0
    H = -p1 * math.log(p1, 2) - (1 - p1) * math.log(1 - p1, 2)
    return H
    

if __name__ == "__main__":
    pass
