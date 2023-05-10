import math


def calculate_entropy(p1):
    """Calculates the entropy of some probability, p1. The negative case is implicit.
    By convention, log(0) is treated as 0.

    Args:
        p1 (float): In range [0, 1]. The ratio of positive cases.
        
    Returns:
        H (float): The calculated entropy.
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
    

if __name__ == "__main__":
    pass
