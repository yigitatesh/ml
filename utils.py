import numpy as np

def euclidean_distance(x, y):
    """Calculate euclidean distance of two numpy arrays or numbers.

    Parameters
    ----------
    x : first array or number

    y : second array or number

    Returns
    -------
    number: calculated distance as float
    """
    return np.sum((x - y) ** 2) ** 0.5

def mode(lst):
    """Get most occured thing in a list.

    Parameters
    ----------
    lst : a list.

    Returns
    -------
    element : most occured thing in the list"""
    return max(set(lst), key=lst.count)
