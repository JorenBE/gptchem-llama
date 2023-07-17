import numpy as np
def array_of_ints_without_nan(arr):
    return arr[~np.isnan(arr)].astype(int)


def try_exccept_nan(f, x):
    try:
        return f(x)
    except:
        return np.nan
    

def get_mode(arr):
    return np.argmax(np.bincount(array_of_ints_without_nan(arr)))