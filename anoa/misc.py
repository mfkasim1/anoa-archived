import numbers
import numpy as np

def _positive_axis(axis, ndim):
    return axis if axis >= 0 else ndim+axis

def _normalise_axis(axis, ndim):
    if not hasattr(axis, '__iter__'): axis = [axis]
    return sorted([_positive_axis(a, ndim) for a in axis])

def _is_number(x):
    return isinstance(x, float) or isinstance(x, numbers.Integral)
    
def _is_constant(x):
    return _is_number(x) or isinstance(x, np.ndarray) or isinstance(x, tuple) or isinstance(x, list)

def _set_val_with_def(dict, key, def_val):
    return dict[key] if key in dict.keys() else def_val
