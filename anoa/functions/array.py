import anoa.core.ops as ops
import numpy as np
from exceptions import *
import anoa.core.array as pa
from anoa.functions.decorator import unary_function, binary_function

__all__ = ["sum", "shear", "shift", "flip", "rot90", "transpose", "max", "min", "sort", "matmul"]

@unary_function(np.sum, "sum")
def sum(x, axis=None):
    return x._add_unary_op(pa._Sum, input_shape=x.shape, axis=axis)

@unary_function(None, "shear")
def shear(x, shift_per_pixel=1, direction_axis=-1, surface_normal_axis=-2):
    return x._add_unary_op(pa._Shear, normal_shape=x.shape, shift_per_pixel=shift_per_pixel, direction_axis=direction_axis, surface_normal_axis=surface_normal_axis)

@unary_function(np.roll, "shift")
def shift(x, shift, axis=-1, boundary="periodic"):
    return x._add_unary_op(pa._Shift, shift=shift, axis=axis, boundary=boundary)

@unary_function(np.flip, "flip")
def flip(x, axis=-1):
    return x._add_unary_op(pa._Flip, axis=axis)

@unary_function(np.rot90, "rot90")
def rot90(x, k=1, axis=(0,1)):
    return x._add_unary_op(pa._Rot90, k=k, axis=axis)

@unary_function(np.transpose, "transpose")
def transpose(x, axes=None):
    return x._add_unary_op(pa._Transpose, axes=axes)

@unary_function(np.max, "max")
def max(x, axis=None):
    return x._add_unary_op(pa._Max, input_shape=x.shape, axis=axis)

@unary_function(np.min, "min")
def min(x, axis=None):
    return x._add_unary_op(pa._Min, input_shape=x.shape, axis=axis)

@unary_function(np.sort, "sort")
def sort(x, axis=-1):
    return x._add_unary_op(pa._Sort, input_shape=x.shape, axis=axis)

@binary_function(np.matmul, "matmul")
def matmul(x, y): # TODO: implement this!
    if isinstance(x, ops.Op):
        return x._add_op("matmul", y, _Matmul_Const, _Matmul_Op)
    else: # y is an instance and x is a constant (the decorator eliminates the possibility of x & y are constants)
        return y._add_op("matmul", x, _RMatmul_Const)
