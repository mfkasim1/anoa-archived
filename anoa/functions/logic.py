import numpy as np
from exceptions import *
import anoa.operators.logic as pm
from anoa.functions.decorator import unary_function, binary_function

__all__ = ["logical_not",
           "logical_and", "logical_or", "logical_xor"]

@unary_function(np.logical_not, "logical not")
def logical_not(x):
    return x._add_unary_op(pm._Logical_Not)

@binary_function(np.logical_and, "logical and", is_commutative=True)
def logical_and(x, y):
    return x._add_op(y, pm._Logical_And_Const, pm._Logical_And_Op)

@binary_function(np.logical_or, "logical or", is_commutative=True)
def logical_or(x, y):
    return x._add_op(y, pm._Logical_Or_Const, pm._Logical_Or_Op)

@binary_function(np.logical_xor, "logical xor", is_commutative=True)
def logical_xor(x, y):
    return x._add_op(y, pm._Logical_Xor_Const, pm._Logical_Xor_Op)
