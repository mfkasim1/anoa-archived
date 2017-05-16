import numpy as np
from exceptions import *
import anoa.operators.math as pm
from anoa.functions.decorator import unary_function, binary_function

__all__ = ["sin", "cos", "tan",
           "arcsin", "arccos", "arctan",
           "exp", "log",
           "sinh", "cosh", "tanh",
           "arcsinh", "arccosh", "arctanh",
           "sqrt", "square"]

# trigonometry
@unary_function(np.sin, "sin")
def sin(x):
    return x._add_unary_op(pm._Sin)

@unary_function(np.cos, "cos")
def cos(x):
    return x._add_unary_op(pm._Cos)

@unary_function(np.tan, "tan")
def tan(x):
    return x._add_unary_op(pm._Tan)

@unary_function(np.arcsin, "arcsin")
def arcsin(x):
    return x._add_unary_op(pm._ArcSin)

@unary_function(np.arccos, "arccos")
def arccos(x):
    return x._add_unary_op(pm._ArcCos)

@unary_function(np.arctan, "arctan")
def arctan(x):
    return x._add_unary_op(pm._ArcTan)


# exponential
@unary_function(np.exp, "exp")
def exp(x):
    return x._add_unary_op(pm._Exp)

@unary_function(np.log, "log")
def log(x):
    return x._add_unary_op(pm._Log)


# hyperbolic trigonometry
@unary_function(np.sinh, "sinh")
def sinh(x):
    return x._add_unary_op(pm._Sinh)

@unary_function(np.cosh, "cosh")
def cosh(x):
    return x._add_unary_op(pm._Cosh)

@unary_function(np.tanh, "tanh")
def tanh(x):
    return x._add_unary_op(pm._Tanh)

@unary_function(np.arcsinh, "arcsinh")
def arcsinh(x):
    return x._add_unary_op(pm._ArcSinh)

@unary_function(np.arccosh, "arccosh")
def arccosh(x):
    return x._add_unary_op(pm._ArcCosh)

@unary_function(np.arctanh, "arctanh")
def arctanh(x):
    return x._add_unary_op(pm._ArcTanh)



@unary_function(np.sqrt, "sqrt")
def sqrt(x):
    return x._add_unary_op(pm._Sqrt)

@unary_function(np.square, "square")
def square(x):
    return x._add_unary_op(pm._Square)

