import anoa.misc as misc
import anoa.core.ops as ops
from exceptions import *
from importlib import import_module
import functools

# class to change the function representation
class reprwrapper(object):
    def __init__(self, func, name):
        self._func = func
        self._name = name
        functools.update_wrapper(self, func)
    
    def __call__(self, *args, **kw):
        return self._func(*args, **kw)
    
    def __repr__(self):
        return "<Function 'anoa.%s'>" % (self._name)

# function decorators
def unary_function(numpy_function, operationStr):
    def decorator_function(func):
        # do type checking
        def intended_function(x, **kwargs):
            if misc._is_constant(x) and numpy_function != None:
                return numpy_function(x, **kwargs)
            elif isinstance(x, ops.Op):
                return func(x, **kwargs)
            else:
                raise TypeError("undefined %s function with type %s" % (operationStr, type(x)))
        
        return reprwrapper(intended_function, func.__name__)
        
    return decorator_function

def binary_function(numpy_function, operationStr, is_commutative=0):
    def decorator_function(func):
        # do type checking (only execute func if at least one of the arguments is an op)
        def intended_function(x, y, **kwargs):
            if misc._is_constant(x):
                if misc._is_constant(y): return numpy_function(x, y, **kwargs)
                elif isinstance(y, ops.Op):
                    
                    # if the function is commutative, then put the op at the beginning
                    if is_commutative: return func(y, x, **kwargs)
                    else: return func(x, y, **kwargs)
                    
                else: 
                    raise TypeError("undefined %s function with type %s and %s" % (operationStr, type(x), type(y)))
                
            elif isinstance(x, ops.Op):
                if misc._is_constant(y):
                    return func(x, y, **kwargs)
                elif isinstance(y, ops.Op):
                    return func(x, y, **kwargs)
                else:
                    raise TypeError("undefined %s function with type %s and %s" % (operationStr, type(x), type(y)))
            else:
                raise TypeError("undefined %s function with type %s and %s" % (operationStr, type(x), type(y)))
        
        return reprwrapper(intended_function, func.__name__)
        
    return decorator_function
