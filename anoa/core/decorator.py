import numpy as np

# function decorators
def make_output_array(func):
    # make the output as an array
    def intended_adjoint(obj, x, *op_vals):
        return [func(obj, x, *op_vals)]
    return intended_adjoint

def put_child_values_arguments(func):
    # adding *op_vals argument at the end and output a list
    def intended_adjoint(obj, x, *op_vals):
        return func(obj, x)
    return intended_adjoint

def put_gradient_argument_and_output(func):
    # multiply each element in the array to the previous gradient
    def intended_adjoint(obj, previous_gradient, *op_vals):
        return [previous_gradient * i for i in func(obj, *op_vals)]
    return intended_adjoint

def normalise_axis(func):
    # normalise the axis parameter in the object
    def intended_function(obj, *args):
        import anoa.misc as misc
        # normalise axis
        if obj.axis == None: obj.axis = tuple(np.arange(len(obj.input_shape)))
        else:                obj.axis = tuple(misc._normalise_axis(obj.axis, len(obj.input_shape)))
        return func(obj, *args)
    return intended_function

def linear_transform_initialisation(const_or_unary):
    const_or_unary = const_or_unary.lower()
    def decorator(func):
        if const_or_unary == "unary":
        
            # adding *op argument at the beginning
            def intended_init(obj, *op, **kwargs):
                import anoa.core.ops as ops
                func(obj, **kwargs)
                ops.Transform.__init__(obj, *op)
            return intended_init
        
        elif const_or_unary == "const":
        
            # adding *op argument at the beginning
            def intended_init(obj, c, *op, **kwargs):
                import anoa.core.ops as ops
                func(obj, c, **kwargs)
                ops.Transform.__init__(obj, *op)
            return intended_init
    
    return decorator
