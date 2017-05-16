import numpy as np
import anoa.core.ops as ops
import anoa.misc as misc

def minimise(loss, alg, reg=None, **kwargs):
    # prepare the regulariser
    if reg == None:
        regulariser_function = lambda x: 0
        thresholding_function = lambda x, a: x
    else:
        regulariser_function = reg.eval
        thresholding_function = reg.soft_thresholding
    
    # list all variables that the loss variable depends on
    vars = loss.vars
    
    # if there's only one variable, it is easy
    if len(vars) == 1:
        var = vars[0]
        input_size = var.shape
        
        def eval_fun(x):
            feval, g = loss.grad({var: x}, return_eval=True)
            return feval, g[var]
        
        return alg(input_size, eval_fun, 1,
                   regulariser_function=regulariser_function,
                   thresholding_function=thresholding_function, **kwargs)
    
    # otherwise, flatten all the variables
    else:
        var_shapes = [var.shape for var in vars]
        var_sizes = [np.prod(vs) for vs in var_shapes]
        starting_idx = [0] + var_sizes[:-1]
        input_size = (np.sum(var_sizes),)
        
        def eval_fun(x):
            # recover the shape of the variables
            values_dict = {}
            for i,var in enumerate(vars):
                values_dict[var] = np.reshape(x[starting_idx[i]:starting_idx[i]+var_sizes[i]], var_shapes[i])
            
            # evaluate and get the gradients
            feval, g = loss.grad(values_dict, return_eval=True)
            
            # rearrange the gradients into a flat shape
            grad = np.zeros(input_size)
            for i,var in enumerate(vars):
                grad[starting_idx[i]:starting_idx[i]+var_sizes[i]] = np.array(g[var]).flatten()
            
            return feval, grad
        
        return alg(input_size, eval_fun, 1,
                   regulariser_function=regulariser_function,
                   thresholding_function=thresholding_function, **kwargs)
    