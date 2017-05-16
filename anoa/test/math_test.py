import anoa as an
import numpy as np

def main():
    # [0,1]
    value1 = np.random.random()
    v1 = an.Variable()
    val_dict1 = {v1:value1}
    
    # [1,2]
    value2 = np.random.random() + 1
    v2 = an.Variable()
    val_dict2 = {v2:value2}
    
    # trigonometry test
    if True:
        funcs = [an.sin, an.cos, an.tan, an.arcsin, an.arccos, an.arctan]
        npfuncs = [np.sin, np.cos, np.tan, np.arcsin, np.arccos, np.arctan]
        derivfuncs = [np.cos,
            lambda x: -np.sin(x), # cos
            lambda x: 1./np.cos(x)**2, # tan
            lambda x: 1./np.sqrt(1 - np.square(x)), # arcsin
            lambda x: -1./np.sqrt(1 - np.square(x)), # arccos
            lambda x: 1./(1 + np.square(x)) #arctan
            ]
        for i, fun in enumerate(funcs):
            p = fun(v1)
            assert np.allclose(p.eval(val_dict1), npfuncs[i](value1))
            assert np.allclose(p.grad(val_dict1)[v1], derivfuncs[i](value1))
    
    # exponential test
    if True:
        funcs = [an.exp, an.log]
        npfuncs = [np.exp, np.log]
        derivfuncs = [np.exp, lambda x: 1./x]
        for i, fun in enumerate(funcs):
            p = fun(v1)
            assert np.allclose(p.eval(val_dict1), npfuncs[i](value1))
            assert np.allclose(p.grad(val_dict1)[v1], derivfuncs[i](value1))
    
    # hyperbolic trigonometry test
    if True:
        funcs = [an.sinh, an.cosh, an.tanh, an.arcsinh, an.arccosh, an.arctanh]
        npfuncs = [np.sinh, np.cosh, np.tanh, np.arcsinh, np.arccosh, np.arctanh]
        vars = [val_dict1, val_dict1, val_dict1, val_dict1, val_dict2, val_dict1]
        derivfuncs = [np.cosh, np.sinh,
            lambda x: 1./(np.cosh(x))**2, # tanh
            lambda x: 1./np.sqrt(1. + x**2), # arcsinh
            lambda x: 1./np.sqrt(-1. + x**2), # arccosh
            lambda x: 1./(1. - x**2) # arctanh
            ]
        for i, fun in enumerate(funcs):
            v = vars[i].keys()[0]
            p = fun(v)
            assert np.allclose(p.eval(vars[i]), npfuncs[i](vars[i][v]))
            assert np.allclose(p.grad(vars[i])[v], derivfuncs[i](vars[i][v]))
    
    print("Math test completed")

if __name__ == "__main__":
    main()
