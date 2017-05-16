import anoa as an
import numpy as np
import scipy.fftpack as ft

def dct2(x, type=2, axes=(-2,-1), norm="ortho"):
    return dctn(x, type=type, axes=axes, norm="ortho")

def idct2(x, type=2, axes=(-2,-1), norm="ortho"):
    return idctn(x, type=type, axes=axes, norm="ortho")

def dctn(x, type=2, axes=None, norm="ortho"):
    if axes == None: axes = np.arange(len(x.shape))
    y = x
    for axis in axes: y = ft.dct(y, type=type, axis=axis, norm="ortho")
    return y

def idctn(x, type=2, axes=None, norm="ortho"):
    if axes == None: axes = np.arange(len(x.shape))
    y = x
    for axis in axes: y = ft.idct(y, type=type, axis=axis, norm="ortho")
    return y

def dst2(x, type=2, axes=(-2,-1), norm="ortho"):
    return dstn(x, type=type, axes=axes, norm="ortho")

def idst2(x, type=2, axes=(-2,-1), norm="ortho"):
    return idstn(x, type=type, axes=axes, norm="ortho")

def dstn(x, type=2, axes=None, norm="ortho"):
    if axes == None: axes = np.arange(len(x.shape))
    y = x
    for axis in axes: y = ft.dst(y, type=type, axis=axis, norm="ortho")
    return y

def idstn(x, type=2, axes=None, norm="ortho"):
    if axes == None: axes = np.arange(len(x.shape))
    y = x
    for axis in axes: y = ft.idst(y, type=type, axis=axis, norm="ortho")
    return y

def main():
    shape1 = (10,)
    shape2 = (10,10)
    shape3 = (10,10,10)
    value1 = np.random.randn(*shape1)
    value2 = np.random.randn(*shape2)
    value3 = np.random.randn(*shape3)
    v1 = an.Variable(shape1)
    v2 = an.Variable(shape2)
    v3 = an.Variable(shape3)
    val_dict = {v1:value1, v2:value2, v3:value3}
    
    # 1D variable test
    if True:
        # DCT and DST
        funcs = [an.dct, an.idct, an.dst, an.idst]
        invfuncs = [an.idct, an.dct, an.idst, an.dst]
        npfuncs = [ft.dct, ft.idct, ft.dst, ft.idst]
        derivfuncs = [ft.idct, ft.dct, ft.idst, ft.dst]
        
        types = [2, 3]
        axes = [0, -1]
        for i in range(len(funcs)):
            # no argument
            p = funcs[i](v1)
            assert np.allclose(p.eval(val_dict), npfuncs[i](value1, norm="ortho"))
            g = an.sum(p)
            assert np.allclose(g.grad(val_dict)[v1], derivfuncs[i](np.ones(shape1), norm="ortho"))
            
            # iterate all the arguments
            for type in types:
                for axis in axes:
                    p = funcs[i](v1, type=type, axis=axis)
                    assert np.allclose(p.eval(val_dict), npfuncs[i](value1, type=type, axis=axis, norm="ortho"))
                    g = an.sum(p)
                    assert np.allclose(g.grad(val_dict)[v1], derivfuncs[i](np.ones(shape1), type=type, axis=axis, norm="ortho"))
                    
                    # inverse
                    p = invfuncs[i](p, type=type, axis=axis)
                    assert np.allclose(p.eval(val_dict), value1)
                    g = an.sum(p)
                    assert np.allclose(g.grad(val_dict)[v1], np.ones(shape1))
    
    # 2D variable test
    if True:
        funcs = [an.dct, an.idct, an.dst, an.idst, an.dct2, an.idct2, an.dst2, an.idst2]
        invfuncs = [an.idct, an.dct, an.idst, an.dst, an.idct2, an.dct2, an.idst2, an.dst2]
        npfuncs = [ft.dct, ft.idct, ft.dst, ft.idst, dct2, idct2, dst2, idst2]
        derivfuncs = [ft.idct, ft.dct, ft.idst, ft.dst, idct2, dct2, idst2, dst2]
        dim = [1, 1, 1, 1, 2, 2, 2, 2]
        
        types = [2, 3]
        axes1 = [0, -1, 1, -2]
        axes2 = [(0, 1), (-2, -1)]
        for i in range(len(funcs)):
            # no argument
            p = funcs[i](v2)
            assert np.allclose(p.eval(val_dict), npfuncs[i](value2, norm="ortho"))
            g = an.sum(p)
            assert np.allclose(g.grad(val_dict)[v2], derivfuncs[i](np.ones(shape2), norm="ortho"))
            
            if dim[i] == 1:
                for type in types:
                    for axis in axes1:
                        # normal 
                        p = funcs[i](v2, type=type, axis=axis)
                        assert np.allclose(p.eval(val_dict), npfuncs[i](value2, type=type, axis=axis, norm="ortho"))
                        g = an.sum(p)
                        assert np.allclose(g.grad(val_dict)[v2], derivfuncs[i](np.ones(shape2), type=type, axis=axis, norm="ortho"))
                        
                        # inverse
                        p = invfuncs[i](p, type=type, axis=axis)
                        assert np.allclose(p.eval(val_dict), value2)
                        g = an.sum(p)
                        assert np.allclose(g.grad(val_dict)[v2], np.ones(shape2))
                        
            elif dim[i] == 2:
                for type in types:
                    for axes in axes2:
                        # normal 
                        p = funcs[i](v2, type=type, axes=axes)
                        assert np.allclose(p.eval(val_dict), npfuncs[i](value2, type=type, axes=axes, norm="ortho"))
                        g = an.sum(p)
                        assert np.allclose(g.grad(val_dict)[v2], derivfuncs[i](np.ones(shape2), type=type, axes=axes, norm="ortho"))
                        
                        # inverse
                        p = invfuncs[i](p, type=type, axes=axes)
                        assert np.allclose(p.eval(val_dict), value2)
                        g = an.sum(p)
                        assert np.allclose(g.grad(val_dict)[v2], np.ones(shape2))
    
    # 3D variable test
    if True:
        funcs = [an.dct, an.idct, an.dst, an.idst, an.dct2, an.idct2, an.dst2, an.idst2, an.dctn, an.idctn, an.dstn, an.idstn]
        invfuncs = [an.idct, an.dct, an.idst, an.dst, an.idct2, an.dct2, an.idst2, an.dst2, an.idctn, an.dctn, an.idstn, an.dstn]
        npfuncs = [ft.dct, ft.idct, ft.dst, ft.idst, dct2, idct2, dst2, idst2, dctn, idctn, dstn, idstn]
        derivfuncs = [ft.idct, ft.dct, ft.idst, ft.dst, idct2, dct2, idst2, dst2, idctn, dctn, idstn, dstn]
        dim = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        
        types = [2, 3]
        axes1 = [0, 1, 2, -3, -2, -1]
        axes2 = [(0, 1), (0, 2), (1, 2), (-3, -2), (-3, -1), (-2, -1)]
        axes3 = [(0, 1, 2), (-3, -2, -1)]
        for i in range(len(funcs)):
            # no argument
            p = funcs[i](v3)
            assert np.allclose(p.eval(val_dict), npfuncs[i](value3, norm="ortho"))
            g = an.sum(p)
            assert np.allclose(g.grad(val_dict)[v3], derivfuncs[i](np.ones(shape3), norm="ortho"))
            
            if dim[i] == 1:
                for type in types:
                    for axis in axes1:
                        # normal 
                        p = funcs[i](v3, type=type, axis=axis)
                        assert np.allclose(p.eval(val_dict), npfuncs[i](value3, type=type, axis=axis, norm="ortho"))
                        g = an.sum(p)
                        assert np.allclose(g.grad(val_dict)[v3], derivfuncs[i](np.ones(shape3), type=type, axis=axis, norm="ortho"))
                        
                        # inverse
                        p = invfuncs[i](p, type=type, axis=axis)
                        assert np.allclose(p.eval(val_dict), value3)
                        g = an.sum(p)
                        assert np.allclose(g.grad(val_dict)[v3], np.ones(shape3))
                        
            else:
                for type in types:
                    axess = axes2 if dim[i] == 2 else axes3
                    for axes in axess:
                        # normal 
                        p = funcs[i](v3, type=type, axes=axes)
                        assert np.allclose(p.eval(val_dict), npfuncs[i](value3, type=type, axes=axes, norm="ortho"))
                        g = an.sum(p)
                        assert np.allclose(g.grad(val_dict)[v3], derivfuncs[i](np.ones(shape3), type=type, axes=axes, norm="ortho"))
                        
                        # inverse
                        p = invfuncs[i](p, type=type, axes=axes)
                        assert np.allclose(p.eval(val_dict), value3)
                        g = an.sum(p)
                        assert np.allclose(g.grad(val_dict)[v3], np.ones(shape3))
    
    print("FFTpack test completed")

if __name__ == "__main__":
    main()
