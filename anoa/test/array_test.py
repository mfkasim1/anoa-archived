import anoa as an
import numpy as np

def main(with_shape=1):
    shape = (3,3)
    # known elements
    value = np.reshape(np.arange(np.prod(shape))+1, shape)
    v = an.Variable(shape if with_shape else None)
    val_dict = {v:value}
    
    # random numbers
    value2 = np.random.randint(0,10, shape)
    v2 = an.Variable(shape if with_shape else None)
    val_dict2 = {v2:value2}
    
    # sum test
    if True:
        p1 = an.sum(v)
        assert np.allclose(p1.eval(val_dict), np.sum(value))
        g1 = p1.grad(val_dict)
        assert np.allclose(g1[v], np.ones(shape))
    
    # shear test
    if True:
        p2 = an.shear(v, shift_per_pixel=1, direction_axis=0, surface_normal_axis=1)
        assert np.allclose(p2.eval(val_dict), [[1,0,0],[4,2,0],[7,5,3],[0,8,6],[0,0,9]])
        p2s = an.sum(an.square(p2))/2.
        assert np.allclose(p2s.grad(val_dict)[v], value)
        
        p2 = an.shear(v, shift_per_pixel=2, direction_axis=0, surface_normal_axis=1)
        assert np.allclose(p2.eval(val_dict), [[1,0,0],[4,0,0],[7,2,0],[0,5,0],[0,8,3],[0,0,6],[0,0,9]])
        p2s = an.sum(an.square(p2))/2.
        assert np.allclose(p2s.grad(val_dict)[v], value)
        
        p2 = an.shear(v, shift_per_pixel=1, direction_axis=1, surface_normal_axis=0)
        assert np.allclose(p2.eval(val_dict), [[1,2,3,0,0],[0,4,5,6,0],[0,0,7,8,9]])
        p2s = an.sum(an.square(p2))/2.
        assert np.allclose(p2s.grad(val_dict)[v], value)
        
        p2 = an.shear(v, shift_per_pixel=-1, direction_axis=1, surface_normal_axis=0)
        assert np.allclose(p2.eval(val_dict), [[0,0,1,2,3],[0,4,5,6,0],[7,8,9,0,0]])
        p2s = an.sum(an.square(p2))/2.
        assert np.allclose(p2s.grad(val_dict)[v], value)
        
        p2 = an.shear(v, shift_per_pixel=-1, direction_axis=0, surface_normal_axis=1)
        assert np.allclose(p2.eval(val_dict), np.reshape([0,0,3, 0,2,6, 1,5,9, 4,8,0, 7,0,0], (5,3)))
        p2s = an.sum(an.square(p2))/2.
        assert np.allclose(p2s.grad(val_dict)[v], value)
    
    # shift test
    if True:
        p3 = an.shift(v, shift=1, axis=0, boundary="periodic")
        assert np.allclose(p3.eval(val_dict), np.reshape([7,8,9, 1,2,3, 4,5,6], shape))
        p3s = an.sum(an.square(p3))/2.
        assert np.allclose(p3s.grad(val_dict)[v], value)
        
        p3 = an.shift(v, shift=2, axis=0, boundary="periodic")
        assert np.allclose(p3.eval(val_dict), np.reshape([4,5,6, 7,8,9, 1,2,3], shape))
        p3s = an.sum(an.square(p3))/2.
        assert np.allclose(p3s.grad(val_dict)[v], value)
        
        p3 = an.shift(v, shift=-1, axis=0, boundary="periodic")
        assert np.allclose(p3.eval(val_dict), np.reshape([4,5,6, 7,8,9, 1,2,3], shape))
        p3s = an.sum(an.square(p3))/2.
        assert np.allclose(p3s.grad(val_dict)[v], value)
        
        p3 = an.shift(v, shift=-2, axis=0, boundary="periodic")
        assert np.allclose(p3.eval(val_dict), np.reshape([7,8,9, 1,2,3, 4,5,6], shape))
        p3s = an.sum(an.square(p3))/2.
        assert np.allclose(p3s.grad(val_dict)[v], value)
        
        p3 = an.shift(v, shift=1, axis=1, boundary="periodic")
        assert np.allclose(p3.eval(val_dict), np.reshape([3,1,2, 6,4,5, 9,7,8], shape))
        p3s = an.sum(an.square(p3))/2.
        assert np.allclose(p3s.grad(val_dict)[v], value)
        
        p3 = an.shift(v, shift=2, axis=1, boundary="periodic")
        assert np.allclose(p3.eval(val_dict), np.reshape([2,3,1, 5,6,4, 8,9,7], shape))
        p3s = an.sum(an.square(p3))/2.
        assert np.allclose(p3s.grad(val_dict)[v], value)
        
        p3 = an.shift(v, shift=-1, axis=1, boundary="periodic")
        assert np.allclose(p3.eval(val_dict), np.reshape([2,3,1, 5,6,4, 8,9,7], shape))
        p3s = an.sum(an.square(p3))/2.
        assert np.allclose(p3s.grad(val_dict)[v], value)
        
        p3 = an.shift(v, shift=1, axis=0, boundary="zeros")
        assert np.allclose(p3.eval(val_dict), np.reshape([0,0,0, 1,2,3, 4,5,6], shape))
        p3s = an.sum(an.square(p3))/2.
        assert np.allclose(p3s.grad(val_dict)[v], value * np.array([[1,1,1],[1,1,1],[0,0,0]]))
        
        p3 = an.shift(v, shift=2, axis=0, boundary="zeros")
        assert np.allclose(p3.eval(val_dict), np.reshape([0,0,0, 0,0,0, 1,2,3], shape))
        p3s = an.sum(an.square(p3))/2.
        assert np.allclose(p3s.grad(val_dict)[v], value * np.array([[1,1,1],[0,0,0],[0,0,0]]))
        
        p3 = an.shift(v, shift=-1, axis=0, boundary="zeros")
        assert np.allclose(p3.eval(val_dict), np.reshape([4,5,6, 7,8,9, 0,0,0], shape))
        p3s = an.sum(an.square(p3))/2.
        assert np.allclose(p3s.grad(val_dict)[v], value * np.array([[0,0,0],[1,1,1],[1,1,1]]))
        
        p3 = an.shift(v, shift=1, axis=1, boundary="zeros")
        assert np.allclose(p3.eval(val_dict), np.reshape([0,1,2, 0,4,5, 0,7,8], shape))
        p3s = an.sum(an.square(p3))/2.
        assert np.allclose(p3s.grad(val_dict)[v], value * np.array([[1,1,0],[1,1,0],[1,1,0]]))
        
        p3 = an.shift(v, shift=-1, axis=1, boundary="zeros")
        assert np.allclose(p3.eval(val_dict), np.reshape([2,3,0, 5,6,0, 8,9,0], shape))
        p3s = an.sum(an.square(p3))/2.
        assert np.allclose(p3s.grad(val_dict)[v], value * np.array([[0,1,1],[0,1,1],[0,1,1]]))
        
        p3 = an.shift(v, shift=1, axis=0, boundary="symmetric")
        assert np.allclose(p3.eval(val_dict), np.reshape([1,2,3, 1,2,3, 4,5,6], shape))
        p3s = an.sum(an.square(p3))/2.
        assert np.allclose(p3s.grad(val_dict)[v], value * np.reshape([2,2,2, 1,1,1, 0,0,0], shape))
        
        p3 = an.shift(v, shift=2, axis=0, boundary="symmetric")
        assert np.allclose(p3.eval(val_dict), np.reshape([4,5,6, 1,2,3, 1,2,3], shape))
        p3s = an.sum(an.square(p3))/2.
        assert np.allclose(p3s.grad(val_dict)[v], value * np.reshape([2,2,2, 1,1,1, 0,0,0], shape))
        
        p3 = an.shift(v, shift=-1, axis=0, boundary="symmetric")
        assert np.allclose(p3.eval(val_dict), np.reshape([4,5,6, 7,8,9, 7,8,9], shape))
        p3s = an.sum(an.square(p3))/2.
        assert np.allclose(p3s.grad(val_dict)[v], value * np.reshape([0,0,0, 1,1,1, 2,2,2], shape))
        
        p3 = an.shift(v, shift=1, axis=1, boundary="symmetric")
        assert np.allclose(p3.eval(val_dict), np.reshape([1,1,2, 4,4,5, 7,7,8], shape))
        p3s = an.sum(an.square(p3))/2.
        assert np.allclose(p3s.grad(val_dict)[v], value * np.reshape([2,1,0, 2,1,0, 2,1,0], shape))
        
        p3 = an.shift(v, shift=-1, axis=1, boundary="symmetric")
        assert np.allclose(p3.eval(val_dict), np.reshape([2,3,3, 5,6,6, 8,9,9], shape))
        p3s = an.sum(an.square(p3))/2.
        assert np.allclose(p3s.grad(val_dict)[v], value * np.reshape([0,1,2, 0,1,2, 0,1,2], shape))
        
        p3 = an.shift(v, shift=1, axis=0, boundary="reflect")
        assert np.allclose(p3.eval(val_dict), np.reshape([4,5,6, 1,2,3, 4,5,6], shape))
        p3s = an.sum(an.square(p3))/2.
        assert np.allclose(p3s.grad(val_dict)[v], value * np.reshape([1,1,1, 2,2,2, 0,0,0], shape))
        
        p3 = an.shift(v, shift=2, axis=0, boundary="reflect")
        assert np.allclose(p3.eval(val_dict), np.reshape([7,8,9, 4,5,6, 1,2,3], shape))
        p3s = an.sum(an.square(p3))/2.
        assert np.allclose(p3s.grad(val_dict)[v], value * np.reshape([1,1,1, 1,1,1, 1,1,1], shape))
        
        p3 = an.shift(v, shift=1, axis=1, boundary="reflect")
        assert np.allclose(p3.eval(val_dict), np.reshape([2,1,2, 5,4,5, 8,7,8], shape))
        p3s = an.sum(an.square(p3))/2.
        assert np.allclose(p3s.grad(val_dict)[v], value * np.reshape([1,2,0, 1,2,0, 1,2,0], shape))
        
        p3 = an.shift(v, shift=-1, axis=1, boundary="reflect")
        assert np.allclose(p3.eval(val_dict), np.reshape([2,3,2, 5,6,5, 8,9,8], shape))
        p3s = an.sum(an.square(p3))/2.
        assert np.allclose(p3s.grad(val_dict)[v], value * np.reshape([0,2,1, 0,2,1, 0,2,1], shape))
    
    # flip test
    if True:
        p4 = an.flip(v, axis=0)
        assert np.allclose(p4.eval(val_dict), np.reshape([7,8,9, 4,5,6, 1,2,3], shape))
        p4s = an.sum(an.square(p4))/2.
        assert np.allclose(p4s.grad(val_dict)[v], value)
        
        p4 = an.flip(v, axis=1)
        assert np.allclose(p4.eval(val_dict), np.reshape([3,2,1, 6,5,4, 9,8,7], shape))
        p4s = an.sum(an.square(p4))/2.
        assert np.allclose(p4s.grad(val_dict)[v], value)
    
    # rot90 test
    if True:
        for k in [0,1,-1,2,-2]:
            for axis in [(0,1), (1,0)]:
                p5 = an.rot90(v, k=k, axis=axis)
                assert np.allclose(p5.eval(val_dict), np.rot90(value, k, axis))
                p5s = an.sum(an.square(p5))/2.
                assert np.allclose(p5s.grad(val_dict)[v], value)
    
    # min test
    if True:
        p = an.min(v, axis=0)
        assert np.allclose(p.eval(val_dict), [1,2,3])
        ps = an.sum(an.square(p))/2.
        assert np.allclose(ps.grad(val_dict)[v], value * np.reshape([1,1,1, 0,0,0, 0,0,0], shape))
        
        p = an.min(v, axis=1)
        assert np.allclose(p.eval(val_dict), [1,4,7])
        ps = an.sum(an.square(p))/2.
        assert np.allclose(ps.grad(val_dict)[v], value * np.reshape([1,0,0, 1,0,0, 1,0,0], shape))
        
        p = an.min(v, axis=(0,1))
        assert np.allclose(p.eval(val_dict), [1])
        ps = an.sum(an.square(p))/2.
        assert np.allclose(ps.grad(val_dict)[v], value * np.reshape([1,0,0, 0,0,0, 0,0,0], shape))
        
        p = an.min(v, axis=None)
        assert np.allclose(p.eval(val_dict), [1])
        ps = an.sum(an.square(p))/2.
        assert np.allclose(ps.grad(val_dict)[v], value * np.reshape([1,0,0, 0,0,0, 0,0,0], shape))
    
    # max test
    if True:
        p = an.max(v, axis=0)
        assert np.allclose(p.eval(val_dict), [7,8,9])
        ps = an.sum(an.square(p))/2.
        assert np.allclose(ps.grad(val_dict)[v], value * np.reshape([0,0,0, 0,0,0, 1,1,1], shape))
        
        p = an.max(v, axis=1)
        assert np.allclose(p.eval(val_dict), [3,6,9])
        ps = an.sum(an.square(p))/2.
        assert np.allclose(ps.grad(val_dict)[v], value * np.reshape([0,0,1, 0,0,1, 0,0,1], shape))
        
        p = an.max(v, axis=(0,1))
        assert np.allclose(p.eval(val_dict), [9])
        ps = an.sum(an.square(p))/2.
        assert np.allclose(ps.grad(val_dict)[v], value * np.reshape([0,0,0, 0,0,0, 0,0,1], shape))
        
        p = an.max(v, axis=None)
        assert np.allclose(p.eval(val_dict), [9])
        ps = an.sum(an.square(p))/2.
        assert np.allclose(ps.grad(val_dict)[v], value * np.reshape([0,0,0, 0,0,0, 0,0,1], shape))
    
    # sort test
    if True:
        p = an.sort(v2, axis=0)
        assert np.allclose(p.eval(val_dict2), np.sort(value2, axis=0))
        ps = an.sum(an.square(v2))/2.
        assert np.allclose(ps.grad(val_dict2)[v2], value2)
        
        p = an.sort(v2, axis=1)
        assert np.allclose(p.eval(val_dict2), np.sort(value2, axis=1))
        ps = an.sum(an.square(p))/2.
        assert np.allclose(ps.grad(val_dict2)[v2], value2)
        
        p = an.sort(v2, axis=-1)
        assert np.allclose(p.eval(val_dict2), np.sort(value2, axis=-1))
        ps = an.sum(an.square(p))/2.
        assert np.allclose(ps.grad(val_dict2)[v2], value2)
    
    
    print("2D array test %s shape completed" % ("with" if with_shape else "without"))

if __name__ == "__main__":
    main()
