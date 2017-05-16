import anoa as an
import numpy as np

def main(with_shape=1):
    # scalar test
    shape = (1,)
    value1 = np.reshape(np.arange(np.prod(shape))+2, shape)
    v1 = an.Variable(shape if with_shape else None)
    val_dict1 = {v1:value1}
    value2 = np.reshape((np.arange(np.prod(shape))+1)*10, shape)
    v2 = an.Variable(shape if with_shape else None)
    val_dict2 = {v1:value1, v2:value2}
    
    # broadcasting test
    shape1 = (1,)
    shape2 = (4,3)
    shape3 = (1,3)
    shape4 = (3,)
    mvalue1 = np.reshape(np.arange(np.prod(shape1))+2, shape1).astype(float)
    m1 = an.Variable(shape1 if with_shape else None)
    mvalue2 = np.reshape((np.arange(np.prod(shape2))+1)*2, shape2).astype(float)
    m2 = an.Variable(shape2 if with_shape else None)
    mvalue3 = np.reshape(np.arange(np.prod(shape3))+4, shape3).astype(float)
    m3 = an.Variable(shape3 if with_shape else None)
    mvalue4 = np.reshape(np.arange(np.prod(shape4))+3, shape4).astype(float)
    m4 = an.Variable(shape4 if with_shape else None)
    mval_dict2 = {m1:mvalue1, m2:mvalue2, m3:mvalue3, m4:mvalue4}
    ms = [m1, m2, m3, m4]
    mvals = [mvalue1, mvalue2, mvalue3, mvalue4]
    shapes = [shape1, shape2, shape3, shape4]
    
    shape = (3,3)
    pval1 = np.reshape(np.arange(np.prod(shape))+1, shape)
    pv1 = an.Variable(shape if with_shape else None)
    pval_dict1 = {pv1:pval1}
    pval2 = np.reshape((np.arange(np.prod(shape))+1)*10, shape)
    pv2 = an.Variable(shape if with_shape else None)
    pval_dict2 = {pv1:pval1, pv2:pval2}
    
    ########################################### SCALAR-SCALAR TEST (OPS & CONST) ###########################################
    # add test
    if True:
        p = v1 + 2
        assert np.allclose(p.eval(val_dict1), value1+2)
        assert np.allclose(p.grad(val_dict1)[v1], 1)
        p = 2 + v1
        assert np.allclose(p.eval(val_dict1), value1+2)
        assert np.allclose(p.grad(val_dict1)[v1], 1)
        p = v1 + v2
        assert np.allclose(p.eval(val_dict2), value1+value2)
        assert np.allclose(p.grad(val_dict2)[v1], 1)
        assert np.allclose(p.grad(val_dict2)[v2], 1)
    
    # subtraction test
    if True:
        p = v1 - 2
        assert np.allclose(p.eval(val_dict1), value1-2)
        assert np.allclose(p.grad(val_dict1)[v1], 1)
        
        p = 2 - v1
        assert np.allclose(p.eval(val_dict1), 2-value1)
        assert np.allclose(p.grad(val_dict1)[v1], -1)
        
        p = v1 - v2
        assert np.allclose(p.eval(val_dict2), value1-value2)
        assert np.allclose(p.grad(val_dict2)[v1], 1)
        assert np.allclose(p.grad(val_dict2)[v2], -1)
    
    # multiplication test
    if True:
        p = v1 * 2
        assert np.allclose(p.eval(val_dict1), value1*2)
        assert np.allclose(p.grad(val_dict1)[v1], 2)
        
        p = 2 * v1
        assert np.allclose(p.eval(val_dict1), 2*value1)
        assert np.allclose(p.grad(val_dict1)[v1], 2)
        
        p = v1 * v2
        assert np.allclose(p.eval(val_dict2), value1*value2)
        assert np.allclose(p.grad(val_dict2)[v1], value2)
        assert np.allclose(p.grad(val_dict2)[v2], value1)
    
    # division test
    if True:
        p = v1 / 2.
        assert np.allclose(p.eval(val_dict1), value1/2.)
        assert np.allclose(p.grad(val_dict1)[v1], 1/2.)
        
        p = 2. / v1
        assert np.allclose(p.eval(val_dict1), 2./value1)
        assert np.allclose(p.grad(val_dict1)[v1], -2./value1/value1)
        
        p = v1 / v2
        assert np.allclose(p.eval(val_dict2), value1*1./value2)
        assert np.allclose(p.grad(val_dict2)[v1], 1./value2)
        assert np.allclose(p.grad(val_dict2)[v2], -value1*1./value2/value2)
    
    # power test
    if True:
        p = v1 ** 2.
        assert np.allclose(p.eval(val_dict1), value1**2.)
        assert np.allclose(p.grad(val_dict1)[v1], 2*value1)
        
        p = v1 ** v1
        assert np.allclose(p.eval(val_dict1), value1**value1)
        assert np.allclose(p.grad(val_dict1)[v1], value1**value1 * (1 + np.log(value1)))
        
        p = v1 ** v2
        assert np.allclose(p.eval(val_dict2), value1**value2)
        assert np.allclose(p.grad(val_dict2)[v1], value2 * value1**(value2-1))
        assert np.allclose(p.grad(val_dict2)[v2], value1**value2 * np.log(value1))
        
        p = 3. ** v1
        assert np.allclose(p.eval(val_dict1), 3.**value1)
        assert np.allclose(p.grad(val_dict1)[v1], 3.**value1 * np.log(3.))
    
    # neg test
    if True:
        p = -v1
        assert np.allclose(p.eval(val_dict1), -value1)
        assert np.allclose(p.grad(val_dict1)[v1], -1.)
    
    # comparison
    if True:
        p = m2 > 8
        assert np.allclose(p.eval(mval_dict2), mvalue2 > 8)
        p = m2 >= 8
        assert np.allclose(p.eval(mval_dict2), mvalue2 >= 8)
        p = m2 < 8
        assert np.allclose(p.eval(mval_dict2), mvalue2 < 8)
        p = m2 <= 8
        assert np.allclose(p.eval(mval_dict2), mvalue2 <= 8)
        mask = np.random.randint(0, 15, shape2)
        p = m2 < mask
        assert np.allclose(p.eval(mval_dict2), mvalue2 < mask)
        mask = np.random.randint(0, 15, (3,))
        p = m2 < mask
        assert np.allclose(p.eval(mval_dict2), mvalue2 < mask)
        mask = np.random.randint(0, 15, (1,3))
        p = m2 < mask
        assert np.allclose(p.eval(mval_dict2), mvalue2 < mask)
    
    ########################################### BROADCASTING TEST (OPS) ###########################################
    # add test
    if True:
        for j in range(len(ms)):
            for i in range(len(ms)):
                p = ms[j] + ms[i]
                assert np.allclose(p.eval(mval_dict2), mvals[j]+mvals[i])
                ps = an.sum(p)
                if i != j:
                    assert np.allclose(ps.grad(mval_dict2)[ms[j]], np.ones(shapes[j]) * (np.prod(shapes[i])/np.prod(shapes[j]) if np.prod(shapes[i]) > np.prod(shapes[j]) else 1))
                    assert np.allclose(ps.grad(mval_dict2)[ms[i]], np.ones(shapes[i]) * (np.prod(shapes[j])/np.prod(shapes[i]) if np.prod(shapes[j]) > np.prod(shapes[i]) else 1))
                else:
                    assert np.allclose(ps.grad(mval_dict2)[ms[i]], np.ones(shapes[i])*2)
    
    # sub test
    if True:
        for j in range(len(ms)):
            for i in range(len(ms)):
                p = ms[j] - ms[i]
                assert np.allclose(p.eval(mval_dict2), mvals[j]-mvals[i])
                ps = an.sum(p)
                if i != j:
                    assert np.allclose(ps.grad(mval_dict2)[ms[j]], np.ones(shapes[j]) * (np.prod(shapes[i])/np.prod(shapes[j]) if np.prod(shapes[i]) > np.prod(shapes[j]) else 1))
                    assert np.allclose(-ps.grad(mval_dict2)[ms[i]], np.ones(shapes[i]) * (np.prod(shapes[j])/np.prod(shapes[i]) if np.prod(shapes[j]) > np.prod(shapes[i]) else 1))
                else:
                    assert np.allclose(ps.grad(mval_dict2)[ms[i]], np.ones(shapes[i])*0)
    
    # multiplication test
    if True: # ???
        for j in range(len(ms)):
            for i in range(len(ms)):
                p = ms[j] * ms[i]
                assert np.allclose(p.eval(mval_dict2), mvals[j]*mvals[i])
                ps = an.sum(p)
                if i == j:
                    assert np.allclose(ps.grad(mval_dict2)[ms[i]], mvals[i]*2)
                # elif np.prod(shapes[i]) >= np.prod(shapes[j]):
                    # print(shapes[j], shapes[i])
                    # assert np.allclose(ps.grad(mval_dict2)[ms[i]], np.broadcast_to(mvals[j], shapes[i]))
    
    # division test
    if True: # ???
        for j in range(len(ms)):
            for i in range(len(ms)):
                p = ms[j] / ms[i]
                assert np.allclose(p.eval(mval_dict2), mvals[j]/mvals[i])
                ps = an.sum(p)
                if i == j:
                    assert np.allclose(ps.grad(mval_dict2)[ms[i]], mvals[i]*0)
                else:
                    pass # ???
    
    # power test
    if True: # ???
        for j in range(len(ms)):
            for i in range(len(ms)):
                p = ms[j] ** ms[i]
                assert np.allclose(p.eval(mval_dict2), mvals[j]**mvals[i])
                ps = an.sum(p)
                if i == j:
                    assert np.allclose(ps.grad(mval_dict2)[ms[i]], mvals[i]**mvals[i] * (1 + np.log(mvals[i])))
                else:
                    pass # ???
    
    # neg test
    if True: 
        for i in range(len(ms)):
            p = -ms[i]
            assert np.allclose(p.eval(mval_dict2), -mvals[i])
            ps = an.sum(p)
            assert np.allclose(ps.grad(mval_dict2)[ms[i]], -np.ones(shapes[i]))
    
    ########################################### BROADCASTING TEST (OPS & CONST) ###########################################
    # add test
    if True:
        for j in range(len(ms)):
            for i in range(len(ms)):
                p = ms[j] + mvals[i]
                assert np.allclose(p.eval(mval_dict2), mvals[j]+mvals[i])
                ps = an.sum(p)
                if i != j: assert np.allclose(ps.grad(mval_dict2)[ms[j]], np.ones(shapes[j]) * (np.prod(shapes[i])/np.prod(shapes[j]) if np.prod(shapes[i]) > np.prod(shapes[j]) else 1))
                else: assert np.allclose(ps.grad(mval_dict2)[ms[j]], np.ones(shapes[j]))
                
                p = mvals[i] + ms[j]
                assert np.allclose(p.eval(mval_dict2), mvals[j]+mvals[i])
                ps = an.sum(p)
                if i != j: assert np.allclose(ps.grad(mval_dict2)[ms[j]], np.ones(shapes[j]) * (np.prod(shapes[i])/np.prod(shapes[j]) if np.prod(shapes[i]) > np.prod(shapes[j]) else 1))
                else: assert np.allclose(ps.grad(mval_dict2)[ms[j]], np.ones(shapes[j]))
    
    # sub test
    if True:
        for j in range(len(ms)):
            for i in range(len(ms)):
                p = ms[j] - mvals[i]
                assert np.allclose(p.eval(mval_dict2), mvals[j]-mvals[i])
                ps = an.sum(p)
                if i != j: assert np.allclose(ps.grad(mval_dict2)[ms[j]], np.ones(shapes[j]) * (np.prod(shapes[i])/np.prod(shapes[j]) if np.prod(shapes[i]) > np.prod(shapes[j]) else 1))
                else: assert np.allclose(ps.grad(mval_dict2)[ms[j]], np.ones(shapes[j]))
                
                p = mvals[i] - ms[j]
                assert np.allclose(p.eval(mval_dict2), mvals[i]-mvals[j])
                ps = an.sum(p)
                if i != j: assert np.allclose(ps.grad(mval_dict2)[ms[j]], -np.ones(shapes[j]) * (np.prod(shapes[i])/np.prod(shapes[j]) if np.prod(shapes[i]) > np.prod(shapes[j]) else 1))
                else: assert np.allclose(ps.grad(mval_dict2)[ms[j]], -np.ones(shapes[j]))
    
    # indexing test
    if True:
        for idx in [np.index_exp[:,0], np.index_exp[:,1], np.index_exp[0,:], np.index_exp[-1,:], np.index_exp[1,0], np.index_exp[:,:], np.index_exp[np.array([0,3]),np.array([1,2])]]:
            p = m2[idx]
            assert np.allclose(p.eval(mval_dict2), mvalue2[idx])
            ps = an.sum(an.square(p)) / 2.
            mask = np.zeros(mvalue2.shape)
            mask[idx] = 1
            assert np.allclose(ps.grad(mval_dict2)[m2], mvalue2 * mask)
        
        p = m2[m2 > 12]
        assert np.allclose(p.eval(mval_dict2), mvalue2[mvalue2 > 12])
        ps = an.sum(an.square(p)) / 2.
        assert np.allclose(ps.grad(mval_dict2)[m2], mvalue2 * [mvalue2 > 12])
    
    # indexing and logical test
    if True:
        funcs = [an.logical_and, an.logical_or, an.logical_xor]
        npfuncs = [np.logical_and, np.logical_or, np.logical_xor]
        for i,f in enumerate(funcs):
            p = m2[f(m2 > 12, m3 < 20)]
            assert np.allclose(p.eval(mval_dict2), mvalue2[npfuncs[i](mvalue2 > 12, mvalue3 < 20)])
            ps = an.sum(an.square(p)) / 2.
            assert np.allclose(ps.grad(mval_dict2)[m2], mvalue2 * npfuncs[i](mvalue2 > 12, mvalue3 < 20))
            assert np.allclose(ps.grad(mval_dict2)[m3], np.zeros(mvalue3.shape))
        
        # logical_not
        p = m2[an.logical_not(m2 > 12)]
        assert np.allclose(p.eval(mval_dict2), mvalue2[np.logical_not(mvalue2 > 12)])
        ps = an.sum(an.square(p)) / 2.
        assert np.allclose(ps.grad(mval_dict2)[m2], mvalue2 * (1-(mvalue2 > 12)))
    
    print("Basic test %s completed" % ("with shape" if with_shape else "without shape"))

if __name__ == "__main__":
    main()
