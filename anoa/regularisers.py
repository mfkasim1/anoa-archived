import anoa.core.ops as ops
import misc
import numpy as np
from exceptions import *

class Regulariser:
    def __init__(self, var, weights=1):
        if not isinstance(var, ops.Variable): raise TypeError("var must be a variable")
        self.var = var
        self.weights = weights
    
    def eval(self, x):
        return 0
    
    def soft_thresholding(self, x, step_size):
        return x
    
    def __mul__(self, x):
        if misc._is_constant(x):
            self.weights *= x
            return self
        else:
            raise NotImplemented
    
    def __rmul__(self, x):
        return self.__mul__(x)

class l1_regulariser(Regulariser):
    def eval(self, x):
        return np.sum(np.abs(x) * self.weights)
    
    def soft_thresholding(self, x, step_size):
        xx = x - step_size * self.weights
        xx[xx < 0] = 0
        xx = np.sign(x) * xx
        return xx

class iso_tv_2d_regulariser(Regulariser):
    def eval(self, x):
        N = x.shape
        id = range(1,N[0]) + [N[0]-1] # 1, 2, 3, ..., N-1
        ir = range(1,N[1]) + [N[1]-1] # 1, 2, 3, ..., N-1
        
        if len(N) == 2:
            dxh = np.sqrt(np.square(x - x[:,ir]) + np.square(x - x[id,:]))
        else:
            dxh = np.sqrt(np.square(x - x[:,ir,:]) + np.square(x - x[id,:,:]))
        
        return np.sum(dxh * self.weights)
    
    def soft_thresholding(self, x, step_size):
        # translation from matlab version from 
        # Pascal Getreuer 2007-2008
        alpha = step_size * self.weights
        n_iter_max = 50
        dt = 0.25
        
        N = x.shape
        id = range(1,N[0]) + [N[0]-1] # 1, 2, 3, ..., N-1
        iu = [0] + range(N[0]-1) # 0, 0, 1, 2, ..., N-2
        ir = range(1,N[1]) + [N[1]-1] # 1, 2, 3, ..., N-1
        il = [1] + range(N[1]-1) # 0, 0, 1, 2, ..., N-2
        
        p1 = np.zeros(x.shape)
        p2 = np.zeros(x.shape)
        divp = np.zeros(x.shape)
        
        if len(N) == 2:
            for i in range(n_iter_max):
                # lastdivp = divp
                z = divp - x*alpha
                z1 = z[:,ir] - z
                z2 = z[id,:] - z
                denom = 1 + dt * np.sqrt(np.square(z1) + np.square(z2))
                p1 = (p1 + dt * z1) / denom
                p2 = (p2 + dt * z2) / denom
                divp = p1 - p1[:,il] + p2 - p2[iu,:]
        
        elif len(N) == 3:
            for i in range(n_iter_max):
                # lastdivp = divp
                z = divp - x*alpha
                z1 = z[:,ir,:] - z
                z2 = z[id,:,:] - z
                denom = 1 + dt * np.sqrt(np.square(z1) + np.square(z2))
                p1 = (p1 + dt * z1) / denom
                p2 = (p2 + dt * z2) / denom
                divp = p1 - p1[:,il,:] + p2 - p2[iu,:,:]
        
        return x - divp/alpha

