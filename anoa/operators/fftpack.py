import anoa.core.ops as ops
import numpy as np
import anoa.core.decorator as decor
import scipy.fftpack as scft

########## 1D transformation ##########
class _DCT(ops.Transform):
    @decor.linear_transform_initialisation("unary")
    def __init__(self, type, axis):
        self.type = type
        self.axis = axis
    
    def forward(self, x):
        return scft.dct(x, type=self.type, axis=self.axis, norm="ortho")
    
    @decor.make_output_array
    @decor.put_child_values_arguments
    def adjoint(self, grad):
        return scft.idct(grad, type=self.type, axis=self.axis, norm="ortho")

class _IDCT(_DCT):
    def forward(self, x):
        return scft.idct(x, type=self.type, axis=self.axis, norm="ortho")
    
    @decor.make_output_array
    @decor.put_child_values_arguments
    def adjoint(self, grad):
        return scft.dct(grad, type=self.type, axis=self.axis, norm="ortho")

class _DST(_DCT):
    def forward(self, x):
        return scft.dst(x, type=self.type, axis=self.axis, norm="ortho")
    
    @decor.make_output_array
    @decor.put_child_values_arguments
    def adjoint(self, grad):
        return scft.idst(grad, type=self.type, axis=self.axis, norm="ortho")

class _IDST(_DCT):
    def forward(self, x):
        return scft.idst(x, type=self.type, axis=self.axis, norm="ortho")
    
    @decor.make_output_array
    @decor.put_child_values_arguments
    def adjoint(self, grad):
        return scft.dst(grad, type=self.type, axis=self.axis, norm="ortho")

class _FFT(ops.Transform):
    @decor.linear_transform_initialisation("unary")
    def __init__(self, axis):
        self.axis = axis
    
    def forward(self, x):
        return scft.fft(x, axis=self.axis)
    
    @decor.make_output_array
    @decor.put_child_values_arguments
    def adjoint(self, grad):
        return scft.ifft(grad, axis=self.axis)

class _IFFT(_FFT):
    def forward(self, x):
        return scft.ifft(x, axis=self.axis)
    
    @decor.make_output_array
    @decor.put_child_values_arguments
    def adjoint(self, grad):
        return scft.fft(grad, axis=self.axis)

########## 2D transformation ##########

class _DCT2(ops.Transform):
    @decor.linear_transform_initialisation("unary")
    def __init__(self, type, axes):
        self.type = type
        self.axes = axes
    
    def forward(self, x):
        return _DCTN.forward_static(x, type=self.type, axes=self.axes)
    
    @decor.make_output_array
    @decor.put_child_values_arguments
    def adjoint(self, grad):
        return _DCTN.adjoint_static(grad, type=self.type, axes=self.axes)

class _IDCT2(_DCT2):
    def forward(self, x):
        return _DCTN.adjoint_static(x, type=self.type, axes=self.axes)
    
    @decor.make_output_array
    @decor.put_child_values_arguments
    def adjoint(self, grad):
        return _DCTN.forward_static(grad, type=self.type, axes=self.axes)

class _DST2(_DCT2):
    def forward(self, x):
        return _DSTN.forward_static(x, type=self.type, axes=self.axes)
    
    @decor.make_output_array
    @decor.put_child_values_arguments
    def adjoint(self, grad):
        return _DSTN.adjoint_static(grad, type=self.type, axes=self.axes)

class _IDST2(_DCT2):
    def forward(self, x):
        return _DSTN.adjoint_static(x, type=self.type, axes=self.axes)
    
    @decor.make_output_array
    @decor.put_child_values_arguments
    def adjoint(self, grad):
        return _DSTN.forward_static(grad, type=self.type, axes=self.axes)

class _FFT2(ops.Transform):
    @decor.linear_transform_initialisation("unary")
    def __init__(self, axes):
        self.axes = axes
    
    def forward(self, x):
        return scft.fft2(x, axes=self.axes)
    
    @decor.make_output_array
    @decor.put_child_values_arguments
    def adjoint(self, grad):
        return scft.ifft2(grad, axes=self.axes)

class _IFFT2(_FFT2):
    def forward(self, x):
        return scft.ifft2(x, axes=self.axes)
    
    @decor.make_output_array
    @decor.put_child_values_arguments
    def adjoint(self, grad):
        return scft.fft2(grad, axes=self.axes)

########## N-D transformation ##########

class _DCTN(_DCT2):
    @staticmethod
    def forward_static(x, type=2, axes=None):
        if axes == None: axes = np.arange(len(x.shape))
        
        # make a copy of the input
        y = x
        
        # apply the dct for each axis
        for axis in axes: y = scft.dct(y, type=type, axis=axis, norm="ortho")
        return y
    
    @staticmethod
    def adjoint_static(x, type=2, axes=None):
        if axes == None: axes = np.arange(len(x.shape))
        
        # make a copy of the input
        y = x
        
        # apply the dct for each axis
        for axis in axes: y = scft.idct(y, type=type, axis=axis, norm="ortho")
        return y

class _IDCTN(_IDCT2):
    # the __init__ function is same as _DCT2
    # the forward and adjoint functions are same as _IDCT2
    pass

class _DSTN(_DST2):
    @staticmethod
    def forward_static(x, type=2, axes=None):
        if axes == None: axes = np.arange(len(x.shape))
        
        # make a copy of the input
        y = x
        
        # apply the dct for each axis
        for axis in axes: y = scft.dst(y, type=type, axis=axis, norm="ortho")
        return y
    
    @staticmethod
    def adjoint_static(x, type=2, axes=None):
        if axes == None: axes = np.arange(len(x.shape))
        
        # make a copy of the input
        y = x
        
        # apply the dct for each axis
        for axis in axes: y = scft.idst(y, type=type, axis=axis, norm="ortho")
        return y

class _IDSTN(_IDST2):
    # the __init__ function is same as _DST2
    # the forward and adjoint functions are same as _IDST2
    pass

class _FFTN(_FFT2):
    def forward(self, x):
        return scft.fftn(x, axes=self.axes)
    
    @decor.make_output_array
    @decor.put_child_values_arguments
    def adjoint(self, grad):
        return scft.ifftn(grad, axes=self.axes)

class _IFFTN(_FFT2):
    def forward(self, x):
        return scft.ifftn(x, axes=self.axes)
    
    @decor.make_output_array
    @decor.put_child_values_arguments
    def adjoint(self, grad):
        return scft.fftn(grad, axes=self.axes)

