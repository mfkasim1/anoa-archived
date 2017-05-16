import numpy as np
import anoa.misc as misc
import anoa.core.ops as ops
import exceptions, copy
import anoa.core.decorator as decor

class _RMatmul_Const(ops.Transform):
    @decor.linear_transform_initialisation("const")
    def __init__(self, matrix):
        self.matrix = matrix
        self.matrixT = np.transpose(matrix)
    
    def forward(self, x):
        return np.matmul(self.matrix, x)
    
    @decor.make_output_array
    @decor.put_child_values_arguments
    def adjoint(self, x):
        return np.matmul(self.matrixT, x)

class _Matmul_Const(ops.Transform):
    @decor.linear_transform_initialisation("const")
    def __init__(self, matrix):
        self.matrix = matrix
        self.matrixT = np.transpose(matrix)
    
    def forward(self, x):
        return np.matmul(x, self.matrix)
    
    @decor.make_output_array
    @decor.put_child_values_arguments
    def adjoint(self, x):
        return np.matmul(x, self.matrixT)

class _Matmul_Op(ops.Transform):
    def forward(self, x, y):
        return np.matmul(x, y)
    
    def adjoint(self, x, *op_vals):
        child0 = op_vals[0]
        child1 = op_vals[1]
        
        grad0 = np.matmul(x, np.transpose(child1))
        grad1 = np.matmul(np.transpose(child0), x)
        return [grad0, grad1]

class _Mask(ops.Transform):
    """
    ???
    """
    @decor.linear_transform_initialisation("unary")
    def __init__(self, mask):
        self.mask = mask
    
    def forward(self, x):
        return x * self.mask
    
    @decor.make_output_array
    @decor.put_child_values_arguments
    def adjoint(self, x):
        return x * self.mask

class _Sum(ops.Transform):
    """
    ???
    """
    @decor.linear_transform_initialisation("unary")
    def __init__(self, input_shape, axis=None):
        self.input_shape = input_shape
        self.axis = axis
    
    def forward(self, x):
        return np.sum(x, axis=self.axis)
    
    @decor.make_output_array
    @decor.put_child_values_arguments
    @decor.normalise_axis
    def adjoint(self, x):
        # expand the axis to have the same dimension as self.shape
        x_expand = x
        for axis in self.axis:
            x_expand = np.expand_dims(x, axis)
        
        # copy the elements into the new axis
        return np.broadcast_to(x_expand, self.input_shape)

class _Shear(ops.Transform):
    """
    ???
    """
    @decor.linear_transform_initialisation("unary")
    def __init__(self, normal_shape=None, sheared_shape=None, shift_per_pixel=1, direction_axis=-1, surface_normal_axis=-2):
        self.shift_per_pixel = int(shift_per_pixel)
        self.direction_axis = direction_axis
        self.surface_normal_axis = surface_normal_axis
        self.shift_per_pixel = shift_per_pixel
    
    def _assign_shape(self, normal_shape):
        # convert the axis to positive
        ndim = len(normal_shape)
        self.normal_shape = normal_shape
        self.direction_axis = misc._positive_axis(self.direction_axis, ndim)
        self.surface_normal_axis = misc._positive_axis(self.surface_normal_axis, ndim)
        
        # calculate the sheared shape
        self.sheared_shape = list(normal_shape)
        self.sheared_shape[self.direction_axis] = normal_shape[self.direction_axis] + (normal_shape[self.surface_normal_axis] - 1) * abs(self.shift_per_pixel)
    
    def _assign_sheared_shape(self, sheared_shape):
        # convert the axis to positive
        ndim = len(sheared_shape)
        self.sheared_shape = sheared_shape
        self.direction_axis = misc._positive_axis(self.direction_axis, ndim)
        self.surface_normal_axis = misc._positive_axis(self.surface_normal_axis, ndim)
        
        # calculate the input shape
        self.normal_shape = list(sheared_shape)
        self.normal_shape[self.direction_axis] = sheared_shape[self.direction_axis] + (sheared_shape[self.surface_normal_axis] - 1) * abs(self.shift_per_pixel)
    
    def forward(self, x):
        self._assign_shape(self.input_shape)
        y = np.zeros(self.sheared_shape)
        
        # copy the input, x, to y first with zero padding in direction_axis
        idx_beginning = 0 if self.shift_per_pixel > 0 else self.sheared_shape[self.direction_axis]-self.normal_shape[self.direction_axis]
        idx_end = self.normal_shape[self.direction_axis] if self.shift_per_pixel > 0 else self.sheared_shape[self.direction_axis]
        index = np.index_exp[:] * (self.direction_axis) + np.index_exp[idx_beginning:idx_end] + np.index_exp[:] * (len(self.normal_shape) - self.direction_axis - 1)
        y[index] = x
        
        # now roll the axis
        # index to access the slice in surface_normal_axis
        index = np.index_exp[:] * self.surface_normal_axis
        index_suffix = np.index_exp[:] * (len(self.normal_shape) - self.surface_normal_axis - 1)
        roll_axis = (self.direction_axis - 1) if self.surface_normal_axis < self.direction_axis else self.direction_axis
        for i in range(self.normal_shape[self.surface_normal_axis]):
            # get the i-th slice of the surface_normal_axis
            index_i = index + (i,) + index_suffix
            y[index_i] = np.roll(y[index_i], i * self.shift_per_pixel, axis=roll_axis)
        
        return y
    
    @decor.make_output_array
    @decor.put_child_values_arguments
    def adjoint(self, y):
        # transpose of shearing is just de-shearing (shear in the opposite direction)
        y_copy = np.copy(y)
        
        # roll back the axis
        # index to access the slice in surface_normal_axis
        index = np.index_exp[:] * self.surface_normal_axis
        index_suffix = np.index_exp[:] * (len(self.normal_shape) - self.surface_normal_axis - 1)
        roll_axis = (self.direction_axis - 1) if self.surface_normal_axis < self.direction_axis else self.direction_axis
        for i in range(self.normal_shape[self.surface_normal_axis]):
            # get the i-th slice of the surface_normal_axis
            index_i = index + (i,) + index_suffix
            y_copy[index_i] = np.roll(y_copy[index_i], -i * self.shift_per_pixel, axis=roll_axis)
        
        # truncate the array
        idx_beginning = 0 if self.shift_per_pixel > 0 else self.sheared_shape[self.direction_axis]-self.normal_shape[self.direction_axis]
        idx_end = self.normal_shape[self.direction_axis] if self.shift_per_pixel > 0 else self.sheared_shape[self.direction_axis]
        index = np.index_exp[:] * (self.direction_axis) + np.index_exp[idx_beginning:idx_end] + np.index_exp[:] * (len(self.normal_shape) - self.direction_axis - 1)
        x = y_copy[index]
        return x

class _Shift(ops.Transform):
    """
    ???
    """
    @decor.linear_transform_initialisation("unary")
    def __init__(self, shift, axis=-1, boundary="periodic"):
        self.shift = shift 
        self.axis = axis
        self.boundary = boundary.lower()
        
        self.list_of_boundaries = ["periodic", "symmetric", "reflect", "zeros", "same"]
        assert self.boundary in self.list_of_boundaries, "the boundary argument must be one of %s or a number" % self.list_of_boundaries
    
    def forward(self, x):
        # input checking
        assert self.shift < x.shape[self.axis], "the input size in axis %d (%d) must be more than the shift: %d" % (self.axis, x.shape[self.axis], self.shift)
        
        # roll the axis
        y = np.roll(x, self.shift, axis=self.axis)
        
        if self.boundary == "periodic":
            return y
        
        else:
            axis = misc._positive_axis(self.axis, len(x.shape))
            
            # index of the newly shifted-in elements
            if self.shift >= 0:
                idx_begin = 0
                idx_end = self.shift
            else:
                idx_begin = x.shape[axis] + self.shift
                idx_end = x.shape[axis]
            
            index = np.index_exp[:] * axis + np.index_exp[idx_begin:idx_end] + np.index_exp[:] * (len(x.shape) - axis - 1)
            
            if self.boundary == "zeros":
                y[index] = 0
                return y
            
            elif self.boundary == "same":
                # obtain the index for the edge of the shifted elements
                if self.shift >= 0: idx_edge = self.shift
                else:               idx_edge = x.shape[axis] + self.shift - 1
                
                index_edge = np.index_exp[:] * axis + np.index_exp[idx_edge:idx_edge+1] + np.index_exp[:] * (len(x.shape) - axis - 1)
                
                # broadcast the edge's value to fill in the shifted-in elements
                y[index] = np.broadcast_to(y[index_edge], y[index].shape)
                return y
            
            elif self.boundary == "symmetric":
                # get the index of the input element in the shifted axis, reversed
                if self.shift >= 0:
                    idx_begin = self.shift - 1
                    idx_end = -1
                else:
                    idx_begin = x.shape[axis] - 1
                    idx_end = x.shape[axis] - 1 + self.shift
                
                if idx_end == -1:
                    index_input = np.index_exp[:] * axis + np.index_exp[idx_begin::-1] + np.index_exp[:] * (len(x.shape) - axis - 1)
                else:
                    index_input = np.index_exp[:] * axis + np.index_exp[idx_begin:idx_end:-1] + np.index_exp[:] * (len(x.shape) - axis - 1)
                
                # fill in the shifted-in element with the input elements
                y[index] = x[index_input]
                return y
            
            elif self.boundary == "reflect":
                # get the index of the input element in the shifted axis, reversed
                if self.shift >= 0:
                    idx_begin = self.shift
                    idx_end = 0
                else:
                    idx_begin = x.shape[axis] - 2
                    idx_end = x.shape[axis] - 2 + self.shift
                    
                index_input = np.index_exp[:] * axis + np.index_exp[idx_begin:idx_end:-1] + np.index_exp[:] * (len(x.shape) - axis - 1)
                
                # fill in the shifted-in element with the input elements
                y[index] = x[index_input]
                return y
    
    @decor.make_output_array
    @decor.put_child_values_arguments
    def adjoint(self, x):
        # input checking
        assert self.shift < x.shape[self.axis], "the input size in axis %d must be more than the shift %d" % (self.axis, self.shift)
        
        # roll the axis
        y = np.roll(x, -self.shift, axis=self.axis)
        
        if self.boundary == "periodic":
            return y
        
        else:
            axis = misc._positive_axis(self.axis, len(x.shape))
            
            # index of the newly shifted-in and shifted-out elements
            if self.shift >= 0:
                idx_begin_in = x.shape[axis] - self.shift
                idx_end_in = x.shape[axis]
                idx_begin_out = 0
                idx_end_out = self.shift
            else:
                idx_begin_in = 0
                idx_end_in = -self.shift
                idx_begin_out = x.shape[axis] + self.shift
                idx_end_out = x.shape[axis]
            
            index_in = np.index_exp[:] * axis + np.index_exp[idx_begin_in:idx_end_in] + np.index_exp[:] * (len(x.shape) - axis - 1)
            index_out = np.index_exp[:] * axis + np.index_exp[idx_begin_out:idx_end_out] + np.index_exp[:] * (len(x.shape) - axis - 1)
            
            # zeroing the newly shifted-in elements
            y[index_in] = 0
            
            if self.boundary == "zeros":
                return y
            
            elif self.boundary == "same":
                # obtain the index for the edge of the shifted elements
                if self.shift >= 0: idx_edge = 0
                else:               idx_edge = x.shape[axis] - 1
                index_edge = np.index_exp[:] * axis + (idx_edge,) + np.index_exp[:] * (len(x.shape) - axis - 1)
                
                y[index_edge] += np.sum(x[index_out], axis=axis)
                return y
            
            elif self.boundary == "symmetric":
                y[index_out] += np.flip(x[index_out], axis=axis)
                return y
            
            elif self.boundary == "reflect":
                # obtain the shifted-out index + 1
                if self.shift >= 0:
                    idx_begin = 1
                    idx_end = self.shift + 1
                else:
                    idx_begin = x.shape[axis] + self.shift - 1
                    idx_end = x.shape[axis] - 1
                
                index_out2 = np.index_exp[:] * axis + np.index_exp[idx_begin:idx_end] + np.index_exp[:] * (len(x.shape) - axis - 1)
                
                y[index_out2] += np.flip(x[index_out], axis=axis)
                return y

class _Flip(ops.Transform):
    """
    ???
    """
    @decor.linear_transform_initialisation("unary")
    def __init__(self, axis=-1):
        self.axis = axis
    
    def forward(self, x):
        return np.flip(x, self.axis)
    
    @decor.make_output_array
    @decor.put_child_values_arguments
    def adjoint(self, x):
        return np.flip(x, self.axis)

class _Rot90(ops.Transform):
    """
    ???
    """
    @decor.linear_transform_initialisation("unary")
    def __init__(self, k=1, axis=(0,1)):
        self.k = k
        self.axis = axis
    
    def forward(self, x):
        return np.rot90(x, self.k, self.axis)
    
    @decor.make_output_array
    @decor.put_child_values_arguments
    def adjoint(self, x):
        return np.rot90(x, -self.k, self.axis)

class _Transpose(ops.Transform):
    @decor.linear_transform_initialisation("unary")
    def __init__(self, axes=None):
        self.axes = axes
    
    def forward(self, x):
        return np.transpose(x, axes=self.axes)
    
    @decor.make_output_array
    @decor.put_child_values_arguments
    def adjoint(self, x):
        if self.axes == None: return np.transpose(x)
        else:
            original_axes = np.zeros((len(self.axes),))
            original_axes[np.array(self.axes)] = np.arange(len(self.axes))
            original_axes = original_axes.astype(int)
            return np.transpose(x, original_axes)

class _MaxOrMin(_Sum): # it has the same __init__ as _Sum
    """
    ???
    """
    @decor.make_output_array
    @decor.normalise_axis
    def adjoint(self, x, *child_values):
        # expand the axis to have the same dimension as self.shape
        x_expand = x
        output_expand = self.value
        for axis in self.axis:
            x_expand = np.expand_dims(x, axis)
            output_expand = np.expand_dims(output_expand, axis)
        
        # get the mask of maximum or minimum element
        child_value = child_values[0]
        mask = (output_expand == child_value)
        
        # broadcast the gradient and apply mask
        return x_expand * mask

class _Max(_MaxOrMin):
    def forward(self, x):
        return np.max(x, axis=self.axis)

class _Min(_MaxOrMin):
    def forward(self, x):
        return np.min(x, axis=self.axis)

class _Sort(ops.Transform): # ???
    @decor.linear_transform_initialisation("unary")
    def __init__(self, input_shape, axis=-1, order=None):
        self.order = order
        if not misc._is_number(axis): raise TypeError("axis must be a number")
        self.input_shape = input_shape
        self.axis = misc._normalise_axis(axis, len(input_shape))[0]
    
    def forward(self, x):
        self.idx_sorted = np.argsort(x, axis=self.axis, order=self.order)
        
        # get the index if it is flatten
        nelmts = 1
        self.idx_sorted_flat = np.zeros(self.idx_sorted.shape)
        for i in range(len(x.shape))[::-1]:
            if i == self.axis:
                self.idx_sorted_flat += nelmts * self.idx_sorted
            else:
                idx = np.index_exp[:] + (None,) * (len(x.shape)-i-1)
                self.idx_sorted_flat += nelmts * np.arange(x.shape[i])[idx]
            
            nelmts *= x.shape[i]
        
        self.idx_sorted_flat = self.idx_sorted_flat.astype(int)
        return np.take(x, self.idx_sorted_flat)
    
    @decor.make_output_array
    def adjoint(self, x, *child_values):
        y = np.zeros(self.input_shape).flatten()
        y[self.idx_sorted_flat.flatten()] = x.flatten()
        return np.reshape(y, self.input_shape)

class _MaskedSum(ops.Transform):
    """
    ???
    """
    @decor.linear_transform_initialisation("unary")
    def __init__(self, masks):
        self.masks = masks
        self.signal_size = np.product(masks[0].shape)
        self.masks_matrix = np.reshape(np.array(masks), (masks.shape[0], self.signal_size))
    
    def forward(self, x):
        return np.matmul(self.masks_matrix, x.flatten()[:,None])
    
    @decor.make_output_array
    @decor.put_child_values_arguments
    def adjoint(self, x):
        sig_flat = np.matmul(np.transpose(self.masks_matrix), x.flatten()[:,None])
        return np.reshape(sig_flat, self.masks[0].shape)

