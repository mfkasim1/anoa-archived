import anoa.core.ops as ops
import numpy as np
import anoa.core.decorator as decor

class _Logical_Not(ops.Transform):
    def forward(self, x):
        return np.logical_not(x)
    
    @decor.make_output_array
    @decor.put_child_values_arguments
    def adjoint(self, x):
        return np.zeros(x.shape)

class _Logical_And_Const(ops._Const):
    def forward(self, x, y):
        return np.logical_and(x, y)
    
    def adjoint(self, grad, *op_vals):
        return [np.zeros(op_vals[0].shape), np.zeros(op_vals[1].shape)]

class _Logical_Or_Const(_Logical_And_Const):
    def forward(self, x, y):
        return np.logical_or(x, y)

class _Logical_Xor_Const(_Logical_And_Const):
    def forward(self, x, y):
        return np.logical_xor(x, y)

class _Logical_And_Op(ops.Transform):
    def forward(self, x, y):
        return np.logical_and(x, y)
    
    def adjoint(self, grad, *op_vals):
        return [np.zeros(op_vals[0].shape), np.zeros(op_vals[1].shape)]

class _Logical_Or_Op(_Logical_And_Op):
    def forward(self, x, y):
        return np.logical_or(x, y)

class _Logical_Xor_Op(_Logical_And_Op):
    def forward(self, x, y):
        return np.logical_xor(x, y)
