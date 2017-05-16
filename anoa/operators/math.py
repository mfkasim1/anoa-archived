import anoa.core.ops as ops
import numpy as np
import anoa.core.decorator as decor

################################ TRIGONOMETRY ################################
class _Sin(ops.Transform):
    def forward(self, x):
        return np.sin(x)
    
    @decor.put_gradient_argument_and_output
    @decor.make_output_array
    def adjoint(self, x):
        return np.cos(x)

class _Cos(ops.Transform):
    def forward(self, x):
        return np.cos(x)
    
    @decor.put_gradient_argument_and_output
    @decor.make_output_array
    def adjoint(self, x):
        return -np.sin(x)

class _Tan(ops.Transform):
    def forward(self, x):
        return np.tan(x)
    
    @decor.put_gradient_argument_and_output
    @decor.make_output_array
    def adjoint(self, x):
        return 1./np.square(np.cos(x))

class _ArcSin(ops.Transform):
    def forward(self, x):
        return np.arcsin(x)
    
    @decor.put_gradient_argument_and_output
    @decor.make_output_array
    def adjoint(self, x):
        return 1./np.sqrt(1-np.square(x))

class _ArcCos(ops.Transform):
    def forward(self, x):
        return np.arccos(x)
    
    @decor.put_gradient_argument_and_output
    @decor.make_output_array
    def adjoint(self, x):
        return -1./np.sqrt(1-np.square(x))

class _ArcTan(ops.Transform):
    def forward(self, x):
        return np.arctan(x)
    
    @decor.put_gradient_argument_and_output
    @decor.make_output_array
    def adjoint(self, x):
        return 1./(1+np.square(x))

################################ EXPONENTIAL AND LOGARITHMIC ################################
class _Exp(ops.Transform):
    def forward(self, x):
        return np.exp(x)
    
    @decor.put_gradient_argument_and_output
    @decor.make_output_array
    def adjoint(self, x):
        return np.exp(x)

class _Log(ops.Transform):
    def forward(self, x):
        return np.log(x)
    
    @decor.put_gradient_argument_and_output
    @decor.make_output_array
    def adjoint(self, x):
        return 1./x

################################ HYPERBOLIC ################################
class _Sinh(ops.Transform):
    def forward(self, x):
        return np.sinh(x)
    
    @decor.put_gradient_argument_and_output
    @decor.make_output_array
    def adjoint(self, x):
        return np.cosh(x)

class _Cosh(ops.Transform):
    def forward(self, x):
        return np.cosh(x)
    
    @decor.put_gradient_argument_and_output
    @decor.make_output_array
    def adjoint(self, x):
        return np.sinh(x)

class _Tanh(ops.Transform):
    def forward(self, x):
        return np.tanh(x)
    
    @decor.put_gradient_argument_and_output
    @decor.make_output_array
    def adjoint(self, x):
        return 1. - np.square(np.tanh(x))

class _ArcSinh(ops.Transform):
    def forward(self, x):
        return np.arcsinh(x)
    
    @decor.put_gradient_argument_and_output
    @decor.make_output_array
    def adjoint(self, x):
        return 1./np.sqrt(1 + np.square(x))

class _ArcCosh(ops.Transform):
    def forward(self, x):
        return np.arccosh(x)
    
    @decor.put_gradient_argument_and_output
    @decor.make_output_array
    def adjoint(self, x):
        return 1./np.sqrt(-1. + np.square(x))

class _ArcTanh(ops.Transform):
    def forward(self, x):
        return np.arctanh(x)
    
    @decor.put_gradient_argument_and_output
    @decor.make_output_array
    def adjoint(self, x):
        return 1./(1. - np.square(x))

################################ MISC ################################
class _Sqrt(ops.Transform):
    def forward(self, x):
        return np.sqrt(x)
    
    @decor.put_gradient_argument_and_output
    @decor.make_output_array
    def adjoint(self, x):
        return 1. / 2 / np.sqrt(x)

class _Square(ops.Transform):
    def forward(self, x):
        return np.square(x)
    
    @decor.put_gradient_argument_and_output
    @decor.make_output_array
    def adjoint(self, x):
        return 2. * x
