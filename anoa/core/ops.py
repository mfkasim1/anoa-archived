import anoa.misc as misc
import numpy as np
from exceptions import *
import anoa.core.decorator as decor

__all__ = ["Op", "Transform", "Variable"]

class Op:
    def __init__(self, shape=None):
        self.children = []
        self.children_edge = []
        self.parents = []
        
        if shape == None: self.shape = (1,)
        else: self.shape = shape
        
        self.value = None
        self.gradient = None
        self.vars = []
        self.grad_from_parents = {}
        self.connection_from_parents = {}
        self.need_children_values = True
    
    # this function is just to override the operators from numpy (solving __radd__, __rsub__, __r***__ problems)
    def __numpy_ufunc__(self, *a, **kw):
        pass
    
    def __repr__(self):
        return "<type 'anoa.%s'>" % (self.__class__.__name__)
    
    def _add_children(self, children):
        # input checking
        if not hasattr(children, '__iter__'): children = [children]
        for child in children:
            if not isinstance(child, Op): raise TypeError("the children must be ops")
            
            # list all variables that this op depends on
            if isinstance(child, Variable) and child not in self.vars:
                self.vars.append(child)
            else:
                for child_var in child.vars:
                    if child_var not in self.vars: self.vars.append(child_var)
        
        
        # assign the children and assign self as the parent
        for child in children:
            self.children.append(child)
            self.children_edge.append(Edge(child, self))
            child.parents.append(self)
    
    def _add_op(self, x, opConst=None, opOp=None):
        # if it is a number or a numpy array
        if opConst != None and misc._is_constant(x):
            newOp = opConst(x, self)
            newOp._add_children(self)
            return newOp
        
        # if the input is another Op
        elif opOp != None and isinstance(x, Op):
            newOp = opOp(self, x)
            newOp._add_children([self, x])
            return newOp
        
        else:
            return NotImplemented
    
    def _add_unary_op(self, opOp, **kwargs):
        newOp = opOp(self, **kwargs)
        newOp._add_children(self)
        return newOp
    
    def __add__(self, x):
        return self._add_op(x, _Add_Const, _Add_Op)
    
    def __radd__(self, x):
        return self.__add__(x)
    
    def __sub__(self, x):
        return self._add_op(-1*x, _Add_Const, _Add_Op)
    
    def __rsub__(self, x):
        if misc._is_constant(x):
            newOp = -self
            return newOp._add_op(x, _Add_Const)
    
    def __mul__(self, x):
        return self._add_op(x, _Mul_Const, _Mul_Op)
    
    def __rmul__(self, x):
        return self.__mul__(x)
    
    def __truediv__(self, x):
        if misc._is_constant(x):
            return self._add_op(1./x, _Mul_Const)
        else:
            return self._add_op(x, opOp=_TrueDiv_Op)
    
    def __floordiv__(self, x):
        return self.__truediv__(x)
    
    def __rtruediv__(self, x):
        return self._add_op(x, _TrueDiv_Const)
    
    def __rfloordiv__(self, x):
        return self.__rtruediv__(x)
    
    def __div__(self, x):
        return self.__truediv__(x)
    
    def __rdiv__(self, x):
        return self.__rtruediv__(x)
    
    def __pow__(self, x):
        return self._add_op(x, _Pow_Const, _Pow_Op)
    
    def __rpow__(self, x):
        return self._add_op(x, _RPow_Const)
    
    def __lt__(self, x):
        return self._add_op(x, _Less_Const, _Less_Op)
    
    def __le__(self, x):
        return self._add_op(x, _LessEq_Const, _LessEq_Op)
    
    def __gt__(self, x):
        return self._add_op(x, _Greater_Const, _Greater_Op)
    
    def __ge__(self, x):
        return self._add_op(x, _GreaterEq_Const, _GreaterEq_Op)
    
    def eq(self, x):
        return self._add_op(x, _Eq_Const, _Eq_Op)
    
    def __ne__(self, x):
        return self._add_op(x, _NotEq_Const, _NotEq_Op)
    
    def __neg__(self):
        return self._add_op(-1., _Mul_Const)
    
    def __getitem__(self, key):
        return self._add_op(key, _Index_Const, _Index_Op)
    
    def eval(self, values):
        """
        Return the evaluated value of the op
        
        # Arguments:
            values: dictionary of variables' values with variables as the keys and their values as values
        
        # Returns:
            the evaluated value of the op
        """
        # set call_id as the id of the object
        call_id = id(self)
        
        return self._eval(values, reset_grad=False, call_id=call_id)
    
    def _eval(self, values, reset_grad=False, call_id=None):
        
        # also assign the shape of the variable
        def assign_value(val):
            self.shape = val.shape
            return val
        
        if set(values.keys()) < set(self.vars): raise TypeError("all variable values must be specified")
        
        if isinstance(self, Variable) and self in values.keys():
            self.value = assign_value(np.array(values[self]))
        
        # if it is an op non variables, then traverse down the tree
        else:
            childValues = []
            input_shapes = []
            for edge in self.children_edge:
                child = edge.child
                child_val = child._eval(values, reset_grad, call_id=call_id)
                childValues.append(child_val) # get the children's values
                input_shapes.append(child_val.shape)
                
                if reset_grad:
                    child.grad_from_parents[edge] = None
                    child.connection_from_parents[edge] = call_id
            
            # assign the input_shape (it is used in some of the functions)
            self.input_shapes = input_shapes
            if len(self.children) == 1: self.input_shape = self.input_shapes[0]
            
            # evaluate the function and assign the shapes
            self.value = assign_value(np.array(self.forward(*childValues)))
            
            # if the node does not need its children value, then remove it to save memory
            if not self.need_children_values:
                for child in self.children: child.value = None
        
        return self.value
    
    def grad(self, values, return_eval=False):
        """
        Get the partial derivative of this op relative to the listed variables.
        
        # Arguments:
            values: dictionary of variables' values with variables as the keys and their values as values
            return_eval: if True, it returns (feval,grad), if False, just returns grad
        
        # Returns:
            a dictionary with variables as keys and their partial derivatives as the values
        """
        call_id = id(self)
        return self._grad(values, return_eval=return_eval, call_id=call_id)
        
    def _grad(self, values, return_eval=False, call_id=None):
        if set(values.keys()) < set(self.vars): raise TypeError("all variable values must be specified")
        
        # evaluate all the children and reset the gradient to None
        feval = self._eval(values, reset_grad=True, call_id=call_id)
        
        # check the size, it must be a scalar or vector with size 1
        assert np.prod(self.shape) == 1, "the variable must be a scalar or vector with size 1"
        
        # initialise the gradient
        gradient = np.ones(self.shape)
        
        # traverses the gradient down to the leaves
        self._assign_gradient_and_traverses_down(values, gradient, call_id)
        
        # construct the output dictionaries
        res = {}
        for var in self.vars:
            var_grad = var.gradient
            
            # if the gradient shape does not match with the variable shape (broadcasting effect), then sum it to match the shape
            if var.shape != var_grad.shape and not (np.prod(var.shape) == 1 and np.prod(var_grad.shape) == 1):
                sum_axis = []
                keep_dim = False
                
                # it was broadcasted but the initial dimensions are same
                if len(var.shape) == len(var_grad.shape):
                    keep_dim = True
                    for i in xrange(len(var.shape)):
                        if var.shape[i] != var_grad.shape[i]: # var.shape[i] must be one in this case
                            sum_axis.append(i)
                
                # if it was broadcasted from a scalar
                elif np.prod(var.shape) == 1:
                    sum_axis = xrange(len(var_grad.shape))
                
                # it was broadcasted, but the initial dimensions are not same
                else:
                    sum_axis = xrange(len(var_grad.shape) - len(var.shape))
                
                var_grad = np.sum(var_grad, axis=tuple(sum_axis), keepdims=keep_dim)
            
            res[var] = var_grad
        
        if return_eval: return feval, res
        else: return res
    
    def _assign_gradient_and_traverses_down(self, values, gradient, call_id, edge_parent=None):
        if edge_parent == None:
            # assign the gradient
            self.gradient = gradient
        else:
            # mark the gradient from the edge parent
            self.grad_from_parents[edge_parent] = gradient
        
        # get the connections for this call
        current_connections = []
        for ep in self.connection_from_parents.keys():
            if self.connection_from_parents[ep] != call_id: continue
            current_connections.append(ep)
        
        # check if all connections from the parent already have the gradient
        all_connections_done = True
        for ep in current_connections:
            if self.grad_from_parents[ep] is None:
                all_connections_done = False
                break
        
        # if all connections already have gradients, calculate the gradient from its parents
        if all_connections_done and edge_parent != None:
            self.gradient = 0
            
            # examine all connections from the parents of this node
            for ep in current_connections:
                
                # add the gradient from the parents
                self.gradient += self.grad_from_parents[ep]
                
                # remove the saved gradient from parents to save memory
                self.grad_from_parents[ep] = None
        
        # propagate the gradient to the children if all gradients from the parents are complete
        if len(self.children) > 0 and all_connections_done:
            
            # get the children's values first
            child_values = [child.value for child in self.children]
            
            # calculate the additional factor for the child gradient
            child_gradient = self.adjoint(self.gradient, *child_values)
            
            # delete the gradient to save memory
            self.gradient = None
            
            # traverses the gradient down to the children
            for i, edge in enumerate(self.children_edge):
                edge.child._assign_gradient_and_traverses_down(values, child_gradient[i], call_id, edge_parent=edge)

class Edge:
    def __init__(self, child, parent):
        self.child = child
        self.parent = parent

class Transform(Op):
    def __init__(self, *op):
        Op.__init__(self)
        self.op = op
        
        # escaping for variable shape (such as _Index_Op)
        self.shape = None
        # try   : self.shape = self._evaluate_shape()
        # except: self.shape = None
    
    def _evaluate_shape(self):
        # generate op values with all zeros (just to check the shape)
        op_vals = []
        for op in self.op:
            op_vals.append(np.ones(op.shape))
        
        ret_val = self.forward(*op_vals)
        return ret_val.shape

class Variable(Op):
    def __init__(self, shape=None, **kwargs):
        Op.__init__(self, shape=shape, **kwargs)

########################## BASIC OPS (used in the overloading operators in Op) ##########################

class _Const(Transform):
    def __init__(self, c, *op):
        self.c = c
        Transform.__init__(self, *op)

class _Add_Const(_Const):
    def forward(self, x):
        return self.c + x
    
    @decor.put_gradient_argument_and_output
    def adjoint(self, x):
        return [1]

class _Mul_Const(_Const):
    def forward(self, x):
        return self.c * x
    
    @decor.put_gradient_argument_and_output
    def adjoint(self, x):
        return [self.c]
    
class _TrueDiv_Const(_Const):
    def forward(self, x):
        return self.c * 1. / x
    
    @decor.put_gradient_argument_and_output
    def adjoint(self, x):
        return [-self.c * 1. / (x * x)]

class _Pow_Const(_Const):
    def forward(self, x):
        return np.power(x, np.array(self.c).astype(float))
    
    @decor.put_gradient_argument_and_output
    def adjoint(self, x):
        return [self.c * np.power(x, np.array(self.c-1).astype(float))]

class _RPow_Const(_Const):
    def forward(self, x):
        return np.power(self.c, np.float(x))
    
    @decor.put_gradient_argument_and_output
    def adjoint(self, x):
        return [np.log(self.c) * np.power(self.c, np.array(x).astype(float))]

class _Comparator(Transform):
    def adjoint(self, x, *op_vals):
        return [np.zeros(op_val.shape) for op_val in op_vals]

class _Less_Const(_Const, _Comparator):
    def forward(self, x):
        return x < self.c

class _LessEq_Const(_Const, _Comparator):
    def forward(self, x):
        return x <= self.c

class _Greater_Const(_Const, _Comparator):
    def forward(self, x):
        return x > self.c

class _GreaterEq_Const(_Const, _Comparator):
    def forward(self, x):
        return x >= self.c

class _NotEq_Const(_Const, _Comparator):
    def forward(self, x):
        return x != self.c

class _Eq_Const(_Const, _Comparator):
    def forward(self, x):
        return x == self.c

class _Index_Const(Transform):
    @decor.linear_transform_initialisation("const")
    def __init__(self, indices, input_shape=None):
        self.indices = indices
        self.input_shape = input_shape
    
    def forward(self, x):
        return np.array(x)[self.indices]
    
    @decor.put_child_values_arguments
    def adjoint(self, x):
        y = np.zeros(self.input_shape)
        y[self.indices] = x
        return y

class _Add_Op(Transform):
    def forward(self, x, y):
        return x + y
    
    @decor.put_gradient_argument_and_output
    def adjoint(self, x, y):
        return [1, 1]

class _Mul_Op(Transform):
    def forward(self, x, y):
        return x * y
    
    @decor.put_gradient_argument_and_output
    def adjoint(self, x, y):
        return [y, x]

class _TrueDiv_Op(Transform):
    def forward(self, x, y):
        return x * 1. / y
    
    @decor.put_gradient_argument_and_output
    def adjoint(self, x, y):
        return [1./y, -x*1./(y * y)]

class _Pow_Op(Transform):
    def forward(self, x, y):
        return np.power(x, np.array(y).astype(float))
    
    @decor.put_gradient_argument_and_output
    def adjoint(self, x, y):
        dzdx = y * np.power(x, np.array(y-1).astype(float))
        dzdy = np.log(x) * np.power(x, np.array(y).astype(float))
        return [dzdx, dzdy]

class _Less_Op(_Const, _Comparator):
    def forward(self, x, y):
        return x < y

class _LessEq_Op(_Const, _Comparator):
    def forward(self, x, y):
        return x <= y

class _Greater_Op(_Const, _Comparator):
    def forward(self, x, y):
        return x > y

class _GreaterEq_Op(_Const, _Comparator):
    def forward(self, x, y):
        return x >= y

class _NotEq_Op(_Const, _Comparator):
    def forward(self, x, y):
        return x != y

class _Eq_Op(_Const, _Comparator):
    def forward(self, x, y):
        return x == y

class _Index_Op(Transform):
    def forward(self, x, y):
        return np.array(x)[np.array(y)]
    
    def adjoint(self, grad, *op_vals):
        idx = op_vals[1]
        arr_grad = np.zeros(op_vals[0].shape)
        arr_grad[idx] = grad
        idx_grad = np.zeros(idx.shape)
        return [arr_grad, idx_grad]

