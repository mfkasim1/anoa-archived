import numpy as np
from exceptions import *
import anoa.operators.fftpack as pm
from anoa.functions.decorator import unary_function, binary_function
import scipy.fftpack as scft

__all__ = ["dct", "idct", "dst", "idst", "fft", "ifft",
           "dct2", "idct2", "dst2", "idst2", "fft2", "ifft2",
           "dctn", "idctn", "dstn", "idstn", "fftn", "ifftn"]

########## 1D transformation ##########

@unary_function(scft.dct, "dct")
def dct(x, type=2, axis=-1):
    return x._add_unary_op(pm._DCT, type=type, axis=axis)

@unary_function(scft.idct, "idct")
def idct(x, type=2, axis=-1):
    return x._add_unary_op(pm._IDCT, type=type, axis=axis)

@unary_function(scft.dst, "dst")
def dst(x, type=2, axis=-1):
    return x._add_unary_op(pm._DST, type=type, axis=axis)

@unary_function(scft.idst, "idst")
def idst(x, type=2, axis=-1):
    return x._add_unary_op(pm._IDST, type=type, axis=axis)

@unary_function(scft.fft, "fft")
def fft(x, axis=-1):
    return x._add_unary_op(pm._FFT, axis=axis)

@unary_function(scft.ifft, "ifft")
def ifft(x, axis=-1):
    return x._add_unary_op(pm._IFFT, axis=axis)

########## 2D transformation ##########

@unary_function(pm._DCTN.forward_static, "dct2")
def dct2(x, type=2, axes=(-2,-1), norm="ortho"):
    return x._add_unary_op(pm._DCT2, type=type, axes=axes)

@unary_function(pm._DCTN.adjoint_static, "idct2")
def idct2(x, type=2, axes=(-2,-1), norm="ortho"):
    return x._add_unary_op(pm._IDCT2, type=type, axes=axes)

@unary_function(pm._DSTN.forward_static, "dct2")
def dst2(x, type=2, axes=(-2,-1), norm="ortho"):
    return x._add_unary_op(pm._DST2, type=type, axes=axes)

@unary_function(pm._DSTN.adjoint_static, "idct2")
def idst2(x, type=2, axes=(-2,-1), norm="ortho"):
    return x._add_unary_op(pm._IDST2, type=type, axes=axes)

@unary_function(scft.fft2, "fft2")
def fft2(x, axes=(-2,-1)):
    return x._add_unary_op(pm._FFT2, axes=axes)

@unary_function(scft.ifft2, "ifft2")
def ifft2(x, axes=(-2,-1)):
    return x._add_unary_op(pm._IFFT2, axes=axes)

########## N-D transformation ##########

@unary_function(pm._DCTN.forward_static, "dctn")
def dctn(x, type=2, axes=None):
    return x._add_unary_op(pm._DCTN, type=type, axes=axes)

@unary_function(pm._DCTN.adjoint_static, "idctn")
def idctn(x, type=2, axes=None):
    return x._add_unary_op(pm._IDCTN, type=type, axes=axes)

@unary_function(pm._DSTN.forward_static, "dctn")
def dstn(x, type=2, axes=None):
    return x._add_unary_op(pm._DSTN, type=type, axes=axes)

@unary_function(pm._DSTN.adjoint_static, "idctn")
def idstn(x, type=2, axes=None):
    return x._add_unary_op(pm._IDSTN, type=type, axes=axes)

@unary_function(scft.fftn, "fftn")
def fftn(x, axes=None):
    return x._add_unary_op(pm._FFTN, axes=axes)

@unary_function(scft.ifftn, "ifftn")
def ifftn(x, axes=None):
    return x._add_unary_op(pm._IFFTN, axes=axes)
