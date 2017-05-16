# import some files to be shared the same namespace
from .functions.math import *
from .functions.logic import *
from .functions.fftpack import *
from .functions.array import *

from .minimise import *
from .core.ops import Variable

# # specifying what to import if called "from anoa import *"
# __all__ = []
# __all__.extend(fun_math.__all__)
# __all__.extend(fun_array.__all__)

