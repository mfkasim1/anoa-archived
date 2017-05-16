import anoa as an
from anoa.core.ops import Variable
import time
import numpy as np
from anoa.algorithms import fista, owlqn, twist
from scipy.ndimage import imread
from anoa.regularisers import l1_regulariser, iso_tv_2d_regulariser
import matplotlib.pyplot as plt

a_val = [2.001, 2]
b_val = [4.001, 4]
a = Variable((2,))
b = Variable((2,))
# e = a - 32
d = a * b + b * (a - b)
e = an.sum((d - 32) ** 2)
val_dict = {a: a_val, b: b_val}

t0 = time.time()
print(e.eval(val_dict))
print(time.time()-t0)

t1 = time.time()
grad = e.grad(val_dict)
print(time.time() - t1)
print("grad a:", grad[a], "grad b:", grad[b])

t2 = time.time()
sz = 100
aa_val = np.random.randn(sz,sz)
aa = Variable((sz,sz))
bb = an.shift(aa, shift=2, axis=0, boundary="zeros")
bb = an.shear(aa, shift_per_pixel=-1, direction_axis=0, surface_normal_axis=1)
cc = an.shear(bb, shift_per_pixel=1, direction_axis=0, surface_normal_axis=1)
dd = cc[sz-1:sz-1+sz,:]
ee = an.cos(dd - aa)
# ee = an.rot90(aa)
print(ee.eval({aa: aa_val}))
print(time.time() - t2)

######################## compressed sensing test ########################
# load the image and set the randomised picture
img = imread("examples/escher-waterfall.bmp")
# img = imread("examples/red-makassar.jpg")
shape = img.shape
sample = 1./10
randomPos = (np.random.random(img.shape) < sample)
observation = img[randomPos]

# set the variables
img_var = Variable(shape)
idx = img_var[randomPos]
loss = an.sum(an.square(observation - idx))
regulariser = iso_tv_2d_regulariser(img_var, weights=5)
img_val = an.minimise(loss, fista, regulariser)

plt.imshow(img_val, cmap="gray")
plt.show()









