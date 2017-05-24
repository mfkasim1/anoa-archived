import anoa as an
import time, os
import numpy as np
from scipy.ndimage import imread
import matplotlib.pyplot as plt

######################## compressed sensing test ########################
dir_path = os.path.dirname(os.path.realpath(__file__))
# load the image and set the randomised picture
fname = "escher-waterfall.bmp" # "red-makassar.jpg"
img = imread(os.path.join(dir_path, fname))
shape = img.shape
sample = 1./10
randomPos = (np.random.random(img.shape) < sample)
observation = img[randomPos]

# set the variables
img_var = an.Variable(shape)
idx = img_var[randomPos]
loss = an.sum(an.square(observation - idx))
# regulariser = an.l1_regulariser(img_var, weights=0)
regulariser = an.iso_tv_2d_regulariser(img_var, weights=5)
img_val = an.minimise(loss, an.fista, regulariser)

plt.imshow(img_val, cmap="gray")
plt.show()









