import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import numpy as np
from scipy.ndimage import imread
from algorithms import fista, owlqn, twist
from basis import identity, dct2, wavedec2
import matplotlib.pyplot as plt

transform = dct2
regulariser = "l1"

def evaluation(img, observation, randomPos):
    # calculate the function (Ax-b)^2
    img = transform(img, inverse=1)
    Axb = img[randomPos] - observation
    f = np.sum(np.square(Axb))
    
    # get the gradient 2*A.T*(Ax-b)
    ATAxb = np.zeros(img.shape)
    ATAxb[randomPos] = Axb
    ATAxb = transform(ATAxb)
    grad = 2*ATAxb
    
    return f, grad

def main():
    # img = imread("escher-waterfall.bmp")
    img = imread("red-makassar.jpg")
    if (len(img.shape) > 2): img = np.mean(img, axis=2)
    
    sample = 1./100
    randomPos = (np.random.random(img.shape) < sample)
    observation = img[randomPos]
    
    evalFun = lambda x: evaluation(x, observation, randomPos)
    imgRetrieved = fista(img.shape, evalFun, 1., regulariser_function=regulariser)
    
    # get the retrieved image
    imgRetrieved = transform(imgRetrieved, inverse=1)
    
    # get the masked image
    maskedImg = img * randomPos
    
    # show the images
    plt.subplot(1,3,2)
    plt.imshow(maskedImg, cmap="gray")
    plt.title("Sample")
    
    plt.subplot(1,3,3)
    plt.imshow(imgRetrieved, cmap="gray")
    plt.title("Retrieved")
    
    plt.subplot(1,3,1)
    plt.imshow(img, cmap="gray")
    plt.title("Original")
    plt.show()

if __name__ == "__main__":
    main()
