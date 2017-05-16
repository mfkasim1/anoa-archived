from scipy.fftpack import dct, idct
import numpy as np
try: import pywt
except: pass

def identity(x, inverse=0):
    return x

def dct1(x, inverse=0):
    if not inverse: return dct(x)
    else: return idct(x)

def dct2(x, inverse=0):
    if not inverse:
        return dct(dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)
    else:
        return idct(idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def dwt2(x, wavelet='haar', inverse=0):
    if x.shape[0] % 2 != 0 or x.shape[1] % 2 != 0:
        raise("Input Error: all input dimensions must be even")
    
    if not inverse:
        cA, (cH, cV, cD) = pywt.dwt2(x, wavelet)
        cAH = np.concatenate((cA, cH), axis=1)
        cVD = np.concatenate((cV, cD), axis=1)
        return np.concatenate((cAH, cVD), axis=0)
    else:
        cA = x[:x.shape[0]/2, :x.shape[1]/2]
        cH = x[:x.shape[0]/2, x.shape[1]/2:]
        cV = x[x.shape[0]/2:, :x.shape[1]/2]
        cD = x[x.shape[0]/2:, x.shape[1]/2:]
        coeffs = (cA, (cH, cV, cD))
        img = pywt.idwt2(coeffs, wavelet, **kwargs)
        return img

def wavedec2(x, wavelet='haar', inverse=0):
    if not inverse:
        # get the coefficients
        coeffs = pywt.wavedec2(x, wavelet)
        
        # arrange the coefficients to form an image
        imgg = coeffs[0]
        for i in range(1, len(coeffs)):
            cH, cV, cD = coeffs[i]
            
            try:
                imgg = np.concatenate((imgg, cH), axis=1)
                imggBelow = np.concatenate((cV, cD), axis=1)
                imgg = np.concatenate((imgg, imggBelow), axis=0)
            except:
                raise("Input Error: the input's shape must be able to be divided by 2^n")
            
        return imgg
    else:
        # collect the images to form the coefficients
        size = 1
        coeffs = [x[:size, :size]]
        while size < x.shape[0]:
            cH = x[:size, size:2*size]
            cV = x[size:2*size, :size]
            cD = x[size:2*size, size:2*size]
            coeffs.append((cH, cV, cD))
            size *= 2
        
        # perform inverse dwt
        img = pywt.waverec2(coeffs, wavelet)
        return img
