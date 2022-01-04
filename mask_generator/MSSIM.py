
import numpy as np
import cv2
from scipy.signal import convolve2d
import random

def random_erode_dilate(mask, ksize=None):
    # Use cv2 erode and dilate for mask processing.
    # Fakes created with this approach should vary in their
    # mask borders
    if random.random() > 0.5:
        if ksize is None:
            ksize = random.randint(1, 5)
        if ksize % 2 == 0:
            ksize += 1
        mask = np.array(mask).astype(np.uint8)*255
        kernel = np.ones((ksize, ksize), np.uint8)
        mask = cv2.erode(mask, kernel, 1)/255
    else:
        if ksize is None:
            ksize = random.randint(1, 5)
        if ksize % 2 == 0:
            ksize += 1
        mask = np.array(mask).astype(np.uint8)*255
        kernel = np.ones((ksize, ksize), np.uint8)
        mask = cv2.dilate(mask, kernel, 1)/255
    return mask

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
 
def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)
 
def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
 
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")
 
    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))
 
    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)
 
    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2
 
    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))
    ssim_map = ((1-ssim_map) * 255).astype("uint8")
    # cv2.namedWindow("diff", cv2.WINDOW_NORMAL)
    # cv2.imshow("diff", ssim_map)
    dst = cv2.threshold(ssim_map, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #cv2.namedWindow("threshold", cv2.WINDOW_NORMAL)
    #cv2.imshow('threshold', dst)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  
    upd=cv2.erode(dst, kernel)
    upd=cv2.dilate(upd, kernel2)
    upd=cv2.erode(upd, kernel3)
    #upd=ssim_map
    #upd=cv2.erode(upd, kernel)
    #upd=cv2.dilate(upd, kernel2)
    #upd=cv2.erode(upd, kernel3)
    upd=cv2.GaussianBlur(upd,(15,15),0,0)
    #upd=random_erode_dilate(upd)
    cv2.namedWindow("threshold2", cv2.WINDOW_NORMAL)
    cv2.imshow('threshold2', upd)
    cv2.waitKey(0)
    return ssim_map#np.mean(np.mean(ssim_map))
 

if __name__ == "__main__":
    #im1 = Image.open("./data/fake/1.jpg")
    #im2 = Image.open("./data/raw/1.jpg")
    src = cv2.imread('./data/fake/1.png')
    img = cv2.imread('./data/raw/1.png')
    print(src)
    grayA = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(compute_ssim(np.array(grayA),np.array(grayB)))