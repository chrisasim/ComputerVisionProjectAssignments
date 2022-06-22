from skimage import data, feature, color, filters, img_as_float, exposure
from skimage.feature import peak_local_max
from skimage.feature import hog
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scipy import signal
from scipy import ndimage as ndi
import cv2
from math import pi,cos,sin


def loadImage(img_source):
    origimg = Image.open(img_source)
    return origimg

origimg = loadImage('C:/Users/Χρήστης/Desktop/ComputerVision/Assignment_3/Lena.jpg')

def imageFloat(origimg):
    img = img_as_float(origimg)
    return img

img = imageFloat(origimg)

def diffGaussians(img,idx,sigma, k):
    s1=filters.gaussian(img, k*sigma)
    s2=filters.gaussian(img, sigma)
    dog=s1-s2
    #print(dog.size)
    return dog

diffGaussians(img, 2.0,2.0,2.0)
###Reduce salt and pepper noise as shown at slide 41 5th set.
##def saltandpepper(dx, sigma, k):
##    img = cv2.imread("C:/Users/Lena.jpg")
##    row,col,ch = img.shape
##    noisy = img
##  # Salt mode
##    num_salt = np.ceil(k * img.size * sigma)
##    coords = [np.random.randint(0, i - 1, int(num_salt))
##          for i in img.shape]
##    noisy[coords] = 1
##  # Pepper mode
##    num_pepper = np.ceil(k * img.size * (1 - sigma))
##    coords = [np.random.randint(0, i - 1, int(num_pepper))
##            for i in img.shape]
##    noisy[coords] = 0
##    cv2.imshow('noisy', noisy)
##    cv2.waitKey(5000)
##    cv2.destroyAllWindows()
##    #img.save("C:/Users/Χρήστης/Desktop/ComputerVision/Assignment_3/nonoise.jpg")
##    
###saltandpepper(img)

def displayDiff():
    plt.subplot(2,3,1)
    plt.imshow(origimg)
    plt.title('Original Image')
    for k in [2.0,4.0,6.0,8.0,10.0,12.0,14.0]:
        for idx,sigma in enumerate([1.0,2.0,4.0,8.0,16.0]):
            #diffGaussians(img,idx,sigma, k)
            #saltandpepper(idx, sigma, k)
            #s1 = filters.gaussian(img,k*sigma)
            #s2 = filters.gaussian(img,sigma)
            # multiply by sigma to get scale invariance
            #dog = s1 - s2
            dog=diffGaussians(img, idx, sigma, k)
            plt.subplot(2,3,idx+2)
            plt.imshow(dog,cmap='RdBu')
            plt.title('DoG with sigma=' + str(sigma) + ', k=' + str(k))
            #plt.show()
        ax=plt.subplot(2,3,6)
        blobs_dog=[(x[0],x[1],x[2]) for x in feature.blob_dog(img, min_sigma=4, max_sigma=32,threshold=0.5,overlap=1.0)]
        blobs_dog += [(x[0],x[1],x[2]) for x in feature.blob_dog(-img, min_sigma=4, max_sigma=32,threshold=0.5,overlap=1.0)]
        blobs_dog = set(blobs_dog)
        img_blobs = color.gray2rgb(img)
        for blob in blobs_dog:
              y, x, r = blob
              c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
              ax.add_patch(c)
        plt.imshow(img_blobs)
        plt.title('Detected DoG Maxima')
        plt.show()

displayDiff()

def computeSameConvolution():
    sameConvolution=[]
    for k in [2.0,3.0,4.0,5.0,6.0,7.0,8.0]:
        #print(k)
        for idx,sigma in enumerate([1.0,2.0,4.0,8.0,16.0]):
                    dog=diffGaussians(img,idx,sigma,k)
                    sameConvolution=np.convolve(dog[:,0],dog[0,:],'same')
                    #print(sameConvolution)
    return sameConvolution
#computeSameConvolution()

def localMaxima(img):
    #image_max is the dilation of img with 3*3 window
    image_max = ndi.maximum_filter(img, size=3, mode='constant')
    #Comparison between image_max and img to find coordinates of local maxima
    coordinates = peak_local_max(img, min_distance=3)
    #Now, I am going to illustrate the figure
    fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('Original')
    ax[1].imshow(image_max, cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[1].set_title('Maximum filter')
    ax[2].imshow(img, cmap=plt.cm.gray)
    ax[2].autoscale(False)
    ax[2].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
    ax[2].axis('off')
    ax[2].set_title('Local max')
    fig.tight_layout()
    plt.show()


localMaxima(img)

def localMaximaScales(img):
    for k in [2.0,3.0,4.0,5.0,6.0,7.0,8.0]:
        #print(k)
        for idx,sigma in enumerate([1.0,2.0,4.0,8.0,16.0]):
            diffgaussian = diffGaussians(img, idx, sigma, k)
            #image_max is the dilation of img with 3*3 window
            image_max = ndi.maximum_filter(diffgaussian, size=3, mode='constant')
            #print(image_max)
            #Comparison between image_max and img to find coordinates of local maxima
            coordinates = peak_local_max(diffgaussian, min_distance=3)
            #print(coordinates)
            #Now, I am going to illustrate the figure
            fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
            ax = axes.ravel()
            ax[0].imshow(img, cmap=plt.cm.gray)
            ax[0].axis('off')
            ax[0].set_title('Original')
            ax[1].imshow(image_max, cmap=plt.cm.gray)
            ax[1].axis('off')
            ax[1].set_title('Maximum filter')
            ax[2].imshow(img, cmap=plt.cm.gray)
            ax[2].autoscale(False)
            ax[2].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
            ax[2].axis('off')
            ax[2].set_title('Local max'+ str(sigma) + ',k='+ str(k))
            fig.tight_layout()
            plt.show()
    
#localMaximaScales(img)
#from array import array

def computeMatrix():
    count=0
    for k in [2.0,3.0,4.0,5.0,6.0,7.0,8.0]:
        #print(k)
        for idx,sigma in enumerate([1.0,2.0,4.0,8.0,16.0]):
            diffgaussian = diffGaussians(img, idx, sigma, k)
            #image_max is the dilation of img with 3*3 window
            image_max = ndi.maximum_filter(diffgaussian, size=6*3, mode='constant')
            #print(image_max)
            #Comparison between image_max and img to find coordinates of local maxima
            coordinates = peak_local_max(diffgaussian)          
            r=10
            trace=np.trace(diffgaussian)
            determinant=np.linalg.det(diffgaussian)
            if ((trace)**2/determinant)<((r+1)**2)/r:
                #print(coordinates[(((trace)**2/determinant)<((r+1)**2)/r)])
                pointcoordinates=coordinates[(((trace)**2/determinant)<((r+1)**2)/r)]
                #print(np.std(pointcoordinates))
                #window=diffGaussians(img, idx, 6*np.std(pointcoordinates), k)
                #print(windows)

#computeMatrix()

##def sliding_window(image, stepSize, windowSize):
##	# slide a window across the image
##	for y in range(0, image.shape[0], stepSize):
##		for x in range(0, image.shape[1], stepSize):
##			# yield the current window
##			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
##
##
##sliding_window(img, 4, 262144)
##


def computeMagnitudeandOrientation():
    # Read image
    im = cv2.imread('C:/Users/Lena.jpg')
    im = np.float32(im) / 255.0
    # Calculate gradient 
    gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    #print(mag)
    #print(angle)
    return mag, angle

#print(computeMagnitudeandOrientation())

def arrayImage(img):
               data = np.array(img)
               return data

def histogramOfGradients():
    #Transform image to array
    image=arrayImage(origimg)
    #calculate histogram of gradients
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')
    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()
    return 

histogramOfGradients()

