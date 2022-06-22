#Asiminidis_D05_Assignment_4_Keypoint_Descriptors
from numpy import loadtxt
from PIL import Image
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.metrics.pairwise import euclidean_distances
import scipy.ndimage
import numpy
import math

imagepath="lena.txt"

def loadtxtfile(path):
        image = loadtxt(path, unpack=False)
        return image

image=loadtxtfile(imagepath)
initialkeypointswithscalepath="lenaTr.txt"

def loadinitialkeypointswithscale(path):
        keypointswithscale=loadtxt(path, unpack=False)
        return keypointswithscale

keypoints=loadinitialkeypointswithscale(initialkeypointswithscalepath)

count=0
dlistimage=[]
for keypoint in keypoints:
        for i in image:
                dlist=[]
                if keypoint[0]*12>i[0] and keypoint[1]*12>i[1]:
                          count+=1
                        
                        
print("Number of keypoints that have passed the condition:", count)
print("descriptor that has to be found has to be of", 128*count)
orientationpath="lenaR.txt"

def loadorientation(path):
        orientation=loadtxt(path, unpack=False)
        return orientation
orientation=loadorientation(orientationpath)
def obtainimagesecondway():
        input_img = Image.open('lenaTr.pgm')
        input_img.size
        new_width  = 12
        new_height = 12
        input_img = input_img.resize((new_width*input_img.size[0], new_height*input_img.size[1]), Image.ANTIALIAS)
        input_img.save('output.jpg')
        output_img = Image.open('output.jpg')
        output_img.size
        return output_img

obtainimagesecondway()

def patches(keypoints):
        patch=extract_patches_2d(keypoints, (4,4))#, max_patches=None, random_state=None)
        return patch

patches=patches(keypoints)

sub_image=obtainimagesecondway()


def gaussiansubimageandsccale(sub_image):
        sub_image.size
        new_width=3
        new_height=3
        input_img=sub_image
        input_sub_image=input_img.resize((new_width*sub_image.size[0],new_height*sub_image.size[1]), Image.ANTIALIAS)
        gaussian_filter_sub_image=scipy.ndimage.filters.gaussian_filter(input_sub_image, 3, mode='constant')
        return gaussian_filter_sub_image

gaussianscale=gaussiansubimageandsccale(sub_image)

def computehistogramofitspixelsorientationweigthedwithgaussian(orientation, gaussianscale):
        dlist=[]
        gradients=numpy.gradient(orientation)
        histogram_of_gradients=numpy.histogram(gradients, bins=8, weights=gaussianscale)
        dlist.append(histogram_of_gradients)
        return histogram_of_gradients

def computehistogramofitspixelsorientation(orientation, patches):
        dlist=[]
        gradients=numpy.gradient(orientation)
        histogram_of_gradients=numpy.histogram(gradients, bins=8)
        dlist.append(histogram_of_gradients)
        histogram_of_gradients_normalized=numpy.histogram(gradients, bins=8, density=True)
        return histogram_of_gradients_normalized


siftfeaturedescriptor=computehistogramofitspixelsorientation(orientation,patches)

def computehistogramofitspixelsorientationweigthedwithgaussian(orientation, gaussianscale):
        dlist=[]
        gradients=numpy.gradient(orientation)
        histogram_of_gradients=numpy.histogram(gradients, bins=8, weights=gaussianscale)
        dlist.append(histogram_of_gradients)
        print(histogram_of_gradients)
        return histogram_of_gradients


def euclideandistancebetweenimages(image1,image2):

        distance=euclidean_distances(image1,image2)
        return distance

euclideandistanceimages=euclideandistancebetweenimages(image,keypoints)

def computetheminimumvalue(euclideanmatrix, threshold):
        minimumvalue=euclideanmatrix.min(axis=1)
        
        return minimumvalue
        
computetheminimumvalue(euclideandistanceimages, 1)
