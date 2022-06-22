import numpy as np
from matplotlib import pyplot as plt
import os
import cv2
from scipy import ndimage
from PIL import Image

#Synarthsh pou fortwnei thn eikona apo to directory
def loadImage(infilename):
        img = Image.open(infilename)
        img.load()
        return img

#Synarthsh poy metatrepei thn eikona se pinaka
def arrayImage(img):
               data = np.array(img)
               return data
        
#Synarth pou ypologizei thn paragwgo ws pros x               
def compute_gradient_x(img):
        sx = ndimage.sobel(img, axis=0, mode='constant')
        return sx

#Synarthsh pou ypologizei thn paragwgo ws pros y
def compute_gradient_y(img):
        sy = ndimage.sobel(img, axis=0, mode='constant')
        return sy

#fortwnw tis eikones wa morfh JPEG  
imgaustin=loadImage('C:/Users/Χρήστης/Desktop/ComputerVision/car.jpg')
       
imgdisney = loadImage('C:/Users/Χρήστης/Desktop/ComputerVision/disney.jpg')

#Synarthsh pou meiwnei to megethos ths eikonas
def compute_reduce_size(img,width, height):
        image=img.resize((width,height))
        return image

#Kanw reduce to width ths austin se 100
imgaustinreducewidth=compute_reduce_size(imgaustin, 100,322)

#kanw reduce to height ths disney se 100
imgdisneyreduceheight = compute_reduce_size(imgdisney,617,100)

#metatrepw tis eikones se pinakes
imgaustinarray=arrayImage(imgaustin)
imgdisneyarray=arrayImage(imgdisney)

#Synarthsh pou metatrepei thn eikona se grayscale 
def grayscale_converted_scale(img):
               return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Synarthsh pou kanei Gaussiano filtrarisma
def gaussian_blur(img):
               return cv2.GaussianBlur(img, (3,3),0,0)

#synarthsh pou ypologizei thn eksiswsh tou paper Avidan seam carving
def compute_energy_function(img):
        blur = gaussian_blur(img)
        gray = grayscale_converted_scale(blur)
        sy = compute_gradient_y(gray)  
        sx = compute_gradient_x(gray)
        fabsx = np.absolute(sx)
        fabsy = np.absolute(sy)
        return cv2.add(fabsx,fabsy)



energy=compute_energy_function(imgaustinarray)

#Synarthsh pou ypologizei thn athroistikh synarthsh katakoryfa
def cumulative_energies_vertical(energy):
    height, width = energy.shape[:2]
    energies = np.zeros((height, width))

    for i in range(1, height):
        for j in range(width):
            left = energies[i - 1, j - 1] if j - 1 >= 0 else 1e6
            middle = energies[i - 1, j]
            right = energies[i - 1, j + 1] if j + 1 < width else 1e6
            energies[i, j] = energy[i, j] + min(left, middle, right)

    return energies

energies_vertical=cumulative_energies_vertical(energy)

#Synarthsh pou ypologizei thn atroistikh synarthsh orizontia
def cumulative_energies_horizontal(energy):
        height, width = energy.shape[:2]
        energies=np.zeros((height, width))
        for j in range(1, width):
                for i in range(height):
                        top = energies[i-1,j-1] if i-1>=0 else 1e6
                        middle = energies[i, j-1]
                        bottom = energies[i+1, j-1] if i+1<height else 1e6
                        energies[i,j] = energy[i,j]+min(top, middle, bottom)

        return energies

energies_horizontal=cumulative_energies_horizontal(energy)

#synarthsh pou vriskei ta orizontia seams
def find_horizontal_seam(energies):
    height, width = energies.shape[:2]
    previous = 0
    seam = []

    for i in range(width - 1, -1, -1):
        col = energies[:, i]

        if i == width - 1:
            previous = np.argmin(col)

        else:
            top = col[previous - 1] if previous - 1 >= 0 else 1e6
            middle = col[previous]
            bottom = col[previous + 1] if previous + 1 < height else 1e6

            previous = previous + np.argmin([top, middle, bottom]) - 1

        seam.append([i, previous])

    return seam

horizontal_seam=find_horizontal_seam(energies_horizontal)

#synarthsh pou vriskei ta katakoryfa seams
def find_vertical_seam(energies_vertical): 
        im = np.transpose(energies_vertical)
        u = find_horizontal_seam(im)
        for i in range(len(u)):
                temp = list(u[i])
                temp.reverse()
                u[i] = tuple(temp)
        return u

vertical_seam=find_vertical_seam(energies_vertical)

#Synarthsh pou afairei ta seams orizontia
def remove_horizontal_seam(imgarray,horizontal_seam):
        height, width, bands = imgarray.shape
        removed=np.zeros((height-1,width,bands), np.uint8)
        for x,y in reversed(horizontal_seam):
                removed[0:y,x]=imgarray[0:y,x]
                removed[y:height-1,x]=imgarray[y+1:height,x]
        return removed

#synarthsh pou afairei ta seams katakoryfa
def remove_vertical_seam(imgarray,vertical_seam):
        height, width, bands = imgarray.shape
        removed=np.zeros((height,width-1,bands),np.uint8)
        for x,y in reversed(vertical_seam):
                removed[y,0:x]=imgarray[y,0:x]
                removed[y,x:width-1]=imgarray[y,x+1:width]
        return removed

photo_horizontal_seam=remove_horizontal_seam(imgaustinarray,horizontal_seam)
photo_vertical_seam=remove_vertical_seam(imgaustinarray,vertical_seam)
cv2.imwrite('resized_horizontally.jpg',photo_horizontal_seam)
cv2.imwrite('resized_vertical.jpg',photo_vertical_seam)
cv2.imshow('resized_horizontal',photo_horizontal_seam)
cv2.imshow('resized_vertical',photo_vertical_seam)

#Synarthsh pou provalei ta seams
def polylines(imgarray,seam):
        polylines=cv2.polylines(imgarray, np.int32([np.asarray(seam)]), False, (0,255,0))
        cv2.imshow('seam',polylines)

#Emfanish tou orizontiou curve seaming
polylines(imgaustinarray, horizontal_seam)
#Emfanish tou katakorufou curve seaming
polylines(imgaustinarray,vertical_seam)
#Emfanish tou horizontal and vertical seams gia 5 seconds
#cv2.waitKey(10000000)
#meta katastrofh tou parathurou gia na emfanistei to median filtering
#cv2.destroyAllWindows()

#Using Median filtering
#Vazw ws parameters ton pinaka ths eikonas kai ton akeraio arithmo poy thelw na ginei blurring
def median_filtering(imagearray,int):
        blur=cv2.medianBlur(imagearray,int)
        plt.subplot(121),plt.imshow(imgaustin),plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(blur),plt.title('Median Filtering 30% noise')
        plt.xticks([]), plt.yticks([])
        plt.show()
        plt.close('all')

median_filtering(imgaustinarray,3)



