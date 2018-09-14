#%% Importing dependencies
import numpy as np
from PIL import Image
import glob
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
#%% Defining functions


#%% Camera callibration
# Load images and concvert to grayscale

ny = 6
nx = 9 
imgpoints = []                                        # Images
objpoints = []
images = []
objp = np.zeros((ny*nx,3), np.float32)
for filename in glob.glob('./camera_cal/*.jpg'):
    #print(filename)
    # Loading the image
    im = cv2.imread(filename)
    # Convert to grayscale
    im_gry = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #plt.imshow(im_gry, cmap="gray")
    # Appending to the list of images
    #image.append(im_gry)
    images.append(im_gry)
    ret, corners = cv2.findChessboardCorners(im_gry, (nx,ny), None)
    
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
#images = np.array(images) 
       
shape = (im.shape[1],im.shape[0])
ret, cameraMatrix, distortionCoeffs, rvecs, tvecs = \
cv2.calibrateCamera(objpoints, imgpoints, shape,None,None)

im_undist = []
for i in images:
    #plt.subplot(2,1,1)
    #plt.imshow(i, cmap = "gray")
    undist = cv2.undistort(i, cameraMatrix, distortionCoeffs, None, cameraMatrix)
    im_undist.append(undist)
    #plt.subplot(2,1,2)
    #plt.imshow(undist, cmap = "gray")
    
plt.subplot(2,1,1)
plt.imshow(images[18],cmap = "gray")
plt.subplot(2,1,2)
plt.imshow(im_undist[18],cmap = "gray")
plt.show()
 

print('======================================================================')
print('Done')
print('======================================================================')

