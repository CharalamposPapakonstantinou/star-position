import cv2
import numpy as np
import random
from sklearn.metrics import mean_absolute_error as mae


def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a
# path

N=400 #image size, let image be rectangle (NxN)
NumSTAR=8 # Select number of Stars
radius = 5

starimage = np.zeros((N,N), np.uint8)

window_name = 'Star Image'

cc=np.zeros((NumSTAR,2))
for i in range(0,NumSTAR):
    cc[i,:] = (random.randint(radius, N-radius), random.randint(radius, N-radius)) # Prevent from positioning the stars to the edges of the image


color = (255, 255, 255)
thickness = -1

for i in range(0,NumSTAR):
    starimage = cv2.circle(starimage, (int(cc[i,0]),int(cc[i,1])) , radius, color, thickness)

# Displaying the image
cv2.imshow(window_name, starimage)
cv2.waitKey(1)
cv2.destroyAllWindows()


circles = cv2.HoughCircles(starimage,cv2.HOUGH_GRADIENT,1,5,param1=3,param2=5,minRadius=3,maxRadius=5)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the center of the circle
    cv2.circle(starimage,(i[0],i[1]),2,(0,0,255),1)

cv2.imshow('detected circles',starimage)
cv2.waitKey(1)
cv2.destroyAllWindows()

print('ground-truth star position\n',np.sort(cc,axis=0))
print('===================')
print('predicted star position\n',np.sort(circles[0,:,0:2],axis=0))

print('Mean Absolute Error is : ', mae(np.sort(cc,axis=0), np.sort(circles[0,:,0:2],axis=0)))
print('==== better parameter tuning -> improving performance ====')