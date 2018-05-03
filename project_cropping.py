import numpy as np
import cv2
import sys
from PIL import Image
import imutils
import os.path


if len(sys.argv) != 3:
    print ("%s input_file output_file" , sys.argv[0])
    sys.exit()
else:
    input_file = sys.argv[1]
    output_file = sys.argv[2]

if not os.path.isfile(input_file):
    print ("No such file '%s'" , input_file)
    sys.exit()

#input_file="image1.jpg"
img = cv2.imread(input_file)
img = imutils.resize(img, height=500)

## Change to LAB space
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l,a,b = cv2.split(lab)
imglab = np.hstack((l,a,b))
#cv2.imwrite("region_lab.png", imglab)

## normalized the a channel to all dynamic range
na = cv2.normalize(a, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
#cv2.imwrite("region_a_normalized.png", na)

## Threshold to binary
retval, threshed = cv2.threshold(na, thresh = 180,  maxval=255, type=cv2.THRESH_BINARY)

## Do morphology
kernel = cv2.getStructuringElement( cv2.MORPH_ELLIPSE , (3,3))
opened = cv2.morphologyEx(threshed, cv2.MORPH_OPEN,kernel)
res = np.hstack((threshed, opened))
#cv2.imwrite("region_a_binary.png", res)

## Find contours
_, contours, hierarchy = cv2.findContours(opened, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

## Draw Contours
res = img.copy()
cv2.drawContours(res, contours, -1, (255,0,0), 10)
#cv2.imwrite("region_contours.png", res)

## Filter Contours
for idx, contour in enumerate(contours):
    bbox = cv2.boundingRect(contour)
    area = bbox[-1]*bbox[-2]
    if area < 2150:
        continue
    (x, y, w, h) = cv2.boundingRect(contour)
    cv2.rectangle(res, (0, y), (x + w, y + h*2), (0, 255, 0), 4)
    crop_img = img[y:y+h*2, 0:x+w]

#cv2.imshow("cropped", crop_img)

cv2.imwrite(output_file, crop_img)

#cv2.waitKey(0)   
#cv2.destroyAllWindows()
