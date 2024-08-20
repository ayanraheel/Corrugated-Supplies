#pixelsPerMetric
# Without a reference object, Size of objects: 3588.0100000000007
# https://pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/
# https://www.geeksforgeeks.org/measure-size-of-an-object-using-python-opencv/amp/
# https://blog.roboflow.com/dimension-measurement/amp/

import cv2 

  
# Load the image 

img = cv2.imread('/Users/colleenjung/Downloads/cardboard-box-closed-L8ekeqC-600.jpg') 

  
# Convert to grayscale 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

  
#to separate the object from the background 
ret, thresh = cv2.threshold(gray, 127, 255, 0) 

  
# Find the contours of the object  
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

  
# Draw the contours on the original image 
cv2.drawContours(img, contours, -1, (0,255,0), 3) 

  
# Get the area of the object in pixels 
area = cv2.contourArea(contours[0]) 

  
# Convert the area from pixels to a real-world unit of measurement (e.g. cm^2) 
scale_factor = 0.1 # 1 pixel = 0.1 cm 
size = area * scale_factor ** 2

  
# Print the size of the object 
print('Size:', size) 

  
# Display the image with the contours drawn 
cv2.imwrite('Object.jpeg', img) 
cv2.waitKey(0) 

  
# Save the image with the contours drawn to a file 
cv2.imwrite('object_with_contours.jpg', img)


# With a reference object, Size:


import cv2
import numpy as np

# Load the image
image_path = '/Users/colleenjung/Downloads/measuring-box-step1.jpg'
img = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold the image to get binary image
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the contour that likely corresponds to the ruler
ruler_contour = None
for contour in contours:
    # Get the bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)
    # Check if the contour has an aspect ratio close to 1 (since the ruler is 20cm x 20cm)
    if 0.9 < aspect_ratio < 1.1 and w > 50 and h > 50:
        ruler_contour = contour
        break

# Calculate the scale factor (pixels per centimeter)
if ruler_contour is not None:
    x, y, w, h = cv2.boundingRect(ruler_contour)
    ruler_length_in_pixels = max(w, h)
    ruler_length_in_cm = 20  # 20 cm as per the given reference object
    pixels_per_cm = ruler_length_in_pixels / ruler_length_in_cm
else:
    raise Exception("Ruler not found")

# Draw the ruler contour for verification
cv2.drawContours(img, [ruler_contour], -1, (0, 255, 0), 3)

# Find the contour that likely corresponds to the box
box_contour = None
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w > 100 and h > 100 and contour is not ruler_contour:
        box_contour = contour
        break

# Calculate the size of the box in cm
if box_contour is not None:
    x, y, w, h = cv2.boundingRect(box_contour)
    box_width_cm = w / pixels_per_cm
    box_height_cm = h / pixels_per_cm
else:
    raise Exception("Box not found")

# Draw the box contour for verification
cv2.drawContours(img, [box_contour], -1, (255, 0, 0), 3)

# Save the image with drawn contours
cv2.imwrite('/mnt/data/object_with_contours.jpg', img)

# Print the size of the box
print(f'Box Size: Width = {box_width_cm:.2f} cm, Height = {box_height_cm:.2f} cm')

# Display the image with the contours drawn
cv2.imshow('Image with Contours', img)
cv2
