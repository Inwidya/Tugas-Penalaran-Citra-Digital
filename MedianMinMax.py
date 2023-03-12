import cv2
import numpy as np

# Load the image
img = cv2.imread('Original.jpg')

# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Buat kernel filter dengan ukuran jendela
kernel = np.ones((3, 3), np.uint8)

# # Apply the median filter
filtered_img = cv2.medianBlur(gray_img, 5) # 5 is the size of the filter kernel

# Aplikasikan filter minimum
min_filtered = cv2.erode(gray_img, kernel)
# Aplikasikan filter maximum
img_max = cv2.dilate(img, kernel, iterations = 1)

#menampilkan data citra
cv2.imshow('Median_image.jpg', filtered_img)
cv2.imshow('Min_Filtered', min_filtered)
cv2.imshow('Max filter',img_max)
cv2.waitKey(0)
cv2.destroyAllWindows()
