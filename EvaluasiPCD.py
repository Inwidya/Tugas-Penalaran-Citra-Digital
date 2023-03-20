import cv2

# load citra

img = cv2.imread('Original.jpg')

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

filtered_img = cv2.medianBlur(gray_img, 5)

blur_img = cv2.GaussianBlur(filtered_img, (0, 0), 3)
sharp_img = cv2.addWeighted(filtered_img, 1.5, blur_img, -0.5, 0)

cv2.imshow('Original Image', gray_img)
cv2.imshow('Filtered Image', filtered_img)

cv2.imshow('Sharpened Image', sharp_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
