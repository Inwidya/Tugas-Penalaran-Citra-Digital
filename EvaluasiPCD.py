import cv2

# load citra
img = cv2.imread('Original.jpg')

# konversi ke grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# terapkan filter median dengan kernel 3x3
filtered_img = cv2.medianBlur(gray_img, 5)

# terapkan filter Unsharp Masking
blur_img = cv2.GaussianBlur(filtered_img, (0, 0), 3)
sharp_img = cv2.addWeighted(filtered_img, 1.5, blur_img, -0.5, 0)

# tampilkan citra asli dan hasil filter median
cv2.imshow('Original Image', gray_img)
cv2.imshow('Filtered Image', filtered_img)
# tampilkan hasil filter Unsharp Masking
cv2.imshow('Sharpened Image', sharp_img)

# tunggu hingga tombol keyboard ditekan
cv2.waitKey(0)

# tutup jendela tampilan citra
cv2.destroyAllWindows()
