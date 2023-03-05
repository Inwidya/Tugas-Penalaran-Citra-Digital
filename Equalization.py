import cv2
import numpy as np
from matplotlib import pyplot as plt

# membaca gambar
img = cv2.imread('gambar.webp', 0)

# menghitung histogram gambar asli
hist, bins = np.histogram(img.flatten(), 256, [0, 256])

# melakukan ekualisasi histogram
equ_img = cv2.equalizeHist(img)

# menghitung histogram gambar yang telah di-ekualisasi
equ_hist, equ_bins = np.histogram(equ_img.flatten(), 256, [0, 256])

# menampilkan gambar asli dan gambar yang telah di-ekualisasi
cv2.imshow('Original Image', img)
cv2.imshow('Equalized Image', equ_img)

# menampilkan grafik histogram sebelum dan sesudah ekualisasi
plt.figure(figsize=(8, 6))
plt.hist(img.flatten(), 256, [0, 256], color='r')
plt.hist(equ_img.flatten(), 256, [0, 256], color='b')
plt.xlim([0, 256])
plt.legend(('Original Image', 'Equalized Image'), loc='upper left')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()