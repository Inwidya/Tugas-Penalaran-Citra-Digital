import cv2
import numpy as np

# Load image
img = cv2.imread('Original2.jpg', cv2.IMREAD_GRAYSCALE)

# Perform FFT
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

# Butterworth Highpass Filter
n = 2 # order of filter
W = 30 # cutoff frequency
rows, cols = img.shape
crow, ccol = rows//2, cols//2
butterworth_highpass, gaussian_highpass = np.zeros((rows, cols), np.float32), np.zeros((rows, cols), np.float32)
for i in range(rows):
    for j in range(cols):
        dist = np.sqrt((i-crow)**2 + (j-ccol)**2)
        butterworth_highpass[i,j] = 1 / (1 + (W/dist)**(2*n)) 
butterworth_highpass_filter = fshift * butterworth_highpass
butterworth_highpass_spectrum = 20*np.log(np.abs(butterworth_highpass_filter))
gaussian_highpass_filter = fshift * gaussian_highpass
gaussian_highpass_spectrum = 20*np.log(np.abs(gaussian_highpass_filter))

# Butterworth Lowpass Filter
D = 30 # cutoff frequency
butterworth_lowpass, ideal_lowpass = np.zeros((rows, cols), np.float32), np.zeros((rows, cols), np.float32)
for i in range(rows):
    for j in range(cols):
        dist = np.sqrt((i-crow)**2 + (j-ccol)**2)
        butterworth_lowpass[i,j] = 1 / (1 + (dist/D)**(2*n)) 
butterworth_lowpass_filter = fshift * butterworth_lowpass
butterworth_lowpass_spectrum = 20*np.log(np.abs(butterworth_lowpass_filter))
gaussian_highpass_filter = fshift * gaussian_highpass
gaussian_highpass_spectrum = 20*np.log(np.abs(gaussian_highpass_filter))
for i in range(rows):
    for j in range(cols):
        dist = np.sqrt((i-crow)**2 + (j-ccol)**2)
        if dist < D:
            ideal_lowpass[i,j] = 1
ideal_lowpass_filter = fshift * ideal_lowpass
ideal_lowpass_spectrum = 20*np.log(np.abs(ideal_lowpass_filter))

# Ideal Highpass Filter
D = 30 # cutoff frequency
rows, cols = img.shape
crow, ccol = rows//2, cols//2
ideal_highpass, gaussian_lowpass = np.ones((rows, cols), np.float32), np.ones((rows, cols), np.float32)
for i in range(rows):
    for j in range(cols):
        dist = np.sqrt((i-crow)**2 + (j-ccol)**2)
        if dist < D:
            ideal_highpass[i,j] = 0
ideal_highpass_filter = fshift * ideal_highpass
ideal_highpass_spectrum = 20*np.log(np.abs(ideal_highpass_filter))
for i in range(rows):
    for j in range(cols):
        dist = np.sqrt((i-crow)**2 + (j-ccol)**2)
        gaussian_lowpass[i,j] = np.exp(-(dist**2) / (2*(D**2)))
gaussian_lowpass_filter = fshift * gaussian_lowpass
gaussian_lowpass_spectrum = 20*np.log(np.abs(gaussian_lowpass_filter))


# Perform IFFT
butterworth_highpass_ifft = np.fft.ifftshift(butterworth_highpass_filter)
butterworth_highpass_ifft = np.fft.ifft2(butterworth_highpass_ifft)
butterworth_highpass_ifft = np.abs(butterworth_highpass_ifft)

butterworth_lowpass_ifft = np.fft.ifftshift(butterworth_lowpass_filter)
butterworth_lowpass_ifft = np.fft.ifft2(butterworth_lowpass_ifft)
butterworth_lowpass_ifft = np.abs(butterworth_lowpass_ifft)

# Perform IFFT
ideal_lowpass_ifft = np.fft.ifftshift(ideal_lowpass_filter)
ideal_lowpass_ifft = np.fft.ifft2(ideal_lowpass_ifft)
ideal_lowpass_ifft = np.abs(ideal_lowpass_ifft)

ideal_highpass_ifft = np.fft.ifftshift(ideal_highpass_filter)
ideal_highpass_ifft = np.fft.ifft2(ideal_highpass_ifft)
ideal_highpass_ifft = np.abs(ideal_highpass_ifft)

# Perform IFFT
gaussian_lowpass_ifft = np.fft.ifftshift(gaussian_lowpass_filter)
gaussian_lowpass_ifft = np.fft.ifft2(gaussian_lowpass_ifft)
gaussian_lowpass_ifft = np.abs(gaussian_lowpass_ifft)

gaussian_highpass_ifft = np.fft.ifftshift(gaussian_highpass_filter)
gaussian_highpass_ifft = np.fft.ifft2(gaussian_highpass_ifft)
gaussian_highpass_ifft = np.abs(gaussian_highpass_ifft)


# Display images
cv2.imshow('Original Image', img)
cv2.imshow('Butterworth Highpass Filtered Image', butterworth_highpass_ifft.astype(np.uint8))
cv2.imshow('Butterworth Lowpass Filtered Image', butterworth_lowpass_ifft.astype(np.uint8))

cv2.imshow('Ideal Lowpass Filtered Image', ideal_lowpass_ifft.astype(np.uint8))
cv2.imshow('Ideal Highpass Filtered Image', ideal_highpass_ifft.astype(np.uint8))

cv2.imshow('Gaussian Lowpass Filtered Image', gaussian_lowpass_ifft.astype(np.uint8))
cv2.imshow('Gaussian Highpass Filtered Image', gaussian_highpass_ifft.astype(np.uint8))

cv2.waitKey(0)
cv2.destroyAllWindows()

