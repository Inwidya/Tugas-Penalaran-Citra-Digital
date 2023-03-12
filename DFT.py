import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Generate test signal
t = np.linspace(0, 1, 1000, endpoint=False)
sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t)

# Compute DFT
dft = np.fft.fft(sig)

# Compute frequency domain
freq = np.fft.fftfreq(len(sig), t[1]-t[0])

# Generate filter kernels
fs = 1000
cutoff = 30
order = 4
Wn = 2*cutoff/fs

# Gaussian filter
b = signal.gaussian(len(sig), std=100)
gaussian_filt = np.fft.fft(b)
gaussian_filt = gaussian_filt/np.max(gaussian_filt)

# Butterworth filter
butter_filt = signal.butter(order, Wn, btype='low', analog=False)
butter_filt = signal.freqz(butter_filt[0], butter_filt[1], worN=len(sig))
butter_filt = np.abs(butter_filt[1])
butter_filt = butter_filt/np.max(butter_filt)

# Ideal Lowpass filter
ideal_filt = np.abs(freq) <= cutoff
ideal_filt = ideal_filt.astype(np.float)

# Ideal Highpass filter
ideal_hp_filt = np.abs(freq) >= cutoff
ideal_hp_filt = ideal_hp_filt.astype(np.float)

# Apply filters to DFT
dft_gaussian_filt = dft*gaussian_filt
dft_butter_filt = dft*butter_filt
dft_ideal_filt = dft*ideal_filt
dft_ideal_hp_filt = dft*ideal_hp_filt

# Compute inverse DFT to obtain filtered signals
gaussian_filt_sig = np.real(np.fft.ifft(dft_gaussian_filt))
butter_filt_sig = np.real(np.fft.ifft(dft_butter_filt))
ideal_filt_sig = np.real(np.fft.ifft(dft_ideal_filt))
ideal_hp_filt_sig = np.real(np.fft.ifft(dft_ideal_hp_filt))

# Plot results
fig, axs = plt.subplots(2, 2, figsize=(10, 6))

axs[0, 0].plot(t, sig)
axs[0, 0].set_title('Original Signal')

axs[0, 1].plot(freq, np.abs(dft))
axs[0, 1].set_xlim(-50, 50)
axs[0, 1].set_title('Frequency Domain')

axs[1, 0].plot(t, gaussian_filt_sig)
axs[1, 0].set_title('Gaussian Filtered Signal')

axs[1, 1].plot(t, butter_filt_sig)
axs[1, 1].set_title('Butterworth Filtered Signal')

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

axs[0].plot(t, ideal_filt_sig)
axs[0].set_title('Ideal Lowpass Filtered Signal')

axs[1].plot(t, ideal_hp_filt_sig)
axs[1].set_title('Ideal Highpass Filtered Signal')

plt.show()
