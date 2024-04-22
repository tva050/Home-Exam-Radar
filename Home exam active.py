import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.signal import convolve2d
from scipy.fft import fft2, fftshift, ifft2, ifftshift


# Paths to data and header file
data_file = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\FYS-3001\Home exam active\image.dat"
header_file = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\FYS-3001\Home exam active\image.txt"

# Number of pixels from meta data 
Ny = 1759
Nx = 501

# constants
f_sf = 19207680.0
f_prf = 1686.0674
V = 6716.7298
theta = np.deg2rad(22.779148)
c = 3e8


# Read the image data as vector and reshape to image 
dat = np.fromfile(data_file, dtype=np.csingle)
img = dat.reshape(Ny, Nx)


""" ----------------------------------- Read Plot Data ----------------------------------- """ 
def task_0():
    # plot SLC image
    plt.pcolormesh(np.abs(img), cmap='gray')
    plt.colorbar()
    plt.xlabel("Range (pixels)")
    plt.ylabel("Azimuth (pixels)")
    plt.title("SLC image")
    plt.show()
    

""" ---------------------------------- A. Image Statistics ----------------------------------- """


# Real and imaginary part of the image
img_real = np.real(img)
img_imag = np.imag(img)

img_real_flat = img_real.flatten()
img_imag_flat = img_imag.flatten()


real_hist, real_bins = np.histogram(img_real_flat, bins=6000)
imag_hist, imag_bins = np.histogram(img_imag_flat, bins=6000)


def task_1A():
    # Plot histogram of real and imaginary part of the image
    plt.style.use("ggplot")
    
    figure, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True) # maybe us 65535 as bins for 16 bit images
    axes[0].hist(img_real_flat, bins=6000)
    axes[0].set_title("Real part")
    axes[0].set_xlabel("Intensity")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(True)
    
    axes[1].hist(img_imag_flat, bins=6000, color="teal")
    axes[1].set_title("Imaginary part")
    axes[1].set_xlabel("Intensity")
    #axes[1].set_ylabel("Frequency")
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("max real: ", np.max(img_real_flat))
    print("min real: ", np.min(img_real_flat))
    print("max imag: ", np.max(img_imag_flat))
    print("min imag: ", np.min(img_imag_flat))
    print("max y real value", np.max(real_hist))
    print("max y imag value", np.max(imag_hist))

slc_mag = np.abs(img_real + 1j*img_imag)
def task_2A():
    plt.style.use("ggplot")
    intensity_hist, intensity_bins = np.histogram(slc_mag.flatten(), bins=6000)
    
    plt.hist(slc_mag.flatten(), bins=6000)
    plt.title("Intensity histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    #plt.grid(True)
    plt.show()
    
    normalized_variance = np.var(slc_mag) / np.mean(slc_mag)
    print("Normalized variance: ", normalized_variance)
    
    
def task_3A():
    plt.style.use("ggplot")
    kernel = np.ones((5, 5)) / 25  # Normalized kernel for averaging

    # Perform 2D convolution to apply the smoothing operation
    smoothed_intensity = convolve2d(slc_mag, kernel, mode='same')

    # Compute histogram of the smoothed intensity image
    smoothed_intensity_hist, smoothed_intensity_bins = np.histogram(smoothed_intensity.flatten(), bins=100)

    # Plot histogram of the smoothed intensity image
    plt.figure(figsize=(8, 6))
    plt.hist(smoothed_intensity.flatten(), bins=6000)
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram of Smoothed Intensity Image')
    plt.show()

    # Compute normalized variance of the smoothed intensity image
    normalized_variance_smoothed = np.var(smoothed_intensity) / np.mean(smoothed_intensity)

    print("Normalized variance of the smoothed intensity image:", normalized_variance_smoothed)
    
    
""" ---------------------------------- B. Look extraction and Fourier Spectral Estimation ----------------------------------- """

slc_magnitude = np.sqrt(img_real**2 + img_imag**2)

img_normalized = slc_magnitude / np.mean(slc_magnitude)

img_fft = np.fft.fft2(img_normalized)
img_fft_shifted = np.fft.fftshift(img_fft)
mag_spec = np.log(np.abs(img_fft_shifted))

""" delta_x = c / (2*f_sf*np.sin(theta)) # Resolution or pixel size in range (x)
delta_y = V / f_prf # Resolution or pixel size in azimuth (y)

# N_X and N_Y are the sizes of our Fourier transformed image in range and azimuth respectively
N_X = np.fft.fftfreq(Nx)
print(N_X)
N_Y = mag_spec.shape[0]
print(N_Y)

delta_kx = (2*np.pi) / (N_X*delta_x) # Resolution in kx
delta_ky = (2*np.pi) / (N_Y*delta_y) # Resolution in ky

x_range = np.arange(N_X) * delta_x
y_range = np.arange(N_Y) * delta_y
 """
NX = np.fft.fftfreq(Nx)
NY = np.fft.fftfreq(Ny)

x_range = 2*np.pi*NX
y_range = 2*np.pi*NY

def task_1B():
    # Print the 10 values around the center of the array
    
    plt.style.use("ggplot")
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(x_range, y_range, mag_spec, cmap="gray")
    plt.colorbar()
    #plt.xlim(-0.0010, 0.0010)
    #plt.ylim(-0.00025, 0.00025)
    #plt.xscale("linear")
    plt.xlabel('Frequency (radians per minute)')
    plt.ylabel('Frequency (radians per minute)')
    plt.title('Magnitude Spectrum of the Image')
    plt.show()

#def task_2B():
    



if __name__ == "__main__":
    #task_0()
    
# ______A. Image Statistics______ #
    #task_1A()
    #task_2A()
    #task_3A()
    
# ______B. Look extraction and Fourier Spectral Estimation______ #
    task_1B()