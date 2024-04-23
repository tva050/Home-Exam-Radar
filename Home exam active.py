import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase




# Paths to data and header file
data_file = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\FYS-3001\Home exam active\image.dat"
header_file = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\FYS-3001\Home exam active\image.txt"

# Number of pixels from meta data 
Ny = 1759
Nx = 501

# constants
f_sf = 19207680.0 # Sampling frequency in Hz (info.xsamplefreq)
f_prf = 1686.0674 # Pulse repetition frequency in Hz (info.ysamplefreq)
V = 6716.7298 # Velocity in m/s (geo.groundvel)
theta = np.deg2rad(22.779148) # Convert to radians (geo.incangle)
c = 3e8 # Speed of light in m/s


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

#mean_int = np.mean(np.abs(img))
#img_normalized = img / mean_int

slc_magnitude = np.sqrt(img_real**2 + img_imag**2)
img_normalized = slc_magnitude / np.mean(slc_magnitude)

img_fft = np.fft.fft2(img_normalized)
img_fft_shifted = np.fft.fftshift(img_fft)
#mag_spec = np.log(np.abs(img_fft_shifted))

#Ny = np.arange(1, 1759+1, 1)
#Nx = np.arange(1, 501+1, 1)

Ny, Nx = img_normalized.shape

dx = c / (2*f_sf*np.sin(theta)) # Resolution or pixel size in range (x)
dy = V / f_prf # Resolution or pixel size in azimuth (y)

delta_kx = (2*np.pi) / (Nx*dx) # Resolution in kx
delta_ky = (2*np.pi) / (Ny*dy) # Resolution in ky

d_kx_max = np.pi / dx
d_ky_max = np.pi / dy


d_kx_min = -d_kx_max
d_ky_min = -d_ky_max

kx = np.linspace(d_kx_min, d_kx_max, Nx)
ky = np.linspace(d_ky_min, d_ky_max, Ny)
Kx, Ky = np.meshgrid(kx, ky)

def task_1B():
    # Print the 10 values around the center of the array
    
    plt.style.use("ggplot")
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(Kx, Ky, np.log(np.abs(img_fft_shifted)), cmap="gray")
    plt.colorbar()
    plt.xlim(-0.08, 0.08)
    plt.ylim(-0.08, 0.08)
    plt.xlabel(r'Range Wavenumber $[rad/m]$')
    plt.ylabel(r'Azimuth Wavenumber $[rad/m]$')
    plt.title('Magnitude Spectrum of the Image')
    plt.show()

def task_2B():
    plt.style.use("ggplot")
    
    spec_profile_azimuth = np.mean(np.abs(img_fft_shifted), axis=0)
    print(spec_profile_azimuth.max(), spec_profile_azimuth.min())

    min_azimuth = -d_ky_max
    max_azimuth = d_ky_max
    Ny = img_fft_shifted.shape[1]
    azimuth_freqs = np.linspace(min_azimuth, max_azimuth, Ny)
    
    spec_profile_azimuth_normalized = spec_profile_azimuth / spec_profile_azimuth.max()
    
    plt.figure(figsize=(8, 6))
    plt.plot(azimuth_freqs, spec_profile_azimuth_normalized)
    plt.xlabel(r'Azimuth Wavenumber $[rad/m]$')
    plt.ylabel(r'Azimuth Spectral Profile $[Norm]$')
    plt.title('Azimuth Spectrum Profile')
    plt.show()
    
    is_symmetric = np.allclose(spec_profile_azimuth, spec_profile_azimuth[::-1], atol=1e-10)
    if is_symmetric:
        print("The spectral profile is symmetric around the zero frequency.")
    else:
        print("The spectral profile is shifted.")

def task_3B():
    Ny = img_fft_shifted.shape[1]
    
    num_parts = 3
    part_size = Ny // num_parts

    intensity_images = []
    
    for i in range(num_parts):
        start_idx = i*part_size
        end_idx = (i +1) * part_size
        complex_look = img_fft_shifted[:, start_idx:end_idx]
        
        complex_image  = np.fft.ifftshift(complex_look, axes=1)
        spatial_image = np.fft.ifft2(complex_image)
        
        intensity_image = np.abs(spatial_image)**2
        intensity_images.append(intensity_image)
        
    plt.figure(figsize=(8, 6))
    for i, intensity_image in enumerate(intensity_images):
        plt.subplot(1, num_parts, i+1)
        plt.pcolormesh(intensity_image, cmap='gray')
        plt.title(f'Look {i+1}')
        plt.axis('off')
    
    # Create a new axes at the bottom of current figure, with 10% height and 100% width relative to the figure
    colorbar_axes = plt.gcf().add_axes([0.138, 0.06, 0.75, 0.04])

    # Create a Normalize instance to normalize data to [0-1] range
    norm = mcolors.Normalize(vmin=np.min(intensity_images), vmax=np.max(intensity_images))
    # Create a ColorbarBase instance with the 'gray' colormap
    colorbar = ColorbarBase(colorbar_axes, cmap='gray', norm=norm, orientation='horizontal')
    plt.show()
    
#def task_4B():    
    
if __name__ == "__main__":
    #task_0()
    
# ______A. Image Statistics______ #
    #task_1A()
    #task_2A()
    #task_3A()
    
# ______B. Look extraction and Fourier Spectral Estimation______ #
    #task_1B()
    #task_2B()
    task_3B()
    #task_4B()