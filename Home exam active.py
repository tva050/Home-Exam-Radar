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

img_int = np.mean(img)
img_normalized = img / img_int


img_fft = np.fft.fft2(img_normalized)
img_fft_shifted = np.fft.fftshift(img_fft)
img_spec = np.abs(img_fft_shifted)


dx = c / (2*f_sf*np.sin(theta)) # Resolution or pixel size in range (x)
dy = V / f_prf # Resolution or pixel size in azimuth (y)

delta_kx = (2*np.pi) / (Nx*dx) # Resolution in kx
delta_ky = (2*np.pi) / (Ny*dy) # Resolution in ky

d_kx_max = np.pi / dx
d_ky_max = np.pi / dy

kx = np.linspace(-d_kx_max, d_kx_max, Nx)
ky = np.linspace(-d_ky_max, d_ky_max, Ny) 

def task_1B():
    """ 
    Plots the magnitude spectrum of the image in the Fourier/Freq domain.
    ~ Complex valued 2D spectra
    """
    #plt.style.use("ggplot")
    plt.figure(figsize=(8, 6))
    plt.pcolor(kx, ky, img_spec, cmap='gray')
    plt.colorbar()
    plt.xlabel(r'Range Wavenumber $[rad/m]$')
    plt.ylabel(r'Azimuth Wavenumber $[rad/m]$')
    plt.title('Spectrum of the Image')
    plt.show()

spec_profile_azimuth = np.mean(np.abs(img_fft_shifted), axis=0)
def task_2B():
    """ 
    Plots the azimuth profile of the spectral profile.
    ~ Azimuth Spectral Profile
    """
    plt.style.use("ggplot")
    
    freqs = np.fft.fftfreq(Ny, d=1/f_prf)
    freqs = np.fft.fftshift(freqs)
    
    spec_profile_azimuth_normalized = spec_profile_azimuth / spec_profile_azimuth.max()
    
    plt.figure(figsize=(8, 6))
    plt.plot(freqs, spec_profile_azimuth_normalized)
    plt.xlabel(r'Azimuth Frequency $[Hz]$')
    plt.ylabel(r'Intensity $[Norm]$')
    plt.title('Azimuth Fourier Domain')
    plt.show()
    
    is_symmetric = np.allclose(spec_profile_azimuth, spec_profile_azimuth[::-1], atol=1e-10)
    if is_symmetric:
        print("The spectral profile is symmetric around the zero frequency.")
    else:
        print("----|The spectral profile is shifted|----")
        # find the amount of shift
        print("Shifted by: ", np.argmax(spec_profile_azimuth) - Ny // 2)


azimuth_shift = len(spec_profile_azimuth) // 2
shifted_azimuth_pixels = np.argmax(spec_profile_azimuth) - Ny // 2
shifted_img_fft = np.roll(img_fft_shifted, shift=-azimuth_shift, axis=0)
def task_3B():
    print("Azimuth shift in pixels:", shifted_img_fft)
    
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(kx, ky, np.abs(shifted_img_fft), cmap='gray')
    plt.colorbar()
    plt.xlabel(r'Range Wavenumber $[rad/m]$')
    plt.ylabel(r'Azimuth Wavenumber $[rad/m]$')
    plt.title('Spectrum of the Image with Azimuth Shift')
    plt.show()
    
Ny = shifted_img_fft.shape[1]
print(shifted_img_fft.shape)
num_parts = 3
part_size = Ny // num_parts
intensity_images = []

for i in range(num_parts):
    start_idx = i*part_size
    end_idx = (i+1) * part_size
    complex_look = shifted_img_fft[:, start_idx:end_idx]
    
    complex_image  = np.fft.ifftshift(complex_look, axes=0)
    spatial_image = np.fft.ifft2(complex_image)
    
    intensity_image = np.abs(spatial_image)
    intensity_images.append(intensity_image)    
def task_4B():
        
    plt.figure(figsize=(8, 6))
    for i, intensity_image in enumerate(intensity_images):
        plt.subplot(1, num_parts, i+1)
        plt.pcolormesh(intensity_image, cmap='gray')
        plt.title(f'Look {i+1}')
        #plt.colorbar()
        plt.axis('off')
    
    colorbar_axes = plt.gcf().add_axes([0.138, 0.06, 0.75, 0.04])
    norm = mcolors.Normalize(vmin=np.min(intensity_images), vmax=np.max(intensity_images))
    colorbar = ColorbarBase(colorbar_axes, cmap='gray', orientation="horizontal", norm=norm) 
    
    plt.show()
    
def task_5B():
    normalized_intensity_images = [(img - np.mean(img)) / np.mean(img) for img in intensity_images]
    
    fft_intensity_images = [np.fft.fft2(img) for img in normalized_intensity_images]
    
    co_spectra = []
    cross_spectra = []
    
    for i in range(len(fft_intensity_images)):
        for j in range(i, len(fft_intensity_images)):
            product_spectrum = np.conj(fft_intensity_images[i]) * fft_intensity_images[j]
            if i == j:
                co_spectra.append(product_spectrum)
            else:
                cross_spectra.append(product_spectrum)
    
    print("Number of Co-spectra: ",len(co_spectra))
    print("Number of Cross-spectra: ",len(cross_spectra))
    co_spectrum_avg = np.mean(co_spectra, axis=0)
    #co_spectrum_avg = np.roll(co_spectrum_avg, shift=azimuth_shift_pixels, axis=0)
    co_spectrum_avg = np.fft.fftshift(co_spectrum_avg, axes=(0,1))
    
    cross_spectrum_avg = np.mean(cross_spectra, axis=0)
    #cross_spectrum_avg = np.roll(cross_spectrum_avg, shift=shifted_azimuth_pixels, axis=0)
    #cross_spectrum_avg = np.fft.fftshift(cross_spectrum_avg, axes=(0,1))
    
    complex_spectra = [co_spectrum_avg] + [cross_spectrum_avg]*2
    #complex_spectra = np.roll(complex_spectra, shift=azimuth_shift_pixels, axis=1)
    
    """ plt.pcolormesh(np.log(np.abs(complex_spectra[0])), cmap='gray', vmin=6, vmax=15, shading="gouraud")
    plt.colorbar()
    plt.xlabel(r'Range Wavenumber $[rad/m]$')
    plt.ylabel(r'Azimuth Wavenumber $[rad/m]$')
    plt.title('Co-spectrum')
    plt.show() """
    
    plt.figure(figsize=(8, 6))
    for i, complex_spectrum in enumerate(complex_spectra):
        plt.subplot(1, 3, i+1)
        plt.pcolormesh(np.log(np.abs(complex_spectrum)), cmap='gray', vmin=6, vmax=15, shading="gouraud")
        plt.title(f'Spectrum {i+1}')
        plt.colorbar()
        plt.xlabel(r'Range Wavenumber $[rad/m]$')
        plt.ylabel(r'Azimuth Wavenumber $[rad/m]$')
        plt.axis('off')
    
    # Create a new axes at the bottom of current figure, with 10% height and 100% width relative to the figure
    #colorbar_axes = plt.gcf().add_axes([0.138, 0.06, 0.75, 0.04])
    
    # Create a Normalize instance to normalize data to [0-1] range
    #norm = mcolors.Normalize(vmin=np.min(complex_spectra), vmax=np.max(complex_spectra))
    
    # Create a ColorbarBase instance with the 'gray' colormap
    #colorbar = ColorbarBase(colorbar_axes, cmap='gray', norm=norm, orientation='horizontal')
    plt.show()
    
""" ---------------------------------- C. Analysis of 2D Spectra ----------------------------------- """

def task_1C():
    Ny, Nx = shifted_img_fft.shape
    
    shifted_spectrum = np.fft.fftshift(shifted_img_fft)
    
    
    # plot only on surface plot 
    """ fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(kx, ky)
    ax.plot_surface(X, Y, np.imag(shifted_spectrum), cmap='viridis')
    ax.set_title('Magnitude Spectrum')
    ax.set_xlabel('Range Wavenumber $[rad/m]$')
    ax.set_ylabel('Azimuth Wavenumber $[rad/m]$')
    ax.set_zlabel('Magnitude')
    plt.show() """
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    X, Y = np.meshgrid(kx, ky)
    ax1.plot_surface(X, Y, np.real(shifted_spectrum), cmap='viridis')
    ax1.set_title('Magnitude Spectrum')
    ax1.set_xlabel('Range Wavenumber $[rad/m]$')
    ax1.set_ylabel('Azimuth Wavenumber $[rad/m]$')
    ax1.set_zlabel('Magnitude')
    
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.plot_surface(X, Y, np.imag(shifted_spectrum), cmap='viridis')
    ax2.set_title('Phase Spectrum')
    ax2.set_xlabel('Range Wavenumber $[rad/m]$')
    ax2.set_ylabel('Azimuth Wavenumber $[rad/m]$')
    ax2.set_zlabel('Phase')
    
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.plot_surface(X, Y, np.angle(shifted_spectrum), cmap='viridis')
    ax3.set_title('Phase Spectrum')
    ax3.set_xlabel('Range Wavenumber $[rad/m]$')
    ax3.set_ylabel('Azimuth Wavenumber $[rad/m]$')
    ax3.set_zlabel('Phase')
    
    plt.show()
    

if __name__ == "__main__":
    #task_0()
    
# ______A. Image Statistics______ #
    #task_1A()
    #task_2A()
    #task_3A()
    
# ______B. Look extraction and Fourier Spectral Estimation______ #
    #task_1B()
    #task_2B()
    #task_3B()
    #task_4B()
    task_5B()
    
# ______C. Analysis of 2D Spectra______ #
    #task_1C()