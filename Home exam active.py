import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
from scipy.stats import rayleigh, norm, gamma, expon, laplace


# Paths to data and header file
data_file = r"C:\Home exam active\image.dat"
header_file = r"C:\Home exam active\image.txt"

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
    """ ploting SLC image """
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

real_hist, real_bins = np.histogram(img_real, bins=100)
imag_hist, imag_bins = np.histogram(img_imag, bins=100)


def task_1A():
    """ Histogram of real and imaginary part of the image """
    plt.style.use("ggplot")
    
    figure, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=False) 
    axes[0].hist(img_real_flat, bins=100)
    axes[0].set_title("Real part")
    axes[0].set_xlabel("Intensity")
    axes[0].set_ylabel("Frequency")
    
    axes[1].hist(img_imag_flat, bins=100, color="teal")
    axes[1].set_title("Imaginary part")
    axes[1].set_xlabel("Intensity")
    axes[1].set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()
    
    # Estimate parameters for Gaussian distribution real part
    mu_r, sigma_r = norm.fit(img_real_flat)
    x_r = np.linspace(min(real_bins), max(real_bins), 100)
    gaussian_pdf_r = norm.pdf(x_r, mu_r, sigma_r)
    gaussian_pdf_normalized_r = gaussian_pdf_r * len(img_real_flat) * np.diff(real_bins)[0]
    
    # Estimate parameters for Laplace distribution real part
    loc_laplace_r, scale_laplace_r = laplace.fit(img_real_flat)
    laplace_pdf_r = laplace.pdf(x_r, loc_laplace_r, scale_laplace_r)
    laplace_pdf_normalized_r = laplace_pdf_r * len(img_real_flat) * np.diff(real_bins)[0]
    
    # Estimate parameters for Gaussian distribution imaginary part
    mu_i, sigma_i = norm.fit(img_imag_flat)
    x_i = np.linspace(min(imag_bins), max(imag_bins), 100)
    gaussian_pdf_i = norm.pdf(x_i, mu_i, sigma_i)
    gaussian_pdf_normalized_i = gaussian_pdf_i * len(img_imag_flat) * np.diff(imag_bins)[0]
    
    # Estimate parameters for Laplace distribution imaginary part
    loc_laplace_i, scale_laplace_i = laplace.fit(img_imag_flat)
    laplace_pdf_i = laplace.pdf(x_i, loc_laplace_i, scale_laplace_i)
    laplace_pdf_normalized_i = laplace_pdf_i * len(img_imag_flat) * np.diff(imag_bins)[0]
    
    figure, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=False)
    axes[0].hist(img_real_flat, bins=100, color="gray")
    axes[0].plot(x_r, gaussian_pdf_normalized_r, 'r--', label='Gaussian Distribution')
    axes[0].plot(x_r, laplace_pdf_normalized_r, 'g--', label='Laplace Distribution')
    axes[0].set_title("Real part")
    axes[0].set_xlabel("Intensity")
    axes[0].set_ylabel("Frequency")
    
    axes[1].hist(img_imag_flat, bins=100, color="gray")
    axes[1].plot(x_i, gaussian_pdf_normalized_i, 'r--', label='Gaussian Distribution')
    axes[1].plot(x_i, laplace_pdf_normalized_i, 'g--', label='Laplace Distribution')
    axes[1].set_title("Imaginary part")
    axes[1].set_xlabel("Intensity")
    axes[1].set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.legend()
    plt.show()

    
    


slc_mag = np.abs(img_real + 1j*img_imag)
def task_2A():
    """ histogram of the magnitude of the image """
    plt.style.use("ggplot")
    intensity_hist, intensity_bins = np.histogram(slc_mag.flatten(), bins=100)
    
    plt.hist(slc_mag.flatten(), bins=6000)
    plt.title("Intensity histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    #plt.grid(True)
    plt.show()
    
    normalized_variance = np.var(slc_mag) / np.mean(slc_mag)
    print("Normalized variance: ", normalized_variance)
    # Histogram Normalized
    plt.hist(slc_mag.flatten(), bins=6000, density=True)
    plt.title("Intensity histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.show()
    
    # Estimate parameters for Gaussian, Gamma and exponential distribution
    mu, sigma = norm.fit(slc_mag.flatten())
    alpha, loc, beta = gamma.fit(slc_mag.flatten())
    scale_exp = 1 / np.mean(slc_mag.flatten())
    scale_param = np.mean(slc_mag.flatten())
    
    plt.figure(figsize=(8, 6))
    plt.hist(slc_mag.flatten(), bins=100, color='gray', label='Intensity Histogram')

    # Plot normalized Gaussian distribution function
    x = np.linspace(min(intensity_bins), max(intensity_bins), 100)
    gaussian_pdf = norm.pdf(x, mu, sigma)
    gaussian_pdf_normalized = gaussian_pdf * len(slc_mag.flatten()) * np.diff(intensity_bins)[0]
    plt.plot(x, gaussian_pdf_normalized, 'r--', label='Gaussian Distribution')

    # Plot normalized Rayleigh distribution function
    rayleigh_pdf = rayleigh.pdf(x, scale_param)
    rayleigh_pdf_normalized = rayleigh_pdf * len(slc_mag.flatten()) * np.diff(intensity_bins)[0]
    plt.plot(x, rayleigh_pdf_normalized, 'b--', label='Rayleigh Distribution')
    
    # Plot normalized Gamma distribution function
    gamma_pdf = gamma.pdf(x, alpha, loc, beta)
    gamma_pdf = gamma_pdf * len(slc_mag.flatten()) * np.diff(intensity_bins)[0]
    plt.plot(x, gamma_pdf, 'g--', label='Gamma Distribution')

    # Plot normalized Exponential distribution function
    exponential_pdf = expon.pdf(x, scale=scale_exp)
    exponential_pdf = exponential_pdf * len(slc_mag.flatten()) * np.diff(intensity_bins)[0]
    plt.plot(x, exponential_pdf, 'm--', label='Exponential Distribution')

    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram of Intensity Image with Distribution Functions')
    plt.legend()
    plt.show()
    
    
def task_3A():
    """ 2D convolution to smooth the intensity image """
    plt.style.use("ggplot")
    kernel = np.ones((5, 5)) / 25  # Normalized kernel for averaging  (5x5)

    # Perform 2D convolution to apply the smoothing operation
    smoothed_intensity = convolve2d(slc_mag, kernel, mode='same')
    smoothed_intensity_hist, smoothed_intensity_bins = np.histogram(smoothed_intensity.flatten(), bins=100)


    # Plot histogram of the smoothed intensity image
    plt.figure(figsize=(8, 6))
    plt.hist(smoothed_intensity.flatten(), bins=6000, density=True)
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram of Smoothed Intensity Image')
    plt.show()

    # Compute normalized variance of the smoothed intensity image
    normalized_variance_smoothed = np.var(smoothed_intensity) / np.mean(smoothed_intensity)

    print("Normalized variance of the smoothed intensity image:", normalized_variance_smoothed)
    
    # Estimate parameters for Gaussian distribution
    mu, sigma = norm.fit(smoothed_intensity.flatten())

    # Estimate parameter for Rayleigh distribution (using mean as scale parameter)
    scale_param = np.mean(smoothed_intensity.flatten())
    
    alpha, loc, beta = gamma.fit(smoothed_intensity.flatten())

    plt.figure(figsize=(8, 6))
    plt.hist(smoothed_intensity.flatten(), bins=100, color='gray', label='Intensity Histogram')

    # Plot normalized Gaussian distribution function
    x = np.linspace(min(smoothed_intensity_bins), max(smoothed_intensity_bins), 100)
    gaussian_pdf = norm.pdf(x, mu, sigma)
    gaussian_pdf_normalized = gaussian_pdf * len(smoothed_intensity.flatten()) * np.diff(smoothed_intensity_bins)[0]
    plt.plot(x, gaussian_pdf_normalized, 'r--', label='Gaussian Distribution')

    # Plot normalized Rayleigh distribution function
    rayleigh_pdf = rayleigh.pdf(x, scale_param)
    rayleigh_pdf_normalized = rayleigh_pdf * len(smoothed_intensity.flatten()) * np.diff(smoothed_intensity_bins)[0]
    plt.plot(x, rayleigh_pdf_normalized, 'b--', label='Rayleigh Distribution')

    gamma_pdf = gamma.pdf(x, alpha, loc, beta)
    gamma_pdf = gamma_pdf * len(smoothed_intensity.flatten()) * np.diff(smoothed_intensity_bins)[0]
    plt.plot(x, gamma_pdf, 'g--', label='Gamma Distribution')

    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram of Intensity Image with Distribution Functions')
    plt.legend()
    plt.show()
    
    
    
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

spec_profile_azimuth = np.mean(np.abs(img_fft_shifted), axis=1)
spec_profile_azimuth_normalized = spec_profile_azimuth / spec_profile_azimuth.max()
def task_2B():
    """ 
    Plots the azimuth profile of the spectral profile.
    ~ Azimuth Spectral Profile
    """
    plt.style.use("ggplot")
    
    freqs = np.fft.fftfreq(Ny, d=1/f_prf)
    freqs = np.fft.fftshift(freqs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(freqs ,spec_profile_azimuth_normalized)
    plt.xlabel(r'Azimuth Frequency $[Hz]$')
    plt.ylabel(r'Intensity $[Norm]$')
    plt.title('Azimuth Fourier Domain')
    plt.show()
    
    # print the maximum and minimum values of azimuth frequency
    print("Max azimuth frequency: ", freqs.max())
    print("Min azimuth frequency: ", freqs.min())
    
    # shift the azimuth profile maximum to the center of the array for symmetry using np.roll
    max_idx = np.argmax(spec_profile_azimuth_normalized)
    
    shift = np.roll(spec_profile_azimuth_normalized, -max_idx + Ny//2 + 60)
    plt.plot(freqs, shift)
    plt.xlabel(r'Azimuth Frequency $[Hz]$')
    plt.ylabel(r'Intensity $[Norm]$')
    plt.title('Azimuth Fourier Domain')
    plt.show()
    
    
    is_symmetric = np.allclose(spec_profile_azimuth_normalized, spec_profile_azimuth_normalized[::-1], atol=1e-10)
    if is_symmetric:
        print("The spectral profile is symmetric around the zero frequency.")
    else:
        print("----|The spectral profile is shifted|----")
        # find the amount of shift
        print("Shifted by: ", np.argmax(spec_profile_azimuth_normalized) - Ny // 2)

    
# plot so that the maximum is in the center 
idx = len(spec_profile_azimuth_normalized)
max_idx = np.argmax(spec_profile_azimuth_normalized) + idx // 2 - 60

shifted_img_fft = np.roll(img_fft_shifted, -max_idx, axis=0)

def task_3B():
    # Plot the shifted spectrum
    plt.figure(figsize=(8, 6))
    plt.pcolor(kx, ky, np.abs(shifted_img_fft), cmap='gray')
    plt.colorbar()
    plt.xlabel(r'Range Wavenumber $[rad/m]$')
    plt.ylabel(r'Azimuth Wavenumber $[rad/m]$')
    plt.title('Shifted Spectrum of the Image')
    plt.show()
    
Ny = shifted_img_fft.shape[1]
print(shifted_img_fft.shape)
num_parts = 3
part_size = Ny // num_parts
intensity_images = [] 

for i in range(num_parts): # Extracting the looks from the shifted spectrum and computing the intensity image 
    start_idx = i*part_size
    end_idx = (i+1) * part_size
    complex_look = shifted_img_fft[:, start_idx:end_idx]
    
    complex_image  = np.fft.ifftshift(complex_look, axes=0)
    spatial_image = np.fft.ifft2(complex_image)
    
    intensity_image = np.abs(spatial_image)
    intensity_images.append(intensity_image)    
def task_4B():
    """ Plot the intensity images of the looks (3 complex images -> 3 intensity images) """   
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

mean_intensity1 = np.mean(intensity_images[0]) # Mean intensity of the first look
mean_intensity2 = np.mean(intensity_images[1]) # Mean intensity of the second look
mean_intensity3 = np.mean(intensity_images[2]) # Mean intensity of the third look
normalized_intensity1 = (intensity_images[0]-mean_intensity1) / mean_intensity1 # Normalized intensity of the first look
normalized_intensity2 = (intensity_images[1]-mean_intensity2) / mean_intensity2 # Normalized intensity of the second look
normalized_intensity3 = (intensity_images[2]-mean_intensity3) / mean_intensity3 # Normalized intensity of the third look

fourier_tr1 = np.fft.fftshift(np.fft.fft2(normalized_intensity1))
fourier_tr2 = np.fft.fftshift(np.fft.fft2(normalized_intensity2))
fourier_tr3 = np.fft.fftshift(np.fft.fft2(normalized_intensity3))

""" 
Co-spectrum: sub1*sub1, sub2*sub2, sub3*sub3
Cross-spectrum: sub1*sub2, sub2*sub3, sub1*sub3
"""

co_spectrum1_1 = fourier_tr1 * np.conj(fourier_tr1) # Co-spectrum 1*1
co_spectrum2_2 = fourier_tr2 * np.conj(fourier_tr2) # Co-spectrum 2*2
co_spectrum3_3 = fourier_tr3 * np.conj(fourier_tr3) # Co-spectrum 3*3

cross_spectrum1_2 = fourier_tr1 * np.conj(fourier_tr2) # Cross-spectrum 1*2
cross_spectrum2_3 = fourier_tr2 * np.conj(fourier_tr3) # Cross-spectrum 2*3
cross_spectrum1_3 = fourier_tr1 * np.conj(fourier_tr3) # Cross-spectrum 1*3

avg_co_spectrum = (co_spectrum1_1 + co_spectrum2_2 + co_spectrum3_3) / 3
avg_cross_spectrum = (cross_spectrum1_2 + cross_spectrum2_3) / 2

mag_cross_spectrum1_3 = np.abs(cross_spectrum1_3)
phase_cross_spectrum1_3 = np.angle(cross_spectrum2_3)


Ny, Nx = normalized_intensity1.shape
dx = c / (2*f_sf*np.sin(theta)) # Resolution or pixel size in range (x)
dy = V / f_prf # Resolution or pixel size in azimuth (y)
delta_kx = (2*np.pi) / (Nx*dx) # Resolution in kx
delta_ky = (2*np.pi) / (Ny*dy) # Resolution in ky
d_kx_max = np.pi / dx
d_ky_max = np.pi / dy

kx = np.linspace(-d_kx_max, d_kx_max, Nx)
ky = np.linspace(-d_ky_max, d_ky_max, Ny) 

def task_5B():
    """ Plot the co-spectra, cross-spectra and cross-sub1*sub3-spectra """
    figure, axes = plt.subplots(1, 3, figsize=(8, 6))
    axes[0].pcolormesh(kx, ky, np.log(np.abs(avg_co_spectrum)), cmap='gray', shading='nearest')
    axes[0].set_ylim(-0.1, 0.1)
    axes[0].set_title('Co-spectrum')
    axes[0].set_xlabel(r'$k_x$ $[rad/m]$')
    axes[0].set_ylabel(r'$k_y$ $[rad/m]$')
    
    axes[1].pcolormesh(kx, ky, np.log(np.abs(avg_cross_spectrum)), cmap='gray', shading='nearest')
    axes[1].set_ylim(-0.1, 0.1)
    axes[1].tick_params(axis='y', which='both', left=True, labelleft=False)
    axes[1].set_title('Cross-spectrum')
    axes[1].set_xlabel(r'$k_y$ $[rad/m]$')
    
    axes[2].pcolormesh(kx, ky, np.log(np.abs(cross_spectrum1_3)), cmap='gray', shading='nearest')
    axes[2].set_ylim(-0.1, 0.1)
    axes[2].tick_params(axis='y', which='both', left=True, labelleft=False)
    axes[2].set_title(r'sub1$\times$sub3')
    axes[2].set_xlabel(r'$k_y$ $[rad/m]$')
    
    # make one colorbar for all subplots, and place to the right of the last plot 
    cbar_ax = figure.add_axes([0.92, 0.115, 0.02, 0.81])
    cbar = figure.colorbar(axes[0].pcolormesh(kx, ky, np.log(np.abs(avg_co_spectrum)), cmap='gray', shading='nearest'), cax=cbar_ax)
   
    plt.tight_layout()
    plt.show() 
    
    
""" ---------------------------------- C. Analysis of 2D Spectra ----------------------------------- """

def task_1C():
    """ Plot the real and imaginary part of the co-spectrum, cross-spectrum and cross-spectrum 1-3 """
    plt.subplot(2,3,1) 
    plt.pcolormesh(kx, ky, np.real(avg_co_spectrum), shading="gouraud")
    plt.xlim(-0.05, 0.05)
    plt.ylim(-0.05, 0.05)
    plt.title("Real part of Co-Spectrum")
    plt.xlabel("Range Wavenumber [rad/m]")
    plt.ylabel("Azimuth Wavenumber [rad/m]")
    
    plt.subplot(2,3,4) 
    plt.pcolormesh(kx, ky, np.imag(avg_co_spectrum), shading="gouraud")
    plt.xlim(-0.05, 0.05)
    plt.ylim(-0.05, 0.05)
    plt.title("Imaginary part of Co-Spectrum")
    plt.xlabel("Range Wavenumber [rad/m]")
    plt.ylabel("Azimuth Wavenumber [rad/m]")
    
    plt.subplot(2,3,2) 
    plt.pcolormesh(kx, ky, np.real(avg_cross_spectrum), shading="gouraud")
    plt.xlim(-0.05, 0.05)
    plt.ylim(-0.05, 0.05)
    plt.title("Real part of Cross-Spectrum")
    plt.xlabel("Range Wavenumber [rad/m]")
    plt.ylabel("Azimuth Wavenumber [rad/m]")
    
    plt.subplot(2,3,5)  
    plt.pcolormesh(kx, ky, np.imag(avg_cross_spectrum), shading="gouraud")
    plt.xlim(-0.05, 0.05)
    plt.ylim(-0.05, 0.05)
    plt.title("Imaginary part of Cross-Spectrum")
    plt.xlabel("Range Wavenumber [rad/m]")
    plt.ylabel("Azimuth Wavenumber [rad/m]")
    
    plt.subplot(2,3,3) 
    plt.pcolormesh(kx, ky, np.abs(np.real(cross_spectrum1_3)), shading="gouraud")
    plt.xlim(-0.05, 0.05)
    plt.ylim(-0.05, 0.05)
    plt.title("Real part of Cross-Spectrum 1-3")
    plt.xlabel("Range Wavenumber [rad/m]")
    plt.ylabel("Azimuth Wavenumber [rad/m]")
    
    plt.subplot(2,3,6)
    plt.pcolormesh(kx, ky, np.imag(cross_spectrum1_3), shading="gouraud")
    plt.xlim(-0.05, 0.05)
    plt.ylim(-0.05, 0.05)
    plt.title("Imaginary part of Cross-Spectrum 1-3")
    plt.xlabel("Range Wavenumber [rad/m]")
    plt.ylabel("Azimuth Wavenumber [rad/m]")
    
    plt.tight_layout(pad=100)
    plt.show()
    
    plt.subplot(1,3,1)
    plt.pcolormesh(kx, ky, np.imag(avg_co_spectrum), shading="gouraud")
    plt.xlim(-0.05, 0.05)
    plt.ylim(-0.05, 0.05)
    plt.title("Co-Spectrum")
    plt.xlabel("Range Wavenumber [rad/m]")
    plt.ylabel("Azimuth Wavenumber [rad/m]")
    
    plt.subplot(1,3,2)
    plt.pcolormesh(kx, ky, np.imag(avg_cross_spectrum), shading="gouraud")
    plt.xlim(-0.05, 0.05)
    plt.ylim(-0.05, 0.05)
    plt.title("Cross-Spectrum")
    plt.xlabel("Range Wavenumber [rad/m]")
    plt.ylabel("Azimuth Wavenumber [rad/m]")
    
    plt.subplot(1,3,3)
    plt.pcolormesh(kx, ky, np.imag(cross_spectrum1_3), shading="gouraud")
    plt.xlim(-0.05, 0.05)
    plt.ylim(-0.05, 0.05)
    plt.title("Cross-Spectrum 1-3")
    plt.xlabel("Range Wavenumber [rad/m]")
    plt.ylabel("Azimuth Wavenumber [rad/m]")
    
    plt.tight_layout(pad=100)
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
    #task_5B()
    
# ______C. Analysis of 2D Spectra______ #
    task_1C()