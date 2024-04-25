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


img_mean_intensity = np.mean(img)
img_norm = img / img_mean_intensity

fft_img = np.fft.fft2(img_norm)
fft_img_shifted = np.fft.fftshift(fft_img)
img_spec = np.abs(fft_img_shifted)

dx = c / (2*f_sf*np.sin(theta)) # Resolution or pixel size in range (x)
dy = V / f_prf # Resolution or pixel size in azimuth (y)

d_kx = (2*np.pi) / (Nx*dx)
d_ky = (2*np.pi) / (Ny*dy)
print(d_kx, d_ky)

d_kx_max = np.pi / dx
d_ky_max = np.pi / dy
print(d_kx_max, d_ky_max)

Kx = np.linspace(-d_kx_max, d_kx_max, 501)
Ky = np.linspace(-d_ky_max, d_ky_max, 1759)

plt.pcolormesh(Kx, Ky, img_spec, cmap='gray')
#plt.xlim(-0.4, 0.4)
#plt.ylim(-0.6, 0.7)
plt.colorbar()
plt.show()