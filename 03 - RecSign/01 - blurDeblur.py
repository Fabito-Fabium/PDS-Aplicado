# %%
import matplotlib
matplotlib.use('TkAgg')
# %%
import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
# %%
f = plt.imread('03 - RecSign/cameraman.pgm')/255.0

Nh = 9
h = np.ones((Nh, Nh)) / (Nh**2)
g1 = ndi.convolve(f, h)


F = np.fft.rfft2(f)
H = np.fft.rfft2(h, f.shape)
g2clean = np.fft.irfft2(F * H)
# %%
Pg2clean = np.mean(g2clean**2)

SNR = 40
sw2 = Pg2clean / (10**(SNR / 10))
w = np.sqrt(sw2) * np.random.randn(*f.shape)

g2 = g2clean + w
Pw = np.mean(w**2)
SNR_obs = 10*np.log10(Pg2clean/Pw)

print(f'{SNR_obs}')
# %% Blurring
plt.figure(figsize=(12, 7))
plt.subplot(141)
plt.imshow(f, cmap='gray')

plt.subplot(142)
plt.title('clean w/ conv')
plt.imshow(g1, cmap='gray')

plt.subplot(143)
plt.title('clean')
plt.imshow(g2clean, cmap='gray')

plt.subplot(144)
plt.title('w/ noise')
plt.imshow(g2, cmap='gray')
# %% Reconstruction
G2 = np.fft.rfft2(g2)
Hwiener = np.conj(H)/(np.abs(H)**2 + Pw /(np.abs(G2) ** 2))
fh2 =  np.fft.irfft2(G2 * Hwiener)

plt.figure(figsize=(12, 7))
plt.suptitle(f'SNR: {SNR} [dB]')
plt.subplot(131)
plt.imshow(f, cmap='gray')

plt.subplot(132)
plt.imshow(fh2, cmap='gray')

plt.subplot(133)
plt.imshow(f - fh2, cmap='gray')
# %%