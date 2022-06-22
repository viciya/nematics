# %%
import numpy as np
import matplotlib.pyplot as plt

from skimage import feature
import sys
import os

sys.path.insert(1, os.path.dirname(os.getcwd()))


def orientation_analysis(img, sigma):
    """ Orientation Analysis as in OrientationJ (Fiji). See https://forum.image.sc/t/orientationj-or-similar-for-python/51767/3 for a discussion.
    Also check theoretical-background-orientation.pdf for mathematical concepts. """

    # Compute structure tensor (upper diagonal elements) for every pixel in the image. Structre tensor is postive-definite symmetric matrix (positive eigenvalues)
    Axx, Axy, Ayy = feature.structure_tensor(img.astype(np.float32), sigma=sigma, mode='reflect', order='xy')

    # Get orientation for every pixel in the image
    ori = np.arctan2(2*Axy, Ayy - Axx) / 2

    # Compute coherence (local anisotropy of the image). If both eigenvalues of the structure tensor are equal, the coherence is zero (isotropic image). 
    # If the smaller eigenvalue tends to zero, the coherence tends to 1 (see math formulas).
    l1, l2 = feature.structure_tensor_eigenvalues([Axx, Axy, Ayy])
    eps = 1e-3 # to avoid division by zero?
    coh = ((l2-l1) / (l2+l1+eps)) ** 2

    # Finally, compute energy as trace of the structure tensor
    E = np.sqrt(Axx + Ayy)
    E /= np.max(E)

    return ori, coh, E

# %%
data_path = r'C:\Users\USER'

# file = r'\4.jpg'
# file = r"\CODES\nematics\example_images\raw\1_X1.tif"
file = r"..\example_images\raw\1_X1.tif"
# file = r"\Downloads\BEER\pep 858\1-20\1-0001.tif"

# TODO works only for reqtangular images.
img = plt.imread(file)#[:,:900]
img = img[-np.min(img.shape):,-np.min(img.shape):]

ori, coh, E = orientation_analysis(img, 12)

fig, axs  = plt.subplots(1,2,figsize=(20,10))
axs[0].imshow(img,cmap='gray')
axs[1].imshow(img,cmap='gray')
axs[1].imshow(ori,cmap='hsv', alpha=.2)   
axs[0].set_title('Raw ')
axs[1].set_title('Orientation') 

# %%
fig, axs  = plt.subplots(1,2,figsize=(20,10))
axs[0].imshow(coh,cmap='gray')
axs[1].imshow(E,cmap='gray')
axs[0].set_title('Coherency')
axs[1].set_title('Energy')
# %%

pix_x = img.shape[0]
pix_y = img.shape[1]

x = np.arange(0,pix_x)
y = np.arange(0,pix_y)

xx, yy = np.meshgrid(x, y)
fig, axs  = plt.subplots(1,2,figsize=(20,10))
s = 15
axs[0].imshow(img,cmap='gray')
axs[1].imshow(img,cmap='gray')
axs[1].quiver(xx[::s,::s], yy[::s,::s], 
    np.cos(ori)[::s,::s], np.sin(ori)[::s,::s], 
    headaxislength=0, headwidth=0, headlength=0, 
    color='lawngreen',scale=60, pivot='mid', alpha=.8)