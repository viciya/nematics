# %% 
import numpy as np
import matplotlib.pyplot as plt
import cv2
from cellpose import plot
import os
# %matplotlib qt

im_path = r"C:\Users\USER\Downloads\HBEC\s2_250_500\Trans__s2_0024.tif"
os.path.splitext(f)[0] 
seg = np.load(r"C:\Users\USER\Downloads\HBEC\s2_250_500\masks\Trans__s2_0024_seg.npy", allow_pickle=True).item()
masks= seg['masks'].squeeze()
rgb = plot.mask_rgb(masks, colors=None)


img = cv2.imread(im_path)[:,:,0]
plt.imshow(img)
plt.imshow(rgb, alpha=0.4)

# plot.mask_overlay(
#     io.imread(r"C:\Users\USER\Downloads\HBEC\s2_250_500\Trans__s2_0024.tif"),  
#     io.imread(r"C:\Users\USER\Downloads\HBEC\s2_250_500\masks\Trans__s2_0024_cp_masks.png"), 
#     colors=None)
# %% 
# area in pixels
plt.figure(figsize=(6,6))
plt.hist(np.unique(masks, return_counts=True)[1][1:]*.74**2, bins=30, rwidth=.9)
plt.xlabel('$Area~(\mu m^2)$', fontsize=16)
plt.ylabel("$Count$", fontsize=16)

