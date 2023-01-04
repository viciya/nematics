# %% 
import numpy as np
import matplotlib.pyplot as plt
import cv2
from cellpose import plot
# %matplotlib qt


im_path = r"C:\Users\USER\Downloads\HBEC\s2_250_500\Trans__s2_0024.tif"
img = cv2.imread(im_path)[:,:,0]

seg = np.load(r"C:\Users\USER\Downloads\HBEC\s2_250_500\Trans__s2_0024_seg.npy", allow_pickle=True).item()
masks= seg['masks'].squeeze()
plt.imshow(img)
plt.imshow(masks, cmap='Pastel1', alpha=0.3)
# %% 
# area in pixels
plt.figure(figsize=(6,6))
plt.hist(np.unique(masks, return_counts=True)[1][1:]*.74**2, bins=30, rwidth=.9)
plt.xlabel('$Area~(\mu m^2)$', fontsize=16)
plt.ylabel("$Count$", fontsize=16)

