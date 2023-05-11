# %%
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pathlib
import glob
import cv2
import pickle 

file_path = pathlib.Path(__file__).parent.resolve()
print("********", os.path.join(file_path.parents[0],"vasco_scripts"))
sys.path.insert(1, os.path.join(file_path.parents[0],"vasco_scripts"))

from defects import *

# %matplotlib qt

# %%
image_list = glob.glob(r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)\*.tif")
from natsort import natsorted
image_list = natsorted(image_list, key=lambda y: y.lower())

for (i,im) in enumerate(image_list):
    img = cv2.imread(im)[:,:,0]

    pix_x = img.shape[0]
    pix_y = img.shape[1]

    x = np.arange(0,pix_x)
    y = np.arange(0,pix_y)

    xx, yy = np.meshgrid(x, y)

    ori, coh, E = orientation_analysis(img, 31)
    k = compute_topological_charges(ori, int_area='cell', origin='upper')
    defects = localize_defects(k, x_grid=xx, y_grid=yy)
    compute_defect_orientations(ori, defects)    
    plushalf = defects[defects['charge']==.5]
    minushalf = defects[defects['charge']==-.5]

    save_path = os.path.join(
        os.path.dirname(im), 
        'Nematic', 
        os.path.splitext(os.path.basename(im))[0]
        )
    break
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path+ '_orient.tif', ori[::3,::3])
    with open(save_path+ "_defects.pkl", 'wb') as f:
            pickle.dump(dict(plus=plushalf, minus=minushalf), f)

    


# %%

# Due to the choice of the 2x2 grid (neigbors in the northeastern corner), 
# we move the detected defect to the middle of the cell
img = cv2.imread(image_list[300])[:,:,0]

shift = 1/2 
pix_x = img.shape[0]
pix_y = img.shape[1]

x = np.arange(0,pix_x)
y = np.arange(0,pix_y)

xx, yy = np.meshgrid(x, y)

ori, coh, E = orientation_analysis(img, 31)
cv2.imwrite(r"C:\Users\victo\OneDrive\Desktop\orient.tif", ori[::3,::3])


k = compute_topological_charges(ori, int_area='cell', origin='upper')
defects = localize_defects(k, x_grid=xx, y_grid=yy)
compute_defect_orientations(ori, defects)
plushalf = defects[defects['charge']==.5]
minushalf = defects[defects['charge']==-.5]
fig, ax  = plt.subplots(figsize=(16,16))
s = 30
ax.imshow(img, cmap='gray')
ax.quiver(xx[::s,::s], yy[::s,::s], 
    np.cos(ori)[::s,::s], np.sin(ori)[::s,::s], 
    headaxislength=0, headwidth=0, headlength=0, 
    color='lawngreen', scale=60, pivot='mid', alpha=.5)

ax.plot(plushalf['x'], plushalf['y'],'ro',markersize=10,label=r'+1/2 defect')
ax.quiver(plushalf['x'], plushalf['y'], 
    np.cos(plushalf['ang1']), np.sin(plushalf['ang1']), 
    headaxislength=0, headwidth=0, headlength=0, color='r', scale=50)

for i in range(3):
    ax.quiver(minushalf['x'], minushalf['y'], 
        np.cos(minushalf['ang'+str(i+1)]), np.sin(minushalf['ang'+str(i+1)]), 
        headaxislength=0, headwidth=0, headlength=0, color='b', scale=50)

ax.set_xlabel('x (in pixels)')
ax.set_ylabel('y (in pixels)')

px = np.argwhere(np.abs(k)>.1)
plt.plot(px[:,1],px[:,0],"+")