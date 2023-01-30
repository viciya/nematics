# %% 
import numpy as np
import matplotlib.pyplot as plt
import cv2
from cellpose import plot
import os
from skimage.measure import label, regionprops, regionprops_table
from skimage.color import label2rgb
import pandas as pd
import glob
from vasco_scripts.defects  import *
import pathlib
import seaborn as sns
from pip._internal.cli.progress_bars import get_download_progress_renderer


# %matplotlib qt

import sys,time,random
def progressBar(count_value, total, suffix=''):
    bar_length = 100
    filled_up_Length = int(round(bar_length* count_value / float(total)))
    percentage = round(100.0 * count_value/float(total),1)
    bar = '=' * filled_up_Length + '-' * (bar_length - filled_up_Length)
    sys.stdout.write('[%s] %s%s ...%s\r' %(bar, percentage, '%', suffix))
    sys.stdout.flush()

def find_centers(masks):
    '''find centers in segmented image b/w or labels'''
    regions = regionprops(masks)
    x_cent = []
    y_cent = []
    label_num = []
    for i,props in enumerate(regions):
        yi, xi = props.centroid
        x_cent.append(xi)
        y_cent.append(yi)
        label_num.append(props.label)
        # plt.text(xi, yi, str(i))
    return x_cent, y_cent, label_num

def center_pairs(Xlong, Xshort):
    '''find indexes of Xshort in Xlong'''
    from scipy import spatial
    tree = spatial.KDTree(Xlong)
    _, minid = tree.query(Xshort)   
    return minid

#TODO brightness deterction works, but take too long
# replace with local threshold
# than look for pairs between centers of bright spots and ...
def brightness(image, mask, labels):
    '''measure relative intenslity for each mask label'''
    from scipy import ndimage
    intensity = []
    for l in labels:
        # print(l, " from ", len(labels))  
        mask_in = (masks==l)
        img_in = image*mask_in
        mask_out = ndimage.binary_dilation(mask_in, iterations=50)
        img_out = image*mask_out
        intensity.append(
            np.true_divide(img_in.sum(),(img_in!=0).sum())/ # mean without zeros
            np.true_divide(img_out.sum(),(img_out!=0).sum())
            )  
        progressBar(l,len(labels))
        
    return intensity    

# folder = r"D:\HBEC\s2_250_500"
folder = r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)"
raw_list = sorted(glob.glob(folder + "/*.tif"))
mask_list = []
for raw in raw_list:
    # mask_folder = os.path.dirname(raw) + "/masks/"
    # mask_name = os.path.basename(raw).split('.')[0] + "_seg.npy"
    # mask_list.append(mask_folder + mask_name)
    mask_path = os.path.join(
        str(pathlib.Path(raw).parents[0]), 
        "Mask", 
        raw.split(os.sep)[-1].split(".")[0]+"_cp_masks.png"
        )    
    mask_list.append(mask_path)

# %% Tracking Data Analysis
track_path = r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)\Tracking\spots_100_200.csv"
track_df = pd.read_csv(track_path, skiprows=[1,2,3]).reset_index()
track_df = track_df.drop(columns=['MANUAL_SPOT_COLOR']).dropna()
# track_df["DIVIDING"][track_df["POSITION_T"]==0] = 0
track_df.head
# %%
# plt.hist(track_df["AREA"][:], bins=60, rwidth=.8, range=[0,4000], density=True)
# plt.hist(track_df["CIRCULARITY"][:], bins=60, rwidth=.8, density=True)
parameter = "AREA"
r_max = track_df[parameter].max()
# plt.hist(track_df[parameter], range=[0,r_max], bins=60, rwidth=.8, density=True, alpha=.6)
# plt.hist(track_df[parameter][np.logical_and(track_df["CIRCULARITY"]>.95, track_df["AREA"]<500)], range=[0,r_max], bins=60, rwidth=.8, density=True, alpha=.6)
# plt.plot(track_df["AREA"][:], track_df["CIRCULARITY"][:], "o", alpha=.01)
# %%

# sns.jointplot(data=track_df, x="ELLIPSE_ASPECTRATIO", y="CIRCULARITY")
# sns.pairplot(data=track_df[["AREA", "CIRCULARITY", "PERIMETER", "SHAPE_INDEX"]], kind="kde")

# %%
# from sklearn.cluster import KMeans
# import matplotlib.cm as cm
# from sklearn.decomposition import PCA

# n_clusters = 5
# colors = cm.rainbow(np.linspace(0, 1, n_clusters))
# # X_train = track_df[["AREA", "CIRCULARITY", "PERIMETER", "SHAPE_INDEX"]].dropna()
# X_train = track_df[1:].dropna()
# kmeans = KMeans(n_clusters=5, random_state=0).fit(X_train)
# label = kmeans.fit_predict(X_train)
# # %%
# plt.figure()
# for l,c in zip(np.unique(label), colors):
#     plt.plot(track_df["PERIMETER"][label==l], track_df["AREA"][label==l], "o", color=c, alpha=.005)

# %%


# %% show segmentation on overlay
num = 100 #first frame of TrackMate file
for (frame, r), m in zip(enumerate(raw_list[num:]), mask_list[num:]):
    # seg = np.load(m, allow_pickle=True).item()
    # masks = seg['masks'].squeeze()
    masks = cv2.imread(m,flags=cv2.IMREAD_ANYDEPTH)
    x_cent, y_cent, label_num = find_centers(masks)
    # plt.figure()
    image = cv2.imread(r)[:,:,0]
    # plt.imshow(image, cmap="gray", alpha=1.)
    # plt.imshow(label2rgb(masks, bg_label=0), alpha=0.6)
    # plt.plot(x_cent, y_cent, '*', color="white", alpha=.3) 

    cell_idx = track_df[track_df["POSITION_T"]==frame+1].index#(track_df["POSITION_T"]==num-99)
    
    xr = track_df["POSITION_X"][cell_idx]
    yr = track_df["POSITION_Y"][cell_idx]
    radi = track_df["RADIUS"][cell_idx]
    track_df["DIVIDING"][cell_idx] = -1
    # plt.plot(xr, yr, '*', color="red", alpha=1) 

    # Register btw points (x,y) of two arrays
    # xy_seg[idx[:n]] == xy_track[:n]
    xy_seg = np.array([x_cent, y_cent]).T
    xy_track = np.array([xr, yr]).T
    idx = center_pairs(
        xy_seg, #Long Array
        xy_track #Short Array
        ) 

    intensity = brightness(image, masks, label_num)

    idx1 = np.array(intensity)>1.2 # looks like a good threshold
    xy_seg1 = np.hstack([xy_seg,idx1.reshape(-1,1)]) #xy pos + True/False round cell
    # plt.plot(xy_seg[idx1,0], xy_seg[idx1,1], '*', color="green", alpha=1) 

    idx_r = center_pairs(
        xy_track, #Long Array
        xy_seg1[idx1,:2] #Short Array
        ) 

    track_df["DIVIDING"][cell_idx[idx_r]] = 1
    id = track_df["DIVIDING"][track_df["DIVIDING"]==1].index
    # plt.plot(track_df["POSITION_X"][id], track_df["POSITION_Y"][id], '*', color="red", alpha=1) 

    # plt.title(os.path.basename(r))

    # track_df.to_csv(r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)\Tracking\spots_100_200_1.csv")   

    # progressBar(frame, 100)

    if frame>99:
        break

# exit()

print(xy_seg[idx[:2]])
print(xy_track[:2])

print(xy_track[idx_r[:2]])
print(xy_seg[idx1,:][:2])

#%% check--
MASK = False

# import TrackMate data
track_path = r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)\Tracking\spots_100_200_1_w_divisions.csv"
track_df = pd.read_csv(track_path, skiprows=[1,2,3]).reset_index()

frame = 88 #first frame in TrackMate file starts from 100
im_path = r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)/Trans__"+str(frame+100)+".tif"

fig = plt.figure(figsize=(15,15))
plt.imshow(cv2.imread(im_path)[:,:,0], cmap="gray", alpha=1.)

if MASK:
    mask_path = os.path.join(
        str(pathlib.Path(im_path).parents[0]), 
        "Mask", 
        im_path.split(os.sep)[-1].split(".")[0]+"_cp_masks.png"
        )
    masks = cv2.imread(mask_path, flags=cv2.IMREAD_ANYDEPTH)
    x_cent, y_cent = find_centers(masks)
    plt.imshow(label2rgb(masks, bg_label=0), alpha=0.1)
    plt.plot(x_cent, y_cent, '*', color="white", alpha=.3) 


cell_idx = track_df[track_df["POSITION_T"]==frame].index
x_all = track_df["POSITION_X"][cell_idx]
y_all = track_df["POSITION_Y"][cell_idx]
plt.plot(x_all, y_all, '.', color="red", alpha=.3) 

div_idx = track_df["DIVIDING"][np.logical_and(
        track_df["POSITION_T"]==frame,
        track_df["DIVIDING"]==1)
        ].index
x_all = track_df["POSITION_X"][div_idx]
y_all = track_df["POSITION_Y"][div_idx]
plt.plot(x_all, y_all, '*', color="green", alpha=.6, )#mfc='none') 

# %%
img = plt.imread(r)
pix_x = img.shape[1]
pix_y = img.shape[0]

x = np.arange(0,pix_x)
y = np.arange(0,pix_y)

xx, yy = np.meshgrid(x, y)

ori, coh, E = orientation_analysis(img, 31)
k = compute_topological_charges(-ori, int_area='cell', origin='lower')
defects = localize_defects(k, x_grid=xx, y_grid=yy)
compute_defect_orientations(-ori, defects, method='interpolation', x_grid=x, y_grid=y, interpolation_radius=5,  min_sep=1)
# %%
plushalf = defects[defects['charge']==.5]
minushalf = defects[defects['charge']==-.5]
fig, ax  = plt.subplots(figsize=(16,16))
s = 31
ax.imshow(img, cmap='gray', origin='lower')

ax.quiver(xx[::s,::s], yy[::s,::s], 
    np.cos(ori)[::s,::s], -np.sin(ori)[::s,::s], 
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

# ax.imshow(label2rgb(masks, bg_label=0), alpha=0.3)
# ax.plot(x_cent, y_cent, '*r', alpha=.5) 

ax.set_xlabel('x (in pixels)')
ax.set_ylabel('y (in pixels)')
# %% 
# area in pixels
plt.figure(figsize=(6,6))
plt.hist(np.unique(masks, return_counts=True)[1][1:]*.74**2, bins=30, rwidth=.9)
plt.xlabel('$Area~(\mu m^2)$', fontsize=16)
plt.ylabel("$Count$", fontsize=16)

# %%
import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import sobel
from skimage.measure import label
from skimage.segmentation import watershed, expand_labels
from skimage.color import label2rgb
from skimage import data

coins = data.coins()

# Make segmentation using edge-detection and watershed.
edges = sobel(coins)

# Identify some background and foreground pixels from the intensity values.
# These pixels are used as seeds for watershed.
markers = np.zeros_like(coins)
foreground, background = 1, 2
markers[coins < 30.0] = background
markers[coins > 150.0] = foreground

ws = watershed(edges, markers)
seg1 = label(ws == foreground)

expanded = expand_labels(seg1, distance=10)

# Show the segmentations.
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 5),
                         sharex=True, sharey=True)

color1 = label2rgb(seg1, image=coins, bg_label=0)
axes[0].imshow(color1)
axes[0].set_title('Sobel+Watershed')

color2 = label2rgb(expanded, image=coins, bg_label=0)
axes[1].imshow(color2)
axes[1].set_title('Expanded labels')

for a in axes:
    a.axis('off')
fig.tight_layout()
plt.show()