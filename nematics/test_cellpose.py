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
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


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


def brightness(image, mask, labels):
    '''measure raw image intenslity for each mask label'''
    intensity = []
    for l in labels:
        intensity.append(image[mask==l].mean())
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

# %%
# %matplotlib qt
num = int(350)
mask = cv2.imread(r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)\Mask\Trans__"+str(num)+"_cp_masks.png", flags=cv2.IMREAD_ANYDEPTH)
div_masks = cv2.imread(r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)\DivMask\Trans__"+str(num)+"_div_mask.png",flags=cv2.IMREAD_ANYDEPTH)
img = cv2.imread(r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)\Trans__"+str(num)+".tif")
x_cent, y_cent, label_num = find_centers(mask)
intensity = brightness(div_masks, mask, label_num)
x_cent, y_cent, intensity = np.array(x_cent), np.array(y_cent), np.array(intensity)
plt.figure()
plt.imshow(img, cmap='gray')
plt.imshow(label2rgb(mask, bg_label=0), alpha=0.2)
plt.imshow(div_masks, cmap='Oranges', alpha=.6)
plt.figure()
plt.hist(div_masks.ravel(), bins=60)
plt.figure()
plt.hist(intensity[intensity>0].ravel(), bins=60)

# %%
track_df = pd.read_csv(r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)\Tracking\spots_100_500_3s_wdiv.csv")
df = track_df[track_df["FRAME"]==num-100]
# Register btw points (x,y) of two arrays
# xy_seg[idx[:n]] == xy_track[:n]
xy_seg = np.array([x_cent, y_cent]).T
xy_track = np.array([df['POSITION_X'].values, df['POSITION_Y'].values]).T
idx = center_pairs(
    xy_seg, #Long Array
    xy_track #Short Array
    )       
xy_seg[idx[:3]]
xy_track[:3]

# use index to insert "1" fo deviding cells
track_df['POSITION_X'][df['POSITION_X'][intensity[idx]>60].index]
track_df['DIVIDING'] = 0
# TODO 
# %%
plt.figure()
# plt.imshow(label2rgb(mask, bg_label=0), alpha=0.2)

tresh = 60
d = 20

for x,y,inten in zip(x_cent[idx][intensity[idx]>tresh], 
                     y_cent[idx][intensity[idx]>tresh], 
                     intensity[idx][intensity[idx]>tresh]):
    cv2.rectangle(img, 
        (int(x-d),int(y-d)), (int(x+d),int(y+d)), 
        (100/1.2,165/1.2,0), 2)
    # cv2.circle(img, (int(x),int(y)), radius=5, color=(100/1.2,165/1.2,0), thickness=2)
    # cv2.putText(img, str((int(inten))), (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    # plt.text(x,y, str(inten//1), color=(100/1.2,165/1.2,0), 2]) fontsize=8)
plt.imshow(img, cmap='gray')
# plt.imshow(div_masks, cmap='Oranges', alpha=.3)


for x,y in zip(df['POSITION_X'], df['POSITION_Y']):
    plt.plot(x,y,"*", color="white", alpha=.3)  

for x,y in zip(df['POSITION_X'][intensity[idx]>tresh], 
               df['POSITION_Y'][intensity[idx]>tresh]):
    plt.plot(x,y,".", color="red", alpha=.6)  
# %% Tracking Data Analysis
track_path = r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)\Tracking\spots_100_300.csv"
track_df = pd.read_csv(track_path, skiprows=[1,2,3]).reset_index()
track_df = track_df.drop(columns=['MANUAL_SPOT_COLOR']).dropna()
track_df["DIVIDING"] = 0
track_df["INTENSITY"] = 0
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

# %% unsupervised search of deviding cells *KMeans
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




# %% brightness-based check of deviding cells 
num = 99 #first frame of TrackMate file
for (frame, r), m in zip(enumerate(raw_list[num:]), mask_list[num:]):
    print(r)
    # seg = np.load(m, allow_pickle=True).item()
    # masks = seg['masks'].squeeze()
    masks = cv2.imread(m,flags=cv2.IMREAD_ANYDEPTH)
    x_cent, y_cent, label_num = find_centers(masks)
    # plt.figure()
    image = cv2.imread(r)[:,:,0]
    plt.imshow(image, cmap="gray", alpha=1.)
    # plt.imshow(label2rgb(masks, bg_label=0), alpha=0.3)
    plt.plot(x_cent, y_cent, '*', color="white", alpha=.3) 

    cell_idx = track_df[track_df["POSITION_T"]==frame].index#(track_df["POSITION_T"]==num-99)
    
    xr = track_df["POSITION_X"][cell_idx]
    yr = track_df["POSITION_Y"][cell_idx]
    radi = track_df["RADIUS"][cell_idx]
    # track_df["DIVIDING"][cell_idx] = -1
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
    # track_df["INTENSITY"][cell_idx] = np.array(intensity)[idx]

    # idx1 = np.array(intensity)>1.2 # looks like a good threshold
    # xy_seg1 = np.hstack([xy_seg, idx1.reshape(-1,1)]) #xy pos + True/False round cell
    # plt.plot(xy_seg[idx1,0], xy_seg[idx1,1], '*', color="green", alpha=1) 

    # idx_r = center_pairs(
    #     xy_track, #Long Array
    #     xy_seg1[idx1,:2] #Short Array
    #     ) 

    # track_df["DIVIDING"][cell_idx[idx_r]] = 1
    # track_df["INTENSITY"][cell_idx] = np.array(intensity)[idx]
    # np.unique(track_df["DIVIDING"][track_df["INTENSITY"]!=0], return_counts=True)
    # track_df["INTENSITY"][track_df["DIVIDING"]==1].min()

    # id = track_df["DIVIDING"][track_df["INTENSITY"]>1.2].index
    # plt.plot(track_df["POSITION_X"][id], track_df["POSITION_Y"][id], '*', color="green", alpha=1)
    # plt.plot(track_df["POSITION_X"][id], track_df["POSITION_Y"][id], '*', color="green", alpha=1) 

    # plt.title(os.path.basename(r))

    # track_df.to_csv(r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)\Tracking\spots_100_300_1.csv")   

    # progressBar(frame, 100)
    break


    if frame==track_df["POSITION_T"].max():
        break
    
# exit()

print(xy_seg[idx[:2]])
print(xy_track[:2])

print(xy_track[idx_r[:2]])
print(xy_seg[idx1,:][:2])

#%% check--
MASK = False

# import TrackMate data
track_path = r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)\Tracking\spots_100_300_2.csv"
track_df = pd.read_csv(track_path)
PARAM = track_df.keys()[-1]
print("Max frame with divisions: ",
    track_df["POSITION_T"][track_df[PARAM]>0].max())

frame = 180 #first frame in TrackMate file starts from 100
im_path = r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)/Trans__"+str(frame+100)+".tif"

fig = plt.figure(figsize=(15,15))

if MASK:
    mask_path = os.path.join(
        str(pathlib.Path(im_path).parents[0]), 
        "Mask", 
        im_path.split(os.sep)[-1].split(".")[0]+"_cp_masks.png"
        )
    masks = cv2.imread(mask_path, flags=cv2.IMREAD_ANYDEPTH)
    x_cent, y_cent = find_centers(masks)
    img = cv2.imread(im_path)
    plt.imshow(img, cmap="gray", alpha=1.)
    plt.imshow(label2rgb(masks, bg_label=0), alpha=0.1)
    plt.plot(x_cent, y_cent, '*', color="white", alpha=.3) 


cell_idx = track_df[track_df["POSITION_T"]==frame].index
x_all = track_df["POSITION_X"][cell_idx]
y_all = track_df["POSITION_Y"][cell_idx]


div_idx = track_df["DIVIDING"][np.logical_and.reduce((
        track_df["POSITION_T"]==frame,
        # track_df["DIVIDING"]==1,
        track_df[PARAM]>track_df[PARAM][track_df["FRAME"]==frame].median()))
        ].index
x_div = track_df["POSITION_X"][div_idx]
y_div = track_df["POSITION_Y"][div_idx]
   

img = cv2.imread(im_path)
d = 20
for xi, yi in zip(x_div, y_div):
    cv2.rectangle(img, 
    (int(xi-d),int(yi-d)), (int(xi+d),int(yi+d)), 
    (100/1.1,165/1.1,0), 2)
plt.imshow(img, cmap="gray", alpha=1.)    
plt.plot(x_all, y_all, '.', color="red", alpha=.3) 
plt.plot(x_div, y_div, '*', color="green", alpha=1, )#mfc='none') 

#%% looking for first appearance of round cell
# import TrackMate data
track_path = r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)\Tracking\spots_100_500_2s.csv"
track_df = pd.read_csv(track_path)
PARAM = track_df.keys()[-1]
track_df["DIVISION_T35"] = 0
print("Max frame with divisions: ",
    track_df["POSITION_T"][track_df[PARAM]>0].max())

# track_df[['index', 'ID', 'TRACK_ID',
#        'POSITION_X', 'POSITION_Y', 'POSITION_Z', 'POSITION_T', 'FRAME',
#        'RADIUS', 'ELLIPSE_X0', 'ELLIPSE_Y0', 'ELLIPSE_MAJOR', 'ELLIPSE_MINOR', 'ELLIPSE_THETA',
#        'ELLIPSE_ASPECTRATIO', 'AREA', 'PERIMETER', 'CIRCULARITY', 'SOLIDITY',
#        'SHAPE_INDEX', 'DIVIDING', 'INTENSITY_UNNORM']].to_csv(r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)\Tracking\spots_100_500_2s.csv")  
# %%
df = track_df.groupby(['TRACK_ID']).median()
plt.plot(df['FRAME'], df['INTENSITY_UNNORM'], ".k", alpha=.005)
df = track_df.groupby(['TRACK_ID']).median().groupby(['POSITION_T']).mean()

plt.plot(df['FRAME'][1:], df['INTENSITY_UNNORM'][1:].rolling(window=10).mean(), "-r", alpha=.6)
plt.xlabel("$Frame$", fontsize=14)
plt.ylabel("$Average ~Intensity$", fontsize=14)
plt.title("$Grey ~level ~intensity ~of ~raw ~images$")
plt.gca().set_box_aspect(1)
plt.ylim(20, 80)
# %%
min_track_len = 15
track_df['count'] = track_df.groupby('TRACK_ID')['TRACK_ID'].transform('count')
track_df1 = track_df[np.logical_and.reduce((
    track_df['count']>min_track_len,
    track_df['INTENSITY_UNNORM']!=0
    ))]
print(
    "Remained track number is ", 
    len(np.unique(track_df1["TRACK_ID"])), 
    "from ",  
    len(np.unique(track_df["TRACK_ID"])), "[=",
    int(100*len(np.unique(track_df1["TRACK_ID"]))/len(np.unique(track_df["TRACK_ID"]))),
    "%]"
    )
# %%
sns.pairplot(data=track_df1[[
    "AREA", "SHAPE_INDEX","FRAME", "INTENSITY_UNNORM"
    ]][track_df1["FRAME"]<100], kind="hist") #, kind="kde", plot_kws={'alpha': 0.1}
sns.jointplot(data=track_df1[track_df1["FRAME"]<200], x="SHAPE_INDEX", y="INTENSITY_UNNORM", kind="hist")
# %%
t_ids, t_len = np.unique(track_df["TRACK_ID"], return_counts=True)
t_ids_long = t_ids[t_len>15]
print(len(t_ids_long))

t_std_div = []

t_std_ = np.array([np.std(track_df['INTENSITY_UNNORM'][track_df['TRACK_ID']==t]) for t in t_ids_long])
t_mean_ = np.array([np.mean(track_df['INTENSITY_UNNORM'][track_df['TRACK_ID']==t]) for t in t_ids_long])
t_diff_sum_ = np.array([np.abs(np.diff(track_df['INTENSITY_UNNORM'][track_df['TRACK_ID']==t])).sum() for t in t_ids_long])


grey_levels = [track_df['INTENSITY_UNNORM'][track_df['TRACK_ID']==t] for t in t_ids_long]

tt = [np.std(grey) for grey in grey_levels if np.std(grey)>6]

fig, ax  = plt.subplots(1,2)
for grey in grey_levels[1000:1300]:
    if np.std(grey)>6:
        ax[0].plot(np.arange(len(grey)), grey/np.mean(grey), color="red", alpha=.3)
    else:
        ax[1].plot(np.arange(len(grey)), grey/np.mean(grey), color="blue", alpha=.3)



plt.hist(tt, bins=60, alpha=.6)
plt.figure()
plt.hist(t_std_, bins=60, alpha=.6, rwidth=.85)
plt.hist(t_mean_, bins=60, alpha=.6, rwidth=.85)
plt.hist(t_diff_sum_, bins=60, alpha=.6, rwidth=.85)
plt.plot(track_df['POSITION_T'], track_df['INTENSITY_UNNORM'], ".", alpha=.01)
plt.plot(t_mean_, t_std_, ".", alpha=.1)
plt.plot(t_mean_, t_std_, ".", alpha=.1)
# %%
ave_intensity = track_df[PARAM].median()

div_traks = track_df["TRACK_ID"][track_df[PARAM]>35].unique()   
for tnum, tid in enumerate(div_traks[:]):
    print(tnum, "form: ", len(div_traks))
    idx = track_df["TRACK_ID"]==tid
    t_intensity = track_df[PARAM][idx]/ave_intensity
    x_y_t_i = track_df[["POSITION_X", "POSITION_Y", "POSITION_T", PARAM]][idx]

    t_time = track_df["POSITION_T"][idx]
    
    if any(t_intensity>1.2):
        # plt.plot(t_time, t_intensity,"-",  alpha=.5)
        # plt.plot(t_time[t_intensity>1.2].iloc[1:5], t_intensity[t_intensity>1.2].iloc[1:5],"o",  alpha=.5)
        if len((t_intensity>1.2).index[1:5])>3:
            track_df["DIVISION_T35"][(t_intensity>1.2).index[1:5]] = 1
        # break

    # if tnum>50:
    #     break

track_df.to_csv(r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)\Tracking\spots_100_300_2_1.csv") 
# %%
PLOT = True
track_path = r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)\Tracking\spots_100_300_2_1.csv"
track_df = pd.read_csv(track_path)

if PLOT:
    fig = plt.figure(figsize=(10,10))

frames = track_df["POSITION_T"][track_df["DIVISION_T35"]!=0].unique()
for frame in frames[:]:
    print(frame)
    im_path = os.path.join(
            folder,
            "Trans__"+str(int(frame)+100)+".tif")
    
    cell_idx = track_df[track_df["POSITION_T"]==frame].index
    x_all = track_df["POSITION_X"][cell_idx]
    y_all = track_df["POSITION_Y"][cell_idx]

    ave_intensity = track_df['INTENSITY_UNNORM'][cell_idx].median()
    # TODO !!!!!!!!!!!!!


    div_idx = track_df["DIVIDING"][np.logical_and.reduce((
            track_df["POSITION_T"]==frame,
            # track_df["DIVIDING"]==1,
            track_df["DIVISION_T35"]!=0
            ))].index
    x_div = track_df["POSITION_X"][div_idx]
    y_div = track_df["POSITION_Y"][div_idx]    

    img = cv2.imread(im_path)
    d = 20
    for xi, yi in zip(x_div, y_div):
        cv2.rectangle(img, 
            (int(xi-d),int(yi-d)), (int(xi+d),int(yi+d)), 
            (100/1.2,165/1.2,0), 2)
        # cv2.circle(img, (int(xi),int(yi)), radius=5, color=(0, 0, 255/1.5), thickness=2)
    
    if PLOT:
        plt.imshow(img, cmap="gray", alpha=1.)   

        plt.plot(x_all, y_all, '.', color="red", alpha=.3) 
        # plt.plot(x_div, y_div, '*', color="green", alpha=1, ) #mfc='none') 
        break

    div_path = os.path.join(
        str(pathlib.Path(im_path).parents[0]), 
        "Div", 
        im_path.split(os.sep)[-1].split(".")[0]+"_div.png"
        )
    os.makedirs(os.path.dirname(div_path), exist_ok=True)
    cv2.imwrite(div_path,img)
    # break

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