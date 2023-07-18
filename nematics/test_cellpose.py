# %% 
import numpy as np
import matplotlib.pyplot as plt
import pickle
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

# %%
%matplotlib qt

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



# %% TEST CELL CENTERS AND BOUNDARIES WITH VORONOI
# NEXT: COUNT NEIGHBOUR NUMBER FOR EACH CELL
from scipy.spatial import Voronoi, voronoi_plot_2d

colors = plt.cm.tab10(np.arange(10))

fig, ax = plt.subplots(1,1, figsize=(8,8))
h, w, dw = 200, 600, 500
num = 200
dt = 1
for (i,praw), pmask in zip(enumerate(raw_list[num::dt]),mask_list[num::dt]):
    print(i, praw,"\n", pmask)
    if dw==-1:
        img = cv2.imread(praw)[:,:,0]
        mask = cv2.imread(pmask, flags=cv2.IMREAD_ANYDEPTH)    
    else: 
        img = cv2.imread(praw)[h:h+dw,w:w+dw,0]
        mask = cv2.imread(pmask, flags=cv2.IMREAD_ANYDEPTH)[h:h+dw,w:w+dw]
    if i==0:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_clahe = clahe.apply(img)
        ax.imshow(255-img_clahe, "gray")         
        # ax.imshow(img)
        # ax.imshow(label2rgb(mask, bg_label=0), alpha=0.2)

    x_cent, y_cent,_ = find_centers(mask)
    ax.plot(x_cent, y_cent, 'o', color=colors[i], alpha=.3) 

    points = np.vstack((x_cent, y_cent)).T
    vor = Voronoi(points)
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors="red", line_width=2, line_alpha=0.5, point_size=2)


    ax.set_xlim([0,img.shape[0]])
    ax.set_ylim([0,img.shape[0]])
    break
    if i==1:
        break
# %%
phi = mask
width = 1
e = np.roll(phi,-width,axis=1) # eastern neighbor
w = np.roll(phi,width,axis=1) # western neighbor
n = np.roll(phi,width,axis=0) # northern neighbor
s = np.roll(phi,-width,axis=0) # southern neighbor
ne = np.roll(e,width,axis=0)
se = np.roll(e,-width,axis=0)
nw = np.roll(w,width,axis=0)
sw = np.roll(w,-width,axis=0)
int_angles = [e, ne, n, nw, w, sw, s, se, e]

k = np.zeros_like(phi).astype(np.int8)
for i in range(len(int_angles)-2):
    print(int_angles[i+1].shape)

    k += (np.abs(int_angles[i+1]-int_angles[i])>0).astype(np.int8)

# plt.imshow(k)
# plt.imshow(k, "jet")
plt.imshow(label2rgb(mask, bg_label=0), alpha=0.5)
plt.imshow(k, "jet", alpha=0.5)

from skimage.feature import peak_local_max
coordinates = peak_local_max(k, min_distance=11)
plt.plot(coordinates[:, 1], coordinates[:, 0], 'ro')
# %%
from skimage.morphology import skeletonize
from skimage.segmentation import find_boundaries

def count_different_neighbors(arr):
    """
    Counts the number of neighbors that have different values in a 2D NumPy array.
    Returns a 2D array of the same size as the input array.
    """
    rows, cols = arr.shape
    output_arr = np.zeros((rows, cols), dtype=np.int8)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            neighbors = [arr[i-1,j-1], arr[i-1,j], arr[i-1,j+1], arr[i,j-1], arr[i,j+1], arr[i+1,j-1], arr[i+1,j], arr[i+1,j+1]]
            # neighbors = [x for x in neighbors if x != 0]
            if len(set(neighbors)) > 2:
                output_arr[i, j] = 1
    return output_arr

a = find_boundaries(mask, mode='inner')
a = cv2.dilate(a.astype(np.float32), None)
skeleton = skeletonize(a)

# dst = cv2.cornerHarris(skeleton.astype("float32"),2,3,0.04)
# dst = cv2.dilate(dst,None)
# skeleton[dst>0.01*dst.max()]=255
plt.figure()
plt.imshow(label2rgb(mask, bg_label=0), alpha=0.5)
plt.imshow(skeleton, "gray", alpha=0.5)
# plt.imshow(count_different_neighbors(mask), alpha=.6)
# dst_th = dst>0.02*dst.max()
# xor = np.logical_and(dst_th, skeleton)
# # plt.imshow(xor, alpha=.6)
# plt.imshow(dst_th, alpha=.6)

# %%
def pad_array_with_nans(arr, mid_idx, padd_arr_len=41):
    """
    Pads an array with NaN values around "start_idx" and "end_idx" index
    """
    start_idx = max(padd_arr_len//2 - mid_idx, 0)
    # if left side doesn't fit cut and shift mid_idx
    if mid_idx >= padd_arr_len//2: 
        left_shift = mid_idx - padd_arr_len//2
        arr = arr[left_shift:]
        mid_idx = mid_idx - left_shift

    end_idx = min(padd_arr_len//2 + (len(arr) - mid_idx), padd_arr_len)
    if end_idx == padd_arr_len:
        arr = arr[:padd_arr_len-start_idx]
    
    padded_arr = np.full(padd_arr_len, np.nan)
    padded_arr[start_idx:end_idx] = arr
    return padded_arr

import numpy as np

def arrays_average_std(arrays):
    """
    Calculates average and standard deviation of multiple equally sized numpy arrays with NaNs using a loop.

    Parameters:
    arrays (list): List of numpy arrays.

    Returns:
    result (tuple): Tuple of average and standard deviation arrays for each index, with NaNs where these statistics cannot be calculated.
    """
    # Replace NaNs with zeros temporarily to calculate sum and count
    sums = np.nan_to_num(arrays[0])
    counts = np.array(~np.isnan(arrays[0]), dtype=int)
    for i in range(1, len(arrays)):
        sums += np.nan_to_num(arrays[i])
        counts += np.array(~np.isnan(arrays[i]), dtype=int)

    # Calculate average and std of each element
    result_avg = np.empty([arrays[0].size])
    result_std = np.empty([arrays[0].size])
    for i in range(arrays[0].size):
        # Calculate average and std if all elements are not NaN, otherwise set them to NaN
        if np.all(np.isnan([arr[i] for arr in arrays])):
            result_avg[i] = np.nan
            result_std[i] = np.nan
        else:
            temp_sum = np.sum([np.nan_to_num(arr[i]) for arr in arrays])
            temp_count = np.sum(~np.isnan([arr[i] for arr in arrays]))
            avg = temp_sum / temp_count
            std = np.sqrt(np.sum([(np.nan_to_num(arr[i]) - avg) ** 2 for arr in arrays if not np.isnan(arr[i])]) / (temp_count - 1))
            result_avg[i] = avg
            result_std[i] = std

    return result_avg, result_std

def arrays_average_sem(arrays):
    '''Parameters:
    arrays (list): List of numpy arrays.

    Returns:
    result (tuple): Tuple of average and standard error of the mean arrays for each index, with NaNs where these statistics cannot be calculated.
    '''
    # Replace NaNs with zeros temporarily to calculate sum and count
    sums = np.nan_to_num(arrays[0])
    counts = np.array(~np.isnan(arrays[0]), dtype=int)
    for i in range(1, len(arrays)):
        sums += np.nan_to_num(arrays[i])
        counts += np.array(~np.isnan(arrays[i]), dtype=int)

    # Calculate average and SEM of each element
    result_avg = np.empty([arrays[0].size])
    result_sem = np.empty([arrays[0].size])
    for i in range(arrays[0].size):
        # Calculate average and SEM if all elements are not NaN, otherwise set them to NaN
        if np.all(np.isnan([arr[i] for arr in arrays])):
            result_avg[i] = np.nan
            result_sem[i] = np.nan
        else:
            temp_sum = np.sum([np.nan_to_num(arr[i]) for arr in arrays])
            temp_count = np.sum(~np.isnan([arr[i] for arr in arrays]))
            avg = temp_sum / temp_count
            sem = np.sqrt(np.sum([(np.nan_to_num(arr[i]) - avg) ** 2 for arr in arrays if not np.isnan(arr[i])]) / (temp_count * (temp_count - 1)))
            result_avg[i] = avg
            result_sem[i] = sem

    return result_avg, result_sem



track_df1 = pd.read_csv(r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)\Tracking\spots_100_500_wdivs.csv")
# %%
min_track_len = 50
track_df1['count'] = track_df1.groupby('TRACK_ID')['TRACK_ID'].transform('count')
track_df2 = track_df1.loc[track_df1['count']>min_track_len]
# %%
tracks_all = track_df2["TRACK_ID"][track_df2['T0']==1].unique()

params = ['AREA', 'PERIMETER','SHAPE_INDEX', 'CIRCULARITY', 'RADIUS' , #'ELLIPSE_MAJOR', 'ELLIPSE_ASPECTRATIO'     
# 'RADIUS', 'ELLIPSE_X0', 'ELLIPSE_Y0', 'ELLIPSE_MAJOR', 'ELLIPSE_MINOR', 
# 'ELLIPSE_THETA', 'ELLIPSE_ASPECTRATIO', 'AREA', 'PERIMETER',
# 'CIRCULARITY', 'SOLIDITY', 'SHAPE_INDEX', 'DIVIDING'
]
PARAM = 'ELLIPSE_MINOR'
%matplotlib qt

frame2hr = 1/12
padd_arr_len = 400

data_dict = {}

plt.figure(PARAM)
fig, axs  = plt.subplots(1,len(params), num=PARAM)
axs = axs.ravel()

for ax,PARAM in zip(axs, params):
    param_arr = []
    for tid in tracks_all[:5000:10]:
        div_idx = track_df2[PARAM][track_df2["TRACK_ID"]==tid].index    
        param_val = track_df2[PARAM][div_idx].rolling(window=5).mean()
        # param_val = np.diff(param_val)
        t0 = track_df2["T0"][div_idx].values.astype(int)
        
        if not any(t0[0:3]):
            t0_val = t0
            # print(len(t0))

            mid_idx = int(np.where(t0>0)[0])
            #  collect param_val arrays for averaging
            param_arr.append(pad_array_with_nans(param_val, mid_idx, padd_arr_len=padd_arr_len))            

            # print(np.where(t0>0)[0])
            # ax.plot(np.arange(len(param_val)) - mid_idx, param_val, alpha=.03, linewidth=3, color='r')
            # ax.plot(np.arange(len(param_val)), param_val, alpha=.03, linewidth=3, color='r')
            
    # param_ave, param_std = arrays_average_std(param_arr)
    param_ave, param_std = arrays_average_sem(param_arr)

    print(PARAM, param_ave.shape)
    x = np.arange(len(param_ave))-padd_arr_len//2
    x = x * frame2hr
    ax.axvline(x=0, color='red', linestyle='--', linewidth=3, alpha=0.3)
    ax.plot(x, param_ave, alpha=.3, linewidth=3, color='b')
    ax.fill_between(x, param_ave - param_std, param_ave + param_std, color='blue', alpha=0.2)
    ax.set_xlabel("$hours$") 
    ax.set_title(PARAM)
    ax.set_xlim([-padd_arr_len//2 * frame2hr, padd_arr_len//2 * frame2hr])
    # ax.set_ylim([param_val.min(),param_val.max()])
    data_dict[PARAM] = {"_ave": param_ave, "std": param_std}

# %%
save_path = r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)\Tracking\spots_100_500_wdivs_division_dynamics"+str(min_track_len)+".pkl"
with open(save_path, 'wb') as f:
    pickle.dump(data_dict, f)
# %%
with open(save_path, 'rb') as f:
    data_dict1 = pickle.load(f)

fig, axs  = plt.subplots(1,len(data_dict1.keys()), num="1")
axs = axs.ravel()
print(data_dict1["AREA"].keys())

frame2hr = 1/12
for ax,param_name in zip(axs, data_dict1):
    param_ave = data_dict1[param_name]["_ave"]
    param_std = data_dict1[param_name]["std"]

    x = np.arange(len(param_ave))-len(param_ave)//2
    x = x * frame2hr
    ax.axvline(x=0, color='red', linestyle='--', linewidth=3, alpha=0.3)
    ax.plot(x, param_ave, alpha=.3, linewidth=3, color='b')
    ax.fill_between(x, param_ave - param_std, param_ave + param_std, color='blue', alpha=0.2)
    ax.set_xlabel("$hours$") 
    ax.set_title(param_name)
    ax.set_xlim([-padd_arr_len//2 * frame2hr, padd_arr_len//2 * frame2hr])
# %%
def pad_array_with_nans(arr, mid_idx, padd_arr_len=41):

    start_idx = max(padd_arr_len//2 - mid_idx, 0)
    # if left side doesn't fit cut and shift arr
    if mid_idx > padd_arr_len//2: 
        left_shift = mid_idx - padd_arr_len//2
        arr = arr[left_shift:]
        mid_idx = mid_idx - left_shift

    end_idx = min(padd_arr_len//2 + (len(arr) - mid_idx), padd_arr_len)
    if len(arr) > padd_arr_len:
        arr = arr[:padd_arr_len-start_idx]
    
    padded_arr = np.full(padd_arr_len, np.nan)
    padded_arr[start_idx:end_idx] = arr
    return padded_arr


mid_idx = 3
b = np.zeros((55))
b[:5] = [np.nan] * 5
b[mid_idx] = 1
print(b)
c = pad_array_with_nans(b, mid_idx, padd_arr_len=13)
print(c)
# c = a
# c =   
# %%
import numpy as np

# example data
arrays = []
arrays.append(np.array([1, 2, np.nan, 4, 5]))
arrays.append(np.array([6, np.nan, 8, 9, 10]))
arrays.append(np.array([np.nan, 3, 4, np.nan, 6]))

# replace NaNs with zeros temporarily to calculate sum and count
sums = np.nan_to_num(arrays[0])
counts = np.array(~np.isnan(arrays[0]), dtype=int)
for i in range(1, len(arrays)):
    sums += np.nan_to_num(arrays[i])
    counts += np.array(~np.isnan(arrays[i]), dtype=int)

# calculate average of each element
result = np.empty([arrays[0].size])
for i in range(arrays[0].size):
    # calculate average if all elements are not NaN, otherwise set to NaN
    if np.all(np.isnan([arr[i] for arr in arrays])):
        result[i] = np.nan
    else:
        temp_sum = np.sum([np.nan_to_num(arr[i]) for arr in arrays])
        temp_count = np.sum(~np.isnan([arr[i] for arr in arrays]))
        result[i] = temp_sum / temp_count

# print result array
print(result)
# %%
# # %matplotlib qt
# num = int(350)
# mask = cv2.imread(r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)\Mask\Trans__"+str(num)+"_cp_masks.png", flags=cv2.IMREAD_ANYDEPTH)
# div_masks = cv2.imread(r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)\DivMask\Trans__"+str(num)+"_div_mask.png",flags=cv2.IMREAD_ANYDEPTH)
# img = cv2.imread(r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)\Trans__"+str(num)+".tif")
# x_cent, y_cent, label_num = find_centers(mask)
# intensity = brightness(div_masks, mask, label_num)
# x_cent, y_cent, intensity = np.array(x_cent), np.array(y_cent), np.array(intensity)
# plt.figure()
# plt.imshow(img, cmap='gray')
# plt.imshow(label2rgb(mask, bg_label=0), alpha=0.2)
# plt.imshow(div_masks, cmap='Oranges', alpha=.6)
# plt.figure()
# plt.hist(div_masks.ravel(), bins=60)
# plt.figure()
# plt.hist(intensity[intensity>0].ravel(), bins=60)

# # %%
# track_df = pd.read_csv(r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)\Tracking\spots_100_500_3s_wdiv.csv")
# df = track_df[track_df["FRAME"]==num-100]
# # Register btw points (x,y) of two arrays
# # xy_seg[idx[:n]] == xy_track[:n]
# xy_seg = np.array([x_cent, y_cent]).T
# xy_track = np.array([df['POSITION_X'].values, df['POSITION_Y'].values]).T
# idx = center_pairs(
#     xy_seg, #Long Array
#     xy_track #Short Array
#     )       
# xy_seg[idx[:3]]
# xy_track[:3]

# # use index to insert "1" fo deviding cells
# track_df['POSITION_X'][df['POSITION_X'][intensity[idx]>60].index]
# track_df['DIVIDING'] = 0
# # TODO 
# # %%
# plt.figure()
# # plt.imshow(label2rgb(mask, bg_label=0), alpha=0.2)

# tresh = 60
# d = 20

# for x,y,inten in zip(x_cent[idx][intensity[idx]>tresh], 
#                      y_cent[idx][intensity[idx]>tresh], 
#                      intensity[idx][intensity[idx]>tresh]):
#     cv2.rectangle(img, 
#         (int(x-d),int(y-d)), (int(x+d),int(y+d)), 
#         (100/1.2,165/1.2,0), 2)
#     # cv2.circle(img, (int(x),int(y)), radius=5, color=(100/1.2,165/1.2,0), thickness=2)
#     # cv2.putText(img, str((int(inten))), (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
#     # plt.text(x,y, str(inten//1), color=(100/1.2,165/1.2,0), 2]) fontsize=8)
# plt.imshow(img, cmap='gray')
# # plt.imshow(div_masks, cmap='Oranges', alpha=.3)


# for x,y in zip(df['POSITION_X'], df['POSITION_Y']):
#     plt.plot(x,y,"*", color="white", alpha=.3)  

# for x,y in zip(df['POSITION_X'][intensity[idx]>tresh], 
#                df['POSITION_Y'][intensity[idx]>tresh]):
#     plt.plot(x,y,".", color="red", alpha=.6)  
# # %% Tracking Data Analysis
# track_path = r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)\Tracking\spots_100_300.csv"
# track_df = pd.read_csv(track_path, skiprows=[1,2,3]).reset_index()
# track_df = track_df.drop(columns=['MANUAL_SPOT_COLOR']).dropna()
# track_df["DIVIDING"] = 0
# track_df["INTENSITY"] = 0
# track_df.head
# # %%
# # plt.hist(track_df["AREA"][:], bins=60, rwidth=.8, range=[0,4000], density=True)
# # plt.hist(track_df["CIRCULARITY"][:], bins=60, rwidth=.8, density=True)
# parameter = "AREA"
# r_max = track_df[parameter].max()
# # plt.hist(track_df[parameter], range=[0,r_max], bins=60, rwidth=.8, density=True, alpha=.6)
# # plt.hist(track_df[parameter][np.logical_and(track_df["CIRCULARITY"]>.95, track_df["AREA"]<500)], range=[0,r_max], bins=60, rwidth=.8, density=True, alpha=.6)
# # plt.plot(track_df["AREA"][:], track_df["CIRCULARITY"][:], "o", alpha=.01)
# # %%

# # sns.jointplot(data=track_df, x="ELLIPSE_ASPECTRATIO", y="CIRCULARITY")
# # sns.pairplot(data=track_df[["AREA", "CIRCULARITY", "PERIMETER", "SHAPE_INDEX"]], kind="kde")

# # %% unsupervised search of deviding cells *KMeans
# # from sklearn.cluster import KMeans
# # import matplotlib.cm as cm
# # from sklearn.decomposition import PCA

# # n_clusters = 5
# # colors = cm.rainbow(np.linspace(0, 1, n_clusters))
# # # X_train = track_df[["AREA", "CIRCULARITY", "PERIMETER", "SHAPE_INDEX"]].dropna()
# # X_train = track_df[1:].dropna()
# # kmeans = KMeans(n_clusters=5, random_state=0).fit(X_train)
# # label = kmeans.fit_predict(X_train)
# # # %%
# # plt.figure()
# # for l,c in zip(np.unique(label), colors):
# #     plt.plot(track_df["PERIMETER"][label==l], track_df["AREA"][label==l], "o", color=c, alpha=.005)




# # %% brightness-based check of deviding cells 
# num = 99 #first frame of TrackMate file
# for (frame, r), m in zip(enumerate(raw_list[num:]), mask_list[num:]):
#     print(r)
#     # seg = np.load(m, allow_pickle=True).item()
#     # masks = seg['masks'].squeeze()
#     masks = cv2.imread(m,flags=cv2.IMREAD_ANYDEPTH)
#     x_cent, y_cent, label_num = find_centers(masks)
#     # plt.figure()
#     image = cv2.imread(r)[:,:,0]
#     plt.imshow(image, cmap="gray", alpha=1.)
#     # plt.imshow(label2rgb(masks, bg_label=0), alpha=0.3)
#     plt.plot(x_cent, y_cent, '*', color="white", alpha=.3) 

#     cell_idx = track_df[track_df["POSITION_T"]==frame].index#(track_df["POSITION_T"]==num-99)
    
#     xr = track_df["POSITION_X"][cell_idx]
#     yr = track_df["POSITION_Y"][cell_idx]
#     radi = track_df["RADIUS"][cell_idx]
#     # track_df["DIVIDING"][cell_idx] = -1
#     # plt.plot(xr, yr, '*', color="red", alpha=1) 


#     # Register btw points (x,y) of two arrays
#     # xy_seg[idx[:n]] == xy_track[:n]
#     xy_seg = np.array([x_cent, y_cent]).T
#     xy_track = np.array([xr, yr]).T
#     idx = center_pairs(
#         xy_seg, #Long Array
#         xy_track #Short Array
#         ) 

#     intensity = brightness(image, masks, label_num)
#     # track_df["INTENSITY"][cell_idx] = np.array(intensity)[idx]

#     # idx1 = np.array(intensity)>1.2 # looks like a good threshold
#     # xy_seg1 = np.hstack([xy_seg, idx1.reshape(-1,1)]) #xy pos + True/False round cell
#     # plt.plot(xy_seg[idx1,0], xy_seg[idx1,1], '*', color="green", alpha=1) 

#     # idx_r = center_pairs(
#     #     xy_track, #Long Array
#     #     xy_seg1[idx1,:2] #Short Array
#     #     ) 

#     # track_df["DIVIDING"][cell_idx[idx_r]] = 1
#     # track_df["INTENSITY"][cell_idx] = np.array(intensity)[idx]
#     # np.unique(track_df["DIVIDING"][track_df["INTENSITY"]!=0], return_counts=True)
#     # track_df["INTENSITY"][track_df["DIVIDING"]==1].min()

#     # id = track_df["DIVIDING"][track_df["INTENSITY"]>1.2].index
#     # plt.plot(track_df["POSITION_X"][id], track_df["POSITION_Y"][id], '*', color="green", alpha=1)
#     # plt.plot(track_df["POSITION_X"][id], track_df["POSITION_Y"][id], '*', color="green", alpha=1) 

#     # plt.title(os.path.basename(r))

#     # track_df.to_csv(r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)\Tracking\spots_100_300_1.csv")   

#     # progressBar(frame, 100)
#     break


#     if frame==track_df["POSITION_T"].max():
#         break
    
# # exit()

# print(xy_seg[idx[:2]])
# print(xy_track[:2])

# print(xy_track[idx_r[:2]])
# print(xy_seg[idx1,:][:2])

# #%% check--
# MASK = False

# # import TrackMate data
# track_path = r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)\Tracking\spots_100_300_2.csv"
# track_df = pd.read_csv(track_path)
# PARAM = track_df.keys()[-1]
# print("Max frame with divisions: ",
#     track_df["POSITION_T"][track_df[PARAM]>0].max())

# frame = 180 #first frame in TrackMate file starts from 100
# im_path = r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)/Trans__"+str(frame+100)+".tif"

# fig = plt.figure(figsize=(15,15))

# if MASK:
#     mask_path = os.path.join(
#         str(pathlib.Path(im_path).parents[0]), 
#         "Mask", 
#         im_path.split(os.sep)[-1].split(".")[0]+"_cp_masks.png"
#         )
#     masks = cv2.imread(mask_path, flags=cv2.IMREAD_ANYDEPTH)
#     x_cent, y_cent = find_centers(masks)
#     img = cv2.imread(im_path)
#     plt.imshow(img, cmap="gray", alpha=1.)
#     plt.imshow(label2rgb(masks, bg_label=0), alpha=0.1)
#     plt.plot(x_cent, y_cent, '*', color="white", alpha=.3) 


# cell_idx = track_df[track_df["POSITION_T"]==frame].index
# x_all = track_df["POSITION_X"][cell_idx]
# y_all = track_df["POSITION_Y"][cell_idx]


# div_idx = track_df["DIVIDING"][np.logical_and.reduce((
#         track_df["POSITION_T"]==frame,
#         # track_df["DIVIDING"]==1,
#         track_df[PARAM]>track_df[PARAM][track_df["FRAME"]==frame].median()))
#         ].index
# x_div = track_df["POSITION_X"][div_idx]
# y_div = track_df["POSITION_Y"][div_idx]
   

# img = cv2.imread(im_path)
# d = 20
# for xi, yi in zip(x_div, y_div):
#     cv2.rectangle(img, 
#     (int(xi-d),int(yi-d)), (int(xi+d),int(yi+d)), 
#     (100/1.1,165/1.1,0), 2)
# plt.imshow(img, cmap="gray", alpha=1.)    
# plt.plot(x_all, y_all, '.', color="red", alpha=.3) 
# plt.plot(x_div, y_div, '*', color="green", alpha=1, )#mfc='none') 

# #%% looking for first appearance of round cell
# # import TrackMate data
# track_path = r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)\Tracking\spots_100_500_2s.csv"
# track_df = pd.read_csv(track_path)
# PARAM = track_df.keys()[-1]
# track_df["DIVISION_T35"] = 0
# print("Max frame with divisions: ",
#     track_df["POSITION_T"][track_df[PARAM]>0].max())

# # track_df[['index', 'ID', 'TRACK_ID',
# #        'POSITION_X', 'POSITION_Y', 'POSITION_Z', 'POSITION_T', 'FRAME',
# #        'RADIUS', 'ELLIPSE_X0', 'ELLIPSE_Y0', 'ELLIPSE_MAJOR', 'ELLIPSE_MINOR', 'ELLIPSE_THETA',
# #        'ELLIPSE_ASPECTRATIO', 'AREA', 'PERIMETER', 'CIRCULARITY', 'SOLIDITY',
# #        'SHAPE_INDEX', 'DIVIDING', 'INTENSITY_UNNORM']].to_csv(r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)\Tracking\spots_100_500_2s.csv")  
# # %%
# df = track_df.groupby(['TRACK_ID']).median()
# plt.plot(df['FRAME'], df['INTENSITY_UNNORM'], ".k", alpha=.005)
# df = track_df.groupby(['TRACK_ID']).median().groupby(['POSITION_T']).mean()

# plt.plot(df['FRAME'][1:], df['INTENSITY_UNNORM'][1:].rolling(window=10).mean(), "-r", alpha=.6)
# plt.xlabel("$Frame$", fontsize=14)
# plt.ylabel("$Average ~Intensity$", fontsize=14)
# plt.title("$Grey ~level ~intensity ~of ~raw ~images$")
# plt.gca().set_box_aspect(1)
# plt.ylim(20, 80)
# # %%
# min_track_len = 15
# track_df['count'] = track_df.groupby('TRACK_ID')['TRACK_ID'].transform('count')
# track_df1 = track_df[np.logical_and.reduce((
#     track_df['count']>min_track_len,
#     track_df['INTENSITY_UNNORM']!=0
#     ))]
# print(
#     "Remained track number is ", 
#     len(np.unique(track_df1["TRACK_ID"])), 
#     "from ",  
#     len(np.unique(track_df["TRACK_ID"])), "[=",
#     int(100*len(np.unique(track_df1["TRACK_ID"]))/len(np.unique(track_df["TRACK_ID"]))),
#     "%]"
#     )
# # %%
# sns.pairplot(data=track_df1[[
#     "AREA", "SHAPE_INDEX","FRAME", "INTENSITY_UNNORM"
#     ]][track_df1["FRAME"]<100], kind="hist") #, kind="kde", plot_kws={'alpha': 0.1}
# sns.jointplot(data=track_df1[track_df1["FRAME"]<200], x="SHAPE_INDEX", y="INTENSITY_UNNORM", kind="hist")
# # %%
# t_ids, t_len = np.unique(track_df["TRACK_ID"], return_counts=True)
# t_ids_long = t_ids[t_len>15]
# print(len(t_ids_long))

# t_std_div = []

# t_std_ = np.array([np.std(track_df['INTENSITY_UNNORM'][track_df['TRACK_ID']==t]) for t in t_ids_long])
# t_mean_ = np.array([np.mean(track_df['INTENSITY_UNNORM'][track_df['TRACK_ID']==t]) for t in t_ids_long])
# t_diff_sum_ = np.array([np.abs(np.diff(track_df['INTENSITY_UNNORM'][track_df['TRACK_ID']==t])).sum() for t in t_ids_long])


# grey_levels = [track_df['INTENSITY_UNNORM'][track_df['TRACK_ID']==t] for t in t_ids_long]

# tt = [np.std(grey) for grey in grey_levels if np.std(grey)>6]

# fig, ax  = plt.subplots(1,2)
# for grey in grey_levels[1000:1300]:
#     if np.std(grey)>6:
#         ax[0].plot(np.arange(len(grey)), grey/np.mean(grey), color="red", alpha=.3)
#     else:
#         ax[1].plot(np.arange(len(grey)), grey/np.mean(grey), color="blue", alpha=.3)



# plt.hist(tt, bins=60, alpha=.6)
# plt.figure()
# plt.hist(t_std_, bins=60, alpha=.6, rwidth=.85)
# plt.hist(t_mean_, bins=60, alpha=.6, rwidth=.85)
# plt.hist(t_diff_sum_, bins=60, alpha=.6, rwidth=.85)
# plt.plot(track_df['POSITION_T'], track_df['INTENSITY_UNNORM'], ".", alpha=.01)
# plt.plot(t_mean_, t_std_, ".", alpha=.1)
# plt.plot(t_mean_, t_std_, ".", alpha=.1)
# # %%
# ave_intensity = track_df[PARAM].median()

# div_traks = track_df["TRACK_ID"][track_df[PARAM]>35].unique()   
# for tnum, tid in enumerate(div_traks[:]):
#     print(tnum, "form: ", len(div_traks))
#     idx = track_df["TRACK_ID"]==tid
#     t_intensity = track_df[PARAM][idx]/ave_intensity
#     x_y_t_i = track_df[["POSITION_X", "POSITION_Y", "POSITION_T", PARAM]][idx]

#     t_time = track_df["POSITION_T"][idx]
    
#     if any(t_intensity>1.2):
#         # plt.plot(t_time, t_intensity,"-",  alpha=.5)
#         # plt.plot(t_time[t_intensity>1.2].iloc[1:5], t_intensity[t_intensity>1.2].iloc[1:5],"o",  alpha=.5)
#         if len((t_intensity>1.2).index[1:5])>3:
#             track_df["DIVISION_T35"][(t_intensity>1.2).index[1:5]] = 1
#         # break

#     # if tnum>50:
#     #     break

# track_df.to_csv(r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)\Tracking\spots_100_300_2_1.csv") 
# # %%
# PLOT = True
# track_path = r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)\Tracking\spots_100_300_2_1.csv"
# track_df = pd.read_csv(track_path)

# if PLOT:
#     fig = plt.figure(figsize=(10,10))

# frames = track_df["POSITION_T"][track_df["DIVISION_T35"]!=0].unique()
# for frame in frames[:]:
#     print(frame)
#     im_path = os.path.join(
#             folder,
#             "Trans__"+str(int(frame)+100)+".tif")
    
#     cell_idx = track_df[track_df["POSITION_T"]==frame].index
#     x_all = track_df["POSITION_X"][cell_idx]
#     y_all = track_df["POSITION_Y"][cell_idx]

#     ave_intensity = track_df['INTENSITY_UNNORM'][cell_idx].median()
#     # TODO !!!!!!!!!!!!!


#     div_idx = track_df["DIVIDING"][np.logical_and.reduce((
#             track_df["POSITION_T"]==frame,
#             # track_df["DIVIDING"]==1,
#             track_df["DIVISION_T35"]!=0
#             ))].index
#     x_div = track_df["POSITION_X"][div_idx]
#     y_div = track_df["POSITION_Y"][div_idx]    

#     img = cv2.imread(im_path)
#     d = 20
#     for xi, yi in zip(x_div, y_div):
#         cv2.rectangle(img, 
#             (int(xi-d),int(yi-d)), (int(xi+d),int(yi+d)), 
#             (100/1.2,165/1.2,0), 2)
#         # cv2.circle(img, (int(xi),int(yi)), radius=5, color=(0, 0, 255/1.5), thickness=2)
    
#     if PLOT:
#         plt.imshow(img, cmap="gray", alpha=1.)   

#         plt.plot(x_all, y_all, '.', color="red", alpha=.3) 
#         # plt.plot(x_div, y_div, '*', color="green", alpha=1, ) #mfc='none') 
#         break

#     div_path = os.path.join(
#         str(pathlib.Path(im_path).parents[0]), 
#         "Div", 
#         im_path.split(os.sep)[-1].split(".")[0]+"_div.png"
#         )
#     os.makedirs(os.path.dirname(div_path), exist_ok=True)
#     cv2.imwrite(div_path,img)
#     # break

# # %%
# img = plt.imread(r)
# pix_x = img.shape[1]
# pix_y = img.shape[0]

# x = np.arange(0,pix_x)
# y = np.arange(0,pix_y)

# xx, yy = np.meshgrid(x, y)

# ori, coh, E = orientation_analysis(img, 31)
# k = compute_topological_charges(-ori, int_area='cell', origin='lower')
# defects = localize_defects(k, x_grid=xx, y_grid=yy)
# compute_defect_orientations(-ori, defects, method='interpolation', x_grid=x, y_grid=y, interpolation_radius=5,  min_sep=1)
# # %%
# plushalf = defects[defects['charge']==.5]
# minushalf = defects[defects['charge']==-.5]
# fig, ax  = plt.subplots(figsize=(16,16))
# s = 31
# ax.imshow(img, cmap='gray', origin='lower')

# ax.quiver(xx[::s,::s], yy[::s,::s], 
#     np.cos(ori)[::s,::s], -np.sin(ori)[::s,::s], 
#     headaxislength=0, headwidth=0, headlength=0, 
#     color='lawngreen', scale=60, pivot='mid', alpha=.5)

# ax.plot(plushalf['x'], plushalf['y'],'ro',markersize=10,label=r'+1/2 defect')
# ax.quiver(plushalf['x'], plushalf['y'], 
#     np.cos(plushalf['ang1']), np.sin(plushalf['ang1']), 
#     headaxislength=0, headwidth=0, headlength=0, color='r', scale=50)

# for i in range(3):
#     ax.quiver(minushalf['x'], minushalf['y'], 
#         np.cos(minushalf['ang'+str(i+1)]), np.sin(minushalf['ang'+str(i+1)]), 
#         headaxislength=0, headwidth=0, headlength=0, color='b', scale=50)

# # ax.imshow(label2rgb(masks, bg_label=0), alpha=0.3)
# # ax.plot(x_cent, y_cent, '*r', alpha=.5) 

# ax.set_xlabel('x (in pixels)')
# ax.set_ylabel('y (in pixels)')
# # %% 
# # area in pixels
# plt.figure(figsize=(6,6))
# plt.hist(np.unique(masks, return_counts=True)[1][1:]*.74**2, bins=30, rwidth=.9)
# plt.xlabel('$Area~(\mu m^2)$', fontsize=16)
# plt.ylabel("$Count$", fontsize=16)

# # %%
# %matplotlib qt
# import numpy as np
# import scipy.ndimage as ndi

# img = np.random.randint(0,100, size=(50,50))
# img1 = ndi.gaussian_filter(img, (5,5))
# fig, ax  = plt.subplots(1,2,figsize=(8,8))
# ax[0].imshow(img)
# ax[1].imshow(img1)
# plt.tight_layout()
# %%
import matplotlib
import numpy as np
# from matplotlib.mlab import griddata
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import math
from scipy.spatial import KDTree
import time
import scipy.ndimage as ndi
# %matplotlib qt


def grid_density_kdtree(xl, yl, xi, yi, dfactor):
    zz = np.empty([len(xi),len(yi)], dtype=np.uint8)
    zipped = zip(xl, yl)
    kdtree = KDTree(zipped)
    for xci in range(0, len(xi)):
        xc = xi[xci]
        for yci in range(0, len(yi)):
            yc = yi[yci]
            density = 0.
            retvalset = kdtree.query((xc,yc), k=5)
            for dist in retvalset[0]:
                density = density + math.exp(-dfactor * pow(dist, 2)) / 5
            zz[yci][xci] = min(density, 1.0) * 255
    return zz

def grid_density(xl, yl, xi, yi):
    ximin, ximax = min(xi), max(xi)
    yimin, yimax = min(yi), max(yi)
    xxi,yyi = np.meshgrid(xi,yi)
    #zz = np.empty_like(xxi)
    zz = np.empty([len(xi),len(yi)])
    for xci in range(0, len(xi)):
        xc = xi[xci]
        for yci in range(0, len(yi)):
            yc = yi[yci]
            density = 0.
            for i in range(0,len(xl)):
                xd = math.fabs(xl[i] - xc)
                yd = math.fabs(yl[i] - yc)
                if xd < 1 and yd < 1:
                    dist = math.sqrt(math.pow(xd, 2) + math.pow(yd, 2))
                    density = density + math.exp(-5.0 * pow(dist, 2))
            zz[yci][xci] = density
    return zz

def boxsum(img, w, h, r):
    st = [0] * (w+1) * (h+1)
    for x in range(w):
        st[x+1] = st[x] + img[x]
    for y in range(h):
        st[(y+1)*(w+1)] = st[y*(w+1)] + img[y*w]
        for x in range(w):
            st[(y+1)*(w+1)+(x+1)] = st[(y+1)*(w+1)+x] + st[y*(w+1)+(x+1)] - st[y*(w+1)+x] + img[y*w+x]
    for y in range(h):
        y0 = max(0, y - r)
        y1 = min(h, y + r + 1)
        for x in range(w):
            x0 = max(0, x - r)
            x1 = min(w, x + r + 1)
            img[y*w+x] = st[y0*(w+1)+x0] + st[y1*(w+1)+x1] - st[y1*(w+1)+x0] - st[y0*(w+1)+x1]

def grid_density_boxsum(x0, y0, x1, y1, w, h, data):
    kx = (w - 1) / (x1 - x0)
    ky = (h - 1) / (y1 - y0)
    r = 15
    border = r * 2
    imgw = (w + 2 * border)
    imgh = (h + 2 * border)
    img = [0] * (imgw * imgh)
    for x, y in data:
        ix = int((x - x0) * kx) + border
        iy = int((y - y0) * ky) + border
        if 0 <= ix < imgw and 0 <= iy < imgh:
            img[iy * imgw + ix] += 1
    for p in range(4):
        boxsum(img, imgw, imgh, r)
    a = np.array(img).reshape(imgh,imgw)
    b = a[border:(border+h),border:(border+w)]
    return b

def grid_density_gaussian_filter(x0, y0, x1, y1, w, h, data):
    kx = (w - 1) / (x1 - x0)
    ky = (h - 1) / (y1 - y0)
    r = 20
    border = r
    imgw = (w + 2 * border)
    imgh = (h + 2 * border)
    img = np.zeros((imgh,imgw))
    for x, y in data:
        ix = int((x - x0) * kx) + border
        iy = int((y - y0) * ky) + border
        if 0 <= ix < imgw and 0 <= iy < imgh:
            img[iy][ix] += 1
    return ndi.gaussian_filter(img, (r,r))  ## gaussian convolution

def generate_graph():    
    n = 1000
    # data points range
    data_ymin = -2.
    data_ymax = 2.
    data_xmin = -2.
    data_xmax = 2.
    # view area range
    view_ymin = -.5
    view_ymax = .5
    view_xmin = -.5
    view_xmax = .5
    # generate data
    xl = np.random.uniform(data_xmin, data_xmax, n)    
    yl = np.random.uniform(data_ymin, data_ymax, n)
    zl = np.random.uniform(0, 1, n)

    # get visible data points
    xlvis = []
    ylvis = []
    for i in range(0,len(xl)):
        if view_xmin < xl[i] < view_xmax and view_ymin < yl[i] < view_ymax:
            xlvis.append(xl[i])
            ylvis.append(yl[i])

    fig = plt.figure()

    # boxsum smoothing
    plt3 = plt.figure()
    # plt3.set_axis_off()
    t0 = time.process_time()
    zd = grid_density_boxsum(view_xmin, view_ymin, view_xmax, view_ymax, 256, 256, zip(xl, yl))
    plt.title('boxsum smoothing - '+str(time.process_time()-t0)+"sec")
    plt.imshow(zd)#, origin='lower', extent=[view_xmin, view_xmax, view_ymin, view_ymax])
    plt.scatter(255*(np.array(xlvis)+.5), 255*(np.array(ylvis)+.5))


generate_graph()
