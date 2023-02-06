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
from scipy import ndimage


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
    '''measure raw image intenslity for each mask label'''
    intensity = []
    for l in labels:
        intensity.append(image[masks==l].mean())
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
track_path = r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)\Tracking\spots_100_300.csv"
track_df = pd.read_csv(track_path, skiprows=[1,2,3]).reset_index()
track_df = track_df.drop(columns=['MANUAL_SPOT_COLOR']).dropna()
track_df = pd.read_csv(track_path).reset_index()
track_df["DIVIDING"] = 0
track_df["INTENSITY_UNNORM"] = 0
track_df.head

# %% show segmentation on overlay
num = 99 #first frame of TrackMate file
for (frame, r), m in zip(enumerate(raw_list[num:]), mask_list[num:]):
    if frame>0:
        print("------------ F R A M E:  >", frame, "< --------------")
        masks = cv2.imread(m,flags=cv2.IMREAD_ANYDEPTH)
        x_cent, y_cent, label_num = find_centers(masks)
        # plt.figure()
        image = cv2.imread(r)[:,:,0]
        # plt.imshow(image, cmap="gray", alpha=1.)
        # plt.imshow(label2rgb(masks, bg_label=0), alpha=0.3)
        # plt.plot(x_cent, y_cent, '*', color="white", alpha=.3) 

        cell_idx = track_df[track_df["POSITION_T"]==frame].index#(track_df["POSITION_T"]==num-99)
        
        xr = track_df["POSITION_X"][cell_idx]
        yr = track_df["POSITION_Y"][cell_idx]
        radi = track_df["RADIUS"][cell_idx]
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
        track_df["INTENSITY_UNNORM"][cell_idx] = np.array(intensity)[idx]

        # id = track_df["DIVIDING"][track_df["INTENSITY"]>1.2].index
        # plt.plot(track_df["POSITION_X"][id], track_df["POSITION_Y"][id], '*', color="green", alpha=1)
    
        if (frame % 10)==0:
            track_df.to_csv(r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)\Tracking\spots_100_300_2.csv")   


    if frame==track_df["POSITION_T"].max():
        break

track_df.to_csv(r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)\Tracking\spots_100_300_2.csv") 
exit()
