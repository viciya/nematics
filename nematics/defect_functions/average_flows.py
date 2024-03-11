import matplotlib.pyplot as plt
import cv2
import numpy as np
from natsort import natsorted
import sys
import pandas as pd
from scipy.ndimage import rotate, gaussian_filter
from scipy.stats import circmean, circstd, sem
import time

sys.path.append('../vasco_scripts')  # add the relative path to the folder
sys.path.append('../defect_functions') 
from defects import *  # import the module from the folder
from defect_pairs import * 


def divergence_npgrad(flow):
    flow = np.swapaxes(flow, 0, 1)
    Fx, Fy = flow[:, :, 0], flow[:, :, 1]
    dFx_dx = np.gradient(Fx, axis=0)
    dFy_dy = np.gradient(Fy, axis=1)
    return (dFx_dx + dFy_dy).T

def curl_npgrad(flow):
    flow = np.swapaxes(flow, 0, 1)
    Fx, Fy = flow[:, :, 0], flow[:, :, 1]
    dFx_dy = np.gradient(Fx, axis=1)
    dFy_dx = np.gradient(Fy, axis=0)
    curl = dFy_dx - dFx_dy
    return curl.T

def crop(img, center, width, height):
    ulx, uly = max(int(center[0] - width//2), 0), max(int(center[1] - height//2), 0)
    lrx, lry = min(int(center[0] + width//2), img.shape[1]), min(int(center[1] + height//2), img.shape[0])
    new_center = ((lrx-ulx)/2 , (lry-uly)/2)
    return img[uly:lry,ulx:lrx], new_center

def rotate_vector(vector, angle):
    '''rotate vectors'''
    x = vector[0] * np.cos(angle) - vector[1] * -np.sin(angle)
    y = vector[0] * -np.sin(angle) + vector[1] * np.cos(angle)
    return [x, y]    

def rotate_flow_field(flow, angle):
    '''rotate flow field'''
    uv_rot = rotate_vector(flow, angle)
    u = rotate(uv_rot[0], angle * 180/np.pi)
    v = rotate(uv_rot[1], angle * 180/np.pi)
    return [u, v]  

def defect_flow_frame_average(img1,img2, df_frame, defect_type="up", 
                              box=(300,300), filt=1, sigma=15):
    im_h, im_w = img1.shape
    width, height = box[0], box[1]
    width1, height1 = int(width/2**.5), int(height/2**.5)
    flow = cv2.calcOpticalFlowFarneback(img1,img2, None, 0.5, 3, 
        winsize=sigma, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    if filt !=1:
        flow = gaussian_filter(flow, sigma=filt)

    if defect_type=="up":
        df = df_frame[df_frame.fuse_up].copy()
    elif defect_type=="down":
        df = df_frame[~df_frame.fuse_up].copy()   
    else:
        df = df_frame.copy()


    u_frame = np.zeros((height1, width1), dtype=np.float32)
    v_frame = u_frame
    count = 0

    x,y,th = ['xm', 'ym', 'angm1'] if defect_type=="minus" else ['xp', 'yp', 'angp1']

    for i in range(len(df[x])):
        try:
            # center at defect position
            cnt = (int(df[x].iloc[i]), int(df[y].iloc[i]))
            if (cnt[0]>width//2) and (cnt[0]<im_w-width//2) and (cnt[1]>height//2) and (cnt[1]<im_h-height//2):
                #1 crop each component of velocity field
                
                # image_crop = crop(255-img_clahe, cnt, width, height)[0] *** image
                u,_ = crop(flow[:,:,0], cnt, width, height)
                v,_ = crop(flow[:,:,1], cnt, width, height)

                #2 rotate velocity field (1. rotate vectors 2. rotate positions) 
                # image_rot = rotate(image_crop, df["angp1"].iloc[i] * 180/np.pi) *** image
                uv_rot = rotate_flow_field((u,v), df[th].iloc[i])

                #3 crop again to smaller box (box**0.5)
                cnt_crop = uv_rot[0].shape[1]/2, uv_rot[0].shape[0]/2
                # image_rot_crop = crop(image_rot, cnt_crop, width1, height1)[0] *** image
                u_frame = u_frame + crop(uv_rot[0], cnt_crop, width1, height1)[0]
                v_frame = v_frame + crop(uv_rot[1], cnt_crop, width1, height1)[0]
                count += 1 
        except:
            pass
        #      break

    if count:
        print(u_frame.shape[1], u_frame.shape[0])
        return u_frame/count, v_frame/count, count
    

def orienatation_frame_average(img1, df_frame, defect_type="up", 
                              box=(300,300), sigma=11):
    im_h, im_w = img1.shape
    width, height = box[0], box[1]
    width1, height1 = int(width/2**.5), int(height/2**.5)
    ori = analyze_defects(img1, sigma=sigma)[0]


    if defect_type=="up":
        df = df_frame[df_frame.fuse_up].copy()
    elif defect_type=="down":
        df = df_frame[~df_frame.fuse_up].copy()   
    else:
        df = df_frame.copy()


    ori_list = []
    count = 0

    x,y,th = ['xm', 'ym', 'angm1'] if defect_type=="minus" else ['xp', 'yp', 'angp1']

    for i in range(len(df[x])):
        try:
            # center at defect position
            cnt = (int(df[x].iloc[i]), int(df[y].iloc[i]))
            if (cnt[0]>width//2) and (cnt[0]<im_w-width//2) and (cnt[1]>height//2) and (cnt[1]<im_h-height//2):
                #1 crop the orientation field
                crop_ori = crop(ori, cnt, width, height)[0]

                #2 rotate the orientation field (1. rotate the angle) 
                rot_ori = rotate(crop_ori + df[th].iloc[i], df[th].iloc[i]*180/np.pi)

                #3 crop again to smaller box (box**0.5)
                cnt_crop = rot_ori.shape[1]/2, rot_ori.shape[0]/2

                ori_list.append(crop(rot_ori, cnt_crop, width1, height1)[0])
                count += 1 
        except:
            pass
        #      break

    if count:
        return circmean(np.stack(ori_list, axis=-1), axis=-1, low=-np.pi/2, high=np.pi/2), count