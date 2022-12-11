# %%
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
import scipy.io
import os
import sys
import pandas as pd
from PIL import Image

import scipy.io

# self_path = os.path.dirname(__file__)
self_path = os.path.join(os.path.dirname(__file__))
sys.path.insert(1, self_path)

# %matplotlib qt

# %%
# image_list = glob.glob(r"C:\Users\USER\Downloads\B-sub-sur-minus-in-supernatant-40X-100fps\flow_test\*tif")
# image_list = glob.glob(r"C:\Users\USER\Downloads\B-sub-sur-minus-in-supernatant-40X-100fps\B-sub-sur-minus-in-supernatant-40X-100fps(raw)\*tif")
# image_list = glob.glob(r"C:\Users\USER\Downloads\BEER\March 1st 100fps 40X 50-50 5um gap\1_X*.tif")
# image_list = glob.glob(r"C:\Users\USER\Downloads\BEER\March 1st 100fps 40X 50-50 1um gap\1_X*.tif")
image_list = glob.glob(r"D:\work\nematics\nematics\nematics\_bacteria\hbec_images\raw\test_flow\*tif")

from natsort import natsorted
image_list = natsorted(image_list, key=lambda y: y.lower())
# %%
PLOT  = True
SAVE = False

x, y = 0, 0
win1, win2 = 600, 600

for (i,im1), im2 in zip(enumerate(image_list[:-1]),image_list[1:]):

    img1 = cv2.imread(im1)[:,:,0]
    img2 = cv2.imread(im2)[:,:,0]

    # fig, axs = plt.subplots(2,1)
    # axs = axs.flatten()
    # axs[0].imshow(img1, cmap="gray")
    # axs[1].imshow(img2, cmap="gray")


    flow = cv2.calcOpticalFlowFarneback(img1,img2, None, 0.5, 3, 
        winsize=61, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    if PLOT:
        fig = plt.figure(figsize=(15,10))
        step = 15
        plt.imshow(img1, cmap="gray")
        # x, y = np.arange(0, flow.shape[1], step), np.arange(flow.shape[0]-step, -1, -step)
        x = np.arange(0, flow.shape[1], step, dtype=np.int32)
        y = np.arange(0, flow.shape[0], step, dtype=np.int32)
        plt.quiver(x,y, 
                flow[::step, ::step, 0], -flow[::step, ::step, 1], color="red")
        # plt.tight_layout()
        plt.show()
        save_path = os.path.join(os.path.dirname(image_list[i]),'flow_%d.png' % i)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        plt.cla()
    
    if SAVE:
        save_path = os.path.join(
            os.path.dirname(image_list[i]), 
            'OptFlow', 
            os.path.splitext(os.path.basename(image_list[i]))[0] + '.mat'
            )
        # break
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        scipy.io.savemat(save_path, dict(u=flow[:,:,0], v=flow[:,:,1]))
        # "C:\Users\USER\Downloads\B-sub-sur-minus-in-supernatant-40X-100fps\OptFlow\test.mat"


    # break



#%% 
'''CHECK LEFT-RIGHT SHIFT ()'''
image_list = glob.glob(r"C:\Users\USER\Downloads\BEER\March 1st 100fps 40X 50-50 1um gap\1_X*.tif")

from natsort import natsorted
image_list = natsorted(image_list, key=lambda y: y.lower())
image_list = image_list[::10]

methods = ['cv2.TM_CCOEFF']
pad = 200
# w1, h1 = orient_im_left.shape[::-1]
# shift = np.array((w1//2, h1//2)) - np.mean(center,axis=0)
center = []

for (i,im1), im2 in zip(enumerate(image_list[:-1]),image_list[1:]):

    img1 = cv2.imread(im1)[:,:,0]
    img2 = cv2.imread(im2)[:,:,0]    
    flow = cv2.calcOpticalFlowFarneback(img1,img2, None, 0.5, 3, 
        winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        
    # left = flow[:,:900,0]
    # right = flow[:,900:,0]
    left = np.arctan2(flow[:,:900,0],flow[:,:900,0])
    right = np.arctan2(flow[:,900:,0],flow[:,900:,0])

    template = right[pad:-pad,pad:-pad]  
    w, h = template.shape[::-1]  

    res = cv2.matchTemplate(left, template, eval(methods[0]))
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc

    print(top_left[0] + w/2, top_left[1] + h/2) 
    center.append([top_left[0] + w/2, top_left[1] + h/2])
    
    # if i>10:
    #     break  

center = np.array(center)
xc, yc = np.mean(center,axis=0)
w1, h1 = left.shape[::-1]
shift = np.array((w1//2, h1//2)) - np.mean(center,axis=0)

plt.figure(figsize=(5,5))
plt.plot(center[:,0], center[:,1], 'o', alpha=.1)
plt.plot(w1//2, h1//2, '+', markersize=20)
plt.plot(xc, yc, '+', markersize=20)
plt.plot([w1//2,xc], [h1//2,yc], '--')
plt.title("$L-R ~Shift:$ \n($\Delta X,\Delta Y$)=("+ '%.2f' %shift[0]+ ","+ '%.2f' %shift[1]+ ")")
plt.axis([0,w1,0,h1])