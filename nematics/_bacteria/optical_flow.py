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
import pickle 
from natsort import natsorted

import scipy.io

# self_path = os.path.dirname(__file__)
self_path = os.path.join(os.path.dirname(__file__))
sys.path.insert(1, self_path)

# %matplotlib qt
import sys,time,random
def progressBar(count_value, total, suffix=''):
    bar_length = 100
    filled_up_Length = int(round(bar_length* count_value / float(total)))
    percentage = round(100.0 * count_value/float(total),1)
    bar = '=' * filled_up_Length + '-' * (bar_length - filled_up_Length)
    sys.stdout.write('[%s] %s%s ...%s\r' %(bar, percentage, '%', suffix))
    sys.stdout.flush()

def flow_time_series(path_list, start=int(0), last=None, winsize=15):
    if not last:
        last = len(path_list) - start

    flows = []
    for (i,path1), path2 in zip(enumerate(path_list[start:last]), path_list[start+1:last+1]):
        im1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        im2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
        flow = cv2.calcOpticalFlowFarneback(im1, im2, None, 0.5, 3, 
            winsize=winsize, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        flows.append(flow[:, :, 1])

        progressBar(i, last-start)

    return flows

# # %%
# image_list = glob.glob(r"C:\Users\USER\Downloads\B-sub-sur-minus-in-supernatant-40X-100fps\flow_test\*tif")
# image_list = glob.glob(r"C:\Users\USER\Downloads\B-sub-sur-minus-in-supernatant-40X-100fps\B-sub-sur-minus-in-supernatant-40X-100fps(raw)\*tif")
# image_list = glob.glob(r"C:\Users\USER\Downloads\BEER\March 1st 100fps 40X 50-50 5um gap\1_X*.tif")
# image_list = glob.glob(r"C:\Users\USER\Downloads\BEER\March 1st 100fps 40X 50-50 1um gap\1_X*.tif")
image_list = glob.glob(r"C:\Users\victo\Downloads\SB_lab\RPE1_C2C12_\Test\*.tif")

from natsort import natsorted
image_list = natsorted(image_list, key=lambda y: y.lower())
# %%
PLOT  = True
SAVE = not PLOT

x, y = 0, 0
w, h = -1, -1

for (i,im1), im2 in zip(enumerate(image_list[:-1]),image_list[1:]):

    img1 = cv2.imread(im1)[:,:,0]
    img2 = cv2.imread(im2)[:,:,0]

    # fig, axs = plt.subplots(2,1)
    # axs = axs.flatten()
    # axs[0].imshow(img1, cmap="gray")
    # axs[1].imshow(img2, cmap="gray")


    flow = cv2.calcOpticalFlowFarneback(img1,img2, None, 0.5, 3, 
        winsize=21, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
      

    if PLOT:
        save_path = os.path.join(
            os.path.dirname(image_list[i]), 
            'OptFlow', 
            os.path.splitext(os.path.basename(image_list[i]))[0] + '_u_v.pkl'
            )        
        with open(save_path, 'rb') as f:
            loaded_dict = pickle.load(f) 

        fig = plt.figure(figsize=(15,10))
        step = 15
        plt.imshow(img1, cmap="gray")
        # x, y = np.arange(0, flow.shape[1], step), np.arange(flow.shape[0]-step, -1, -step)
        x = np.arange(0, flow.shape[1], step, dtype=np.int16)
        y = np.arange(0, flow.shape[0], step, dtype=np.int16)
        plt.quiver(x,y, 
                flow[::step, ::step, 0], -flow[::step, ::step, 1], color="red")

        # plt.tight_layout()
        fig1 = plt.figure(figsize=(15,10))
        plt.imshow(img2, cmap="gray")
        plt.quiver(x,y, 
                loaded_dict["u"][::step, ::step], -loaded_dict["v"][::step, ::step], color="red")
                # loaded_dict["u"][::step//3, ::step//3], -loaded_dict["v"][::step//3, ::step//3], color="red")
                #  loaded_dict["u"][:300//3:step//3, :300//3:step//3], -loaded_dict["v"][:300//3:step//3, :300//3:step//3], color="red")
 
        plt.show()
        break
    
    if SAVE:
        # save_path = os.path.join(os.path.dirname(image_list[i]),'flow_%d.png' % i)
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # fig.savefig(save_path)
        # plt.cla()        
        save_path = os.path.join(
            os.path.dirname(image_list[i]), 
            'OptFlow', 
            os.path.splitext(os.path.basename(image_list[i]))[0] + '_u_v.pkl'
            )
        # break
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(dict(u=flow[...,0], v=flow[...,1]), f)
        # ===== read =======
        # with open(save_path, 'rb') as f:
        #     loaded_dict = pickle.load(f) 
        #    
        # scipy.io.savemat(save_path, dict(u=flow[:,:,0], v=flow[:,:,1]))
        # "C:\Users\USER\Downloads\B-sub-sur-minus-in-supernatant-40X-100fps\OptFlow\test.mat"

    
    # if i==2:
    #     break

    # break
# %%
import cv2
import glob
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from natsort import natsorted

image_list = glob.glob(r"C:/Users/victo/Downloads/SB_lab/RPE1_C2C12_/Test/*.tif")
image_list = natsorted(image_list, key=lambda y: y.lower())

fig, ax = plt.subplots(figsize=(10,10))

def update(frame):
    ax.clear()
    im1 = image_list[frame]
    im2 = image_list[frame+1]
    img1 = cv2.imread(im1)[:,:,0]
    img2 = cv2.imread(im2)[:,:,0]
    flow = cv2.calcOpticalFlowFarneback(img1,img2, None, 0.5, 3, 
        winsize=21, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)       
    step = 15
    ax.imshow(img1, cmap="gray")
    x = np.arange(0, flow.shape[1], step, dtype=np.int16)
    y = np.arange(0, flow.shape[0], step, dtype=np.int16)
    return ax.quiver(x, y, flow[::step, ::step, 0], -flow[::step, ::step, 1], color="red")

anim = FuncAnimation(fig, update, frames=len(image_list)-1, interval=100)
anim.save(r"C:/Users/victo/Downloads/SB_lab/RPE1_C2C12_/test.gif") 
plt.show()




# #%% 
# # '''CHECK LEFT-RIGHT SHIFT ()'''
# # image_list = glob.glob(r"C:\Users\USER\Downloads\BEER\March 1st 100fps 40X 50-50 1um gap\1_X*.tif")

# # from natsort import natsorted
# # image_list = natsorted(image_list, key=lambda y: y.lower())
# # image_list = image_list[::10]

# # methods = ['cv2.TM_CCOEFF']
# # pad = 200
# # # w1, h1 = orient_im_left.shape[::-1]
# # # shift = np.array((w1//2, h1//2)) - np.mean(center,axis=0)
# # center = []

# # for (i,im1), im2 in zip(enumerate(image_list[:-1]),image_list[1:]):

# #     img1 = cv2.imread(im1)[:,:,0]
# #     img2 = cv2.imread(im2)[:,:,0]    
# #     flow = cv2.calcOpticalFlowFarneback(img1,img2, None, 0.5, 3, 
# #         winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        
# #     # left = flow[:,:900,0]
# #     # right = flow[:,900:,0]
# #     left = np.arctan2(flow[:,:900,0],flow[:,:900,0])
# #     right = np.arctan2(flow[:,900:,0],flow[:,900:,0])

# #     template = right[pad:-pad,pad:-pad]  
# #     w, h = template.shape[::-1]  

# #     res = cv2.matchTemplate(left, template, eval(methods[0]))
# #     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
# #     top_left = max_loc

# #     print(top_left[0] + w/2, top_left[1] + h/2) 
# #     center.append([top_left[0] + w/2, top_left[1] + h/2])
    
# #     # if i>10:
# #     #     break  

# # center = np.array(center)
# # xc, yc = np.mean(center,axis=0)
# # w1, h1 = left.shape[::-1]
# # shift = np.array((w1//2, h1//2)) - np.mean(center,axis=0)

# # plt.figure(figsize=(5,5))
# # plt.plot(center[:,0], center[:,1], 'o', alpha=.1)
# # plt.plot(w1//2, h1//2, '+', markersize=20)
# # plt.plot(xc, yc, '+', markersize=20)
# # plt.plot([w1//2,xc], [h1//2,yc], '--')
# # plt.title("$L-R ~Shift:$ \n($\Delta X,\Delta Y$)=("+ '%.2f' %shift[0]+ ","+ '%.2f' %shift[1]+ ")")
# # plt.axis([0,w1,0,h1])

# # %%

from skimage import io
# im = io.imread(r"C:\Users\victo\Downloads\optical_flow.tif")
# im = io.imread(r"C:\Users\victo\Downloads\SB_lab\HT1080\26_07_2018_HT1080_10x_15min_1_s8_4.tif")
# im = io.imread(r"C:\Users\victo\Downloads\HT1080\27072018_TIFF\s29_3.tif")
im = io.imread(r"C:\Users\victo\Downloads\SB_lab\TestFlow\rotations.tif")[:,:-100, :]
# %%
# %matplotlib qt
xi = int(0)
xj = int (-1)
im_num = im.shape[0]-1 #9
colors = plt.cm.jet(np.linspace(0, 1, num=im_num))

fig = plt.figure(figsize=(10,10))
plt.imshow(im[0], cmap="gray")
step = 10
flows = []
for i in range(im_num):
    flow = cv2.calcOpticalFlowFarneback(im[i,xi:xj,xi:xj],im[i+1,xi:xj,xi:xj], None, 0.5, 3, 
        winsize=10, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)      
    flows.append(flow[:, :, 1])  

    # 
    # x, y = np.arange(0, flow.shape[1], step), np.arange(flow.shape[0]-step, -1, -step)
    x = np.arange(0, flow.shape[1], step, dtype=np.int32)
    y = np.arange(0, flow.shape[0], step, dtype=np.int32)
    # fig = plt.figure(figsize=(15,15))
    plt.imshow(im[i], cmap="gray")
    plt.quiver(x,y, 
            flow[::step, ::step, 0], -flow[::step, ::step, 1],  scale=150, width=0.01, 
            color=colors[-1])
            # color=colors[i], alpha=.6)
    
    # if i>10:
    #      break
    # break
    save_path = os.path.join(r"C:\Users\victo\Downloads\SB_lab\TestFlow\rotations",'flow_%d.png' % i)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.cla()

    save_path = os.path.join(
        r"C:\Users\victo\Downloads\SB_lab\TestFlow\rotations",
        str(i) + '_u_v.pkl'
        )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)    
    with open(save_path, 'wb') as f:
        pickle.dump({'u': flow[:,:,0], 'v': flow[:,:,1]}, f)

    
plt.tight_layout()
plt.show()
# %%
vel_list = glob.glob(r"C:\Users\victo\Downloads\SB_lab\TestFlow\rotations" + "/*.pkl")
vel_list = natsorted(vel_list, key=lambda y: y.lower())

with open(vel_list[11], 'rb') as f:
    loaded_dict = pickle.load(f) 

u = loaded_dict["u"]
v = -np.flipud(loaded_dict["v"])
from scipy.signal import convolve2d

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')

u_x, u_y = np.gradient(u)
v_x, v_y = np.gradient(v)
vorticity = (u_x - v_y)
filtN = 11
filt = np.outer(np.exp(np.linspace(-1, 1, filtN)), np.exp(np.linspace(-1, 1, filtN)))
filt = filt / filt.sum()
u1 = convolve2d(vorticity, filt, mode='same')
Xu, Yu = np.meshgrid(np.arange(1, u1.shape[1]+1), np.arange(1, u1.shape[0]+1))
ax.plot_surface(Xu, Yu, u1, cmap='jet', rstride=1, cstride=1)
ax.view_init(elev=90., azim=-90.)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Vorticity')


plt.figure()
plt.imshow(u1, cmap="seismic") #, vmin=u1.min()/10, vmax=u1.max()/10)
plt.figure()
plt.imshow(u, cmap="gray"); 
plt.figure()
plt.imshow(v, cmap="gray")
step = 10
x = np.arange(0, u.shape[1], step, dtype=np.int32)
y = np.arange(0, u.shape[0], step, dtype=np.int32)
plt.quiver(x, y, 
    u[::step, ::step], v[::step, ::step],  
    scale=150, color='red')



# # %%
# from skimage import io
# # im = io.imread(r"C:\Users\victo\Downloads\optical_flow.tif")
# im = io.imread(r"C:\Users\victo\Downloads\SB_lab\HT1080\26_07_2018_HT1080_10x_15min_1_s8_4.tif")
# # im = io.imread(r"C:\Users\victo\Downloads\HT1080\27072018_TIFF\s29_3.tif")
# # im = io.imread(r"C:\Users\victo\Downloads\HT1080\27072018_TIFF\s29_3.tif")

# xi = int(0)
# xj = int (-1)

# flows = []
# for i in range(im_num):
#     flow = cv2.calcOpticalFlowFarneback(im[i,xi:xj,xi:xj],im[i+1,xi:xj,xi:xj], None, 0.5, 3, 
#         winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)      
#     flows.append(flow[:, :, 1])  

# # %%
# im_list = glob.glob(r"C:\Users\victo\Downloads\HT1080\26_07_2018_HT1080_stripes_20_600\HT1080_10x_15min_1_s9_*")
# im_list = natsorted(im_list, key=lambda y: y.lower())
# plt.imshow(cv2.imread(im_list[200]), "gray")  
# # %%
# start = 100
# last = start + 80
# flows = []
# for path1, path2 in zip(im_list[start:last], im_list[start+1:last+1]):
#     im1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
#     im2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
#     flow = cv2.calcOpticalFlowFarneback(im1, im2, None, 0.5, 3, 
#         winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
#     flows.append(flow[:, :, 1]) 
# # %%
# dt = 50
# # fig, axs  = plt.subplots(1,im_num//dt, figsize=(20, 5))
# fig, axs  = plt.subplots(1,2)#, figsize=(20, 5))
# axs = axs.ravel()

# for i,ax in enumerate(axs):
#     first = int(dt * i)
#     last = int(first + dt)
#     mean_flow = np.mean(np.array(flows)[first:last,:,:], axis=0) - np.mean(np.array(flows)[first:last,:,:])
#     q = ax.imshow(mean_flow, cmap="seismic", vmin=-10, vmax=10)
#     # Add colorbar
#     cbar = fig.colorbar(q, ax=ax)
#     ax.axis("off")
#     ax.set_title("$frames:~$"+str(first)+"-"+str(last))
#     # break
    
# # %%
# # Loop through the rest of the images and compute auto-correlation
# corr = []
# for i in range(0, 49):
#     # vel = 
#     corr.append(np.mean(flows[i] * flows[0])/ np.mean(flows[0] * flows[0]))
#     print(i)

# # Plot the correlation matrix
# plt.figure()
# plt.plot(corr,"--o")
# plt.title('Auto-Correlation of Image Sequence')

# %%
folder = r"C:\Users\victo\Downloads\HT1080\26_07_2018_HT1080_stripes_20_600"
EXPERIMENTS = {
            # "2": (0,60),
            # "3": (30,90),
            # "8": (70,150),
            # "9": (100,160), 
            "11": (100,160),               
            # "12": (140,200),          
            "13": (100,200), 
            "14": (150,250),
            "15": (100,200), 
            "16": (100,180),
                }

for exp_num, frame_range in EXPERIMENTS.items():
    print(exp_num, frame_range)

    im_list = glob.glob(folder + "\HT1080_10x_15min_1_s" + exp_num + "_t*")

    # im_list = glob.glob(r"C:\Users\victo\Downloads\HT1080\26_07_2018_HT1080_stripes_20_600\HT1080_10x_15min_1_s9_*")
    im_list = natsorted(im_list, key=lambda y: y.lower())

    # dt = 10
    # start = 100
    # last = start + dt
    start, last = frame_range

    flows = flow_time_series(im_list, start=start, last=last)

    # fig, axs  = plt.subplots(1,im_num//dt, figsize=(20, 5))
    fig, axs  = plt.subplots(figsize=(13, 10))

    mean_flow = np.mean(np.array(flows[:]), axis=0)# - np.mean(np.array(flows))
    mean_flow = mean_flow *.74 * 4
    q = axs.imshow(mean_flow, cmap="seismic", vmin=-30, vmax=30)
    # Add colorbar
    cbar = fig.colorbar(q, ax=axs)
    cbar.set_label('$\mu m~/~hr$', fontsize=20)
    axs.axis("off")
    axs.set_title("$average ~over~$"+str((last-start)/4)+"$ ~hours$", fontsize=20)


    plt.savefig(r"C:\Users\victo\Downloads/" + os.path.basename(folder) + "__s" + exp_num +".png", format="png")
    break

# %% OPTION 1
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
%matplotlib qt

def divergence_npgrad(flow):
    # flow = np.swapaxes(flow, 0, 1)
    Fx, Fy = flow[:, :, 0], flow[:, :, 1]
    dFx_dx = np.gradient(Fx, axis=0)
    dFy_dy = np.gradient(Fy, axis=1)
    return dFx_dx + dFy_dy

def curl_npgrad(flow):
    # flow = np.swapaxes(flow, 0, 1)
    Fx, Fy = flow[:, :, 0], flow[:, :, 1]
    dFx_dy = np.gradient(Fx, axis=1)
    dFy_dx = np.gradient(Fy, axis=0)
    curl = dFy_dx - dFx_dy
    return curl

def smooth_field(f, filter_size=11):
    filter_size = 11
    filt = np.outer(np.exp(np.linspace(-1, 1, filter_size)), np.exp(np.linspace(-1, 1, filter_size)))
    filt = filt / filt.sum()
    return convolve2d(f, filt, mode='same')

y, x = np.mgrid[0:150, 0:300]
u = np.cos((x + y)/10) #* np.random.rand(*x.shape)
v = np.sin((x - y)/10) #* np.random.rand(*x.shape)
field = np.stack((u, v), axis=-1)
divergence = smooth_field(divergence_npgrad(field), filter_size=10)
vorticity = smooth_field(curl_npgrad(field), filter_size=10)

plt.figure()
step = 5
plt.quiver(x[::step, ::step], y[::step, ::step], field[::step, ::step, 0], field[::step, ::step, 1])
plt.imshow(divergence, cmap="jet")
plt.title("$Divergence$")
plt.colorbar()

plt.figure()
plt.quiver(x[::step, ::step], y[::step, ::step], field[::step, ::step, 0], field[::step, ::step, 1])
plt.imshow(vorticity, cmap="seismic")
plt.title("$Curl$")
plt.colorbar()

#  Varying line width along a streamline
speed = np.sqrt(u**2 + v**2)
lw = 3*speed / speed.max()
step = 5
plt.figure()
plt.quiver(x[::step, ::step], y[::step, ::step], field[::step, ::step, 0], field[::step, ::step, 1], color="k")#, scale=30
plt.streamplot(x, y, field[..., 0], -field[..., 1], density=3., color='white', linewidth=lw)
plt.imshow(divergence, cmap="jet")
plt.colorbar()
# %% OPTION 2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
%matplotlib qt

def divergence_npgrad(flow):
    flow = np.swapaxes(flow, 0, 1)
    Fx, Fy = flow[:, :, 0], flow[:, :, 1]
    dFx_dx = np.gradient(Fx, axis=0)
    dFy_dy = np.gradient(Fy, axis=1)
    return dFx_dx + dFy_dy

def curl_npgrad(flow):
    flow = np.swapaxes(flow, 0, 1)
    Fx, Fy = flow[:, :, 0], flow[:, :, 1]
    dFx_dy = np.gradient(Fx, axis=1)
    dFy_dx = np.gradient(Fy, axis=0)
    curl = dFy_dx - dFx_dy
    return curl

def smooth_field(f, filter_size=11):
    filter_size = 11
    filt = np.outer(np.exp(np.linspace(-1, 1, filter_size)), np.exp(np.linspace(-1, 1, filter_size)))
    filt = filt / filt.sum()
    return convolve2d(f, filt, mode='same')

y, x = np.mgrid[0:150, 0:300]
u = np.cos((x + y)/10) #* np.random.rand(*x.shape)
v = np.sin((x - y)/10) #* np.random.rand(*x.shape)
field = np.stack((u, v), axis=-1)
divergence = smooth_field(divergence_npgrad(field), filter_size=10)
vorticity = smooth_field(curl_npgrad(field), filter_size=10)

plt.figure()
step = 5
plt.quiver(x[::step, ::step], y[::step, ::step], field[::step, ::step, 0], -field[::step, ::step, 1])
plt.imshow(divergence.T, cmap="jet")
plt.title("$Divergence$")
plt.colorbar()

plt.figure()
plt.quiver(x[::step, ::step], y[::step, ::step], field[::step, ::step, 0], -field[::step, ::step, 1])
plt.imshow(vorticity.T, cmap="seismic")
plt.title("$Curl$")
plt.colorbar()

#  Varying line width along a streamline
speed = np.sqrt(u**2 + v**2)
lw = 3*speed / speed.max()
step = 5
plt.figure()
plt.quiver(x[::step, ::step], y[::step, ::step], field[::step, ::step, 0], -field[::step, ::step, 1], color="k")#, scale=30
plt.streamplot(x, y, field[..., 0], field[..., 1], density=3., color='white', linewidth=lw)
plt.imshow(divergence.T, cmap="jet")
plt.colorbar()

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib qt

def divergence_npgrad(flow):
    flow = np.swapaxes(flow, 0, 1)
    Fx, Fy = flow[:, :, 0], flow[:, :, 1]
    dFx_dx = np.gradient(Fx, axis=0)
    dFy_dy = np.gradient(Fy, axis=1)
    return dFx_dx + dFy_dy

def curl_npgrad(flow):
    flow = np.swapaxes(flow, 0, 1)
    Fx, Fy = flow[:, :, 0], flow[:, :, 1]
    dFx_dy = np.gradient(Fx, axis=1)
    dFy_dx = np.gradient(Fy, axis=0)
    curl = dFy_dx - dFx_dy
    return curl

def smooth_field(f, filter_size=11):
    filter_size = 11
    filt = np.outer(np.exp(np.linspace(-1, 1, filter_size)), np.exp(np.linspace(-1, 1, filter_size)))
    filt = filt / filt.sum()
    return convolve2d(f, filt, mode='same')

# Load the images
from skimage import io
im = io.imread(r"C:\Users\victo\Downloads\SB_lab\TestFlow\rotations.tif")[:,:-100, :]
im_num = 10
im1 = im[im_num]
im2 = im[im_num+1]

# Calculate the optical flow using Lucas-Kanade method
flow = cv2.calcOpticalFlowFarneback(im1,im2, None, 0.5, 3, 
        winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)  
u, v = flow[:,:,0], flow[:,:,1]
y, x = np.mgrid[0:im1.shape[0], 0:im1.shape[1]]
field = np.stack((u, v), axis=-1)
divergence = smooth_field(divergence_npgrad(field), filter_size=50)
vorticity = smooth_field(curl_npgrad(field), filter_size=3)

plt.figure()
step = 10
speed = np.sqrt(u**2 + v**2)
lw = 3*speed / speed.max()
plt.quiver(x[::step, ::step], y[::step, ::step], field[::step, ::step, 0], -field[::step, ::step, 1])
plt.streamplot(x, y, field[..., 0], field[..., 1], density=3., color='k', linewidth=lw)
plt.imshow(vorticity.T, cmap="jet", vmin=-.2, vmax=.2)
plt.title("$vorticity$")
plt.colorbar()
# %%
flow = np.load(r"C:\Users\victo\Downloads\SB_lab\HBEC\KenTest_s3\velocity\velocity_from_Trans__300.npy")
print(
    flow[...,0].mean(),
    flow[...,1].mean(),"\n",
    flow[...,0].max(),
    flow[...,1].max(),"\n",
    flow[...,0].min(),
    flow[...,1].min(),

)


# %%
