# %%
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt
import cv2
import os
import glob
import scipy.io
import os
import sys
from sklearn.neighbors import KDTree, NearestNeighbors
from scipy.spatial.distance import cdist
import pandas as pd
from scipy.spatial import distance
# %matplotlib qt

def mean_square_displacement(x,y):
    r = np.sqrt(x**2 + y**2)
    diff = np.diff(r)
    diff_sq = diff**2
    return np.mean(diff_sq)

# def slope()
# %%
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

DURATION_MINIMUM = 10
folder = r"C:\Users\USER\Downloads\BEER\March 1st 100fps 40X 50-50 5um gap/"
spots = pd.read_csv(folder + "TrackMate/n_def_spots.csv", skiprows=[1,2,3])
tracks_all = pd.read_csv(folder + "TrackMate/n_def_tracks.csv", skiprows=[1,2,3])
# spots = pd.read_csv(r"C:\Users\USER\Downloads\B-sub-sur-minus-in-supernatant-40X-100fps\TrackMate1-1\spots_p.csv", skiprows=[1,2,3])
# tracks_all = pd.read_csv(r"C:\Users\USER\Downloads\B-sub-sur-minus-in-supernatant-40X-100fps\TrackMate1-1\tracks_p.csv", skiprows=[1,2,3])
tracks_to_remove = tracks_all["TRACK_ID"][tracks_all["TRACK_DURATION"]<10].unique()
tracks_ids = tracks_all["TRACK_ID"][tracks_all["TRACK_DURATION"]>=DURATION_MINIMUM].unique()

for t_id in tracks_to_remove:
    # spots["TRACK_ID"][spots["TRACK_ID"]==t_id] = np.nan 
    spots.loc[spots["TRACK_ID"]==t_id, "TRACK_ID"] = np.nan 
    tracks_all.loc[tracks_all["TRACK_ID"]==t_id, "TRACK_ID"] = np.nan 

spots = spots.dropna(subset=["TRACK_ID"])
tracks_all = tracks_all.dropna(subset=["TRACK_ID"])

# check size 
print(len(spots["TRACK_ID"].unique()))
print(len(tracks_ids))

frames = []
msd = []
handness = []
phi = []
fig, axs = plt.subplots(1,1, figsize=(5,5))
for i,t_id in enumerate(tracks_ids[:]):
    # if i>0 and i<1000:
    idx = spots["TRACK_ID"]==t_id
    frame_idx = spots["FRAME"][spots["TRACK_ID"]==t_id].argsort()
    frame_num = spots["FRAME"][idx].iloc[frame_idx]
    frames.append(len(frame_num))
    x,y = spots["POSITION_X"][idx].iloc[frame_idx], spots["POSITION_Y"][idx].iloc[frame_idx]
    if x.max()<800:
        # print('>>', x.min(), x.max(), np.any(x)>400)
        # continue
        x0,y0 = spots["POSITION_X"][idx].iloc[frame_idx.iloc[0]], spots["POSITION_Y"][idx].iloc[frame_idx.iloc[0]]
        axs.plot(
            x-x0,
            y-y0,
            # frame_num - frame_num.iloc[0],
            # np.sqrt((x-x0)**2 + (y-y0)**2),
            "-X", alpha=.3, linewidth=3, color="orange", 
            )
        msd.append(mean_square_displacement(x,y))
        xx = [1.]
        # res_lsq = least_squares(linear_fit, xx, loss='soft_l1', f_scale=0.001,
                    # args=(ratio, sysmex*1000))
        
        # check trajectory handness
        # collect displacements np.diff(x), np.diff(y)
        a = np.vstack((np.diff(x),np.diff(y))).T
        # cross product is handness +/-
        # cross = segmentA X segmentB
        # cross = ||segmentA|| ||segmentB|| sin(Theta)
        # norm_cross = cross/sqrt(||segmentA||^2 + ||segmentB||^2)
        v1 = a[:-1]/np.linalg.norm(a[:-1], axis=1).reshape(-1,1)
        v2 = a[1:]/np.linalg.norm(a[1:], axis=1).reshape(-1,1)
        cross = np.cross(v1, v2) 
        handness.append(np.arcsin(np.mean(cross))/len(cross))

        # show trajectory handness example
        PLOT_TRAJ = False
        if PLOT_TRAJ:
            v1 = a[:-1]/np.linalg.norm(a[:-1], axis=1).reshape(-1,1)
            v2 = a[1:]/np.linalg.norm(a[1:], axis=1).reshape(-1,1)
            cross = np.cross(v1, v2)
            for xi,yi, c in zip(x[1:]-x0, y[1:]-y0, cross):            
                if c<-.1:
                    axs.plot(xi,yi, marker='$R$', c="red")
                    axs.text(xi+1,yi, "{:.1f}".format(np.arcsin(c)*180/np.pi), color="red")
                elif c>.1:
                    axs.plot(xi,yi, marker='$L$', c="blue")
                    axs.text(xi,yi, "{:.1f}".format(np.arcsin(c)*180/np.pi), color="blue")
  
            plt.title("$Mean ~Cross ~Prod.~$"+ "{:.3f}".format(np.nanmean(cross)))
            plt.grid(True)
            break

        # plt.figure()
        # r, rho = cart2pol(x-x0,y-y0)
        # plt.plot(r[1:], np.diff(rho)*180/np.pi, "--D")
        # break
        r, rho = cart2pol(x-x0,y-y0)
        phi.append(np.diff(rho)*180/np.pi)

phi = [item for sublist in phi for item in sublist]
print("mean (deg) >>", np.mean(phi))
# plt.plot(np.diff(x),np.diff(y), "o")
# for xi,yi,p in zip(np.diff(x),np.diff(y),phi[1:]):
#     plt.text(xi,yi, "{:.2f}".format(p*180/np.pi))


plt.xlabel("$X ~(px)$", fontsize=16)
plt.ylabel("$Y ~(px)$", fontsize=16)
plt.gca().set_aspect('equal', 'box')

plt.tight_layout()


plt.figure(figsize=(6,4))
plt.hist(np.array(handness)*180/np.pi, 30, range=(-4.5,4.5), rwidth=.9, 
    label="$N_{trajectories}=$"+"{}\n $Bias=${:.2f}$^o/frame$".format(
        len(handness), np.nanmean(handness)*180/np.pi
        ), alpha=.8)
plt.legend(loc="best", fontsize=10)
plt.xlabel(r"$\theta~(deg^o)=sin^{-1}(\frac{\Delta D_t\ ~\times ~\Delta D_{t+1}}{||D_t||~||D_{t+1}||})$", fontsize=20)
plt.ylabel("$Number ~of ~Tracks$\n $~(>$"+str(DURATION_MINIMUM)+"$ ~frames)$", fontsize=16)
plt.title("$Mean ~angle ~bias.~$"+ "{:.2f}$^o$ $\pm${:.2f}$^o$ $~per ~step$".format(
    np.nanmean(handness)*180/np.pi, np.nanstd(handness)*180/np.pi/len(handness)**.5))
plt.tight_layout()
# plt.figure(figsize=(5,5))        
# plt.hist(msd, 20, rwidth=.9)
# %%
'''
https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
    plt.suptitle(meth)
'''

# TODO: check cross correlation of nematic fields
import glob
import cv2
# take maximum
#'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 
# If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
# 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
methods = ['cv2.TM_CCOEFF']

pad = 200

im_path_list = glob.glob(r"C:\Users\USER\Downloads\BEER\March 1st 100fps 40X 50-50 5um gap\orient\*.tif")

orient_im =  cv2.imread(im_path_list[0], -1)
# plt.imread(im_path_list[0])[:,:,0].astype(np.uint16) 
orient_im_left = (np.cos(orient_im[:,:900])*255).astype(np.uint8)
# orient_im_left -= orient_im_left.mean()

orient_im_right = (np.cos(orient_im[:,900:])*255).astype(np.uint8) 
# orient_im_right -= orient_im_right.mean()

# corr = signal.correlate2d(orient_im_left, orient_im_right, boundary='symm', mode='same')
# res = cv2.matchTemplate(orient_im_left, orient_im_right[100:-100,100:-100], cv2.TM_CCOEFF)
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
center = []
for im_path in im_path_list[::10]:
    orient_im =  cv2.imread(im_path, -1)
    orient_im_left = (np.cos(orient_im[:,:900])*255).astype(np.uint8)
    orient_im_right = (np.cos(orient_im[:,900:])*255).astype(np.uint8)
    
    template = orient_im_right[pad:-pad,pad:-pad]  
    w, h = template.shape[::-1]  

    res = cv2.matchTemplate(orient_im_left, template, eval(methods[0]))
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc

    print(top_left[0] + w/2, top_left[1] + h/2) 
    center.append([top_left[0] + w/2, top_left[1] + h/2])

center = np.array(center)
xc, yc = np.mean(center,axis=0)
w1, h1 = orient_im_left.shape[::-1]
shift = np.array((w1//2, h1//2)) - np.mean(center,axis=0)

plt.figure(figsize=(5,5))
plt.plot(center[:,0], center[:,1], 'o', alpha=.1)
plt.plot(w1//2, h1//2, '+', markersize=20)
plt.plot(xc, yc, '+', markersize=20)
plt.plot([w1//2,xc], [h1//2,yc], '--')
plt.title("$L-R ~Shift:$ \n($\Delta X,\Delta Y$)=("+ '%.2f' %shift[0]+ ","+ '%.2f' %shift[1]+ ")")
plt.axis([0,w1,0,h1])

# %% check orientation alignment after shift
quiveropts = dict(color='red', headlength=0, pivot='middle', scale=1,
    linewidth=.0001, width=.0005, headwidth=.10
    ) # common options
frame = 50    
plt_dim = 8
step = 20
# f, ax = plt.subplots(1,1, sharex=True, sharey=True)

plt.figure(figsize=(plt_dim,plt_dim))
for im_path in im_path_list[frame:frame+10]:
    orient_im =  cv2.imread(im_path, -1)
    left = orient_im[:,:900]
    right = orient_im[:,900:]
    corr0 = cv2.matchTemplate(left, right, eval(methods[0]))
    right = np.roll(right, int(shift[0]), axis=0)
    right = np.roll(right, int(shift[1]), axis=1)
    corr1 = cv2.matchTemplate(left, right, eval(methods[0]))
    print(corr1-corr0)

    x = np.arange(0, left.shape[1], step, dtype=np.int32)
    y = np.arange(0, left.shape[0], step, dtype=np.int32)
    plt.quiver(x,y, 
            np.cos(left[::step, ::step]), np.sin(left[::step, ::step]), 
            color='red')
    plt.quiver(x,y, 
            np.cos(right[::step, ::step]), np.sin(right[::step, ::step]), 
            color='green')
    # TODO check correff value
# %%
scale = 400/900
plt.figure(figsize=(5,5))
plt.axis([0,400,0,400])
for t in range(30):
    x = spots["POSITION_X"][spots["FRAME"]==t]
    y = spots["POSITION_Y"][spots["FRAME"]==t]
    xl, yl = x[x<400], y[x<400]
    xr, yr = x[x>400]-400, y[x>400]
    xr1, yr1 = x[x>400]-400 - scale*shift[0], y[x>400] - scale*shift[1]

    plt.plot(xl, yl, "or", alpha=.4)
    plt.plot(xr, yr, "Xb", alpha=.4)
    plt.plot(xr1, yr1, "Xg", alpha=.4)



# %%
max_num = 50
colors = plt.cm.jet(np.linspace(0,1,len(tracks_ids)))
# colors = plt.cm.jet(np.linspace(0,1,max_num))
fig, axs = plt.subplots(1,1, figsize=(5,5))
frame_num = sorted(spots["FRAME"].unique())

for i,t_id in enumerate(tracks_ids):
    idx = spots["TRACK_ID"]==t_id
    frame_idx = spots["FRAME"][spots["TRACK_ID"]==t_id].argsort()
    frame_num = spots["FRAME"][idx].iloc[frame_idx]   

    if frame_num.min()<50:
        px,py = spots["POSITION_X"][idx].iloc[frame_idx], spots["POSITION_Y"][idx].iloc[frame_idx]
        pxl = px[px<400]
        pyl = py[px<400]
        pxr = px[px>400] - 400
        pyr = py[px>400]
        axs.plot(pxl,pyl,"-", c=colors[i], alpha=.3)
        axs.plot(pxr,pyr,"--+", c=colors[i], alpha=.3)


    # if i>max_num:
    #     break

        
    # 



# %%
mat = scipy.io.loadmat(r"C:\Users\USER\Downloads\BEER\March 1st 100fps 40X 50-50 1um gap\SUMMARY.mat")
for i,t in enumerate(mat["defNum"]):
    print(t[0].shape)
    if len(t[0])>2 and i==230:
        px,py,pteta = t[0],t[1],t[2]
        # px[px>900] = px[px>900] - 900
        pxl = px[px<900]
        pyl = py[px<900]
        pxr = px[px>900] - 900
        pyr = py[px>900]

        nx,ny,nteta = t[3],t[4],t[5]
        nx[nx>900] = nx[nx>900] - 900
        
        print(i)
        break
    
plt.figure()    
plt.plot(pxl,pyl,"P")
plt.plot(pxr,pyr,"P")
# plt.plot(nx,ny,"X")
plt.title("frame: "+ str(i))    
# mat["defNum"][-1][2].shape
# mat.keys()
# x, y = np.unique(mat["x"][0][0]), np.unique(mat["y"][0][0])
plt.axes().set_aspect('equal')




# %%
ff = 0
for i,t_id in enumerate(tracks_ids):
    idx = spots["TRACK_ID"]==t_id
    frame_idx = spots["FRAME"][spots["TRACK_ID"]==t_id].argsort()
    frame_num = spots["FRAME"][idx].iloc[frame_idx]
    frame_num = frame_num - frame_num.iloc[0]
    # print(np.diff(frame_num))
    if ff<frame_num.max():
        ff = frame_num.max()

    # x,y = spots["POSITION_X"][idx].iloc[frame_idx], spots["POSITION_Y"][idx].iloc[frame_idx]

# %%



# %%
sc = 2048/1153
tracks = pd.read_csv(r"C:\Users\USER\Downloads\Trinish\mov1\Track statistics_all3.csv")
spots = pd.read_csv(r"C:\Users\USER\Downloads\Trinish\mov1\Spots in tracks statistics_all3.csv")
multilayer = pd.read_excel(r"C:\Users\USER\Downloads\POV1mul.xlsx").reset_index()


plt.figure(figsize=(10, 10))
plt.imshow(cv2.imread(r"C:\Users\USER\Downloads\mov1.tif"))


START_FRAME = 12 *4
DURATION = 15
tracks1 = tracks[np.logical_and.reduce((
    # tracks["TRACK_DURATION"]<15,
    tracks["TRACK_DURATION"]>DURATION,
    # tracks["TRACK_START"]>START_FRAME, 
    # tracks["TRACK_START"]<START_FRAME + 8 *4 - DURATION, 
    tracks["TRACK_STOP"]>START_FRAME+DURATION,
    # tracks["TRACK_DURATION"]<200
))]
c = sc*4*1.8 * (tracks1["TRACK_DISPLACEMENT"]/tracks1["TRACK_DURATION"])
stop_frame = tracks1["TRACK_STOP"]/4
plt.plot(stop_frame,c,"o")
plt.xlabel("$hours$")
plt.ylabel("$speed$")
plt.gca().set_aspect('equal', 'box')
# TRACK_MEAN_SPEED, MEAN_STRAIGHT_LINE_SPEED
idx = c<1000

# track_ids = tracks1["TRACK_ID"]
# spot_last_x = []
# spot_last_y = []
# spot_x = []
# spot_y = []
# for tid in track_ids:
#     spot_last_x.append(spots["POSITION_X"][spots["TRACK_ID"]==tid].iloc[-1])
#     spot_last_y.append(spots["POSITION_Y"][spots["TRACK_ID"]==tid].iloc[-1])
#     spot_x.append(spots["POSITION_X"][spots["TRACK_ID"]==tid])
#     spot_y.append(spots["POSITION_Y"][spots["TRACK_ID"]==tid])    


# plt.scatter(sc* np.array(spot_last_x), sc* np.array(spot_last_y), c=c, cmap="jet", )
# plt.scatter(tracks1["TRACK_DISPLACEMENT"]/tracks1["TRACK_DURATION"], tracks1["TRACK_DURATION"])
plt.scatter(multilayer["X"], multilayer["Y"], s=1000, color="red", alpha=.2)
plt.scatter(sc* tracks1["TRACK_X_LOCATION"][idx], sc* tracks1["TRACK_Y_LOCATION"][idx], s=100, marker='^', c=c[idx], cmap="jet", )


plt.axis([0,2048,0,2048])
plt.gca().set_aspect('equal', 'box')
cbar = plt.colorbar()
cbar.set_label('$Mean~Dispalcement$')
plt.gca().invert_yaxis()
plt.show()

# for x,y, vel in zip(tracks1["TRACK_X_LOCATION"], tracks1["TRACK_Y_LOCATION"], c):
#     if vel<.4:
#         print("*")
#         plt.text(int(sc*x), int(sc*y), str([int(sc*x), int(sc*y)]),  alpha=.8)
#         # plt.scatter(int(sc*x), int(sc*y),  alpha=.8)

plt.figure()
plt.hist(c, rwidth=.9)
plt.title(["#",len(c),"| V=",int(np.nanmean(c))])
# %%
import matplotlib.cm as cm
colors = cm.rainbow(np.linspace(0, 1, len(track_ids)))

idx_fast = c>6
idx_slow = c<6
spot_sx, spot_sy = np.array(spot_x)[idx_fast], np.array(spot_y)[idx_fast]
# for x,y in zip(spot_sx, spot_sy):
#     plt.plot(sc*1.8*(x-x.iloc[0]),sc*1.8*(y-y.iloc[0]), alpha=.3)

spot_sx, spot_sy = np.array(spot_x)[idx_slow], np.array(spot_y)[idx_slow]
for x,y in zip(spot_sx, spot_sy):
    plt.plot(sc*1.8*(x-x.iloc[0]),sc*1.8*(y-y.iloc[0]), alpha=.3)
    # if count>5:
    #     break
plt.title("$All$")
plt.gca().set_aspect('equal', 'box')
plt.show()

# %%
xi = sc* tracks1["TRACK_X_LOCATION"][idx] #sc* np.array(spot_last_x)[idx]
yi = sc* tracks1["TRACK_Y_LOCATION"][idx] # sc* np.array(spot_last_y)[idx]
# xi = sc* np.array(spot_last_x)[idx]
# yi = sc* np.array(spot_last_y)[idx]
xyi = np.vstack((xi,yi)).T

xj = np.array(multilayer["X"])
yj = np.array(multilayer["Y"])
xyj = np.vstack((xj,yj)).T
# Distance between the array and itself
dists = cdist(xyi, xyj)
dists.sort()
print(dists[:,0])

cthresh = 6
plt.figure()
# plt.hist(c, rwidth=.8, alpha=.5, bins=np.arange(0,5,.1))
x1, bins1, p1 = plt.hist(sc* dists[:,0][c<cthresh], rwidth=.8, alpha=.5, bins=np.array([0., 150, 500]))
# x2, _, p2 = plt.hist(sc*dists[:,0][c>cthresh], rwidth=.8, alpha=.5, density=True, bins=bins1)
x2, _, p2 = plt.hist(sc* dists[:,0], rwidth=.8, alpha=.5, bins=bins1)
plt.ylabel('$Count$', fontsize=16)   
plt.xlabel('$Distance~from~crisscross~(\mu m)$', fontsize=16)
plt.tight_layout()
 
# tresh = 5

# plt.figure()
# plt.plot(dists[:,0][c<tresh], c[c<tresh], "o")
# np.corrcoef(dists[:,0][c<tresh], c[c<tresh])[0,1]
# # plt.hist(c)

# plt.figure()
# plt.scatter(sc* np.array(spot_last_x)[c<tresh], sc* np.array(spot_last_y)[c<tresh], c=c[c<tresh], cmap="jet", )
# plt.scatter(multilayer["X"], multilayer["Y"], s=500, color="red", alpha=.3)

# plt.axis([0,2048,0,2048])
# plt.gca().set_aspect('equal', 'box')
# cbar = plt.colorbar()
# cbar.set_label('$Mean~Dispalcement$')
# plt.show()

# step = .01
# corr = [np.corrcoef(dists[:,0][c<t], c[c<t])[0,1] for t in np.arange(0,10,step)]
# plt.plot(np.arange(0,10,step), corr, '.')

# %% EXPORT
df = pd.DataFrame({
    'TRACK_DISPLACEMENT': tracks1["TRACK_DISPLACEMENT"],
    'TRACK_DURATION': tracks1["TRACK_DURATION"],
    'mean velocity (mic/hour)': sc*4*1.8* c,
    'NEAREST CRISSCROSS REGION': 1.7* dists[:,0]
                   })
# pd.DataFrame(df).to_excel(r"C:\Users\USER\Downloads\defects_mov1.xlsx")

