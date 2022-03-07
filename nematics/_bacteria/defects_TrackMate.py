# %%
import numpy as np
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

def slope()
# %%
folder = r"C:\Users\USER\Downloads\BEER\March 1st 100fps 40X 50-50 1um gap/"
spots = pd.read_csv(folder + "TrackMate/p_def_spots.csv", skiprows=[1,2,3])
tracks_all = pd.read_csv(folder + "TrackMate/p_def_tracks.csv", skiprows=[1,2,3])
tracks_to_remove = tracks_all["TRACK_ID"][tracks_all["TRACK_DURATION"]<10].unique()
tracks_ids = tracks_all["TRACK_ID"][tracks_all["TRACK_DURATION"]>=20].unique()

for t_id in tracks_to_remove:
    spots["TRACK_ID"][spots["TRACK_ID"]==t_id]=np.nan 

spots = spots.dropna(subset=["TRACK_ID"])

# check size 
print(len(spots["TRACK_ID"].unique()))
print(len(tracks_ids))


msd = []
# plt.figure(figsize=(5,5))
for i,t_id in enumerate(tracks_ids):
    # if i>120 and i<150:
    idx = spots["TRACK_ID"]==t_id
    frame_idx = spots["FRAME"][spots["TRACK_ID"]==t_id].argsort()
    frame_num = spots["FRAME"][idx].iloc[frame_idx]
    x,y = spots["POSITION_X"][idx].iloc[frame_idx], spots["POSITION_Y"][idx].iloc[frame_idx]
    x0,y0 = spots["POSITION_X"][idx].iloc[frame_idx.iloc[0]], spots["POSITION_Y"][idx].iloc[frame_idx.iloc[0]]
    plt.plot(
        # x0 - x,
        # y0 - y,
        frame_num - frame_num.iloc[0],
        np.sqrt((x-x0)**2 + (y-y0)**2),
        "-", alpha=.05, color="red", linewidth=3
        )
    msd.append(mean_square_displacement(x,y))
        # if i>150:
        #     break
plt.xlabel("$Time ~(frame)$", fontsize=16)
plt.ylabel("$Displacement ~(px)$", fontsize=16)
plt.tight_layout()

# plt.figure(figsize=(5,5))        
# plt.hist(msd, 20, rwidth=.9)
# %%
ff = 0
for i,t_id in enumerate(tracks_ids):
    idx = spots["TRACK_ID"]==t_id
    frame_idx = spots["FRAME"][spots["TRACK_ID"]==t_id].argsort()
    frame_num = spots["FRAME"][idx].iloc[frame_idx]
    frame_num = frame_num - frame_num.iloc[0]
    print(np.diff(frame_num))
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
pd.DataFrame(df).to_excel(r"C:\Users\USER\Downloads\defects_mov1.xlsx")

