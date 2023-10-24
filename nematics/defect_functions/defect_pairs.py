# %%
import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted
import sys
import pandas as pd
from scipy.ndimage import rotate, gaussian_filter
import pickle 
from scipy.stats import circmean, circstd, sem
from joblib import Parallel, delayed


sys.path.append('../vasco_scripts')  # add the relative path to the folder
from defects import *  # import the module from the folder


def analyze_defects(img, sigma=15):
    # Calculate mgrid
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    
    # Calculate orientation analysis
    ori, coh, E = orientation_analysis(img, sigma)
    
    # Compute topological charges
    k = compute_topological_charges(-ori, int_area='cell', origin='lower')
    
    # Localize defects
    defects = localize_defects(k, x_grid=xx, y_grid=yy)
    
    # Compute defect orientation
    compute_defect_orientations(-ori, defects, method='interpolation', x_grid=xx[0,:], y_grid=yy[:,0], interpolation_radius=5, min_sep=1)
    
    # Filter defects by charge
    plushalf = defects[defects['charge']==.5]
    minushalf = defects[defects['charge']==-.5]
    
    return ori, plushalf, minushalf

from scipy import spatial
def center_pairs(Xlong, Xshort):
    '''find indexes of Xshort in Xlong'''
    tree = spatial.KDTree(Xlong)
    return tree.query(Xshort)   

def plot_orientation_analysis(img, ori, df_plus, df_minus, ax=False, colors=["lawngreen","r","b"], alpha=.8, imshow=True):
    '''If color is False skip do not plot the feature'''
    alpha_half, scale_half, s = alpha, 25 , 15
    y, x = np.mgrid[0:img.shape[0], 0:img.shape[1]]

    if not ax:
        fig, ax = plt.subplots(1,1, figsize=(6,6))

    if imshow:
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # img_clahe = clahe.apply(img)
        # ax.imshow(img_clahe, cmap="gray")  
        ax.imshow(np.zeros_like(img, dtype=np.float32), cmap="gray")

    if colors[0]:
        ax.quiver(x[::s,::s], y[::s,::s],
            np.cos(ori)[::s,::s], np.sin(ori)[::s,::s], 
            headaxislength=0, headwidth=0, headlength=0, 
            color=colors[0], scale=60, pivot='mid', alpha=.7)
  
    if colors[1]:
        ax.plot(df_plus['x'], df_plus['y'],'o',markersize=6, alpha=alpha_half, color=colors[1])
        ax.quiver(df_plus['x'], df_plus['y'], 
            np.cos(df_plus['ang1']), -np.sin(df_plus['ang1']), 
            headaxislength=0, headwidth=0, headlength=0, color=colors[1], scale=scale_half, alpha=alpha_half)

    if colors[2]:
        ax.plot(df_minus['x'], df_minus['y'],'o',markersize=6, alpha=alpha_half, color=colors[2])
        for j in range(3):
            ax.quiver(df_minus['x'], df_minus['y'], 
                np.cos(df_minus['ang'+str(j+1)]), -np.sin(df_minus['ang'+str(j+1)]), 
                headaxislength=0, headwidth=0, headlength=0, color=colors[2], scale=scale_half+10, alpha=alpha_half)

import sys,time,random
def progressBar(count_value, total, suffix=''):
    bar_length = 100
    filled_up_Length = int(round(bar_length* count_value / float(total)))
    percentage = round(100.0 * count_value/float(total),1)
    bar = '=' * filled_up_Length + '-' * (bar_length - filled_up_Length)
    sys.stdout.write('[%s] %s%s ...%s\r' %(bar, percentage, '%', suffix))
    sys.stdout.flush()


def roll_func(what,basis,window,func,*args,**kwargs):
    '''https://stackoverflow.com/questions/14300768/pandas-rolling-computation-with-window-based-on-values-instead-of-counts'''
    
    '''see more examples:
    https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy/14314054#14314054'''
    # from scipy.stats import circmean, circstd
    #note that basis must be sorted in order for this to work properly     
    indexed_what = pd.Series(what.values,index=basis.values)
    def applyToWindow(val):
        # using slice_indexer rather that what.loc [val:val+window] allows
        # window limits that are not specifically in the index
        indexer = indexed_what.index.slice_indexer(val-window,val+window,1)
        chunk = indexed_what.iloc[indexer]
        return func(chunk,*args,**kwargs)
    rolled = basis.apply(applyToWindow)
    return rolled

def plot_rolling_average(df,ax, what_key, basis_key, show=True, win=15, color="red", avfunc=circmean, stdfunc=circstd, *args,**kwargs):
    rad2deg = 180/np.pi if avfunc==circmean else 1.
    df = df.sort_values(by=basis_key)
    df[what_key+"_ave"] = roll_func(df[what_key], df[basis_key], win, avfunc, *args,**kwargs)*rad2deg
    # TODO test if same *args work for std and average
    df[what_key+"_std"] = roll_func(df[what_key], df[basis_key], win, stdfunc, *args)*rad2deg
    df[what_key+"_count"] = roll_func(1.*df[what_key].abs(), df[basis_key], win, np.sum)
    if show:
        ax.plot(df[basis_key], df[what_key+"_ave"], "-", color=color, alpha=.6, linewidth=3)
        ax.fill_between(df[basis_key], 
                        df[what_key+"_ave"]-df[what_key+"_std"]/df[what_key+"_count"]**.5, 
                        df[what_key+"_ave"]+df[what_key+"_std"]/df[what_key+"_count"]**.5, 
                        color=color, alpha=.2) 
    return df[[basis_key, what_key+"_ave", what_key+"_std", what_key+"_count"]] 


def msd_from_df(df,xlabel,ylabel,tlabel,id_label,minimal_track=3):
    def msd(x,y):
        return np.cumsum(np.diff(x))**2 + np.cumsum(np.diff(y))**2

    msds = []
    for id in df[id_label].unique():
        idx = (df[id_label]==id)
        df_ = df[idx].copy()
        if idx.sum()>minimal_track:
            t = df_[tlabel].to_numpy()
            x = df_[xlabel].to_numpy()
            y = df_[ylabel].to_numpy()
            msds.append(pd.DataFrame({"FRAME": t[1:]-t[0], 'MSD':msd(x,y)}).dropna())            
    msd_df = pd.concat(msds)
    return msd_df[msd_df["FRAME"]>0]

def ang_msd_from_df(df,xlabel,tlabel,id_label,period=2*np.pi, minimal_track=3):
    def msd(x):
        x = np.unwrap(x, period=period)
        return np.cumsum(np.diff(x))**2

    msds = []
    for id in df[id_label].unique():
        idx = (df[id_label]==id)
        df_ = df[idx].copy()
        if idx.sum()>minimal_track:
            t = df_[tlabel].to_numpy()
            x = df_[xlabel].to_numpy()
            msds.append(pd.DataFrame({"FRAME": t[1:]-t[0], 'MSD':msd(x)}).dropna())   
    msd_df = pd.concat(msds)
    return msd_df[msd_df["FRAME"]>0]  

def equlalize_trajectories(plus_minus_df, p_idx, m_idx):
    ''' makes sure that positive (p_idx) and negative (m_idx) defect trajectory are equal
    according to "FRAME" number
    if missing frame for one of them it removed with dropna()
    '''
    df1 = plus_minus_df[["TRACK_ID","FRAME","x_img1","y_img1","ang1"]][p_idx]
    df1.set_index('FRAME', inplace=True)
    df2 =  plus_minus_df[["TRACK_ID","FRAME", "x_img1","y_img1", "ang1","ang2","ang3" ]][m_idx]
    df2.set_index('FRAME', inplace=True)
    df_ = pd.concat([df1, df2], axis=1).dropna()
    return df_.set_axis(["plus_id","xp","yp","angp1", "min_id","xm","ym", "angm1","angm2","angm3"], axis=1)