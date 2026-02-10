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
import subprocess
import time
import cv2
import trackpy as tp



sys.path.append('../vasco_scripts')  # add the relative path to the folder
from defects import *  # import the module from the folder


def image_list_to_defect_list_with_trajectories(img_list, sigma=15, search_range=60, memory=3, n_jobs=5):
    df_plus, df_minus = image_list_to_defect_list(img_list, sigma=sigma, n_jobs=n_jobs)
    df_plus = track_defects(df_plus, search_range=search_range, memory=memory)
    df_minus = track_defects(df_minus, search_range=search_range, memory=memory)
    return df_plus, df_minus

def image_list_to_defect_list(img_list, sigma=15, n_jobs=4):

    # sort image list
    img_list = natsorted(img_list, key=lambda y: y.lower())
    df_list = Parallel(n_jobs=n_jobs)(delayed(lambda img: analyze_defects(img, sigma=sigma)[1:])
        (
        cv2.imread(img)[:,:,0]
        ) for img in img_list)

    for i, df in enumerate(df_list):
        df[0]['frame'] = i
        df[1]['frame'] = i

    # Concatenate the DataFrames
    df_plus = pd.concat([df[0] for df in df_list], ignore_index=True)
    df_minus = pd.concat([df[1] for df in df_list], ignore_index=True)
    
    return df_plus, df_minus

def track_defects(df_, search_range=45, memory=3):
    return tp.link_df(df_, search_range, memory=memory)


def analyze_defects(img, sigma=15):
    """
    Find orentaion and defects from raw image
    img: image (phase contrast or fluorescence)
    sigma: standard deviation for Gaussian filter used in orientation analysis
    Returns:
        ori: orientation of the image
        plushalf: DataFrame of positive defects with charge +0.5    
        minushalf: DataFrame of negative defects with charge -0.5
    """
    # Calculate mgrid
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    
    # Calculate orientation analysis
    ori, coh, E = orientation_analysis(img, sigma)
    
    # Compute topological charges
    k = compute_topological_charges(-ori, int_area='cell', origin='lower')
    
    # Localize defects
    defects = localize_defects(k)#, x_grid=xx, y_grid=yy)
    
    # Compute defect orientation
    compute_defect_orientations(-ori, defects, method='interpolation', x_grid=xx[0,:], y_grid=yy[:,0], interpolation_radius=5, min_sep=1)
    
    # Filter defects by charge
    plushalf = defects[defects['charge']==.5]
    minushalf = defects[defects['charge']==-.5]
    
    return ori, plushalf, minushalf





def nematic_plot(ax, ori, plus, min, s=11):

    y, x = np.mgrid[0:ori.shape[0], 0:ori.shape[1]]

    ax.quiver(x[::s,::s], y[::s,::s],
        np.cos(ori)[::s,::s], np.sin(ori)[::s,::s], np.arctan2(np.sin(ori), np.cos(ori))[::s,::s],
        headaxislength=0, headwidth=0, headlength=0, width=.008,
        scale=35, pivot='mid', alpha=.3, cmap="hsv")

    alpha_half, scale_half = .8, 20    
    ax.plot(plus['x'], plus['y'],'ro',markersize=8, alpha=alpha_half)
    ax.quiver(plus['x'], plus['y'], 
        np.cos(plus['ang1']), -np.sin(plus['ang1']), 
        headaxislength=0, headwidth=0, headlength=0, color='r', scale=scale_half, alpha=alpha_half)

    ax.plot(min['x'], min['y'],'wo',markersize=6, alpha=alpha_half)
    for j in range(3):
        ax.quiver(min['x'], min['y'], 
            np.cos(min['ang'+str(j+1)]), -np.sin(min['ang'+str(j+1)]), 
            headaxislength=0, headwidth=0, headlength=0, color='w', scale=scale_half, alpha=alpha_half)

from scipy import spatial
def center_pairs(Xlong, Xshort):
    '''find indexes of Xshort in Xlong'''
    tree = spatial.KDTree(Xlong)
    return tree.query(Xshort)   

def plot_orientation_analysis(img, ori, df_plus, df_minus, ax=False, 
                              colors=["lawngreen","r","b"], alpha=.8, imshow=True, 
                              scale=50, scale_half=25, width=.01, markersize=8
                              ):
    '''If color is False skip do not plot the feature'''
    s = 15

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
            color=colors[0], scale=scale, pivot='mid', alpha=.7)
  
    if colors[1]:
        ax.plot(df_plus['x'], df_plus['y'],'o',markersize=markersize, alpha=alpha, color=colors[1])
        ax.quiver(df_plus['x'], df_plus['y'], 
            np.cos(df_plus['ang1']), -np.sin(df_plus['ang1']), 
            headaxislength=0, headwidth=0, headlength=0, color=colors[1], scale=scale_half, alpha=alpha,
            width=width)

    if colors[2]:
        ax.plot(df_minus['x'], df_minus['y'],'o',markersize=markersize, alpha=alpha, color=colors[2])
        for j in range(3):
            ax.quiver(df_minus['x'], df_minus['y'], 
                np.cos(df_minus['ang'+str(j+1)]), -np.sin(df_minus['ang'+str(j+1)]), 
                headaxislength=0, headwidth=0, headlength=0, color=colors[2], scale=scale_half+10, alpha=alpha,
                width=width)

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

def plot_rolling_average(df,ax, what_key, basis_key, show=True, win=15, color="red", avfunc=circmean, stdfunc=circstd, std=True, *args,**kwargs):
    rad2deg = 180/np.pi if avfunc==circmean else 1.
    df = df.sort_values(by=basis_key)
    df[what_key+"_ave"] = roll_func(df[what_key], df[basis_key], win, avfunc, *args,**kwargs)*rad2deg
    # TODO test if same *args work for std and average
    df[what_key+"_std"] = roll_func(df[what_key], df[basis_key], win, stdfunc, *args)*rad2deg
    df[what_key+"_count"] = roll_func(1.*df[what_key].abs(), df[basis_key], win, np.sum)
    if show:
        ax.plot(df[basis_key], df[what_key+"_ave"], "-", color=color, alpha=.6, linewidth=3)
        if std:
            std_low = df[what_key+"_ave"]-df[what_key+"_std"]
            std_high = df[what_key+"_ave"]+df[what_key+"_std"]
        else:
            std_low = df[what_key+"_ave"]-df[what_key+"_std"]/df[what_key+"_count"]**.5
            std_high = df[what_key+"_ave"]+df[what_key+"_std"]/df[what_key+"_count"]**.5
        ax.fill_between(df[basis_key], 
                        std_low, 
                        std_high, 
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

# = = = FIND PAIRS FINCTIONS = = == = = = = = = = = = = = = = = = = = = = = = =
def find_pairs_and_save_to_csv(input_csv, output_csv, 
                    # jar_path = r"C:\Users\victo\OneDrive - BGU\Dov\CreationFusion.jar", 
                    jar_path = "../defect_functions/CreationFusion.jar",
                    dist_threshold=5, time_threshold=2, 
                    dist_from_edge=30, time_from_beginning_and_end=2, 
                    x=0, y=0, width=1608, height=1104):
    
    # TODO add auto detection of width and height (for max(x) and max(y))
    """
    Function to find pairs of objects based on the given parameters.
    The javadoc is my complete code comments. If you’re interested, download the whole thing, and then start with overview-tree.html.  These pages don’t work without being downloaded.

    To run the jar file, make sure you have java installed and the jar file is set as an executable.  

    Creating a File of Pairs
    Here is  a sample run command:
    java -jar CreationFusion.jar ../PlusAndMinusTM.csv RPE1_pairs.csv window 900 0 900 900 threshold 3, 35, 5, 40

    The run command must have:

    java -jar CreationFusion.jar [file to read from] [file to write to]

    Where each argument is separated by a space.  The first argument after CreationFusion.jar should be the name of the file being read from.  The second argument should be the name of the file being written to.  

    If the whole read file is not meant to be used, but only defects within a certain window, then the window argument should be added.  The first two arguments following the window argument should be the x and y values of te rectangle nearest the origin, followed by the width and height of the rectangle.

    java -jar CreationFusion.jar [file to read from] [file to write to] window [x] [y] [width] [height]

    Another argument is the threshold argument, after threshold the next value must be a distance threshold followed by an integer time threshold.  If no threshold is set, default values of dist = 40 and time = 4 are used.  Note, the distance is for pixels in the original frames.

    java -jar CreationFusion.jar [file to read from] [file to write to] threshold [dist threshold] [time threshold] [dist from edge] [time from beginning and end] window [x] [y] [width] [height]

    If you want to read multiple input files, then the first word should be a folder containing exactly the input files.  For example, if 3 files, s1, s11, and s12 are in in a folder called sFolder, and there is nothing else in sFolder, then the following should work:

    java -jar CreationFusion.jar sFolder output.csv threshold 10 2 30 2 window 0 0 900 900

    The code actually has much more functionality, so let me know what you want to do, and I’ll make it command line accessible.  I’m not claiming I’m done, but this is some of what I’ve done so far.

    Creating a File of Images
    The run command must have:

    java -jar CreationFusion.jar [file to read from] [folder to write to] [line of interest index]

    Note, the folder you write to must already be created.  Be sure the location the run common is executed from contains the files “NefDef.png” and “PosDef.png”. Those files are available for download in this folder.

    """
    command = [
        "java",
        "-jar",
        jar_path,
        input_csv,
        output_csv,
        "threshold",
        str(dist_threshold),
        str(time_threshold),
        str(dist_from_edge),
        str(time_from_beginning_and_end),
        "window",
        str(x), str(y), str(width), str(height)
    ]

    # Run the command and capture output
    try:
        result = subprocess.run(command)#, check=True, capture_output=True, text=True)
        print("Java program executed successfully.")
        print("Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        print("Error Output:", e.stderr)
    
    # Pause for 500 milliseconds
    time.sleep(0.5)
    print(pd.read_csv(output_csv).dropna(subset=["min_id","creation"]).shape)
    return None


def pair_list(PlusAndMinusTM, dist_thresh):
    """
    Finds unique labels of plus and minus defect pairs
    
    Returns: dataframe["minusID","plusID","dist"]
    """

    pairs = []

    for frame in PlusAndMinusTM["FRAME"].unique():
        # print(frame)
        p_idx = np.logical_and(PlusAndMinusTM["FRAME"]==frame, PlusAndMinusTM["charge"]==.5)
        m_idx = np.logical_and(PlusAndMinusTM["FRAME"]==frame, PlusAndMinusTM["charge"]==-.5)
        plus_xy = PlusAndMinusTM[["TRACK_ID","x_img1","y_img1","ang1"]][p_idx]
        minus_xy = PlusAndMinusTM[["TRACK_ID","x_img1","y_img1","ang1","ang2","ang3"]][m_idx]

        dist, idx = center_pairs(minus_xy[["x_img1","y_img1"]], plus_xy[["x_img1","y_img1"]])
        # set a distance that define that defects are a pair (50)
        dist_TF = dist<dist_thresh
        # Notice: some TRACK_IDs are nan, meaning that they are not a part from any trajectory
        pairs.append([
            minus_xy["TRACK_ID"].iloc[idx][dist_TF].values,#.astype(np.int_), 
            plus_xy["TRACK_ID"][dist_TF].values,#.astype(np.intc)
            dist[dist_TF]
            ]) 
        # break

    pairs_df = pd.DataFrame(np.concatenate(pairs, axis=1).T, columns=["minus", "plus","dist"]).dropna()#.drop_duplicates()
    _, indices = np.unique(pairs_df["minus"], return_index=True)
    return pairs_df.iloc[indices].copy()    

PAIR_DIST_THRESH = 50
MINIMAL_TRACK_LENGTH = 5

def trajectory_pair_list(PlusAndMinusTM, minimal_track=MINIMAL_TRACK_LENGTH, dist_thresh=PAIR_DIST_THRESH):

    pairs_df = pair_list(PlusAndMinusTM, dist_thresh=dist_thresh)

    all_pairs_df = []
    for m,n in zip(pairs_df["minus"], pairs_df["plus"]):
        m_idx = np.logical_and(PlusAndMinusTM["TRACK_ID"]==m, PlusAndMinusTM["charge"]==-.5)
        p_idx = np.logical_and(PlusAndMinusTM["TRACK_ID"]==n, PlusAndMinusTM["charge"]==.5)

        if m_idx.sum()>minimal_track and p_idx.sum()>minimal_track:

            dff_ = equlalize_trajectories(PlusAndMinusTM, p_idx, m_idx)
            all_pairs_df.append(dff_)

    return pd.concat(all_pairs_df)   