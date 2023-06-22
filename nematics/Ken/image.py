import pickle

from scipy.ndimage import rotate
from numpy import pad
import os
import scipy as sp
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import cv2
import warnings
# from skimage import feature, measure, restoration
from defects import *
from natsort import natsorted
import glob
from tqdm import tqdm

class image:
    """  
    this class is ment to be activated to automaticly apply analisys process to 
    a single image.

    parameters
    ---------
    path :  str
        path to file 

    window_size : int default = 40
        square window size of size window_size

    resize_image :tuple, default = None
        size for image resize to speedup processing time
        """

    def __init__(self, path, resize_image=None, window_size=40):
        self.path = path
        self.name = path.split('\\')[-1].split('.')[0]
        self.img = plt.imread(path)
        self.window_size = window_size



        # make square image
        self.img = self.img[-np.min(self.img.shape):, -np.min(self.img.shape):]

        if resize_image:
            self.img = cv2.resize(self.img, dsize=resize_image)

    def detect_defects(self, orientation_path, save_ori, defects_path, save_defects, orientation_window):
        """
        detects the defects as -0.5/ +0.5

        returns
        -------
        tuple: (plushalf, minushalf)

            plushalf : pd.DataFrame
                df of plus half defects detected

            minushalf : pd.DataFrame
                df of minus half defects detected

        """
        # mute warnings
        warnings.filterwarnings('ignore')

        pic_x = self.img.shape[1]
        pic_y = self.img.shape[0]

        x = np.arange(0, pic_x)
        y = np.arange(0, pic_y)
        xx, yy = np.meshgrid(x, y)

        ori = self.calc_orientation(orientation_path, save_orientation=save_ori, window_size= orientation_window)
        k = compute_topological_charges(ori, int_area='cell', origin='upper')
        defects = localize_defects(k, x_grid=xx, y_grid=yy)
        # try:
        compute_defect_orientations(ori, defects)
        # except:
        #     x=1

        plushalf = defects[defects['charge'] == .5]
        minushalf = defects[defects['charge'] == -.5]
        plushalf.loc[:, 'from_img'] = self.name
        minushalf.loc[:, 'from_img'] = self.name

        # save:
        if save_defects:
            plushalf[['from_img', 'charge', 'x', 'y', 'ang1']].to_csv(defects_path + self.name + '_PlusHalf.csv')
            minushalf[['from_img', 'charge', 'x', 'y', 'ang1', 'ang2', 'ang3']].to_csv(
                defects_path + self.name + '_MinusHalf.csv')

        return plushalf, minushalf

    def calc_orientation(self, orientation_path, save_orientation=False, window_size=None):

        window_size = self.window_size if window_size is None else window_size
        ori, coh, E = orientation_analysis(self.img, window_size)

        # save

        if save_orientation:

            name = 'orientation_from_' + self.name + '.pkl'
            with open(orientation_path + name, 'wb') as f:
                pickle.dump(ori[::3, ::3], f)


        return ori

# trying new way of rotation
    def rotate_window(self,velocities, df, window_size=400):
        results = []
        for i, row in tqdm(df.iterrows()):
            x, y, ang1 = int(row['x']), int(row['y']), row['ang1']
            half_size = window_size // 2

            # crop and rotate the window
            padded_window = crop_window(velocities,x,y,window_size)
            rotated_window = rotate(padded_window, np.degrees(ang1), reshape=False, order=1) # check order = 1 (normally 3 )

            # Crop the rotated window around its center to be 1/sqrt(2) of its former size
            cropped_size = int(np.ceil(half_size / np.sqrt(2)))
            cropped_window = rotated_window[half_size - cropped_size:half_size + cropped_size,
                             half_size - cropped_size:half_size + cropped_size, :]

            # rotate the velocity field by ang1
            cos_ang1, sin_ang1 = np.cos(ang1), np.sin(ang1)
            rotation_matrix = np.array([[cos_ang1, sin_ang1], [-sin_ang1, cos_ang1]])
            rotated_field = np.apply_along_axis(lambda v: (v @ rotation_matrix), 2, cropped_window)
            results.append(rotated_field)

        return (len(results),np.nanmean(results,axis=0))

    def crop_and_tilt(self,
                      defects_df=None,
                      velocity_array=None,
                      orientation_array=None,
                      half_window=200,
                      save=False,
                      path="", plot=False):
        """
        for each defect in the image, crops a window around it and tilts it and the
        surrounding flow.

        note that orientation_array is currently not in use.
        parameters
        ---------
        path :  str
            path to folder that contains the images.

        return:
            number_of_defects, mean_array
            mean_array : an average over the velocities
            number_of_defects : over how many defect was mean array calculated
        """

        second_window = int(np.floor(half_window * np.sqrt(2)))
        number_of_defects = defects_df.shape[0]
        # mean_arr = np.zeros(shape=(number_of_defects,second_window, second_window, 2))
        array_list = []
        for idx,(_, defect) in enumerate(tqdm(defects_df.iterrows())):


            x,y = int(defect['x']), int(defect['y'])
            ang = defect['ang1']
            ang_d = np.rad2deg(ang)

            if x-half_window <= 0 or x+half_window >= velocity_array.shape[1] or\
                y-half_window <=0 or y+half_window >= velocity_array.shape[0]:
                number_of_defects -= 1
                continue

            cropped = crop_window(arr=velocity_array,x=x,y=y,window_size=half_window * 2)

            # rotate the whole image
            rotated_first = sp.ndimage.rotate(cropped, ang_d,
                                        reshape=False)  # -angle is rotating clockwise by angle
            # crop again to make a square
            mid = int((rotated_first.shape[0] // 2) - 0.5)
            delta = int(second_window // 2)
            cropped_after_rotation = rotated_first[mid-delta: mid+delta, mid-delta: mid+delta, :]

            # rotate each PIV by -angle
            rotation_matrix = np.mat(
                [[np.cos(ang), np.sin(ang)],
                 [-np.sin(ang), np.cos(ang)]])

            # get a list of the velocity after rotation
            rot = lambda x: np.matmul(x, rotation_matrix.T)
            rotated_velocity = np.array([*map(rot, cropped_after_rotation)])


            # rotated = sp.ndimage.rotate(rotated_velocity, -ang_d, reshape=False) #-angle is rotating clockwise by angle

            # cropped_after_rotation = crop_window(arr=rotated,
            #                                      x=mid,
            #                                      y=mid,
            #                                      window_size= second_window)

            # save:
            if save:
                name = self.name + f"_{idx}" + '.pkl'
                with open(path + name, 'wb') as f:
                    # pickle.dump(cropped_after_rotation,f)
                    pickle.dump(rotated_velocity, f)
            # try:
                # mean_arr[idx] = rotated_velocity
            array_list.append(rotated_velocity)
            # except ValueError:
                # mean_arr = np.empty(shape = [number_of_defects,*cropped_after_rotation.shape])
                # mean_arr[idx] = rotated_velocity
            # # add to the mean array
            # try:
            #     mean_arr += cropped_after_rotation
            # except ValueError:
            #     mean_arr = np.zeros_like(cropped_after_rotation)
            #     mean_arr += cropped_after_rotation
            if plot:
                ##########plot ###############
                fig, ax = plt.subplots(2, 2, figsize=(20, 20))

                #plot final
                step = 5
                x = np.arange(0, rotated_velocity.shape[1], step, dtype=np.int16)
                y = np.arange(0, rotated_velocity.shape[0], step, dtype=np.int16)
                center = np.array(rotated_velocity.shape) / 2
                ax[1,1].set_title('final')
                ax[1, 1].plot(center[1], center[0], marker='o', markersize=10, label='defect')
                ax[1, 1].quiver(x, y, rotated_velocity[:, :, 0][::step, ::step],
                                rotated_velocity[:, :, 1][::step, ::step])


                # plot the image before the rotation
                x = np.arange(0, rotated_first.shape[1], step, dtype=np.int16)
                y = np.arange(0, rotated_first.shape[0], step, dtype=np.int16)
                center = np.array(rotated_first.shape) / 2
                ax[1, 0].set_title('after rotating the vectors')
                ax[1, 0].scatter(center[1], center[0], marker='o', s=40, color='red')
                ax[1, 0].quiver(x, y, rotated_first[:, :, 0][::step, ::step],
                                rotated_first[:, :, 1][::step, ::step])


                # plot original flow after crop
                x = np.arange(0, cropped.shape[1], step, dtype=np.int16)
                y = np.arange(0, cropped.shape[0], step, dtype=np.int16)
                center = np.array(cropped.shape) / 2
                ax[0, 1].set_title('cropped flow array')
                ax[0, 1].quiver(x, y, cropped[:, :, 0][::step, ::step], cropped[:, :, 1][::step, ::step], scale=200)
                ax[0, 1].plot(center[1], center[0], 'ro', markersize=10, label=r'+1/2 defect')
                ax[0, 1].quiver(center[1], center[0],
                                np.cos(defect['ang1']), np.sin(defect['ang1']),
                                headaxislength=0, headwidth=0, headlength=0, color='r', scale=50)

                # plot original flow

                x = np.arange(0, velocity_array.shape[1], step, dtype=np.int16)
                y = np.arange(0, velocity_array.shape[0], step, dtype=np.int16)
                ax[0, 0].set_title('full flow array')
                ax[0, 0].quiver(x, y, velocity_array[:, :, 0][::step, ::step], velocity_array[:, :, 1][::step, ::step],
                                scale=200)
                ax[0, 0].plot(defect['x'], defect['y'], 'ro', markersize=10, label=r'+1/2 defect')
                ax[0, 0].quiver(defect['x'], defect['y'],
                                np.cos(defect['ang1']), np.sin(defect['ang1']),
                                headaxislength=0, headwidth=0, headlength=0, color='r', scale=50)
                fig.suptitle(fr'angle = {ang_d}')

                # ax[0,0].set_box_aspect(1)
                # ax[0,1].set_box_aspect(1)
                # ax[1,0].set_box_aspect(1)
                # ax[1,1].set_box_aspect(1)

                plt.show()

            #########added for debugging#######
        mean_arr = np.mean(array_list,axis=0)
        return number_of_defects, mean_arr # np.nanmean(mean_arr,axis=0)
def crop_window(arr, x, y, window_size):
    """

    Parameters
    ----------
    velocities
    x: (int) the x coordinate of the defect
    y: (int) the y coordinate of the defect
    window_size: (int) the size of the window that will be cropped around the defect

    Returns
    -------
    padded_window: (np.array) an array of size window_size that contains the velocity
    array around the defect and a propper padding of 0s in case of going over the limit size of
    the original velocity array.
    """


    # replace x,y because of mistake

    window_size = int(window_size - 1 if window_size % 2 == 0 else window_size)
    half_size = int(window_size // 2)

    x_min, y_min = max(x - half_size, 0), max(y - half_size, 0)
    x_max, y_max = min(x + half_size + 1, arr.shape[0]), min(y + half_size + 1, arr.shape[1])

    x_leave_before = max(half_size - (x - x_min), 0)
    x_leave_after = window_size - max(half_size - (x_max - x -1), 0)
    y_leave_before = max(half_size - (y - y_min), 0)
    y_leave_after = window_size - max(half_size - (y_max - y - 1), 0)

    cropped_window = arr[y_min:y_max, x_min:x_max,:]
    padded_window = np.full((half_size*2+1, half_size*2+1,2), np.nan)
    for i, row in enumerate(range(x_leave_before, x_leave_after)):
        for j, col in enumerate(range(y_leave_before, y_leave_after)):
            padded_window[col, row] = cropped_window[j,i]




    return padded_window
# def crop_window(arr, point, window_size):
#     """
#     Crops a window of the array around the point.
#
#     Parameters:
#     arr (numpy.ndarray): The input array.
#     point (tuple): The point around which to crop the window.
#     window_size (int): The size of the window to crop.
#
#     Returns:
#     numpy.ndarray: The cropped window.
#     """
#     row, col = point
#     row, col = int(row), int(col)
#     half_window = int(window_size // 2)
#     cropped = arr[row - half_window:row + half_window, col - half_window:col + half_window]
#     return cropped
