import os
import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
# from skimage import feature, measure, restoration
from defects import *
from natsort import natsorted
import glob


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

    resize_image :tuple, default = None TODO: chack if size is only square
        size for image resize to speedup processing time
        """


    def __init__(self, path, resize_image= None, window_size = 40):
        self.path = path
        self.name = path.split('\\')[-1].split('.')[0]
        self.img = plt.imread(path)
        self.window_size = window_size

        #make square image
        self.img = self.img[-np.min(self.img.shape):,-np.min(self.img.shape):]
        
        if resize_image:
            self.img = cv2.resize(self.img, dsize = resize_image)

    def detect_defects(self, orientation_path, save_ori, defects_path, save_defects):
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
        
        pic_x = self.img.shape[1]
        pic_y = self.img.shape[0]

        x = np.arange(0,pic_x)
        y = np.arange(0,pic_y)
        xx, yy = np.meshgrid(x, y)

        ori = self.calc_orientation(orientation_path, save_orientation= save_ori)
        k = compute_topological_charges(ori, int_area='cell', origin='upper')
        defects = localize_defects(k, x_grid=xx, y_grid=yy)
        compute_defect_orientations(ori, defects)

        plushalf = defects[defects['charge']==.5]
        minushalf = defects[defects['charge']==-.5]
        plushalf['from_img'] = self.name
        minushalf['from_img'] = self.name

        #save:
        if save_defects:
            plushalf[['from_img','charge', 'x','y','ang1']].to_csv(defects_path +self.name + '_PlusHalf.csv')
            minushalf[['from_img','charge', 'x','y','ang1','ang2','ang3']].to_csv(
                defects_path + self.name + '_MinusHalf.csv')
            
        return plushalf, minushalf


    def calc_orientation(self,orientation_path, save_orientation= False):

        ori, coh, E = orientation_analysis(self.img, self.window_size)
        #save
        if save_orientation:
            name = 'orientation_from_' +self.name + '.npy' 
            np.save(orientation_path + name, ori)

        return ori

      
    def crop_and_tilt(self, defects_df = None, velocity_array = None,orientation_array = None, half_window = 200, save = False, path=""):
        """
        for each defect in the image, cropes a window around it and tilts it and the 
        flow around it.

        note that orientation_array is currently not in use.
        parameters
        ---------
        path :  str
            path to folder that contains the images.

        return : number_of_defects, mean_array
            mean_array : an average over the velocities
            number_of_defects : over how many defect was mean array calculated
        """

        # if(defects_df is None):
        #     plus_half, minus_half = self.detect_defects()
        #     defects_df = pd.concat([plus_half,minus_half])

        # if(velocity_array is None):
        #     velocity_array = self.get_flow()
            
        # if(orientation_array is None):
        #     orientation_array = self.calc_orientation()

        second_window = int(np.floor(half_window*np.sqrt(2)))
        mean_arr = np.zeros(shape=(second_window,second_window,2))
        number_of_defects = defects_df.shape[0]
        for idx, defect in defects_df.iterrows():
            # center = (defect['x_ind'],defect['y_ind'])
            center = (defect['x'],defect['y'])

            #TODO edge cases such as padding 
            if center[0] < half_window or center[1] < half_window:
                continue
            if center[0] + half_window > velocity_array.shape[1] or center[1] + half_window > velocity_array.shape[1]:
                continue
            cropped = self.crop(center = center, 
                                velocity_array=velocity_array, 
                                half_window = half_window)
            
            #rotate each PIV by -angle
            ang = -defect['ang1']
            rotation_matrix = np.mat(
                [[np.cos(ang), np.sin(ang)],
                [-np.sin(ang), np.cos(ang)]])

            #get a list of the velocity after rotation
            rot = lambda x: np.matmul(x,rotation_matrix.T)
            rotated_velocity = list(map(rot, cropped))
            
            ang_d = np.rad2deg(ang)
            
            rotated = sp.ndimage.rotate(rotated_velocity, ang_d, reshape = False)
            mid = (rotated.shape[0] / 2) - 0.5
            center = (mid,mid)
            cropped = self.crop(center, rotated, np.floor(second_window/2))

            #save:
            if save:
                name = self.name + f"_{idx}" + '.npy' 
                np.save(path + name, cropped)
            
            #add to the mean array
            mean_arr += cropped / number_of_defects

        return number_of_defects, mean_arr


    def crop(self, center , velocity_array, half_window):
        """
        crops a window around a defect

        parameters
        -------------
        center: array 
        [x_index, y_index]

        velocity_array : array
        the flow array for the image the defect was taken from 

        half_window : int
        half the window size of which the array will be cropped 

        return:
        --------------
        croped : array
        the flow array around the defect

        """

        x_start = int(center[0] - half_window)
        x_stop = int(center[0] + half_window)
        y_start = int(center[1] - half_window)
        y_stop = int(center[1] + half_window)

        cropped = velocity_array[x_start:x_stop,y_start:y_stop,:]
        
        return cropped
