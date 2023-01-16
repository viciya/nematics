# import torch
# import torchvision
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



class image_series:
    """
    this class is ment to be activated to automaticly apply analisys process to 
    image/ video.

    parameters
    ---------
    path :  str
        path to file/folder in case of many

    window_size : int default = 40
        square window size of size window_size.

    resize_image :tuple, default = None TODO: chack if size is only square
        size for image resize to speedup processing time.
    save_defects : saves the defects data frame into "csv" folder inside the images folder
    save_flow : saves the flow array (currently as mat) into "velocity" folder (inside IO folder)
    save_orientation : saves the flow array (currently as mat) into "velocity" folder (inside IO folder)
    """


    def __init__(
        self,
        folder_path,
        window_size = 40, 
        resize_image = None,
        save_all = False,
        save_defects = False,
        save_flow = False,
        save_orientation = False,
        save_velocity = False
        ) -> None:
        
        self.path = folder_path
        self.window_size = window_size

        self.save_defects = True if save_all else save_defects
        self.save_flow = True if save_all else save_flow
        self.resize_image = True if save_all else resize_image
        self.save_orientation = True if save_all else save_orientation
        self.save_velocity = True if save_all else save_velocity

        self.fetch_data()


    def controller(self): #here right now, will probably be a class 
        pass


    def fetch_data(self): 
        """
        fetches the data into self images
        """
        self.images =[]
        image_list = glob.glob(self.path + r"\*tif")
        image_list = natsorted(image_list, key=lambda y: y.lower())

        for image_path in image_list:
            self.images.append( image( 
                image_path,
                self.resize_image,
                self.window_size))


        
    def calc_orientation(self):

        for img in self.images:
            img.calc_orientation(self.save_orientation)


    def detect_minus_plus(self):
        """
        goes over images and detects the defects as -0.5/ +0.5 
        and adds them to df accordingly.

        returns
        -------
        tuple : (PlusHalf, MinusHalf)
        PlusHalf : pd.DataFrame
            df of plus half defects detected
            columns =[]TODO add columns 

        MinusHalf : pd.DataFrame
            df of minus half defects detected
        """
        
        PlusHalf, MinusHalf = pd.DataFrame(),pd.DataFrame()
        for  img in self.images:
            
            name = img.name
            plus, minus = img.detect_minus_plus()
            plus['from_img'] = name
            minus['from_img'] = name
            if self.save_defects:
                
            #save_defects single image 
                plus[['from_img','charge', 'x','y','ang1']].to_csv(self.path + r"\\csv\\" +name + '_PlusHalf.csv')
                minus[['from_img','charge', 'x','y','ang1','ang2','ang3']].to_csv(
                    self.path + r"\\csv\\" + name + '_MinusHalf.csv')
             
            #add to total DF
            

            PlusHalf = pd.concat([PlusHalf, plus])
            MinusHalf = pd.concat([MinusHalf, minus])

        if self.save_defects:
            PlusHalf[['from_img','charge', 'x','y','ang1']].to_csv(self.path + r"\csv\PlusHalf.csv")
            MinusHalf[['from_img','charge', 'x','y','ang1','ang2','ang3']].to_csv(
                self.path + r"\csv\MinusHalf.csv")


            

        return PlusHalf, MinusHalf
                    

    def __repr__(self) -> str:
        return f"workflow : \n  number of images = {len(self.images)}"


    def get_flow(self):

        """
        calculates the optical flow for image_list and save as an np.array
        saves if save_flow == True to the previous directory at folder valocity
        doesn't add the directory
        """
        if self.save_flow:
            path_to_save = os.path.normpath(
                self.path + os.sep + os.pardir
                ) + '\\velocity\\' #go back one dir and go to velocity dir

        #create a big array that will contain all the velocities
        img_size = self.images[0].img.shape
        num_of_velocity_images = len(self.images) -1
        big_arr = np.empty(shape = (num_of_velocity_images,*img_size,2))
        
        
        for (i,im1), im2 in zip(enumerate(self.images[:-1]),self.images[1:]):

            # methods = ['cv2.TM_CCOEFF']
            # pad = 200

            img1 = im1.img
            img2 = im2.img
            
            flow = cv2.calcOpticalFlowFarneback(img1,img2, None, 0.5, 3, 
                winsize=61, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            
            big_arr[i] = flow


            # #save a single image mat 
            # if self.save_flow:
            #     name = 'velocity_from_' +im1.name.split('.')[0] + '.mat' 
            #     sp.io.savemat(path_to_save+name, dict(u=flow[:,:,0], v=flow[:,:,1]))
                #save a single image array
            if self.save_flow:
                name = 'velocity_from_' +im1.name.split('.')[0] + '.npy' 
                np.save(path_to_save+name, flow)
            



        return None #big_arr
            

        

  
        

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

    def detect_minus_plus(self):
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
        
        pix_x = self.img.shape[1]
        pix_y = self.img.shape[0]

        x = np.arange(0,pix_x)
        y = np.arange(0,pix_y)

        xx, yy = np.meshgrid(x, y)


        ori = self.calc_orientation(save_orientation= False)
        k = compute_topological_charges(ori, int_area='cell', origin='upper')
        defects = localize_defects(k, x_grid=xx, y_grid=yy)
        compute_defect_orientations(ori, defects)
        plushalf = defects[defects['charge']==.5]
        minushalf = defects[defects['charge']==-.5]       
        
        return plushalf, minushalf


    def calc_orientation(self, save_orientation= False):
        ori, coh, E = orientation_analysis(self.img, self.window_size)

        if save_orientation:
            path_to_save = os.path.normpath(
                self.path + os.sep + os.pardir + os.sep + os.pardir
                ) + '\\orientation\\'
            name = 'orientation_from_' +self.name + '.npy' 
            np.save(path_to_save + name,ori)


        return ori


      
    def crop_and_tilt(self, defects_df = None, velocity_array = None,orientation_array = None, half_window = 200, save = False):
        """
        for each defect in the image, cropes a window around it and tilts it and the 
        flow around it.
        """
        
        #TODO: should consider batches

        if(defects_df is None):
            plus_half, minus_half = self.detect_minus_plus()
            defects_df = pd.concat([plus_half,minus_half])

        if(velocity_array is None):
            velocity_array = self.get_flow()
            
        if(orientation_array is None):
            orientation_array = self.calc_orientation()

        second_window = int(np.floor(half_window / np.sqrt(2)))
        for idx, defect in defects_df.iterrows():
            center = (defect['x_ind'],defect['y_ind'])
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

            rot = lambda x: np.matmul(x,rotation_matrix.T)
            Rot_corpped = list(map(rot, cropped))
            
            ang_d = np.rad2deg(ang)
            
            rotated = sp.ndimage.rotate(Rot_corpped, ang_d, reshape = False)
            mid = (rotated.shape[0] / 2) - 0.5
            center = (mid,mid)
            cropped = self.crop(center, rotated, second_window)

            #save:
            if save:
                path_to_save = os.path.normpath(
                    self.path + os.sep + os.pardir + os.sep + os.pardir
                    ) + '\\velocity_around_defect\\'
                name = self.name + '.npy' 
                np.save(path_to_save + name,cropped)



        
        


        pass

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

    
