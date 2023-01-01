# import torch
# import torchvision
import os
import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from skimage import feature, measure, restoration
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
        square window size of size window_size

    resize_image :tuple, default = None TODO: chack if size is only square
        size for image resize to speedup processing time
    """


    def __init__(
        self,
        folder_path,
        window_size = 40, 
        resize_image = None,
        save_defects = False,
        save_flow = False
        ) -> None:
        
        self.path = folder_path
        self.window_size = window_size
        self.save = save_defects
        self.save_flow = save_flow
        self.resize_image = resize_image
        self.fetch_data()


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


        

    def detect_minus_plus(self):
        """
        goes over images and detects the defects as -0.5/ +0.5 
        and adds them to df accordingly.

        returns
        -------
        plushalf : pd.DataFrame
            df of plus half defects detected
            columns =[]TODO add columns 

        minushalf : pd.DataFrame
            df of minus half defects detected
        """
        
        plushalf, minushalf = pd.DataFrame(),pd.DataFrame()
        for img_idx, img in enumerate(self.images):
            
            name = img.name
            plus, minus= img.detect_minus_plus()
            if self.save_defects:
            
            #save_defects single image 

                plus[['charge', 'x','y','ang1']].to_csv(self.path + r"\\csv\\" +name + '_PlusHalf.csv')
                minus[['charge', 'x','y','ang1','ang2','ang3']].to_csv(
                    self.path + r"\\csv\\" + name + '_MinusHalf.csv')
             
            #add to total DF
            plus['from_img'] = name
            minus['from_img'] = name

            plushalf = pd.concat([plushalf, plus])
            minushalf = pd.concat([minushalf, minus])

        if self.save_defects:
            plushalf[['from_img','charge', 'x','y','ang1']].to_csv(self.path + r"\csv\PlusHalf.csv")
            minushalf[['from_img','charge', 'x','y','ang1','ang2','ang3']].to_csv(
                self.path + r"\csv\MinusHalf.csv")

            

        return plushalf, minushalf
                    

    def __repr__(self) -> str:
        return f"workflow : \n  number of images = {len(self.images)}"


    def get_flow(self) -> np.array:

        """
        calculates the optical flow for image_list and returns as an np.array
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


            #save a single image mat 
            if self.save_flow:
                name = 'velocity_from_' +im1.name.split('.')[0] + '.mat' #
                sp.io.savemat(path_to_save+name, dict(u=flow[:,:,0], v=flow[:,:,1]))
            
        #save the big array as mat
        # if self.save_flow:
        #     # sp.io.savemat(path_to_save + 'big_array.mat',dict(u=big_arr[:,:,:,0], v=big_arr[:,:,:,1]))
        #     np.save(path_to_save + 'big_array.npy',big_arr)



        return None#big_arr
            

        



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
        self.name = path.split('\\')[-1]
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


        ori, coh, E = orientation_analysis(self.img, self.window_size)
        k = compute_topological_charges(ori, int_area='cell', origin='upper')
        defects = localize_defects(k, x_grid=xx, y_grid=yy)
        compute_defect_orientations(ori, defects)
        plushalf = defects[defects['charge']==.5]
        minushalf = defects[defects['charge']==-.5]

        
        
        return plushalf, minushalf

    
                    
