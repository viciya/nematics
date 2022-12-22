# import torch
# import torchvision
import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from skimage import feature, measure, restoration
from defects import *
from natsort import natsorted
import glob



class workflow:
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
        resize_image = None ) -> None:
        
        self.path = folder_path
        self.window_size = window_size
        
        self.resize_image = resize_image
        self.fetch_data()


    def fetch_data(self): 
        """
        fetches the data into self images
        """
        self.images =[]
        image_list = glob.glob(self.path)
        image_list = natsorted(image_list, key=lambda y: y.lower())

        for image_path in image_list:
            self.images.append( image( image_path))


        

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
            
            plus, minus= img.detect_minus_plus()
            
            plus['from_img'] = img_idx
            minus['from_img'] = img_idx

            plushalf = pd.concat([plushalf, plus])
            minushalf = pd.concat([minushalf, minus])


        return plushalf, minushalf
                    

    def __repr__(self) -> str:
        return f"workflow : \n  number of images = {len(self.images)}"





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
                    
