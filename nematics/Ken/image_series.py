# import os
# import scipy as sp
# import numpy as np
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import cv2
# from defects import *
# from natsort import natsorted
# import glob
from tqdm import tqdm
from image import *
import concurrent.futures
from joblib import Parallel, delayed


class image_series:
    """
    this class is ment to be activated to automatically apply analysis process to
    image/ video.

    parameters
    ---------
    path :  str
        path to folder that contains the images.

    orientation_window : int default = 40
        square window size of size orientation_window.

    resize_image :tuple, default = None
        size for image resize to speedup processing time.
    save_defects : saves the defects data frame into "csv" folder inside the images folder
    save_flow : saves the flow array (currently as mat) into "velocity" folder (inside IO folder)
    save_orientation : saves the flow array (currently as mat) into "velocity" folder (inside IO folder)
    """

    def __init__(
            self,
            folder_path,
            orientation_window=25,
            Optical_Flow_window=15,
            velocity_around_defect_window=600,
            resize_image=None,
            save_all=True,
            save_defects=True,
            save_flow=True,
            save_orientation=True,
            save_velocity=True):

        self.path = folder_path
        self.orientation_window = orientation_window
        self.OF_window = Optical_Flow_window
        self.velocity_around_defect_window = velocity_around_defect_window
        self.resize_image = resize_image

        self.save_defects = True if save_all else save_defects
        self.save_flow = True if save_all else save_flow
        self.save_orientation = True if save_all else save_orientation
        self.save_velocity = True if save_all else save_velocity

        # paths to results folders (in the parent folder to the folder that contains the images)
        self.defects_csv_path = os.path.normpath(
            self.path + os.sep + os.pardir
        ) + '\\defects_csv\\'  # go back one dir and go to velocity dir
        self.velocity_path = os.path.normpath(
            self.path + os.sep + os.pardir
        ) + '\\velocity\\'
        self.orientation_path = os.path.normpath(
            self.path + os.sep + os.pardir
        ) + '\\orientation\\'

        self.velocity_around_minus_path = os.path.normpath(
            self.path + os.sep + os.pardir
        ) + '\\velocity_around_minus\\'
        self.velocity_around_plus_path = os.path.normpath(
            self.path + os.sep + os.pardir
        ) + '\\velocity_around_plus\\'

        self.create_folders()
        self.fetch_data()

    def create_folders(self):
        """
        create the folders for the output in case they don't exist
        """
        if not os.path.exists(self.orientation_path):
            os.makedirs(self.orientation_path)
        if not os.path.exists(self.defects_csv_path):
            os.makedirs(self.defects_csv_path)
        if not os.path.exists(self.velocity_path):
            os.makedirs(self.velocity_path)
        if not os.path.exists(self.velocity_around_minus_path):
            os.makedirs(self.velocity_around_minus_path)
        if not os.path.exists(self.velocity_around_plus_path):
            os.makedirs(self.velocity_around_plus_path)

    def fetch_data(self):
        """
        fetches the data into self images
        """
        self.images = []
        image_list = glob.glob(self.path + r"\*tif")
        image_list = natsorted(image_list, key=lambda y: y.lower())

        for image_path in image_list:
            self.images.append(image(
                image_path,
                self.resize_image,
                self.orientation_window))

    def calc_orientation(self, save=None, window_size=None):
        """
        used to calculate only the orientation and saving.
        note that the orientation should already be saved if you ran detect defects.
        parameters
        ---------
        save :  Bool
            can be used to override the setting in the constructor leave as None if not needed
        window_size : will be as set in the builder unless changed here
        """

        window_size = self.orientation_window if window_size is None else window_size

        self.save_orientation = self.save_orientation if save is None else save
        for img in self.images:
            ori = img.calc_orientation(self.orientation_path, self.save_orientation, window_size)

            if self.save_orientation:
                name = 'orientation_from_' + img.name + '.npy'
                np.save(self.orientation_path + name, ori)

    def detect_defects_serial(self, window_size=None):
        """
        goes over images and detects the defects as -0.5/ +0.5
        and adds them to df accordingly.

        returns
        -------
        tuple : (PlusHalf, MinusHalf)
        PlusHalf : pd.DataFrame
            df of plus half defects detected

        MinusHalf : pd.DataFrame
            df of minus half defects detected

            columns of DF
            'from_img' : tha name that the defect is taken from
            'charge' : -0.5 or 0.5
            'x' : the coordinate of x-axis that the defect was detected
            'y' : the coordinate of y-axis that the defect was detected
            'ang1' : the angle (radians) of the defect
            'ang2' : 2nd angle (only in minus defects)
            'ang3' : 3rd angle (only in minus defects)

        """
        window_size = self.OF_window if window_size is None else window_size

        PlusHalf, MinusHalf = pd.DataFrame(), pd.DataFrame()
        for img in tqdm(self.images):
            plus, minus = img.detect_defects(self.orientation_path,
                                             self.save_orientation,
                                             self.defects_csv_path,
                                             self.save_defects,
                                             window_size)

            # add to total DF
            PlusHalf = pd.concat([PlusHalf, plus])
            MinusHalf = pd.concat([MinusHalf, minus])

        if self.save_defects:
            PlusHalf[['from_img', 'charge', 'x', 'y', 'ang1']].to_csv(self.defects_csv_path + r"PlusHalf.csv")
            MinusHalf[['from_img', 'charge', 'x', 'y', 'ang1', 'ang2', 'ang3']].to_csv(
                self.defects_csv_path + r"MinusHalf.csv")

        return PlusHalf, MinusHalf


    def detect_defects(self, window_size=None, num_threads=8):
        """
        goes over images and detects the defects as -0.5/ +0.5
        and adds them to df accordingly.

        returns
        -------
        tuple : (PlusHalf, MinusHalf)
        PlusHalf : pd.DataFrame
            df of plus half defects detected

        MinusHalf : pd.DataFrame
            df of minus half defects detected

            columns of DF
            'from_img' : tha name that the defect is taken from
            'charge' : -0.5 or 0.5
            'x' : the coordinate of x-axis that the defect was detected
            'y' : the coordinate of y-axis that the defect was detected
            'ang1' : the angle (radians) of the defect
            'ang2' : 2nd angle (only in minus defects)
            'ang3' : 3rd angle (only in minus defects)

        """
        window_size = self.orientation_window if window_size is None else window_size

        PlusHalf, MinusHalf = pd.DataFrame(), pd.DataFrame()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(img.detect_defects, self.orientation_path,
                                       self.save_orientation,
                                       self.defects_csv_path,
                                       self.save_defects,
                                       window_size) for img in self.images]
            for future in concurrent.futures.as_completed(futures):
                plus, minus = future.result()

                # add to total DF
                PlusHalf = pd.concat([PlusHalf, plus])
                MinusHalf = pd.concat([MinusHalf, minus])

        if self.save_defects:
            PlusHalf[['from_img', 'charge', 'x', 'y', 'ang1']].to_csv(self.defects_csv_path + r"PlusHalf.csv",index=False)
            MinusHalf[['from_img', 'charge', 'x', 'y', 'ang1', 'ang2', 'ang3']].to_csv(
                self.defects_csv_path + r"MinusHalf.csv",index=False)

        return PlusHalf, MinusHalf

    def __repr__(self) -> str:
        return f"image_series : \n  number of images = {len(self.images)}"

    def optical_flow(self, window_size=None):

        """
        calculates the optical flow for image_list and save as a np.array
        saves if save_flow == True to the previous directory at folder velocity
        doesn't add the directory
        """

        window = self.OF_window if window_size is None else window_size
        for (i, im1), im2 in (zip(enumerate(tqdm(self.images[:-1])), self.images[1:])):

            img1 = im1.img
            img2 = im2.img

            flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3,
                                                winsize=window,
                                                iterations=3,
                                                poly_n=5,
                                                poly_sigma=1.2,
                                                flags=0)

            if self.save_flow:
                name = 'velocity_from_' + im1.name.split('.')[0] + '.npy'
                np.save(self.velocity_path + name, flow)

        return None

    def velocity_averaging(self, window_size=None, plot=False):
        """
        takes all the images in the velocity and defect folders, crops a window around each 
        defect and tilts the velocity field around the defect.
        save
        """

        window = self.velocity_around_defect_window if window_size is None else window_size
        half_window = np.floor(window / 2)
        final_size = int(np.floor(half_window / np.sqrt(2)))
        # mean_arr_plus = np.zeros(shape=(final_size, final_size, 2))
        # mean_arr_minus = np.zeros_like(mean_arr_plus)
        total_number_of_minus, total_number_of_plus = 0, 0
        minus_defect_num = []
        plus_defect_num = []

        for img_num, img in enumerate(tqdm(self.images)):

            # there is 1 less flow image thus we skip the last image
            if img_num + 1 >= len(self.images):
                break

            # get the dataframe and the array
            min_path = self.defects_csv_path + img.name + '_MinusHalf.csv'
            plus_path = self.defects_csv_path + img.name + '_PlusHalf.csv'
            flow_path = self.velocity_path + 'velocity_from_' + img.name + '.npy'
            minus_df = pd.read_csv(min_path, header=0, index_col=0)
            plus_df = pd.read_csv(plus_path, header=0, index_col=0)
            flow_arr = np.load(flow_path)

            # minus
            # number_of_defects_minus, mean_arr_minus = img.crop_and_tilt(defects_df=minus_df,
            #                                                             velocity_array=flow_arr,
            #                                                             half_window=half_window,
            #                                                             save=False,
            #                                                             path=self.velocity_around_minus_path,
            #                                                             plot=plot)
            #plus
            number_of_defects_plus, mean_arr_plus = img.crop_and_tilt(defects_df=plus_df,
                                                                      velocity_array=flow_arr,
                                                                      half_window=half_window,
                                                                      save=False,
                                                                      path=self.velocity_around_plus_path,
                                                                      plot=plot)

            total_number_of_plus += number_of_defects_plus
            # total_number_of_minus += number_of_defects_minus
            plus_defect_num.append(number_of_defects_plus)
            # minus_defect_num.append(number_of_defects_minus)

            # save
            if self.save_velocity:
                # np.save(self.velocity_around_minus_path + '_mean_arr_from_' + img.name + '.npy', mean_arr_minus)
                np.save(self.velocity_around_plus_path + '_mean_arr_from_' + img.name + '.npy', mean_arr_plus)

        # calculate average over the mean arrays
        final_plus = np.zeros_like(mean_arr_plus)
        # final_minus = np.zeros_like(mean_arr_minus)

        # add plus
        array_list = glob.glob(self.velocity_around_plus_path + r"\*npy")
        array_list = natsorted(array_list, key=lambda y: y.lower())

        arr_shape_for_mean_calc = (len(array_list), *mean_arr_plus.shape)
        agg_plus = np.zeros(shape=arr_shape_for_mean_calc)
        agg_minus = np.zeros(shape=arr_shape_for_mean_calc)

        for idx, arr_path in enumerate(array_list):

            # there is 1 less flow image thus we skip the last image
            if idx + 1 >= len(self.images):
                break

            # agg_plus[idx] = np.load(arr_path)
            arr = np.load(arr_path)
            arr = arr * plus_defect_num[idx] / total_number_of_plus
            final_plus += arr
        # final_plus = agg_plus.mean(axis=0)

        # add minus
        # todo: return before submition
        # array_list = glob.glob(self.velocity_around_minus_path + r"\*npy")
        # array_list = natsorted(array_list, key=lambda y: y.lower())
        #
        # for idx, arr_path in enumerate(array_list):
        #
        #     # there is 1 less flow image thus we skip the last image
        #
        #     if idx + 1 >= len(self.images):
        #         break
        #     # arr = np.load(arr_path)
        #     #change the
        #     # arr = arr * minus_defect_num[idx] / total_number_of_minus
        #     # arr = arr/
        #     agg_minus[idx] = np.load(arr_path)
        # final_minus += agg_minus.mean(axis=0)
        #
        # # save final
        # if self.save_velocity:
        #     np.save(self.velocity_around_minus_path + 'final_average_minus' + '.npy', final_minus)
        #     np.save(self.velocity_around_plus_path + 'final_average_plus' + '.npy', final_plus)


    def velocity_averaging_2(self, window_size=None):

        window = self.velocity_around_defect_window if window_size is None else window_size
        half_window = np.floor(window / 2)
        final_size = int(np.floor(half_window / np.sqrt(2)))
        mean_arr_plus = np.zeros(shape=(final_size, final_size, 2))
        mean_arr_minus = np.zeros_like(mean_arr_plus)
        total_number_of_minus, total_number_of_plus = 0, 0
        minus_defect_num = []
        plus_defect_num = []

        def process_image(image):
            min_path = self.defects_csv_path + image.name + '_MinusHalf.csv'
            plus_path = self.defects_csv_path + image.name + '_PlusHalf.csv'
            flow_path = self.velocity_path + 'velocity_from_' + image.name + '.npy'
            minus_df = pd.read_csv(min_path, header=0, index_col=0)
            plus_df = pd.read_csv(plus_path, header=0, index_col=0)
            flow_arr = np.load(flow_path)

            number_of_defects_minus, mean_arr_minus = image.crop_and_tilt(defects_df=minus_df,
                                                                          velocity_array=flow_arr,
                                                                          half_window=half_window)
            number_of_defects_plus, mean_arr_plus = image.crop_and_tilt(defects_df=plus_df,
                                                                        velocity_array=flow_arr,
                                                                        half_window=half_window)

            return number_of_defects_minus, mean_arr_minus, number_of_defects_plus, mean_arr_plus

        results = Parallel(n_jobs=-1)(delayed(process_image)(image) for image in self.images[:-2])

        for idx, image in enumerate(self.images[:-2]):
            number_of_defects_minus, mean_arr_minus, number_of_defects_plus, mean_arr_plus = results[idx]
            total_number_of_minus += number_of_defects_minus
            total_number_of_plus += number_of_defects_plus
            minus_defect_num.append(number_of_defects_minus)
            plus_defect_num.append(number_of_defects_plus)

            if self.save_velocity:
                np.save(self.velocity_around_minus_path + '_mean_arr_from_' + image.name + '.npy', mean_arr_minus)
                np.save(self.velocity_around_plus_path + '_mean_arr_from_' + image.name + '.npy', mean_arr_plus)

        final_plus = self.calculate_final_average(self.velocity_around_plus_path, plus_defect_num, total_number_of_plus)
        final_minus = self.calculate_final_average(self.velocity_around_minus_path, minus_defect_num,
                                                   total_number_of_minus)

        if self.save_velocity:
            np.save(self.velocity_around_minus_path + 'final_average_minus' + '.npy', final_minus)
            np.save(self.velocity_around_plus_path + 'final_average_plus' + '.npy', final_plus)


    def calculate_final_average(self, path, defect_nums, total_defects):
        final_average = np.zeros_like(np.load(path + "_mean_arr_from_" + self.images[0].name + ".npy"))

        array_list = natsorted(glob.glob(path + r"\*npy"), key=lambda y: y.lower())
        array_list = array_list[:len(self.images) - 2]

        for i, arr_path in enumerate(array_list):
            arr = np.load(arr_path)
            arr = arr * defect_nums[i] / total_defects
            final_average += arr

        return final_average


    def run_all(self):
        self.optical_flow()
        # self.calc_orientation()
        self.detect_defects()
        self.velocity_averaging()

