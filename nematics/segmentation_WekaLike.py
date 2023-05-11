# %% 
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import data, segmentation, feature, future
from sklearn.ensemble import RandomForestClassifier
from functools import partial
import pickle 
import pandas as pd
import glob
import os
from vasco_scripts.defects  import *
import pathlib
import seaborn as sns
from pip._internal.cli.progress_bars import get_download_progress_renderer
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# %matplotlib qt

import sys,time,random
def progressBar(count_value, total, suffix=''):
    bar_length = 100
    filled_up_Length = int(round(bar_length* count_value / float(total)))
    percentage = round(100.0 * count_value/float(total),1)
    bar = '=' * filled_up_Length + '-' * (bar_length - filled_up_Length)
    sys.stdout.write('[%s] %s%s ...%s\r' %(bar, percentage, '%', suffix))
    sys.stdout.flush()




# %% 
'''Train Random Forest Model'''
# full_img = cv2.imread(r"C:\Users\victo\Downloads\Normal_Epidermis_and_Dermis_with_Intradermal_Nevus_10x.jfif")
# full_img = cv2.imread(r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)\DivMask\rand_forest\Trans_400_raw_crop.tif")
# mask = cv2.imread(r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)\DivMask\rand_forest\Trans_400_raw_crop_label.png")

# img = full_img#[:,:,0]#[:900, :900]
# training_labels = (mask[:,:,0]/mask[:,:,0].max()).astype(np.uint8)
# training_labels = -training_labels+2

# sigma_min = 1
# sigma_max = 12
# features_func = partial(feature.multiscale_basic_features,
#                         intensity=True, edges=False, texture=True,
#                         sigma_min=sigma_min, sigma_max=sigma_max,
#                         channel_axis=-1)
# features = features_func(img)
# clf = RandomForestClassifier(n_estimators=50, n_jobs=-1,
#                              max_depth=50, max_samples=0.05)
# clf = future.fit_segmenter(training_labels, features, clf)
# result = future.predict_segmenter(features, clf)

# fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9, 4))
# ax[0].imshow(segmentation.mark_boundaries(img, result, mode='thick'))
# # ax[0].imshow(img)
# ax[0].contour(training_labels)
# # ax[0].imshow(training_labels, alpha=.3, cmap="hot")
# ax[0].set_title('Image, mask and segmentation boundaries')
# ax[1].imshow(img)
# ax[1].imshow(result, alpha=.3)
# ax[1].set_title('Segmentation')
# fig.tight_layout()
# %%
'''Plot Feature importance'''
# fig, ax = plt.subplots(1, 2, figsize=(9, 4))
# l = len(clf.feature_importances_)
# feature_importance = (
#         clf.feature_importances_[:l//3],
#         clf.feature_importances_[l//3:2*l//3],
#         clf.feature_importances_[2*l//3:])
# sigmas = np.logspace(
#         np.log2(sigma_min), np.log2(sigma_max),
#         num=int(np.log2(sigma_max) - np.log2(sigma_min) + 1),
#         base=2, endpoint=True)
# for ch, color in zip(range(3), ['r', 'g', 'b']):
#     ax[0].plot(sigmas, feature_importance[ch][::3], 'o', color=color)
#     ax[0].set_title("Intensity features")
#     ax[0].set_xlabel("$\\sigma$")
# for ch, color in zip(range(3), ['r', 'g', 'b']):
#     ax[1].plot(sigmas, feature_importance[ch][1::3], 'o', color=color)
#     ax[1].plot(sigmas, feature_importance[ch][2::3], 's', color=color)
#     ax[1].set_title("Texture features")
#     ax[1].set_xlabel("$\\sigma$")

# fig.tight_layout()
# %%
'''Make Prediction on Teest Image'''
# full_img = cv2.imread(r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)\Trans__480.tif")
# img_new = full_img[:700, :]

# features_new = features_func(img_new)
# result_new = future.predict_segmenter(features_new, clf)
# fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(6, 6))
# # ax.imshow(segmentation.mark_boundaries(img_new, result_new, mode='thick'))
# ax.imshow(img_new/img_new.max())
# ax.set_title('Image')
# ax.imshow(result_new, alpha=.3)
# # ax[1].imshow(result_new)
# # ax[1].set_title('Segmentation')
# fig.tight_layout()
# %% 
''' Save Model '''
# with open(r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)\DivMask\rand_forest" + "\RandomForestClassifier_seg_divisions_400.pkl", 'wb') as f:
#         pickle.dump([clf, features_func], f)
# %% 
''' Load Model ''' 
with open(r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)\DivMask\rand_forest" + "\RandomForestClassifier_seg_divisions_400.pkl", 'rb') as f:
    clf, features_func = pickle.load(f) 

# %% 
''' Run Model on Folder''' 
# cv2.imwrite(r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)\DivMask\rand_forest\Trans__480_div_mask.png", 255*(result_new-1))

image_list = glob.glob(r"C:\Users\victo\Downloads\SB_lab\HBEC\s2(120-919)\*.tif")
from natsort import natsorted
image_list = natsorted(image_list, key=lambda y: y.lower())

for (i,im) in enumerate(image_list[335:]):
    img = cv2.imread(im)
    features_new = features_func(img)
    result = future.predict_segmenter(features_new, clf)
    # plt.imshow(img/img.max())
    # plt.imshow(result, alpha=.3)

    save_path = os.path.join(
        os.path.dirname(im), 
        'DivMask', 
        os.path.splitext(os.path.basename(im))[0]
        )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path+ '_div_mask.png', 255*(result-1))
    progressBar(i, len(image_list))

    # break
