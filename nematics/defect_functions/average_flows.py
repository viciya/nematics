import matplotlib.pyplot as plt
import cv2
import numpy as np
from natsort import natsorted
import sys
import pandas as pd
from scipy.ndimage import rotate, gaussian_filter
from scipy.stats import circmean, circstd, sem
import time

sys.path.append("../vasco_scripts")  # add the relative path to the folder
sys.path.append("../defect_functions")
from defects import *  # import the module from the folder
from defect_pairs import *


def divergence_npgrad(flow):
    flow = np.swapaxes(flow, 0, 1)
    Fx, Fy = flow[:, :, 0], flow[:, :, 1]
    dFx_dx = np.gradient(Fx, axis=0)
    dFy_dy = np.gradient(Fy, axis=1)
    return (dFx_dx + dFy_dy).T


def curl_npgrad(flow):
    flow = np.swapaxes(flow, 0, 1)
    Fx, Fy = flow[:, :, 0], flow[:, :, 1]
    dFx_dy = np.gradient(Fx, axis=1)
    dFy_dx = np.gradient(Fy, axis=0)
    curl = dFy_dx - dFx_dy
    return curl.T


def shear_rate_2d(flow):
    """
    Calculates the magnitude of the shear rate for a 2D flow field.
    flow: numpy array of shape (height, width, 2)
          where flow[..., 0] is Fx (u) and flow[..., 1] is Fy (v)
    """
    Fx, Fy = flow[:, :, 0], flow[:, :, 1]

    # Partial derivatives
    # Note: np.gradient axis=0 is vertical (y), axis=1 is horizontal (x)
    dFx_dy = np.gradient(Fx, axis=0)  # du/dy
    dFx_dx = np.gradient(Fx, axis=1)  # du/dx
    dFy_dy = np.gradient(Fy, axis=0)  # dv/dy
    dFy_dx = np.gradient(Fy, axis=1)  # dv/dx

    # Calculate the magnitude of the shear rate tensor
    # Formula: sqrt( 2*(du/dx)^2 + 2*(dv/dy)^2 + (du/dy + dv/dx)^2 )
    shear_rate = np.sqrt(2 * (dFx_dx**2) + 2 * (dFy_dy**2) + (dFx_dy + dFy_dx) ** 2)

    return shear_rate


def crop(img, center, width, height):
    ulx, uly = max(int(center[0] - width // 2), 0), max(int(center[1] - height // 2), 0)
    lrx, lry = min(int(center[0] + width // 2), img.shape[1]), min(
        int(center[1] + height // 2), img.shape[0]
    )
    new_center = ((lrx - ulx) / 2, (lry - uly) / 2)
    return img[uly:lry, ulx:lrx], new_center


def rotate_vector(vector, angle):
    """rotate vectors"""
    x = vector[0] * np.cos(angle) - vector[1] * -np.sin(angle)
    y = vector[0] * -np.sin(angle) + vector[1] * np.cos(angle)
    return [x, y]


def rotate_flow_field(flow, angle):
    """rotate flow field"""
    uv_rot = rotate_vector(flow, angle)
    u = rotate(uv_rot[0], angle * 180 / np.pi)
    v = rotate(uv_rot[1], angle * 180 / np.pi)
    return [u, v]


def defect_flow_frame_average(
    img1,
    img2,
    df_frame,
    defect_type="up",
    vorticity=None,
    vortTh=0.001,
    box=(300, 300),
    filt=1,
    sigma=15,
    edge=0,
):
    """
    vorticity = None/right/left
    """

    im_h, im_w = img1.shape
    width, height = box[0], box[1]
    width1, height1 = int(width / 2**0.5), int(height / 2**0.5)
    flow = cv2.calcOpticalFlowFarneback(
        img1,
        img2,
        None,
        0.5,
        3,
        winsize=sigma,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    if filt != 1:
        flow = gaussian_filter(flow, sigma=filt)

    if defect_type == "up":
        df = df_frame[df_frame.fuse_up].copy()
    elif defect_type == "down":
        df = df_frame[~df_frame.fuse_up].copy()
    else:
        df = df_frame.copy()

    u_frame = np.zeros((height1, width1), dtype=np.float16)
    v_frame = np.zeros_like(u_frame)
    count = 0

    x, y, th = (
        ["xm", "ym", "angm1"] if defect_type == "minus" else ["xp", "yp", "angp1"]
    )

    # Selects defect wich close to the edge
    if edge:
        df = df[((df[x] < edge) | (df[x] > im_w - edge))]

    for i in range(len(df[x])):
        try:
            # center at defect position
            cnt = (int(df[x].iloc[i]), int(df[y].iloc[i]))
            if (
                (cnt[0] > width // 2)
                and (cnt[0] < im_w - width // 2)
                and (cnt[1] > height // 2)
                and (cnt[1] < im_h - height // 2)
            ):
                # 1 crop each component of velocity field
                # image_crop = crop(255-img_clahe, cnt, width, height)[0] *** image
                u, _ = crop(flow[:, :, 0], cnt, width, height)
                v, _ = crop(flow[:, :, 1], cnt, width, height)

                if vorticity:
                    vort = curl_npgrad(np.stack((u, v), axis=-1)).mean()
                    if vorticity == "right" and vort < vortTh:
                        continue
                    elif vorticity == "left" and vort > -vortTh:
                        continue

                # 2 rotate velocity field (1. rotate vectors 2. rotate positions)
                # image_rot = rotate(image_crop, df["angp1"].iloc[i] * 180/np.pi) *** image
                uv_rot = rotate_flow_field((u, v), df[th].iloc[i])

                # 3 crop again to smaller box (box**0.5)
                cnt_crop = uv_rot[0].shape[1] / 2, uv_rot[0].shape[0] / 2
                # image_rot_crop = crop(image_rot, cnt_crop, width1, height1)[0] *** image
                u_frame = u_frame + crop(uv_rot[0], cnt_crop, width1, height1)[0]
                v_frame = v_frame + crop(uv_rot[1], cnt_crop, width1, height1)[0]
                count += 1
        except:
            pass
        #      break

    if count:
        print(u_frame.shape[1], u_frame.shape[0])
        print("Exeptions: %s" % (len(df[x]) - count))
        return u_frame / count, v_frame / count, count


def defect_flow_frame_average_with_edge(
    img1,
    img2,
    df_frame,
    defect_type="up",
    vorticity=None,
    vortTh=0.001,
    box=(300, 300),
    filt=1,
    sigma=15,
    edge=0,
):
    """
    Calculate average flow around defects including those near boundaries.
    Pads the velocity field once to avoid per-defect edge checks, and
    vectorizes coordinate access. Returns (u_avg,v_avg,count) or None.
    vorticity: None/"right"/"left" filter based on circulation sign.
    """
    # convert to 8-bit if necessary
    if img1.dtype != np.uint8:
        img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img2.dtype != np.uint8:
        img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    im_h, im_w = img1.shape
    width, height = box
    width1, height1 = int(width / 2**0.5), int(height / 2**0.5)

    flow = cv2.calcOpticalFlowFarneback(
        img1,
        img2,
        None,
        0.5,
        3,
        winsize=sigma,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    # remove global drift
    flow -= flow.mean(axis=(0, 1), keepdims=True)
    if filt != 1:
        flow = gaussian_filter(flow, sigma=filt)

    # subset defects
    if defect_type == "up":
        df = df_frame[df_frame.fuse_up].copy()
    elif defect_type == "down":
        df = df_frame[~df_frame.fuse_up].copy()
    else:
        df = df_frame.copy()

    # keep only edge defects if requested
    x_col, y_col, th_col = (
        ("xm", "ym", "angm1") if defect_type == "minus" else ("xp", "yp", "angp1")
    )
    if edge:
        df = df[(df[x_col] < edge) | (df[x_col] > im_w - edge)]
    xs = df[x_col].to_numpy(dtype=int)
    ys = df[y_col].to_numpy(dtype=int)
    thetas = df[th_col].to_numpy(dtype=float)

    # pad the flow field so that any crop is always valid
    pad_w = width // 2
    pad_h = height // 2
    flow_pad = np.pad(flow, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="constant")

    u_frame = np.zeros((height1, width1), dtype=np.float32)
    v_frame = np.zeros_like(u_frame)
    count = 0

    # iterate over defect positions
    for cx, cy, theta in zip(xs, ys, thetas):
        # map to padded coordinates
        pcx = cx + pad_w
        pcy = cy + pad_h
        # crop straight from padded array (zeros outside original image)
        u = flow_pad[pcy - pad_h : pcy + pad_h, pcx - pad_w : pcx + pad_w, 0]
        v = flow_pad[pcy - pad_h : pcy + pad_h, pcx - pad_w : pcx + pad_w, 1]

        if vorticity:
            vort = curl_npgrad(np.stack((u, v), axis=-1)).mean()
            if vorticity == "right" and vort < vortTh:
                continue
            if vorticity == "left" and vort > -vortTh:
                continue

        uv_rot = rotate_flow_field((u, v), theta)
        # central crop to reduced box size
        hnew, wnew = uv_rot[0].shape
        cyc, cxc = hnew // 2, wnew // 2
        u_frame += uv_rot[0][
            cyc - height1 // 2 : cyc + height1 // 2,
            cxc - width1 // 2 : cxc + width1 // 2,
        ]
        v_frame += uv_rot[1][
            cyc - height1 // 2 : cyc + height1 // 2,
            cxc - width1 // 2 : cxc + width1 // 2,
        ]
        count += 1

    if count:
        return (
            u_frame.astype(np.float16) / count,
            v_frame.astype(np.float16) / count,
            count,
        )


def defect_flow_frame_all_frames(
    img1,
    img2,
    df_frame,
    defect_type="up",
    vorticity=None,
    vortTh=0.001,
    box=(300, 300),
    filt=1,
    sigma=15,
):
    """
    vorticity = None/right/left
    """

    im_h, im_w = img1.shape
    width, height = box[0], box[1]
    width1, height1 = int(width / 2**0.5), int(height / 2**0.5)
    flow = cv2.calcOpticalFlowFarneback(
        img1,
        img2,
        None,
        0.5,
        3,
        winsize=sigma,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    if filt != 1:
        flow = gaussian_filter(flow, sigma=filt)

    if defect_type == "up":
        df = df_frame[df_frame.fuse_up].copy()
    elif defect_type == "down":
        df = df_frame[~df_frame.fuse_up].copy()
    else:
        df = df_frame.copy()

    u_frame = np.zeros((height1, width1), dtype=np.float16)
    v_frame = np.zeros_like(u_frame)
    u_list, v_list, id_list = [], [], []
    count = 0

    x, y, th = (
        ["xm", "ym", "angm1"] if defect_type == "minus" else ["xp", "yp", "angp1"]
    )

    for i in range(len(df[x])):
        try:
            # center at defect position
            cnt = (int(df[x].iloc[i]), int(df[y].iloc[i]))
            if (
                (cnt[0] > width // 2)
                and (cnt[0] < im_w - width // 2)
                and (cnt[1] > height // 2)
                and (cnt[1] < im_h - height // 2)
            ):
                # 1 crop each component of velocity field

                # image_crop = crop(255-img_clahe, cnt, width, height)[0] *** image
                u, _ = crop(flow[:, :, 0], cnt, width, height)
                v, _ = crop(flow[:, :, 1], cnt, width, height)

                if vorticity:
                    vort = curl_npgrad(np.stack((u, v), axis=-1)).mean()
                    if vorticity == "right" and vort < vortTh:
                        continue
                    elif vorticity == "left" and vort > -vortTh:
                        continue

                # 2 rotate velocity field (1. rotate vectors 2. rotate positions)
                # image_rot = rotate(image_crop, df["angp1"].iloc[i] * 180/np.pi) *** image
                uv_rot = rotate_flow_field((u, v), df[th].iloc[i])

                # 3 crop again to smaller box (box**0.5)
                cnt_crop = uv_rot[0].shape[1] / 2, uv_rot[0].shape[0] / 2

                u_list.append(crop(uv_rot[0], cnt_crop, width1, height1)[0])
                v_list.append(crop(uv_rot[1], cnt_crop, width1, height1)[0])
                id_list.append(df["TRACK_ID"].iloc[i])
                count += 1
        except:
            pass
        #      break

    if count:
        return u_list, v_list, id_list


def orienatation_frame_average(
    img1, df_frame, defect_type="up", box=(300, 300), sigma=11
):
    im_h, im_w = img1.shape
    width, height = box[0], box[1]
    width1, height1 = int(width / 2**0.5), int(height / 2**0.5)
    ori = analyze_defects(img1, sigma=sigma)[0]

    if defect_type == "up":
        df = df_frame[df_frame.fuse_up].copy()
    elif defect_type == "down":
        df = df_frame[~df_frame.fuse_up].copy()
    else:
        df = df_frame.copy()

    ori_list = []
    count = 0

    x, y, th = (
        ["xm", "ym", "angm1"] if defect_type == "minus" else ["xp", "yp", "angp1"]
    )

    for i in range(len(df[x])):
        try:
            # center at defect position
            cnt = (int(df[x].iloc[i]), int(df[y].iloc[i]))
            if (
                (cnt[0] > width // 2)
                and (cnt[0] < im_w - width // 2)
                and (cnt[1] > height // 2)
                and (cnt[1] < im_h - height // 2)
            ):
                # 1 crop the orientation field
                crop_ori = crop(ori, cnt, width, height)[0]

                # 2 rotate the orientation field (1. rotate the angle)
                rot_ori = rotate(
                    crop_ori + df[th].iloc[i], df[th].iloc[i] * 180 / np.pi
                )

                # 3 crop again to smaller box (box**0.5)
                cnt_crop = rot_ori.shape[1] / 2, rot_ori.shape[0] / 2

                ori_list.append(crop(rot_ori, cnt_crop, width1, height1)[0])
                count += 1
        except:
            pass
        #      break

    if count:
        return (
            circmean(
                np.stack(ori_list, axis=-1), axis=-1, low=-np.pi / 2, high=np.pi / 2
            ),
            count,
        )


# --------- MultiTiff Tools ------------
from PIL import Image
import glob


def multiTiff_to_list(tiff):
    img_list = []
    # Initialize a counter for the frames
    frame_count = 0
    # Loop through all frames in the TIFF file
    while True:
        try:
            tiff.seek(frame_count)
            img_list.append(np.array(tiff.copy()))
            frame_count += 1
        except EOFError:
            break
    return img_list


def analyze_image_widths(directory, target_width, dw=10, make_plot=False, ax=None):
    """
    Analyzes the widths of TIFF images in a specified directory and plots a histogram of the widths.

    Parameters:
    - directory: str, the directory path containing the TIFF images.
    - target_width: int, the target width for filtering images.
    - dw: int, the width deviation for filtering images.
    """
    img_list = glob.glob(f"{directory}\\*CROPPED*\\*.tif")
    width_all = []
    tiff_list = []

    for filename in img_list:
        tiff = Image.open(filename)
        image_width = tiff.size[0]

        if (target_width - dw) / 0.74 < image_width < (target_width + dw) / 0.74:
            tiff_list.append(filename)
            width_all.append(image_width)

    if make_plot:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.hist(0.74 * np.array(width_all), bins=len(width_all) // 10, rwidth=0.9)
        ax.set_title(f"Total: {len(width_all)}")
        ax.set_xlabel("$Width$", fontsize=12)
        ax.set_ylabel("$Count$", fontsize=12)

    return tiff_list


def uv_time_average(tiff_path):
    """
    calulates average flow vx, vy from mulitiff path
    """

    tiff = Image.open(tiff_path)
    tiff_frame_list = multiTiff_to_list(tiff)

    u = np.zeros_like(tiff_frame_list[0], dtype=np.float32)
    v = np.zeros_like(u)

    for (i, img1), img2 in zip(enumerate(tiff_frame_list[:-1]), tiff_frame_list[1:]):
        flow = cv2.calcOpticalFlowFarneback(
            img1,
            img2,
            None,
            0.5,
            3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        # flow[:,:,0] = gaussian_filter(flow[:,:,0], sigma=15)
        # flow[:,:,1] = gaussian_filter(flow[:,:,1], sigma=15)
        u += flow[..., 0]
        v += flow[..., 1]

        if i == 20:
            vprofile = (v / i).mean(axis=0)
            if (vprofile[-20:].mean() - vprofile[:20].mean()) < 1.0:  # 1.:
                return  # Return None if the condition is true

    return np.stack((u / i, v / i), axis=-1).astype(np.float16)


def plot_flow_and_nematics(
    image_list, im_num, axs=None, flow_sigma=15, ori_sigma=21, box=None, bg=True
):
    if box is not None:
        w0, h0, width, height = box[0], box[1], box[2], box[3]
        img1 = cv2.imread(image_list[im_num])[w0 : w0 + width, h0 : h0 + height, 0]
        img2 = cv2.imread(image_list[im_num + 1])[w0 : w0 + width, h0 : h0 + height, 0]
    else:
        img1 = cv2.imread(image_list[im_num])[:, :, 0]
        img2 = cv2.imread(image_list[im_num + 1])[:, :, 0]

    flow = cv2.calcOpticalFlowFarneback(
        img1,
        img2,
        None,
        0.5,
        3,
        winsize=flow_sigma,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    step = int(flow_sigma * 1.5)

    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    axs[0].axis("off")
    axs[1].axis("off")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img1)
    axs[0].imshow(255 - img_clahe, cmap="gray")

    y, x = np.mgrid[0 : img1.shape[0], 0 : img1.shape[1]]
    axs[0].quiver(
        x[::step, ::step],
        y[::step, ::step],
        flow[::step, ::step, 0],
        -flow[::step, ::step, 1],
        color="red",
        scale=80,
        alpha=0.5,
        width=0.005,
    )

    ori, plus, min = analyze_defects(img1, sigma=ori_sigma)

    s = int(flow_sigma * 1)
    if bg:
        axs[1].imshow(255 - img_clahe, cmap="gray")
    else:
        axs[1].imshow(np.zeros_like(img1, dtype=np.float32), cmap="gray")

    quiver = axs[1].quiver(
        x[::s, ::s],
        y[::s, ::s],
        np.cos(ori)[::s, ::s],
        np.sin(ori)[::s, ::s],
        np.arctan2(np.sin(ori), np.cos(ori))[::s, ::s],
        headaxislength=0,
        headwidth=0,
        headlength=0,
        width=0.005,
        scale=60,
        pivot="mid",
        alpha=0.5,
        cmap="hsv",
    )

    alpha_half, scale_half = 0.8, 15
    axs[1].plot(plus["x"], plus["y"], "ro", markersize=8, alpha=alpha_half)
    axs[1].quiver(
        plus["x"],
        plus["y"],
        np.cos(plus["ang1"]),
        -np.sin(plus["ang1"]),
        headaxislength=0,
        headwidth=0,
        headlength=0,
        color="r",
        scale=scale_half,
        alpha=alpha_half,
    )

    axs[1].plot(
        min["x"], min["y"], "o", markersize=6, alpha=alpha_half, color="dodgerblue"
    )
    for j in range(3):
        axs[1].quiver(
            min["x"],
            min["y"],
            np.cos(min["ang" + str(j + 1)]),
            -np.sin(min["ang" + str(j + 1)]),
            headaxislength=0,
            headwidth=0,
            headlength=0,
            color="dodgerblue",
            scale=scale_half,
            alpha=alpha_half,
        )
