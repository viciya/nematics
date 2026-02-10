import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd
from skimage.measure import regionprops, label
from natsort import natsorted
from cellpose import plot
import trackpy as tp
import sys
import os

def calculate_neighbors(mask_image, area_threshold=60, boundary_margin=5):
    """
    Calculate neighbors and centroids for each region in the mask image.
    
    Parameters:
    - mask_image: The binary mask image where regions are labeled.
    - area_threshold: Minimum area for a region to be considered (default is 60).
    - boundary_margin: Margin from the image boundary to exclude contours (default is 5).
    
    Returns:
    - neighbors: A dictionary containing the centroids and neighboring cells for each region.
    """
    height, width = mask_image.shape
    neighbors = {}
    
    # Get region properties
    regions = regionprops(mask_image, intensity_image=mask_image)
    
    for region in regions:
        area = region.area
        color = int(region.mean_intensity)
        
        if area > area_threshold:
            mask = np.uint8(mask_image == color)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                centroid = region.centroid  # Get the centroid of the region
                area = region.area
                perimeter = region.perimeter
                if color not in neighbors:
                    neighbors[color] = {'Centroid': centroid, 
                                        'Area': area, 
                                        'Perimeter': perimeter, 
                                        'Neighboring Cells': set()}

                for contour in contours:
                    # Check if the contour is within the boundary margin
                    if all(boundary_margin <= point[0][0] < width - boundary_margin and 
                           boundary_margin <= point[0][1] < height - boundary_margin for point in contour):                    
                
                    # for contour in contours:
                        for point in contour:
                            x, y = point[0]
                            # Check neighboring pixels
                            for i in range(max(0, y - 1), min(height, y + 2)):
                                for j in range(max(0, x - 1), min(width, x + 2)):
                                    if mask_image[i, j] != color and mask_image[i, j] != 0:
                                        neighbors[color]['Neighboring Cells'].add(mask_image[i, j])
    return neighbors


def create_dataframe(neighbors):
    """
    Create a DataFrame from the neighbors dictionary, excluding empty neighbors.
    
    Parameters:
    - neighbors: A dictionary containing neighboring cells and centroids.
    
    Returns:
    - A pandas DataFrame with cells, their centroids, number of neighbors, and the list of neighbors.
    """
    data = []
    for k, v in neighbors.items():
        # if v['Neighboring Cells']:  # Only include if there are neighboring cells
        data.append({'Cell': k, 
                        'x': v['Centroid'][1], 
                        'y': v['Centroid'][0], 
                        'Area': v['Area'], 
                        'Perimeter': v['Perimeter'], 
                        'Neighbors Num': len(v['Neighboring Cells']), 
                        'Neighbors': list(v['Neighboring Cells'])})
    return pd.DataFrame(data)