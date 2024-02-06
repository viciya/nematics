#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 08:55:18 2022

@author: worlit01

Detection and classification of topological defects.
Works for polar and nematic defects. 
Default values are set for nematic field obtained from experimental data.
"""

import scipy as sp
import numpy as np
import pandas as pd

from skimage import feature, measure, restoration

# =============================================================================
# Orientation Analysis
# =============================================================================

def orientation_analysis(img, sigma=1):
    """ Orientation Analysis as in OrientationJ (Fiji). See https://forum.image.sc/t/orientationj-or-similar-for-python/51767/3 for a discussion.
    Also check theoretical-background-orientation.pdf for mathematical concepts. """
    # Compute structure tensor (upper diagonal elements) for every pixel in the image. Structre tensor is postive-definite symmetric matrix (positive eigenvalues)
    Axx, Axy, Ayy = feature.structure_tensor(img.astype(np.float32), sigma=sigma, mode='reflect', order='xy')
    # Get orientation for every pixel in the image
    ori = np.arctan2(2*Axy, Ayy - Axx) / 2
    # Compute coherence (local anisotropy of the image). If both eigenvalues of the structure tensor are equal, the coherence is zero (isotropic image). 
    # If the smaller eigenvalue tends to zero, the coherence tends to 1 (see math formulas).
    l1, l2 = feature.structure_tensor_eigenvalues([Axx, Axy, Ayy])
    eps = 1e-3 # to avoid division by zero?
    coh = ((l2-l1) / (l2+l1+eps)) ** 2
    # Finally, compute energy as trace of the structure tensor
    E = np.sqrt(Axx + Ayy)
    E /= np.max(E)
    return ori, coh, E
    
def get_orientation_angle(u, v, type='nematic'):
    if type=='polar': # get full angle in [-pi,pi)
        phi = np.arctan2(v, u)
    if type=='nematic': # get orientation in [-pi/2,pi/2)
        phi = np.arctan(v/u) # TO DO: catch zero exception!!
    return phi

# =============================================================================
# Computation of topological charges
# =============================================================================

def compute_topological_charges(phi, type='nematic', int_area='cell', width=1, boundary='real', origin='upper'):
    # Get angles of grid points on the integration path.
    int_angles = get_angles_on_integration_path(phi, int_area, width, origin)
    # Compute the topological charges for all points by computing differences along the integration path.
    k = np.zeros_like(phi)
    for i in range(len(int_angles)-1):
        k += modulo(int_angles[i+1] - int_angles[i], type)
    # Some boundary values have to be discarded for non-periodic data depending on the integration path
    if boundary!='periodic':
        discard_boundary_values(k, int_area, width)
    return k/(2*np.pi)

def charge_of_entire_domain(phi, type='nematic', origin='upper'):
    """Only works for quadratic domains so far"""
    sum = 0
    for i in range(phi.shape[0]-1):
        sum += modulo_for_single_number(phi[0,i+1] - phi[0,i], type) # left boundary
        sum += modulo_for_single_number(phi[i+1,-1] - phi[i,-1], type) # upper boundary
        sum += modulo_for_single_number(phi[-1,-i-2] - phi[-1,-i-1], type) # right boundary
        sum += modulo_for_single_number(phi[-i-2,0] - phi[-i-1,0], type) # upper boundary
    # Flip charge if origin is upper
    if origin=='upper':
        sum *= -1
    return sum/(2*np.pi)

def get_angles_on_integration_path(phi, int_area='square', width=1, origin='upper'):
    if int_area=='cell': #2x2 square in old reference
        e = np.roll(phi,-1,axis=1) # eastern neighbor
        n = np.roll(phi,1,axis=0) # northern neighbor
        ne = np.roll(e,1,axis=0) # northeastern neighbor
        int_angles = [e, ne, n, phi, e]
    if int_area=='square':
        e = np.roll(phi,-width,axis=1) # eastern neighbor
        w = np.roll(phi,width,axis=1) # western neighbor
        n = np.roll(phi,width,axis=0) # northern neighbor
        s = np.roll(phi,-width,axis=0) # southern neighbor
        ne = np.roll(e,width,axis=0)
        se = np.roll(e,-width,axis=0)
        nw = np.roll(w,width,axis=0)
        sw = np.roll(w,-width,axis=0)
        int_angles = [e, ne, n, nw, w, sw, s, se, e]
    # Check where the origin is (compare imshow vs. plot) and change the integration order accordingly    
    if origin=='lower':
        int_angles.reverse()
    return int_angles

def modulo(x, type='nematic'):
    """ Customized modulo function to remap [-pi,pi) to [-pi/2,pi/2) (nematic case) 
    or [-2pi,2pi) to [-pi,pi) (polar case). """
    if type=='polar':
        x[x>=np.pi] -= 2*np.pi
        x[x<-np.pi] += 2*np.pi
    if type=='nematic':
        x[x>=np.pi/2] -= np.pi
        x[x<-np.pi/2] += np.pi
    return x

def modulo_for_single_number(x, type='nematic'):
    """ Customized modulo function to remap [-pi,pi) to [-pi/2,pi/2) (nematic case) 
    or [-2pi,2pi) to [-pi,pi) (polar case). """
    if type=='polar':
        if x>=np.pi:
            x -= 2*np.pi
        if x<-np.pi:
            x += 2*np.pi
    if type=='nematic':
        if x>=np.pi/2:
            x -= np.pi
        if x<-np.pi/2:
            x += np.pi
    return x

def discard_boundary_values(k, int_area='square', width=1):
    """ Discard values at the boundary depending on the choosen integration path """
    if int_area=='cell':
        k[0,:], k[:,-1] = 0, 0
    if int_area=='square':
        k[:width+1,:], k[-width:,:], k[:,:width+1], k[:,-width:] = 0, 0, 0, 0

# =============================================================================
# Defect localization
# =============================================================================

def localize_defects(k, x_grid=None, y_grid=None, type='nematic', path='cell', thres=.1):
    ''' Localize defects based on the computed charges and the grid '''
    k_in = get_charge_interval(k, type)
    # Prepare data structure
    columns = ['charge','x','y','x_ind','y_ind']
    defects = pd.DataFrame({'charge': pd.Series(dtype=float),
                            'x' : pd.Series(dtype=float),
                            'y' : pd.Series(dtype=float),
                            'x_ind' : pd.Series(dtype=int),
                            'y_ind' : pd.Series(dtype=int),  
                            })
    if y_grid is None: # Check if grid is provided. Otherwise construct pseudo-grid
        x_grid, y_grid = np.meshgrid(np.arange(k.shape[0]),np.arange(k.shape[1]))
    # Iterate over charges
    for c in k_in:
        if c==0: # Skip charge zero, as there is no defect
            continue
        # Find charges with some threshold values and compute connected domains.
        # For each domain key information is stored.
        pos = (k>c-thres) & (k<c+thres)
        pos = measure.label(pos)
        for region in measure.regionprops(pos):
            x_ind = int(region.centroid[1]) # Some flip x/y again?
            y_ind = int(region.centroid[0])
            x = y_grid[x_ind,y_ind]
            y = x_grid[x_ind,y_ind]
            # if path=='cell': # Shift positions if integration area is a single cell.
            #     x += .5
            #     y += .5
            d = pd.DataFrame([[c, x, y, x_ind, y_ind]], columns=columns)
            defects = pd.concat([defects, d], ignore_index=True)
    return defects

def get_charge_interval(k, type='nematic'):
    '''Returns list of exact charges (-1/2,1/2,etc.) charges'''
    if type=='polar':
        min_k, max_k = round(np.min(k)), round(np.max(k))
        n = (max_k - min_k) + 1 # this assumes that min_k <= 0, which should be always the case, as 0 is lower bound.
    if type=='nematic':
        min_k, max_k = round_to_nearest_half_integer(np.min(k)), round_to_nearest_half_integer(np.max(k))
        n = int((max_k - min_k)*2 + 1) # this assumes that min_k <= 0, which should be always the case, as 0 is lower bound.
    return np.linspace(min_k, max_k, n)

def round_to_nearest_half_integer(x):
    return round(x*2)/2

# =============================================================================
# Defect orientation (so far only for (-1/2 and 1/2))
# =============================================================================

def compute_defect_orientations(phi, defects, method='Giomi', **kwargs):
    '''Two methods to compute defect orientation, either based on
    derivatives of the orietation tensor (see Giomi) or comparing interpolated
    orientations with the defect orientation (brute-force).'''
    if method=='Giomi':
        defect_orienation_Q_tensor(phi, defects, **kwargs)
    if method=='interpolation':
        defect_orientation_interpolation(phi, defects, **kwargs)

def defect_orienation_Q_tensor(phi, defects, origin='upper'):
    dQ_xx, dQ_xy, dQ_yy = Q_tensor_gradients(phi)

    # Here is a flip of x and y again, Why?
    plushalf = np.transpose(defects.loc[defects['charge']==.5,['y_ind','x_ind']].to_numpy())
    minushalf = np.transpose(defects.loc[defects['charge']==-.5,['y_ind','x_ind']].to_numpy())

    p_angles = defect_orientation_via_Q_tensor(plushalf, dQ_xx, dQ_xy, dQ_yy, charge=.5)
    m_angles = defect_orientation_via_Q_tensor(minushalf, dQ_xx, dQ_xy, dQ_yy, charge=-.5)

    # Weird flip. Has to be tested in more detail.
    if origin=='upper':
        p_angles += np.pi/2
        m_angles += np.pi/2

    defects.loc[defects['charge']==.5,'ang1'] = p_angles
    defects.loc[defects['charge']==-.5,'ang1'] = m_angles
    defects.loc[defects['charge']==-.5,'ang2'] = m_angles + 2*np.pi/3
    defects.loc[defects['charge']==-.5,'ang3'] = m_angles + 4*np.pi/3

def Q_tensor_gradients(phi):
    n_x = np.cos(phi)
    n_y = np.sin(phi)
    Q_xx = n_x*n_x 
    Q_xy = n_x*n_y
    Q_yy = n_y*n_y
    dQ_xx = np.gradient(Q_xx)
    dQ_xy = np.gradient(Q_xy)
    dQ_yy = np.gradient(Q_yy)
    return dQ_xx, dQ_xy, dQ_yy

def defect_orientation_via_Q_tensor(defect_positions, dQ_xx, dQ_xy, dQ_yy, charge):
    """ Use idea of Vromans/Giomi to compute orientation. Also see Tang for generalization
    The orientation of the defect is described by a vector p proportional to div nn """
    # Manual computation of Q=nn, also see Victors' code
    p_y = np.sign(charge)*dQ_xy[0] + dQ_yy[1]
    p_x = dQ_xx[0] + np.sign(charge)*dQ_xy[1]
    # Giomi's formula, should provide same results
    # p_y = np.sign(charge)*dQ_xy[0] - dQ_xx[1] 
    # p_x = dQ_xx[0] + np.sign(charge)*dQ_xy[1] 

    # Iterate over all defects and compute the average of the vector p.
    # Taking the arctan and rescaling gives the final orientation
    n = len(defect_positions[0])
    angles = np.zeros(n)
    for defect in range(n):
        i, j = defect_positions[0][defect], defect_positions[1][defect]
        num = np.mean(p_y[i-1:i+1, j-1:j+1])
        dem = np.mean(p_x[i-1:i+1, j-1:j+1])
        angles[defect] = np.arctan2(num, dem)
    return (charge/(1-charge))*angles

def defect_orientation_interpolation(phi, defects, x_grid=None, y_grid=None, interpolation_radius=5, interpolation_points=100, interpolation_method='complex', min_sep=1, **kwargs):
    theta = np.linspace(0,2*np.pi,interpolation_points,endpoint=False)
    x_c = interpolation_radius*np.cos(theta)
    y_c = interpolation_radius*np.sin(theta)
    psi = np.arctan(y_c/x_c) # angle from defect to points on the circle
    int_ori = interpolate_orientation_field(phi, x_grid, y_grid, method=interpolation_method)
    plushalf = defects[defects['charge']==.5]
    minushalf = defects[defects['charge']==-.5]
    for index, row in plushalf.iterrows():
        phi_int = get_interpolated_angles(int_ori, plushalf.loc[index,'x']+x_c, plushalf.loc[index,'y']+y_c)
        diff = np.abs(modulo(phi_int-psi))
        indices = find_k_smallest_values(diff, k=1)
        defects.loc[index, 'ang1'] = np.arctan2(y_c[indices[0]],x_c[indices[0]])
    for index, row in minushalf.iterrows():
        phi_int = get_interpolated_angles(int_ori, minushalf.loc[index,'x']+x_c, minushalf.loc[index,'y']+y_c)
        diff = np.abs(modulo(phi_int-psi))
        indices = find_k_smallest_values(diff, k=3, min_sep=min_sep)
        defects.loc[index, ['ang1','ang2','ang3']] = np.arctan2(y_c[indices],x_c[indices])

def interpolate_orientation_field(phi, x, y, method='complex'):
    '''Different approaches to interpolate an orientation field.
    Complex (mapping to a circle) works best. Maybe a better interpolation routine can be used?'''
    if method=='unwrap_np': 
        phi_unwraped = np.unwrap(np.unwrap(phi,period=np.pi,axis=0),period=np.pi,axis=1)
        return sp.interpolate.interp2d(x,y,phi_unwraped)
        # return sp.interpolate.RectBivariateSpline(x,y,phi_unwraped)
    if method=='unwrap_scikit':
        phi_unwraped = restoration.unwrap_phase(2*phi)/2
        return sp.interpolate.interp2d(x,y,phi_unwraped)
    if method=='complex':
        # Strech to [-pi,pi) and interpolate real an imaginary part separetly.
        c = np.exp(2j*phi)
        interp_r = sp.interpolate.interp2d(x,y,np.real(c))
        interp_i = sp.interpolate.interp2d(x,y,np.imag(c))
        return [interp_r, interp_i]

def get_interpolated_angles(interp, x, y, method='complex'):
    '''Evaluate interpolated function depending on the choosen method.'''
    phi = np.zeros(len(x))
    for i in range(len(x)):
        if method=='complex':
            # Combine real and imaginary part and map back to [-pi/2,pi/2) again
            phi[i] = np.angle(interp[0](x[i], y[i]) + 1j*interp[1](x[i], y[i]))/2
        else:
            phi[i] = modulo(interp(x[i],y[i]))
    return phi

def find_k_smallest_values(arr, k=3, min_sep=1):
    '''Custom funktion to find smallest values in an array that are at least min_sep inidices apart from each other. 
    The second criterion is needed to avoid to minima which are too close to each other.'''
    x = np.copy(arr) # this might be slow, but easier to code
    n = len(x)
    indices = []
    # Iterate over indices and find minima which fullfills the criterion.
    while len(indices)<k:
        ind = np.argmin(x)
        is_too_close = 0
        for i in indices:
            for j in range(min_sep):
                if (ind==(i+j+1)%n) | ((ind==(i-j-1)%n)):
                    is_too_close += 1
        if is_too_close==0:
            indices.append(ind)
        x[ind] = np.inf
    return indices