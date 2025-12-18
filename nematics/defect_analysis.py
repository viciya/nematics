"""Optimized nematic defect analysis utilities.

This module contains improved and faster implementations of the functions
from the notebook. Key optimizations:
 - vectorized evaluation of `RectBivariateSpline` using `ev`
 - avoid repeated `pd.concat` in loops by collecting rows then building DataFrame once
 - faster selection of smallest values using `np.argpartition` and a separation filter
 - small NumPy/vectorization improvements and safer dtype handling

API highlights:
 - orientation_analysis(img, sigma=11)
 - compute_topological_charges(phi, ...)
 - localize_defects(k, ...)
 - compute_defect_orientations(phi, defects, ...)

Keep the function signatures compatible with notebook usage.
"""
from typing import Tuple, List, Optional

import numpy as np
from scipy import interpolate
from skimage import feature, measure
import pandas as pd


def orientation_analysis(img: np.ndarray, sigma: int = 11) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute local orientation, coherence and normalized energy from an image."""
    Axx, Axy, Ayy = feature.structure_tensor(img.astype(np.float32), sigma=sigma, mode='reflect', order='xy')
    ori = 0.5 * np.arctan2(2 * Axy, Ayy - Axx)
    l1, l2 = feature.structure_tensor_eigenvalues([Axx, Axy, Ayy])
    eps = 1e-9
    coh = ((l2 - l1) / (l2 + l1 + eps)) ** 2
    E = np.sqrt(Axx + Ayy)
    E = E / (np.nanmax(E) + eps)
    return ori, coh, E


def modulo(x: np.ndarray, type: str = 'nematic') -> np.ndarray:
    """Map angle differences to principal range in a vectorized way."""
    x = np.asarray(x, dtype=float)
    if type == 'polar':
        return (x + np.pi) % (2 * np.pi) - np.pi
    # nematic: map to [-pi/2, pi/2)
    return (x + np.pi / 2) % np.pi - np.pi / 2


def get_angles_on_integration_path(phi: np.ndarray, int_area: str = 'cell', width: int = 1, origin: str = 'upper') -> List[np.ndarray]:
    if int_area == 'cell':
        e = np.roll(phi, -1, axis=1)
        n = np.roll(phi, -1, axis=0)
        ne = np.roll(e, -1, axis=0)
        int_angles = [e, ne, n, phi, e]
    else:
        e = np.roll(phi, -width, axis=1)
        w = np.roll(phi, width, axis=1)
        n = np.roll(phi, -width, axis=0)
        s = np.roll(phi, width, axis=0)
        ne = np.roll(e, -width, axis=0)
        se = np.roll(e, width, axis=0)
        nw = np.roll(w, -width, axis=0)
        sw = np.roll(w, width, axis=0)
        int_angles = [e, ne, n, nw, w, sw, s, se, e]
    if origin == 'lower':
        int_angles = list(reversed(int_angles))
    return int_angles


def discard_boundary_values(k: np.ndarray, int_area: str = 'square', width: int = 1) -> None:
    rows, cols = k.shape
    if int_area == 'cell':
        k[0, :] = 0
        k[:, -1] = 0
    else:
        w = max(1, width)
        k[: w + 1, :] = 0
        k[-w:, :] = 0
        k[:, : w + 1] = 0
        k[:, -w:] = 0


def compute_topological_charges(phi: np.ndarray, type: str = 'nematic', int_area: str = 'cell', width: int = 1, boundary: str = 'real', origin: str = 'upper') -> np.ndarray:
    """Compute topological charge map from an orientation field `phi`.

    Returns charges in units of winding (i.e. integer/half-integer multiples of 1).
    """
    int_angles = get_angles_on_integration_path(phi, int_area=int_area, width=width, origin=origin)
    k = np.zeros_like(phi, dtype=float)
    for i in range(len(int_angles) - 1):
        diff = int_angles[i + 1] - int_angles[i]
        k += modulo(diff, type)
    if boundary != 'periodic':
        discard_boundary_values(k, int_area=int_area, width=width)
    return k / (2 * np.pi)


def round_to_nearest_half_integer(x: float) -> float:
    return round(x * 2) / 2.0


def get_charge_interval(k: np.ndarray, type: str = 'nematic') -> np.ndarray:
    if type == 'polar':
        min_k, max_k = int(np.round(np.nanmin(k))), int(np.round(np.nanmax(k)))
        n = (max_k - min_k) + 1
        return np.linspace(min_k, max_k, n)
    min_k = round_to_nearest_half_integer(float(np.nanmin(k)))
    max_k = round_to_nearest_half_integer(float(np.nanmax(k)))
    n = int((max_k - min_k) * 2 + 1)
    return np.linspace(min_k, max_k, n)


def localize_defects(k: np.ndarray, x_grid: Optional[np.ndarray] = None, y_grid: Optional[np.ndarray] = None, type: str = 'nematic', thres: float = 0.1) -> pd.DataFrame:
    """Localize defects from a topological charge map `k`.

    Faster implementation: accumulate rows then build the DataFrame once.
    Coordinates returned in image convention: x = column index, y = row index.
    """
    charges = get_charge_interval(k, type)
    rows, cols = k.shape
    if x_grid is None or y_grid is None:
        xs = np.arange(cols)
        ys = np.arange(rows)
        xx, yy = np.meshgrid(xs, ys)  # xx[y, x]
    else:
        xx, yy = x_grid, y_grid
    out_rows = []
    for c in charges:
        if c == 0:
            continue
        mask = (k > c - thres) & (k < c + thres)
        labeled = measure.label(mask)
        for region in measure.regionprops(labeled):
            r, cind = region.centroid  # (row, col)
            y_ind = int(round(r))
            x_ind = int(round(cind))
            x_coord = float(xx[y_ind, x_ind])
            y_coord = float(yy[y_ind, x_ind])
            out_rows.append((float(c), x_coord, y_coord, int(x_ind), int(y_ind)))
    if not out_rows:
        return pd.DataFrame(columns=['charge', 'x', 'y', 'x_ind', 'y_ind'])
    df = pd.DataFrame(out_rows, columns=['charge', 'x', 'y', 'x_ind', 'y_ind'])
    return df.astype({'charge': float, 'x': float, 'y': float, 'x_ind': int, 'y_ind': int})


def interpolate_orientation_field(phi: np.ndarray, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None, method: str = 'complex'):
    """Return an interpolator. Uses (y, x) order for RectBivariateSpline."""
    rows, cols = phi.shape
    if x is None or y is None:
        x = np.arange(cols)
        y = np.arange(rows)
    if method == 'unwrap_np':
        phi_unwrap = np.unwrap(np.unwrap(phi, axis=0, period=np.pi), axis=1, period=np.pi)
        return interpolate.RectBivariateSpline(y, x, phi_unwrap)
    if method == 'complex':
        c = np.exp(2j * phi)
        interp_r = interpolate.RectBivariateSpline(y, x, np.real(c))
        interp_i = interpolate.RectBivariateSpline(y, x, np.imag(c))
        return [interp_r, interp_i]
    return interpolate.RectBivariateSpline(y, x, phi)


def get_interpolated_angles(interp, x_coords: np.ndarray, y_coords: np.ndarray, method: str = 'complex') -> np.ndarray:
    """Vectorized evaluation of interpolated angles at (x_coords, y_coords).

    Uses RectBivariateSpline.ev for fast vectorized evaluation.
    x_coords, y_coords should be 1D arrays of same length.
    """
    x_coords = np.asarray(x_coords)
    y_coords = np.asarray(y_coords)
    if method == 'complex' and isinstance(interp, list):
        # interp_r.ev expects (x, y) arrays in the same order used for construction
        r = interp[0].ev(y_coords, x_coords)
        i = interp[1].ev(y_coords, x_coords)
        phi = np.angle(r + 1j * i) / 2.0
        return modulo(phi)
    # fallback: scalar evaluate using ev
    vals = interp.ev(y_coords, x_coords)
    return modulo(vals)


def find_k_smallest_values(arr: np.ndarray, k: int = 3, min_sep: int = 1) -> List[int]:
    """Find indices of k smallest values with a minimum separation constraint.

    Strategy: pick a candidate pool via `argpartition` then scan in ascending order,
    accepting indices that satisfy the `min_sep` circular spacing constraint.
    """
    n = len(arr)
    if k >= n:
        return list(range(n))
    # get candidate indices (oversample factor to increase chance to find k separated minima)
    oversample = max(3, int(np.ceil(k * 2)))
    m = min(n, k * oversample)
    cand = np.argpartition(arr, m - 1)[:m]
    # sort candidates by value
    cand_sorted = cand[np.argsort(arr[cand])]
    out = []
    taken = np.zeros(n, dtype=bool)
    for idx in cand_sorted:
        if len(out) >= k:
            break
        # check circular separation
        sep_ok = True
        for t in out:
            if min((idx - t) % n, (t - idx) % n) <= min_sep:
                sep_ok = False
                break
        if sep_ok:
            out.append(int(idx))
    # if not enough found (rare), fall back to brute force filling
    if len(out) < k:
        for idx in np.argsort(arr):
            if len(out) >= k:
                break
            sep_ok = True
            for t in out:
                if min((idx - t) % n, (t - idx) % n) <= min_sep:
                    sep_ok = False
                    break
            if sep_ok:
                out.append(int(idx))
    return out


def compute_defect_orientations(phi: np.ndarray, defects: pd.DataFrame, method: str = 'interpolation', **kwargs) -> pd.DataFrame:
    if method == 'interpolation':
        defect_orientation_interpolation(phi, defects, **kwargs)
    return defects


def defect_orientation_interpolation(phi: np.ndarray, defects: pd.DataFrame, x_grid: Optional[np.ndarray] = None, y_grid: Optional[np.ndarray] = None, interpolation_radius: int = 5, interpolation_points: int = 100, interpolation_method: str = 'complex', min_sep: int = 1) -> None:
    theta = np.linspace(0, 2 * np.pi, interpolation_points, endpoint=False)
    x_c = interpolation_radius * np.cos(theta)
    y_c = interpolation_radius * np.sin(theta)
    psi = np.arctan2(y_c, x_c)
    interp = interpolate_orientation_field(phi, x_grid, y_grid, method=interpolation_method)
    plushalf = defects[defects['charge'] == 0.5]
    minushalf = defects[defects['charge'] == -0.5]
    # positive half defects
    for idx in plushalf.index:
        xc = plushalf.at[idx, 'x'] + x_c
        yc = plushalf.at[idx, 'y'] + y_c
        phi_int = get_interpolated_angles(interp, xc, yc, method=interpolation_method)
        diff = np.abs(modulo(phi_int - psi))
        ind = find_k_smallest_values(diff, k=1)[0]
        defects.at[idx, 'ang1'] = float(np.arctan2(y_c[ind], x_c[ind]))
    # negative half defects
    for idx in minushalf.index:
        xc = minushalf.at[idx, 'x'] + x_c
        yc = minushalf.at[idx, 'y'] + y_c
        phi_int = get_interpolated_angles(interp, xc, yc, method=interpolation_method)
        diff = np.abs(modulo(phi_int - psi))
        inds = find_k_smallest_values(diff, k=3, min_sep=min_sep)
        angs = np.arctan2(y_c[inds], x_c[inds])
        for j, a in enumerate(angs, start=1):
            defects.at[idx, f'ang{j}'] = float(a)
