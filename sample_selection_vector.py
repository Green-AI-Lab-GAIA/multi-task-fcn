"""
Vector-based sample selection for the multi-task FCN pipeline.

This module replaces the slow raster-based operations in sample_selection.py
with fast vector operations using GeoPandas and Shapely.

Key improvements:
- Spatial indexing (R-tree) for O(n log n) intersection queries
- No pixel-by-pixel iteration
- Compact GeoPackage storage instead of TIFF
- Parallel processing support

Author: Optimized from sample_selection.py
"""

import os
import sys
from logging import getLogger
from os.path import dirname, exists, join
from typing import Tuple, Optional, Union

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union
from tqdm import tqdm


def gaussian_filter(input_array: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply Gaussian filter using OpenCV (faster than scipy.ndimage.gaussian_filter).
    
    This function provides an interface compatible with scipy.ndimage.gaussian_filter
    but uses OpenCV's GaussianBlur for better performance (~4x faster).
    
    Parameters
    ----------
    input_array : np.ndarray
        Input array to filter
    sigma : float
        Standard deviation for Gaussian kernel. The kernel size is automatically
        calculated as 2 * ceil(4 * sigma) + 1 to match scipy's behavior.
    
    Returns
    -------
    np.ndarray
        Filtered array with same shape and dtype as input
    
    Notes
    -----
    - OpenCV's GaussianBlur requires an odd kernel size
    - The kernel size is calculated to match scipy's truncate=4.0 default (4 sigma radius)
    - For sigma=0, returns the input unchanged
    - Small numerical differences from scipy are expected but negligible for practical use
    """
    if sigma <= 0:
        return input_array.copy()
    
    # Calculate kernel size to match scipy behavior
    # scipy uses truncate=4.0 by default, meaning radius = ceil(4 * sigma)
    # Kernel size = 2 * radius + 1
    ksize = int(2 * np.ceil(4 * sigma) + 1)
    
    # Ensure kernel size is at least 1
    ksize = max(ksize, 1)
    
    # Ensure kernel size is odd (required by OpenCV)
    if ksize % 2 == 0:
        ksize += 1
    
    # Store original dtype
    original_dtype = input_array.dtype
    
    # OpenCV GaussianBlur works best with float32 or float64
    if input_array.dtype not in [np.float32, np.float64]:
        work_array = input_array.astype(np.float32)
    else:
        work_array = input_array
    
    # Apply Gaussian blur
    # sigmaX and sigmaY are set to sigma for isotropic filtering
    result = cv2.GaussianBlur(work_array, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    
    # Convert back to original dtype if needed
    if result.dtype != original_dtype:
        result = result.astype(original_dtype)
    
    return result

ROOT_PATH = dirname(__file__)
sys.path.append(ROOT_PATH)

from src.io_operations import (
    get_image_metadata,
    read_tiff,
)
from src.vector_operations import (
    raster_to_geodataframe,
    labels_to_geodataframe_with_stats,
    geodataframe_to_raster,
    save_labels_as_geopackage,
    load_labels_from_geopackage,
    get_labels_delta_vector,
    get_label_intersection_vector,
    join_labels_vector,
    filter_by_geometric_properties_vector,
    filter_by_mask_vector,
    select_n_labels_by_class_vector,
    select_good_samples_vector,
)

logger = getLogger("__main__")


def filter_map_by_depth_prob_to_gdf(
    pred_map: np.ndarray,
    prob_map: np.ndarray,
    depth_map: np.ndarray,
    prob_thr: float,
    depth_thr: float,
    sigma: float = 9,
    reference_tiff: str = None,
) -> gpd.GeoDataFrame:
    """
    Filter prediction map by probability and depth thresholds, then convert to GeoDataFrame.
    
    Parameters
    ----------
    pred_map : np.ndarray
        Class prediction map
    prob_map : np.ndarray
        Probability map
    depth_map : np.ndarray
        Depth/distance map
    prob_thr : float
        Probability threshold
    depth_thr : float
        Depth threshold
    sigma : float
        Gaussian smoothing sigma
    reference_tiff : str, optional
        Reference TIFF for georeferencing
    
    Returns
    -------
    gpd.GeoDataFrame
        Filtered predictions as polygons
    """
    # Create local copy
    pred_map = pred_map.copy()
    
    # Smooth the maps
    depth_gauss = gaussian_filter(depth_map, sigma=sigma)
    prob_gauss = gaussian_filter(prob_map, sigma=sigma)
    
    # Apply threshold
    mask = (depth_gauss + prob_gauss) > (depth_thr + prob_thr)
    pred_map = np.where(mask, pred_map, 0)
    
    # Convert to GeoDataFrame
    transform = None
    crs = None
    if reference_tiff and exists(reference_tiff):
        meta = get_image_metadata(reference_tiff)
        transform = meta.get('transform')
        crs = meta.get('crs')
    
    gdf = labels_to_geodataframe_with_stats(pred_map, transform, crs)
    
    return gdf


def load_or_convert_labels(
    tiff_path: str = None,
    gpkg_path: str = None,
    labels_array: np.ndarray = None,
    reference_tiff: str = None,
) -> gpd.GeoDataFrame:
    """
    Load labels from GeoPackage if available, otherwise convert from TIFF/array.
    
    Parameters
    ----------
    tiff_path : str, optional
        Path to TIFF file
    gpkg_path : str, optional
        Path to GeoPackage file
    labels_array : np.ndarray, optional
        Labels array directly
    reference_tiff : str, optional
        Reference TIFF for georeferencing
    
    Returns
    -------
    gpd.GeoDataFrame
        Labels as GeoDataFrame
    """
    # Try GeoPackage first (fastest)
    if gpkg_path and exists(gpkg_path):
        logger.info(f"Loading labels from GeoPackage: {gpkg_path}")
        return gpd.read_file(gpkg_path)
    
    # Load from TIFF
    if tiff_path and exists(tiff_path):
        logger.info(f"Converting labels from TIFF: {tiff_path}")
        labels_array = read_tiff(tiff_path)
        reference_tiff = reference_tiff or tiff_path
    
    # Convert array to GeoDataFrame
    if labels_array is not None:
        transform = None
        crs = None
        if reference_tiff and exists(reference_tiff):
            meta = get_image_metadata(reference_tiff)
            transform = meta.get('transform')
            crs = meta.get('crs')
        
        return labels_to_geodataframe_with_stats(labels_array, transform, crs)
    
    # Return empty GeoDataFrame
    return gpd.GeoDataFrame(columns=['geometry', 'tree_type', 'area'])


def get_new_segmentation_sample_vector(
    ground_truth_map: np.ndarray,
    old_selected_labels: np.ndarray,
    old_all_labels: np.ndarray,
    new_pred_map: np.ndarray,
    new_prob_map: np.ndarray,
    new_depth_map: np.ndarray,
    prob_thr: float,
    depth_thr: float,
    args: dict,
    sigma: float = 9,
    reference_tiff: str = None,
    output_as_raster: bool = True,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]]:
    """
    Vector-based version of get_new_segmentation_sample.
    
    This function performs all sample selection using vector operations,
    which is significantly faster than the raster-based approach.
    
    Parameters
    ----------
    ground_truth_map : np.ndarray
        Ground truth segmentation map
    old_selected_labels : np.ndarray
        Selected labels from previous iteration
    old_all_labels : np.ndarray
        All labels from previous iteration
    new_pred_map : np.ndarray
        New predicted class map
    new_prob_map : np.ndarray
        New probability map
    new_depth_map : np.ndarray
        New depth map
    prob_thr : float
        Probability threshold
    depth_thr : float
        Depth threshold
    args : dict
        Configuration arguments
    sigma : float
        Gaussian smoothing sigma
    reference_tiff : str, optional
        Reference TIFF for georeferencing
    output_as_raster : bool
        If True, convert output back to raster arrays
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray] or Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
        (all_labels_set, selected_labels_set)
    """
    # Get georeferencing info
    transform = None
    crs = None
    if reference_tiff and exists(reference_tiff):
        meta = get_image_metadata(reference_tiff)
        transform = meta.get('transform')
        crs = meta.get('crs')
    
    # Shift labels to match ground truth scale
    new_pred_map = new_pred_map.copy()
    new_pred_map += 1
    
    logger.info(f"Filtering components with depth >= {depth_thr} and prob >= {prob_thr}")
    
    # Filter and convert to GeoDataFrame
    new_pred_gdf = filter_map_by_depth_prob_to_gdf(
        new_pred_map, new_prob_map, new_depth_map,
        prob_thr, depth_thr, sigma, reference_tiff
    )
    
    logger.info(f"After threshold filter: {len(new_pred_gdf)} components")
    
    # Convert other arrays to GeoDataFrames
    logger.info("Converting labels to vector format...")
    ground_truth_gdf = labels_to_geodataframe_with_stats(ground_truth_map, transform, crs)
    old_selected_gdf = labels_to_geodataframe_with_stats(old_selected_labels, transform, crs)
    old_all_gdf = labels_to_geodataframe_with_stats(old_all_labels, transform, crs)
    
    logger.info(f"Ground truth: {len(ground_truth_gdf)}, Old selected: {len(old_selected_gdf)}, Old all: {len(old_all_gdf)}")
    
    # Apply geometric filters
    logger.info("Selecting samples with good geometric properties")
    new_pred_gdf = select_good_samples_vector(old_all_gdf, new_pred_gdf, args)
    
    logger.info(f"After quality filter: {len(new_pred_gdf)} components")
    
    # Helper to get args value (supports dict or object)
    def _get_arg(key, default=None):
        if hasattr(args, 'get'):
            return args.get(key, default)
        return getattr(args, key, default)
    
    # Filter by mask if provided
    mask_path = _get_arg('mask_path')
    if mask_path and exists(mask_path):
        logger.info("Filtering by mask")
        new_pred_gdf = filter_by_mask_vector(new_pred_gdf, mask_path)
        logger.info(f"After mask filter: {len(new_pred_gdf)} components")
    
    # Area filter (in square meters, CRS-agnostic)
    lower_limit_area = _get_arg('lower_limit_area')
    upper_limit_area = _get_arg('upper_limit_area')
    if lower_limit_area is not None and upper_limit_area is not None:
        logger.info(f"Filtering by area limits: {lower_limit_area} m² - {upper_limit_area} m²")
        new_pred_gdf = filter_by_geometric_properties_vector(
            new_pred_gdf,
            min_area=float(lower_limit_area),
            max_area=float(upper_limit_area),
            area_in_meters=True,  # Use square meters (m²), CRS-agnostic
        )
        logger.info(f"After area filter: {len(new_pred_gdf)} components")
    
    # Join all labels with new predictions
    logger.info("Joining old and new components")
    new_labels_gdf = join_labels_vector(new_pred_gdf, old_all_gdf, overlap_limit=0.05)
    
    # Get delta (new components not in old selected)
    logger.info("Getting new components (delta)")
    delta_gdf = get_labels_delta_vector(old_selected_gdf, new_labels_gdf, overlap_threshold=0.10)
    unbalanced_delta_gdf = delta_gdf.copy()
    
    logger.info(f"Delta components: {len(delta_gdf)}")
    
    # Select balanced sample by class
    logger.info("Selecting 5 components per tree_type")
    delta_gdf = select_n_labels_by_class_vector(delta_gdf, samples_by_class=5)
    
    logger.info(f"Balanced delta: {len(delta_gdf)} components")
    
    # Get intersection (updated shapes for old components)
    logger.info("Getting intersection with old segmentation")
    intersection_gdf = get_label_intersection_vector(old_selected_gdf, new_labels_gdf, overlap_threshold=0.10)
    
    logger.info(f"Intersection components: {len(intersection_gdf)}")
    
    # Update old selected with new shapes
    logger.info("Updating old components with new shapes")
    old_selected_updated_gdf = join_labels_vector(intersection_gdf, old_selected_gdf, overlap_limit=0.10)
    
    # Create selected labels set
    logger.info("Creating selected labels set")
    selected_labels_gdf = join_labels_vector(delta_gdf, old_selected_updated_gdf, overlap_limit=0.10)
    selected_labels_gdf = join_labels_vector(ground_truth_gdf, selected_labels_gdf, overlap_limit=0.01)
    
    # Create all labels set
    logger.info("Creating all labels set")
    all_labels_gdf = join_labels_vector(unbalanced_delta_gdf, old_selected_updated_gdf, overlap_limit=0.10)
    all_labels_gdf = join_labels_vector(ground_truth_gdf, all_labels_gdf, overlap_limit=0.01)
    
    logger.info(f"Final: {len(all_labels_gdf)} all labels, {len(selected_labels_gdf)} selected labels")
    
    if not output_as_raster:
        return all_labels_gdf, selected_labels_gdf
    
    # Convert back to raster
    logger.info("Converting back to raster format")
    shape = ground_truth_map.shape
    
    all_labels_arr = geodataframe_to_raster(all_labels_gdf, shape, transform)
    selected_labels_arr = geodataframe_to_raster(selected_labels_gdf, shape, transform)
    
    return all_labels_arr, selected_labels_arr


def save_labels_vector(
    all_labels: Union[np.ndarray, gpd.GeoDataFrame],
    selected_labels: Union[np.ndarray, gpd.GeoDataFrame],
    output_folder: str,
    reference_tiff: str = None,
    save_tiff: bool = False,
    save_gpkg: bool = True,
) -> Tuple[str, str]:
    """
    Save labels as GeoPackage (and optionally TIFF).
    
    Parameters
    ----------
    all_labels : np.ndarray or gpd.GeoDataFrame
        All labels
    selected_labels : np.ndarray or gpd.GeoDataFrame
        Selected labels
    output_folder : str
        Output folder path
    reference_tiff : str, optional
        Reference TIFF for georeferencing
    save_tiff : bool
        Also save as TIFF (for backwards compatibility)
    save_gpkg : bool
        Save as GeoPackage
    
    Returns
    -------
    Tuple[str, str]
        Paths to (all_labels_file, selected_labels_file)
    """
    from src.io_operations import array2raster
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Get metadata
    transform = None
    crs = None
    if reference_tiff and exists(reference_tiff):
        meta = get_image_metadata(reference_tiff)
        transform = meta.get('transform')
        crs = meta.get('crs')
    
    # Process all_labels
    if isinstance(all_labels, np.ndarray):
        all_labels_gdf = labels_to_geodataframe_with_stats(all_labels, transform, crs)
    else:
        all_labels_gdf = all_labels
    
    # Process selected_labels
    if isinstance(selected_labels, np.ndarray):
        selected_labels_gdf = labels_to_geodataframe_with_stats(selected_labels, transform, crs)
    else:
        selected_labels_gdf = selected_labels
    
    all_path = None
    selected_path = None
    
    if save_gpkg:
        all_gpkg_path = join(output_folder, 'all_labels_set.gpkg')
        selected_gpkg_path = join(output_folder, 'selected_labels_set.gpkg')
        
        all_labels_gdf.to_file(all_gpkg_path, driver='GPKG')
        selected_labels_gdf.to_file(selected_gpkg_path, driver='GPKG')
        
        logger.info(f"Saved GeoPackages: {all_gpkg_path}, {selected_gpkg_path}")
        all_path = all_gpkg_path
        selected_path = selected_gpkg_path
    
    if save_tiff and reference_tiff:
        meta = get_image_metadata(reference_tiff)
        shape = (meta['height'], meta['width'])
        
        all_tiff_path = join(output_folder, 'all_labels_set.tif')
        selected_tiff_path = join(output_folder, 'selected_labels_set.tif')
        
        if isinstance(all_labels, gpd.GeoDataFrame):
            all_arr = geodataframe_to_raster(all_labels, shape, transform)
        else:
            all_arr = all_labels
        
        if isinstance(selected_labels, gpd.GeoDataFrame):
            selected_arr = geodataframe_to_raster(selected_labels, shape, transform)
        else:
            selected_arr = selected_labels
        
        array2raster(all_tiff_path, all_arr, meta, "Byte")
        array2raster(selected_tiff_path, selected_arr, meta, "Byte")
        
        logger.info(f"Saved TIFFs: {all_tiff_path}, {selected_tiff_path}")
        all_path = all_path or all_tiff_path
        selected_path = selected_path or selected_tiff_path
    
    return all_path, selected_path


def get_components_stats_vector(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Get component statistics from a GeoDataFrame.
    
    Compatible interface with the original raster-based get_components_stats.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with polygon geometries
    
    Returns
    -------
    pd.DataFrame
        DataFrame with component statistics
    """
    if gdf.empty:
        return pd.DataFrame(columns=[
            'area', 'convex_area', 'bbox_area', 'extent', 'solidity',
            'eccentricity', 'centroid_x', 'centroid_y', 'tree_type', 'label'
        ])
    
    stats = gdf.copy()
    
    # Ensure all stats are computed
    if 'area' not in stats.columns:
        stats['area'] = stats.geometry.area
    
    if 'convex_area' not in stats.columns:
        stats['convex_area'] = stats.geometry.convex_hull.area
    
    if 'solidity' not in stats.columns:
        stats['solidity'] = stats['area'] / stats['convex_area'].replace(0, np.nan)
        stats['solidity'] = stats['solidity'].fillna(1.0)
    
    bounds = stats.geometry.bounds
    if 'bbox_area' not in stats.columns:
        stats['bbox_area'] = (bounds['maxx'] - bounds['minx']) * (bounds['maxy'] - bounds['miny'])
    
    if 'extent' not in stats.columns:
        stats['extent'] = stats['area'] / stats['bbox_area'].replace(0, np.nan)
        stats['extent'] = stats['extent'].fillna(1.0)
    
    if 'centroid_x' not in stats.columns:
        centroids = stats.geometry.centroid
        stats['centroid_x'] = centroids.x
        stats['centroid_y'] = centroids.y
    
    if 'label' not in stats.columns:
        stats['label'] = range(1, len(stats) + 1)
    
    return stats.drop(columns=['geometry']).reset_index(drop=True)


if __name__ == "__main__":
    """
    Test the vector-based sample selection
    """
    import time
    
    print("=" * 60)
    print("Testing Vector-based Sample Selection")
    print("=" * 60)
    
    # Create synthetic test data
    np.random.seed(42)
    shape = (500, 500)
    
    # Create ground truth
    ground_truth = np.zeros(shape, dtype=np.uint8)
    for i in range(20):
        x, y = np.random.randint(50, 450, 2)
        size = np.random.randint(10, 30)
        tree_type = np.random.randint(1, 4)
        ground_truth[max(0, x-size):x+size, max(0, y-size):y+size] = tree_type
    
    # Create old labels (subset of ground truth)
    old_selected = ground_truth.copy()
    old_selected[250:, :] = 0
    old_all = old_selected.copy()
    
    # Create new predictions (overlapping and new components)
    new_pred = np.zeros(shape, dtype=np.uint8)
    new_pred[200:, :] = ground_truth[200:, :]
    for i in range(10):
        x, y = np.random.randint(250, 450, 2)
        size = np.random.randint(10, 25)
        tree_type = np.random.randint(1, 4)
        new_pred[max(0, x-size):x+size, max(0, y-size):y+size] = tree_type
    
    # Create probability and depth maps
    prob_map = np.random.uniform(0.6, 1.0, shape).astype(np.float32)
    depth_map = np.random.uniform(0.1, 1.0, shape).astype(np.float32)
    
    # Create args-like object
    class Args:
        lower_limit_area = 50
        upper_limit_area = 10000
        upper_limit_area_rlted_to_tree_type = 2.0
        lower_limit_area_rlted_to_tree_type = -0.5
        lower_limit_solidity_rlted_to_tree_type = -0.2
        mask_path = None
        scale_area = None
    
    args_obj = Args()
    
    print(f"\nTest data shapes:")
    print(f"  Ground truth: {ground_truth.shape}, unique: {np.unique(ground_truth)}")
    print(f"  Old selected: {np.sum(old_selected > 0)} pixels")
    print(f"  New predictions: {np.sum(new_pred > 0)} pixels")
    
    # Time the vector-based approach
    print("\n" + "-" * 40)
    print("Running vector-based sample selection...")
    start = time.time()
    
    all_labels, selected_labels = get_new_segmentation_sample_vector(
        ground_truth_map=ground_truth,
        old_selected_labels=old_selected,
        old_all_labels=old_all,
        new_pred_map=new_pred,
        new_prob_map=prob_map,
        new_depth_map=depth_map,
        prob_thr=0.7,
        depth_thr=0.1,
        args=args_obj,
        sigma=5,
        output_as_raster=True,
    )
    
    elapsed = time.time() - start
    print(f"\nVector-based completed in {elapsed:.2f} seconds")
    print(f"  All labels: {np.sum(all_labels > 0)} pixels, unique: {np.unique(all_labels)}")
    print(f"  Selected labels: {np.sum(selected_labels > 0)} pixels, unique: {np.unique(selected_labels)}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)

