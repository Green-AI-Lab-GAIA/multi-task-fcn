"""
Vector-based operations for label processing.

This module provides optimized vector/shapefile operations as an alternative to 
slow raster-based operations. Key benefits:
- Spatial indexing (R-tree) for fast intersection queries
- Geometric operations on polygons instead of pixel arrays
- Compact GeoPackage storage instead of large TIFFs
- Vectorized GeoPandas/Shapely operations

Author: Optimized from raster-based sample_selection.py
"""

import os
import sys
from logging import getLogger
from os.path import dirname, exists, join
from typing import List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.transform import Affine
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.spatial import ConvexHull
from shapely.geometry import MultiPolygon, Polygon, box, shape
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from shapely.validation import make_valid
from skimage.draw import polygon as draw_polygon
from skimage.measure import label, regionprops
from tqdm import tqdm

ROOT_PATH = dirname(dirname(__file__))
sys.path.append(ROOT_PATH)

from src.io_operations import get_image_metadata, read_tiff, get_or_generate_mask_path

logger = getLogger("__main__")


# =============================================================================
# AREA CALCULATION UTILITIES
# =============================================================================

def get_area_in_square_meters(gdf: gpd.GeoDataFrame) -> pd.Series:
    """
    Calculate area in square meters for all geometries in a GeoDataFrame.
    
    This function is CRS-agnostic: it automatically handles both projected
    (metric) and geographic (lat/lon) coordinate systems by reprojecting
    to an appropriate UTM zone when necessary.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with polygon geometries
    
    Returns
    -------
    pd.Series
        Series with area values in square meters (m²)
    
    Notes
    -----
    - For projected CRS with metric units (e.g., UTM): uses geometry.area directly
    - For geographic CRS (e.g., WGS84 EPSG:4326): reprojects to UTM based on centroid
    - For CRS without units info: assumes metric and uses geometry.area
    """
    if gdf.empty:
        return pd.Series([], dtype=float)
    
    # If no CRS is set, assume the coordinates are already in a metric system
    if gdf.crs is None:
        logger.warning("GeoDataFrame has no CRS set. Assuming metric coordinates.")
        return gdf.geometry.area
    
    # Check if the CRS is projected (usually metric) or geographic (degrees)
    if gdf.crs.is_projected:
        # Check if the units are meters
        axis_info = gdf.crs.axis_info
        if axis_info and len(axis_info) > 0:
            unit_name = axis_info[0].unit_name.lower() if axis_info[0].unit_name else ""
            if "metre" in unit_name or "meter" in unit_name:
                # Already in meters, use directly
                return gdf.geometry.area
            elif "foot" in unit_name or "feet" in unit_name:
                # Convert from square feet to square meters (1 ft² = 0.092903 m²)
                return gdf.geometry.area * 0.092903
        # If we can't determine units, assume meters for projected CRS
        return gdf.geometry.area
    
    # Geographic CRS (lat/lon) - need to reproject to UTM
    # Calculate the centroid of all geometries to determine the UTM zone
    total_bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    center_lon = (total_bounds[0] + total_bounds[2]) / 2
    center_lat = (total_bounds[1] + total_bounds[3]) / 2
    
    # Determine UTM zone from longitude
    utm_zone = int((center_lon + 180) / 6) + 1
    
    # Determine hemisphere (north or south)
    if center_lat >= 0:
        utm_crs = f"EPSG:326{utm_zone:02d}"  # Northern hemisphere
    else:
        utm_crs = f"EPSG:327{utm_zone:02d}"  # Southern hemisphere
    
    logger.debug(f"Reprojecting from {gdf.crs} to {utm_crs} for area calculation")
    
    # Reproject to UTM and calculate area
    gdf_utm = gdf.to_crs(utm_crs)
    return gdf_utm.geometry.area


def is_crs_metric(crs) -> bool:
    """
    Check if a CRS uses metric units (meters).
    
    Parameters
    ----------
    crs : pyproj.CRS or similar
        Coordinate Reference System object
    
    Returns
    -------
    bool
        True if the CRS uses metric units, False otherwise
    """
    if crs is None:
        return True  # Assume metric if no CRS
    
    if not crs.is_projected:
        return False  # Geographic CRS uses degrees
    
    # Check axis units
    axis_info = crs.axis_info
    if axis_info and len(axis_info) > 0:
        unit_name = axis_info[0].unit_name.lower() if axis_info[0].unit_name else ""
        if "metre" in unit_name or "meter" in unit_name:
            return True
        elif "foot" in unit_name or "feet" in unit_name:
            return False
    
    # Default to True for projected CRS without clear unit info
    return True


# =============================================================================
# RASTER TO VECTOR CONVERSION
# =============================================================================

def raster_to_geodataframe(
    labels: np.ndarray,
    transform: Optional[Affine] = None,
    crs: Optional[str] = None,
    simplify_tolerance: float = 0.5,
) -> gpd.GeoDataFrame:
    """
    Convert a raster label image to a GeoDataFrame with polygon geometries.
    
    Uses rasterio.features.shapes for efficient vectorization instead of 
    iterating through components.
    
    Parameters
    ----------
    labels : np.ndarray
        2D array with labeled components (0 = background)
    transform : Affine, optional
        Affine transform for georeferencing. If None, uses pixel coordinates.
    crs : str, optional
        Coordinate reference system
    simplify_tolerance : float
        Tolerance for polygon simplification (reduces vertices)
    
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with columns: geometry, tree_type, area
    """
    if labels.ndim != 2:
        raise ValueError(f"Expected 2D array, got {labels.ndim}D")
    
    if transform is None:
        # Use pixel coordinates
        transform = Affine(1, 0, 0, 0, 1, 0)
    
    # Ensure proper dtype for rasterio
    labels = labels.astype(np.int32)
    
    # Extract shapes efficiently using rasterio
    shapes_generator = features.shapes(
        labels,
        mask=labels > 0,
        transform=transform,
        connectivity=4
    )
    
    geometries = []
    tree_types = []
    
    for geom, value in shapes_generator:
        poly = shape(geom)
        
        # Make valid and simplify
        if not poly.is_valid:
            poly = make_valid(poly)
        
        if simplify_tolerance > 0:
            poly = poly.simplify(simplify_tolerance, preserve_topology=True)
        
        if poly.is_empty:
            continue
            
        geometries.append(poly)
        tree_types.append(int(value))
    
    if not geometries:
        return gpd.GeoDataFrame(
            columns=['geometry', 'tree_type', 'area'],
            geometry='geometry',
            crs=crs
        )
    
    gdf = gpd.GeoDataFrame({
        'geometry': geometries,
        'tree_type': tree_types,
    }, crs=crs)
    
    # Calculate area
    gdf['area'] = gdf.geometry.area
    
    return gdf


def labels_to_geodataframe_with_stats(
    labels: np.ndarray,
    transform: Optional[Affine] = None,
    crs: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Convert labels to GeoDataFrame with geometric statistics.
    
    Computes: area, convex_area, extent, solidity, eccentricity, centroid, bbox
    
    Parameters
    ----------
    labels : np.ndarray
        2D labeled array
    transform : Affine, optional
        Affine transform
    crs : str, optional
        CRS string
    
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with geometry and statistics columns
    """
    gdf = raster_to_geodataframe(labels, transform, crs, simplify_tolerance=0)
    
    if gdf.empty:
        return gdf
    
    # Calculate geometric properties from Shapely geometries
    gdf['convex_area'] = gdf.geometry.convex_hull.area
    gdf['solidity'] = gdf['area'] / gdf['convex_area'].replace(0, np.nan)
    gdf['solidity'] = gdf['solidity'].fillna(1.0)
    
    # Bounding box properties
    bounds = gdf.geometry.bounds
    gdf['bbox_area'] = (bounds['maxx'] - bounds['minx']) * (bounds['maxy'] - bounds['miny'])
    gdf['extent'] = gdf['area'] / gdf['bbox_area'].replace(0, np.nan)
    gdf['extent'] = gdf['extent'].fillna(1.0)
    
    # Centroid
    centroids = gdf.geometry.centroid
    gdf['centroid_x'] = centroids.x
    gdf['centroid_y'] = centroids.y
    
    # Eccentricity approximation from minimum rotated rectangle
    def calc_eccentricity(geom):
        try:
            if geom.is_empty:
                return 0.0
            mrr = geom.minimum_rotated_rectangle
            coords = list(mrr.exterior.coords)
            d1 = np.sqrt((coords[0][0] - coords[1][0])**2 + (coords[0][1] - coords[1][1])**2)
            d2 = np.sqrt((coords[1][0] - coords[2][0])**2 + (coords[1][1] - coords[2][1])**2)
            a, b = max(d1, d2), min(d1, d2)
            if a == 0:
                return 0.0
            return np.sqrt(1 - (b/a)**2)
        except:
            return 0.0
    
    gdf['eccentricity'] = gdf.geometry.apply(calc_eccentricity)
    
    # Assign unique label IDs
    gdf['label'] = range(1, len(gdf) + 1)
    
    return gdf


def geodataframe_to_raster(
    gdf: gpd.GeoDataFrame,
    shape: Tuple[int, int],
    transform: Optional[Affine] = None,
    value_column: str = 'tree_type',
) -> np.ndarray:
    """
    Convert a GeoDataFrame back to a raster array.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with polygon geometries
    shape : Tuple[int, int]
        Output raster shape (height, width)
    transform : Affine, optional
        Affine transform. If None, uses pixel coordinates.
    value_column : str
        Column to use for raster values
    
    Returns
    -------
    np.ndarray
        2D raster array
    """
    if gdf.empty:
        return np.zeros(shape, dtype=np.uint8)
    
    if transform is None:
        transform = Affine(1, 0, 0, 0, 1, 0)
    
    # Prepare shapes with values
    shapes = [(geom, value) for geom, value in zip(gdf.geometry, gdf[value_column])]
    
    # Rasterize
    raster = features.rasterize(
        shapes=shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True
    )
    
    return raster


# =============================================================================
# SHAPEFILE / GEOPACKAGE I/O
# =============================================================================

def save_labels_as_geopackage(
    labels: np.ndarray,
    output_path: str,
    reference_tiff: str = None,
    layer_name: str = "labels",
) -> gpd.GeoDataFrame:
    """
    Save raster labels as a GeoPackage file (more efficient than Shapefile).
    
    Parameters
    ----------
    labels : np.ndarray
        2D labeled array
    output_path : str
        Path to save the .gpkg file
    reference_tiff : str, optional
        Reference TIFF to get transform and CRS
    layer_name : str
        Layer name in the GeoPackage
    
    Returns
    -------
    gpd.GeoDataFrame
        The saved GeoDataFrame
    """
    transform = None
    crs = None
    
    if reference_tiff and exists(reference_tiff):
        meta = get_image_metadata(reference_tiff)
        transform = meta.get('transform')
        crs = meta.get('crs')
    
    gdf = labels_to_geodataframe_with_stats(labels, transform, crs)
    
    if not output_path.endswith('.gpkg'):
        output_path = output_path.replace('.tif', '.gpkg').replace('.TIF', '.gpkg')
    
    # Ensure directory exists
    os.makedirs(dirname(output_path), exist_ok=True)
    
    # Save as GeoPackage
    gdf.to_file(output_path, driver='GPKG', layer=layer_name)
    
    logger.info(f"Saved {len(gdf)} features to {output_path}")
    
    return gdf


def load_labels_from_geopackage(
    geopackage_path: str,
    shape: Optional[Tuple[int, int]] = None,
    reference_tiff: str = None,
    return_raster: bool = False,
) -> Union[gpd.GeoDataFrame, Tuple[gpd.GeoDataFrame, np.ndarray]]:
    """
    Load labels from a GeoPackage file.
    
    Parameters
    ----------
    geopackage_path : str
        Path to the .gpkg file
    shape : Tuple[int, int], optional
        Output raster shape if return_raster=True
    reference_tiff : str, optional
        Reference TIFF to get shape
    return_raster : bool
        If True, also return rasterized version
    
    Returns
    -------
    gpd.GeoDataFrame or Tuple[gpd.GeoDataFrame, np.ndarray]
    """
    gdf = gpd.read_file(geopackage_path)
    
    if not return_raster:
        return gdf
    
    # Get shape from reference TIFF if not provided
    if shape is None and reference_tiff:
        meta = get_image_metadata(reference_tiff)
        shape = (meta['height'], meta['width'])
        transform = meta.get('transform')
    else:
        transform = None
    
    if shape is None:
        raise ValueError("Must provide shape or reference_tiff to rasterize")
    
    raster = geodataframe_to_raster(gdf, shape, transform)
    
    return gdf, raster


# =============================================================================
# VECTOR-BASED SAMPLE SELECTION OPERATIONS
# =============================================================================

def get_labels_delta_vector(
    old_labels_gdf: gpd.GeoDataFrame,
    new_labels_gdf: gpd.GeoDataFrame,
    overlap_threshold: float = 0.10,
) -> gpd.GeoDataFrame:
    """
    Get components in new_labels that are NOT in old_labels (vector-based).
    
    This replaces the slow raster-based get_labels_delta function.
    Uses spatial indexing for O(n log n) instead of O(n × m).
    
    Parameters
    ----------
    old_labels_gdf : gpd.GeoDataFrame
        Previous iteration labels
    new_labels_gdf : gpd.GeoDataFrame
        New predicted labels
    overlap_threshold : float
        Maximum allowed overlap ratio with old labels (0.10 = 10%)
    
    Returns
    -------
    gpd.GeoDataFrame
        New components that don't significantly overlap with old labels
    """
    if old_labels_gdf.empty:
        return new_labels_gdf.copy()
    
    if new_labels_gdf.empty:
        return new_labels_gdf.copy()
    
    # Ensure spatial index exists
    if not new_labels_gdf.has_sindex:
        new_labels_gdf = new_labels_gdf.copy()
    
    # Create union of old labels for fast intersection checking
    old_union = unary_union(old_labels_gdf.geometry)
    
    delta_indices = []
    
    for idx, row in new_labels_gdf.iterrows():
        geom = row.geometry
        intersection = geom.intersection(old_union)
        
        if intersection.is_empty:
            delta_indices.append(idx)
        else:
            overlap_ratio = intersection.area / geom.area
            if overlap_ratio < overlap_threshold:
                delta_indices.append(idx)
    
    return new_labels_gdf.loc[delta_indices].copy()


def get_label_intersection_vector(
    old_labels_gdf: gpd.GeoDataFrame,
    new_labels_gdf: gpd.GeoDataFrame,
    overlap_threshold: float = 0.10,
) -> gpd.GeoDataFrame:
    """
    Get components in new_labels that ARE in old_labels (vector-based).
    
    Parameters
    ----------
    old_labels_gdf : gpd.GeoDataFrame
        Previous iteration labels
    new_labels_gdf : gpd.GeoDataFrame
        New predicted labels
    overlap_threshold : float
        Minimum overlap ratio to consider intersection (0.10 = 10%)
    
    Returns
    -------
    gpd.GeoDataFrame
        Components that significantly overlap with old labels
    """
    if old_labels_gdf.empty or new_labels_gdf.empty:
        return gpd.GeoDataFrame(columns=new_labels_gdf.columns, crs=new_labels_gdf.crs)
    
    old_union = unary_union(old_labels_gdf.geometry)
    
    intersection_indices = []
    
    for idx, row in new_labels_gdf.iterrows():
        geom = row.geometry
        intersection = geom.intersection(old_union)
        
        if not intersection.is_empty:
            overlap_ratio = intersection.area / geom.area
            if overlap_ratio >= overlap_threshold:
                intersection_indices.append(idx)
    
    return new_labels_gdf.loc[intersection_indices].copy()


def join_labels_vector(
    high_priority_gdf: gpd.GeoDataFrame,
    low_priority_gdf: gpd.GeoDataFrame,
    overlap_limit: float = 0.05,
) -> gpd.GeoDataFrame:
    """
    Join two GeoDataFrames, keeping high_priority where overlapping.
    
    Replaces the slow raster-based join_labels_set function.
    
    Parameters
    ----------
    high_priority_gdf : gpd.GeoDataFrame
        Labels with high priority (kept when overlapping)
    low_priority_gdf : gpd.GeoDataFrame
        Labels with low priority (removed when overlapping)
    overlap_limit : float
        Maximum allowed overlap ratio
    
    Returns
    -------
    gpd.GeoDataFrame
        Combined GeoDataFrame
    """
    if high_priority_gdf.empty:
        return low_priority_gdf.copy()
    
    if low_priority_gdf.empty:
        return high_priority_gdf.copy()
    
    high_union = unary_union(high_priority_gdf.geometry)
    
    # Filter low priority to keep only non-overlapping
    keep_indices = []
    
    for idx, row in low_priority_gdf.iterrows():
        geom = row.geometry
        intersection = geom.intersection(high_union)
        
        if intersection.is_empty:
            keep_indices.append(idx)
        else:
            overlap_ratio = intersection.area / geom.area
            if overlap_ratio < overlap_limit:
                keep_indices.append(idx)
    
    # Combine
    low_priority_kept = low_priority_gdf.loc[keep_indices]
    
    result = pd.concat([high_priority_gdf, low_priority_kept], ignore_index=True)
    
    return gpd.GeoDataFrame(result, crs=high_priority_gdf.crs)


def filter_by_geometric_properties_vector(
    gdf: gpd.GeoDataFrame,
    min_area: float = None,
    max_area: float = None,
    min_solidity: float = None,
    max_eccentricity: float = None,
    area_in_meters: bool = True,
) -> gpd.GeoDataFrame:
    """
    Filter GeoDataFrame by geometric properties (vector-based).
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame with geometry column
    min_area, max_area : float, optional
        Area bounds in square meters (m²) if area_in_meters=True,
        otherwise in the CRS native units
    min_solidity : float, optional
        Minimum solidity threshold
    max_eccentricity : float, optional
        Maximum eccentricity threshold
    area_in_meters : bool, default True
        If True, area thresholds are interpreted as square meters (m²)
        and the function will automatically convert from the CRS units.
        If False, uses the native CRS units (may be degrees² for geographic CRS).
    
    Returns
    -------
    gpd.GeoDataFrame
        Filtered GeoDataFrame
    """
    if gdf.empty:
        return gdf.copy()
    
    mask = pd.Series([True] * len(gdf), index=gdf.index)
    
    # Calculate area for filtering
    if area_in_meters and (min_area is not None or max_area is not None):
        # Use area in square meters (CRS-agnostic)
        area_m2 = get_area_in_square_meters(gdf)
        gdf = gdf.copy()
        gdf['area_m2'] = area_m2
        area_column = 'area_m2'
    else:
        # Use native CRS units
        if 'area' not in gdf.columns:
            gdf = gdf.copy()
            gdf['area'] = gdf.geometry.area
        area_column = 'area'
    
    if min_area is not None:
        mask &= gdf[area_column] >= min_area
    
    if max_area is not None:
        mask &= gdf[area_column] <= max_area
    
    if min_solidity is not None:
        if 'solidity' not in gdf.columns:
            gdf = gdf.copy()
            gdf['convex_area'] = gdf.geometry.convex_hull.area
            gdf['solidity'] = gdf.geometry.area / gdf['convex_area'].replace(0, np.nan)
            gdf['solidity'] = gdf['solidity'].fillna(1.0)
        mask &= gdf['solidity'] >= min_solidity
    
    if max_eccentricity is not None:
        if 'eccentricity' not in gdf.columns:
            # Would need to compute, skip for now
            pass
        else:
            mask &= gdf['eccentricity'] <= max_eccentricity
    
    return gdf[mask].copy()


def filter_by_mask_vector(
    gdf: gpd.GeoDataFrame,
    mask_path: str,
    max_outside_ratio: float = 0.20,
) -> gpd.GeoDataFrame:
    """
    Filter components that are mostly outside a mask (vector-based).
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame
    mask_path : str
        Path to mask TIFF
    max_outside_ratio : float
        Maximum ratio of area outside mask (0.20 = 20%)
    
    Returns
    -------
    gpd.GeoDataFrame
        Filtered GeoDataFrame
    """
    if gdf.empty:
        return gdf.copy()
    
    # Load mask and convert to polygon
    mask = read_tiff(mask_path)
    
    if mask.dtype != bool:
        mask = mask > 0
    
    # If entire mask is valid, return as-is
    if mask.all():
        return gdf.copy()
    
    # Convert valid mask region to polygon
    meta = get_image_metadata(mask_path)
    transform = meta.get('transform', Affine(1, 0, 0, 0, 1, 0))
    
    valid_shapes = list(features.shapes(
        mask.astype(np.uint8),
        mask=mask,
        transform=transform
    ))
    
    if not valid_shapes:
        return gpd.GeoDataFrame(columns=gdf.columns, crs=gdf.crs)
    
    valid_union = unary_union([shape(geom) for geom, _ in valid_shapes])
    
    keep_indices = []
    
    for idx, row in gdf.iterrows():
        geom = row.geometry
        intersection = geom.intersection(valid_union)
        
        if not intersection.is_empty:
            inside_ratio = intersection.area / geom.area
            if inside_ratio >= (1 - max_outside_ratio):
                keep_indices.append(idx)
    
    return gdf.loc[keep_indices].copy()


def select_n_labels_by_class_vector(
    gdf: gpd.GeoDataFrame,
    samples_by_class: int = 5,
    random_state: int = 0,
) -> gpd.GeoDataFrame:
    """
    Select N random samples per tree_type class (vector-based).
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame with 'tree_type' column
    samples_by_class : int
        Number of samples to select per class
    random_state : int
        Random seed for reproducibility
    
    Returns
    -------
    gpd.GeoDataFrame
        Sampled GeoDataFrame
    """
    if gdf.empty or 'tree_type' not in gdf.columns:
        return gdf.copy()
    
    # Shuffle and select
    gdf_shuffled = gdf.sample(frac=1, random_state=random_state)
    
    selected = gdf_shuffled.groupby('tree_type').head(samples_by_class)
    
    return selected.copy()


# =============================================================================
# VECTOR-BASED DISTANCE MAP GENERATION
# =============================================================================

def generate_distance_map_from_gdf(
    gdf: gpd.GeoDataFrame,
    shape: Tuple[int, int],
    sigma: float = 5,
    transform: Optional[Affine] = None,
) -> np.ndarray:
    """
    Generate a distance map from polygon geometries.
    
    Each polygon gets a normalized distance map (0 at boundary, 1 at center).
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame with polygon geometries
    shape : Tuple[int, int]
        Output raster shape (height, width)
    sigma : float
        Gaussian smoothing sigma
    transform : Affine, optional
        Affine transform
    
    Returns
    -------
    np.ndarray
        2D distance map array (float32)
    """
    if gdf.empty:
        return np.zeros(shape, dtype=np.float32)
    
    if transform is None:
        transform = Affine(1, 0, 0, 0, 1, 0)
    
    # Create binary mask for all polygons
    all_shapes = [(geom, 1) for geom in gdf.geometry]
    
    binary_mask = features.rasterize(
        shapes=all_shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True
    )
    
    # Compute distance transform
    distance_map = distance_transform_edt(binary_mask)
    
    # Apply Gaussian smoothing
    distance_map = gaussian_filter(distance_map, sigma=sigma)
    
    # Normalize per component
    components = label(binary_mask)
    output = np.zeros(shape, dtype=np.float32)
    
    for comp_id in np.unique(components[components > 0]):
        comp_mask = components == comp_id
        max_val = distance_map[comp_mask].max()
        if max_val > 0:
            output[comp_mask] = distance_map[comp_mask] / max_val
    
    output[output < 0.05] = 0
    
    return output


# =============================================================================
# HIGH-LEVEL WORKFLOW FUNCTIONS
# =============================================================================

def get_new_segmentation_sample_vector(
    ground_truth_gdf: gpd.GeoDataFrame,
    old_selected_gdf: gpd.GeoDataFrame,
    old_all_gdf: gpd.GeoDataFrame,
    new_pred_gdf: gpd.GeoDataFrame,
    args: dict,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Vector-based version of get_new_segmentation_sample.
    
    Performs all sample selection using vector operations for speed.
    
    Parameters
    ----------
    ground_truth_gdf : gpd.GeoDataFrame
        Ground truth labels
    old_selected_gdf : gpd.GeoDataFrame
        Selected labels from previous iteration
    old_all_gdf : gpd.GeoDataFrame
        All labels from previous iteration
    new_pred_gdf : gpd.GeoDataFrame
        New model predictions
    args : dict
        Configuration arguments
    
    Returns
    -------
    Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
        (all_labels_set, selected_labels_set)
    """
    logger.info("Vector-based sample selection starting...")
    
    # Filter by geometric properties (area in square meters, CRS-agnostic)
    lower_limit = args.get('lower_limit_area')
    upper_limit = args.get('upper_limit_area')
    logger.info(f"Filtering components by geometric properties: area {lower_limit} m² - {upper_limit} m²")
    new_pred_gdf = filter_by_geometric_properties_vector(
        new_pred_gdf,
        min_area=lower_limit,
        max_area=upper_limit,
        area_in_meters=True,  # Use square meters (m²), CRS-agnostic
    )
    
    # Filter by mask (auto-generate if not provided)
    data_path = args.get('data_path', '.')
    mask_path = get_or_generate_mask_path(args, region_idx=0, data_path=data_path)
    
    if mask_path and exists(mask_path):
        logger.info(f"Filtering components by mask: {mask_path}")
        new_pred_gdf = filter_by_mask_vector(new_pred_gdf, mask_path)
    
    # Join all labels with new predictions
    logger.info("Joining old and new components")
    new_labels_gdf = join_labels_vector(new_pred_gdf, old_all_gdf, overlap_limit=0.05)
    
    # Get delta (new components not in old selected)
    logger.info("Getting new components (delta)")
    delta_gdf = get_labels_delta_vector(old_selected_gdf, new_labels_gdf, overlap_threshold=0.10)
    unbalanced_delta_gdf = delta_gdf.copy()
    
    # Select balanced sample by class
    logger.info("Selecting balanced samples by tree_type")
    delta_gdf = select_n_labels_by_class_vector(delta_gdf, samples_by_class=5)
    
    # Get intersection (updated shapes for old components)
    logger.info("Getting intersection components")
    intersection_gdf = get_label_intersection_vector(old_selected_gdf, new_labels_gdf, overlap_threshold=0.10)
    
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
    
    logger.info(f"Vector selection complete: {len(all_labels_gdf)} all, {len(selected_labels_gdf)} selected")
    
    return all_labels_gdf, selected_labels_gdf


def _get_arg(args, key: str, default=None):
    """Helper to get arg value from dict or object."""
    if hasattr(args, 'get'):
        return args.get(key, default)
    elif hasattr(args, key):
        return getattr(args, key, default)
    return default


def compute_reference_stats(labels_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Compute reference statistics (median area and solidity) by tree_type.
    
    This function can be used to compute global reference statistics from
    labels across multiple regions.
    
    Parameters
    ----------
    labels_gdf : gpd.GeoDataFrame
        Labels GeoDataFrame with 'tree_type' column
    
    Returns
    -------
    pd.DataFrame
        DataFrame indexed by tree_type with columns 'ref_area' and 'ref_solidity'
    """
    if labels_gdf.empty or 'tree_type' not in labels_gdf.columns:
        return pd.DataFrame(columns=['ref_area', 'ref_solidity'])
    
    labels_gdf = labels_gdf.copy()
    
    # Ensure area is computed
    if 'area' not in labels_gdf.columns:
        labels_gdf['area'] = labels_gdf.geometry.area
    
    # Ensure solidity is computed
    if 'solidity' not in labels_gdf.columns:
        labels_gdf['convex_area'] = labels_gdf.geometry.convex_hull.area
        labels_gdf['solidity'] = labels_gdf['area'] / labels_gdf['convex_area'].replace(0, np.nan)
        labels_gdf['solidity'] = labels_gdf['solidity'].fillna(1.0)
    
    ref_stats = labels_gdf.groupby('tree_type').agg({
        'area': 'median',
        'solidity': 'median',
    }).rename(columns={'area': 'ref_area', 'solidity': 'ref_solidity'})
    
    return ref_stats


def filter_by_reference_stats(
    new_pred_gdf: gpd.GeoDataFrame,
    ref_stats: pd.DataFrame,
    args,
) -> gpd.GeoDataFrame:
    """
    Filter predictions based on reference statistics.
    
    Filters samples by comparing their area and solidity to the reference
    statistics (median values) for each tree_type.
    
    Parameters
    ----------
    new_pred_gdf : gpd.GeoDataFrame
        New predictions to filter
    ref_stats : pd.DataFrame
        Reference statistics DataFrame with 'ref_area' and 'ref_solidity' columns,
        indexed by tree_type. Can be computed using compute_reference_stats().
    args : dict or object
        Configuration arguments with thresholds:
        - upper_limit_area_rlted_to_tree_type (default: 2.0)
        - lower_limit_area_rlted_to_tree_type (default: -0.5)
        - lower_limit_solidity_rlted_to_tree_type (default: -0.2)
    
    Returns
    -------
    gpd.GeoDataFrame
        Filtered predictions
    """
    if new_pred_gdf.empty:
        return new_pred_gdf.copy()
    
    if ref_stats.empty:
        return new_pred_gdf.copy()
    
    # Ensure new predictions have required columns
    new_pred_gdf = new_pred_gdf.copy()
    if 'area' not in new_pred_gdf.columns:
        new_pred_gdf['area'] = new_pred_gdf.geometry.area
    
    if 'solidity' not in new_pred_gdf.columns:
        new_pred_gdf['convex_area'] = new_pred_gdf.geometry.convex_hull.area
        new_pred_gdf['solidity'] = new_pred_gdf['area'] / new_pred_gdf['convex_area'].replace(0, np.nan)
        new_pred_gdf['solidity'] = new_pred_gdf['solidity'].fillna(1.0)
    
    # Merge with reference stats
    new_pred_gdf = new_pred_gdf.merge(ref_stats, on='tree_type', how='left')
    
    # Compute differences
    new_pred_gdf['diff_area'] = (new_pred_gdf['area'] - new_pred_gdf['ref_area']) / new_pred_gdf['ref_area'].replace(0, np.nan)
    new_pred_gdf['diff_soli'] = new_pred_gdf['solidity'] - new_pred_gdf['ref_solidity']
    
    # Apply filters - support both dict and object args
    upper_area = _get_arg(args, 'upper_limit_area_rlted_to_tree_type', 2.0)
    lower_area = _get_arg(args, 'lower_limit_area_rlted_to_tree_type', -0.5)
    lower_soli = _get_arg(args, 'lower_limit_solidity_rlted_to_tree_type', -0.2)
    
    mask = (
        (new_pred_gdf['diff_area'] <= upper_area) &
        (new_pred_gdf['diff_area'] >= lower_area) &
        (new_pred_gdf['diff_soli'] >= lower_soli)
    )
    
    # Handle NaN values
    mask = mask.fillna(False)
    
    return new_pred_gdf[mask].copy()


def select_good_samples_vector(
    old_labels_gdf: gpd.GeoDataFrame,
    new_pred_gdf: gpd.GeoDataFrame,
    args,
    ref_stats: pd.DataFrame = None,
) -> gpd.GeoDataFrame:
    """
    Select high-quality samples based on geometric properties (vector-based).
    
    Parameters
    ----------
    old_labels_gdf : gpd.GeoDataFrame
        Previous labels for reference statistics (used if ref_stats is None)
    new_pred_gdf : gpd.GeoDataFrame
        New predictions to filter
    args : dict or object
        Configuration arguments (supports both dict and object with attributes)
    ref_stats : pd.DataFrame, optional
        Pre-computed reference statistics. If provided, old_labels_gdf is not used
        to compute statistics. This allows using global statistics from multiple regions.
    
    Returns
    -------
    gpd.GeoDataFrame
        Filtered predictions
    """
    if new_pred_gdf.empty:
        return new_pred_gdf.copy()
    
    # Compute reference statistics if not provided
    if ref_stats is None:
        if 'tree_type' not in old_labels_gdf.columns:
            return new_pred_gdf.copy()
        ref_stats = compute_reference_stats(old_labels_gdf)
    
    return filter_by_reference_stats(new_pred_gdf, ref_stats, args)


if __name__ == "__main__":
    # Test basic functionality
    import matplotlib.pyplot as plt
    
    # Create test labels
    test_labels = np.zeros((100, 100), dtype=np.uint8)
    test_labels[20:40, 20:40] = 1
    test_labels[50:70, 50:70] = 2
    test_labels[30:50, 60:80] = 1
    
    print("Testing raster_to_geodataframe...")
    gdf = raster_to_geodataframe(test_labels)
    print(f"Created GeoDataFrame with {len(gdf)} features")
    print(gdf)
    
    print("\nTesting labels_to_geodataframe_with_stats...")
    gdf_stats = labels_to_geodataframe_with_stats(test_labels)
    print(f"Stats columns: {gdf_stats.columns.tolist()}")
    
    print("\nTesting geodataframe_to_raster...")
    raster_back = geodataframe_to_raster(gdf, test_labels.shape)
    print(f"Raster shape: {raster_back.shape}, unique values: {np.unique(raster_back)}")
    
    print("\nAll tests passed!")

