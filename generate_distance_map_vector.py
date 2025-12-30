"""
Vector-based distance map generation.

This module provides optimized distance map generation using vector operations.
Instead of iterating through each component on rasters, it uses:
- Vectorized rasterization from GeoPackage/Shapefile
- Batch processing of components
- Efficient distance transform

Author: Optimized from generate_distance_map.py
"""

import os
import sys
from logging import getLogger
from os.path import dirname, exists

import geopandas as gpd
import numpy as np
from rasterio.transform import Affine
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.measure import label
from tqdm import tqdm

ROOT_PATH = dirname(__file__)
sys.path.append(ROOT_PATH)

from src.io_operations import array2raster, get_image_metadata, read_tiff
from src.utils import check_folder
from src.vector_operations import (
    geodataframe_to_raster,
    labels_to_geodataframe_with_stats,
    load_labels_from_geopackage,
    raster_to_geodataframe,
)

logger = getLogger("__main__")


def normalize_each_component_vectorized(
    components: np.ndarray,
    distance_map: np.ndarray,
) -> np.ndarray:
    """
    Normalize distance map per component using vectorized operations.
    
    This is faster than iterating through each component.
    
    Parameters
    ----------
    components : np.ndarray
        Labeled components array
    distance_map : np.ndarray
        Raw distance transform array
    
    Returns
    -------
    np.ndarray
        Normalized distance map (0 at boundary, 1 at center for each component)
    """
    output = np.zeros_like(distance_map, dtype=np.float32)
    
    # Get unique component IDs
    unique_components = np.unique(components[components > 0])
    
    if len(unique_components) == 0:
        return output
    
    # Use ndimage's labeled_comprehension for vectorized computation
    from scipy.ndimage import labeled_comprehension, maximum
    
    # Get maximum distance per component
    max_distances = maximum(distance_map, components, unique_components)
    
    # Create lookup array for max distances
    max_distance_lookup = np.zeros(components.max() + 1, dtype=np.float32)
    for comp_id, max_dist in zip(unique_components, max_distances):
        if max_dist > 0:
            max_distance_lookup[comp_id] = max_dist
    
    # Normalize in one operation
    mask = components > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        output[mask] = distance_map[mask] / max_distance_lookup[components[mask]]
    
    # Handle any NaN/Inf values
    output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
    
    return output


def apply_gaussian_distance_map_fast(
    input_img: np.ndarray,
    sigma: float = 5,
) -> np.ndarray:
    """
    Apply distance transform and gaussian filter efficiently.
    
    Optimized version that uses vectorized normalization.
    
    Parameters
    ----------
    input_img : np.ndarray
        Input segmentation mask (0 = background)
    sigma : float
        Gaussian smoothing sigma
    
    Returns
    -------
    np.ndarray
        Normalized distance map with gaussian smoothing
    """
    # Create binary mask and label components
    binary_mask = input_img > 0
    components = label(binary_mask)
    
    # Apply distance transform
    distance_map = distance_transform_edt(binary_mask).astype(np.float32)
    
    # Apply Gaussian smoothing
    distance_map = gaussian_filter(distance_map, sigma=sigma)
    
    # Normalize per component (vectorized)
    normalized = normalize_each_component_vectorized(components, distance_map)
    
    return normalized


def generate_distance_map_from_geopackage(
    gpkg_path: str,
    output_path: str,
    reference_tiff: str,
    sigma: float = 5,
) -> np.ndarray:
    """
    Generate distance map from a GeoPackage file.
    
    Parameters
    ----------
    gpkg_path : str
        Path to the GeoPackage with label polygons
    output_path : str
        Path to save the distance map TIFF
    reference_tiff : str
        Reference TIFF for shape and georeferencing
    sigma : float
        Gaussian smoothing sigma
    
    Returns
    -------
    np.ndarray
        Distance map array
    """
    if not exists(gpkg_path):
        raise FileNotFoundError(f"GeoPackage not found: {gpkg_path}")
    
    if not exists(reference_tiff):
        raise FileNotFoundError(f"Reference TIFF not found: {reference_tiff}")
    
    # Get reference metadata
    meta = get_image_metadata(reference_tiff)
    shape = (meta['height'], meta['width'])
    transform = meta.get('transform', Affine(1, 0, 0, 0, 1, 0))
    
    # Load GeoPackage
    logger.info(f"Loading labels from {gpkg_path}")
    gdf = gpd.read_file(gpkg_path)
    
    if gdf.empty:
        logger.warning("Empty GeoPackage, returning zero distance map")
        output = np.zeros(shape, dtype=np.float32)
        array2raster(output_path, output, meta, "float32")
        return output
    
    # Rasterize
    logger.info("Rasterizing polygons...")
    labels = geodataframe_to_raster(gdf, shape, transform)
    
    # Generate distance map
    logger.info("Generating distance map...")
    distance_map = apply_gaussian_distance_map_fast(labels, sigma)
    
    # Apply threshold
    distance_map[distance_map < 0.05] = 0
    
    # Save
    check_folder(dirname(output_path))
    array2raster(output_path, distance_map, meta, "float32")
    
    logger.info(f"Saved distance map to {output_path}")
    
    return distance_map


def generate_distance_map_fast(
    input_image_path: str,
    output_image_path: str,
    sigma: float = 5,
) -> np.ndarray:
    """
    Generate distance map from input TIFF (optimized version).
    
    Drop-in replacement for generate_distance_map with vectorized normalization.
    
    Parameters
    ----------
    input_image_path : str
        Path to input segmentation TIFF
    output_image_path : str
        Path to save the distance map
    sigma : float
        Gaussian smoothing sigma
    
    Returns
    -------
    np.ndarray
        Distance map array
    """
    if not exists(input_image_path):
        raise FileNotFoundError(f"Input image not found: {input_image_path}")
    
    if not exists(dirname(output_image_path)):
        raise FileNotFoundError(f"Output folder not found: {dirname(output_image_path)}")
    
    logger.info(f"Loading segmentation from {input_image_path}")
    img_metadata = get_image_metadata(input_image_path)
    input_img = read_tiff(input_image_path).astype('uint16')
    
    logger.info("Generating distance map with vectorized normalization...")
    output_img = apply_gaussian_distance_map_fast(input_img, sigma)
    
    # Apply threshold
    output_img[output_img < 0.05] = 0
    
    # Save
    array2raster(output_image_path, output_img, img_metadata, "float32")
    
    logger.info(f"Saved distance map to {output_image_path}")
    
    return output_img


def generate_distance_map_from_labels(
    labels: np.ndarray,
    output_path: str = None,
    reference_tiff: str = None,
    sigma: float = 5,
) -> np.ndarray:
    """
    Generate distance map from a labels array.
    
    Parameters
    ----------
    labels : np.ndarray
        Segmentation labels array
    output_path : str, optional
        Path to save the distance map
    reference_tiff : str, optional
        Reference TIFF for georeferencing (required if saving)
    sigma : float
        Gaussian smoothing sigma
    
    Returns
    -------
    np.ndarray
        Distance map array
    """
    logger.info("Generating distance map from array...")
    distance_map = apply_gaussian_distance_map_fast(labels, sigma)
    
    # Apply threshold
    distance_map[distance_map < 0.05] = 0
    
    # Save if path provided
    if output_path:
        if not reference_tiff:
            raise ValueError("reference_tiff required to save output")
        
        img_metadata = get_image_metadata(reference_tiff)
        check_folder(dirname(output_path))
        array2raster(output_path, distance_map, img_metadata, "float32")
        logger.info(f"Saved distance map to {output_path}")
    
    return distance_map


def generate_distance_map_from_gdf(
    gdf: gpd.GeoDataFrame,
    shape: tuple,
    sigma: float = 5,
    transform: Affine = None,
) -> np.ndarray:
    """
    Generate distance map from a GeoDataFrame.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with polygon geometries
    shape : tuple
        Output shape (height, width)
    sigma : float
        Gaussian smoothing sigma
    transform : Affine, optional
        Affine transform for rasterization
    
    Returns
    -------
    np.ndarray
        Distance map array
    """
    if gdf.empty:
        return np.zeros(shape, dtype=np.float32)
    
    # Rasterize
    labels = geodataframe_to_raster(gdf, shape, transform)
    
    # Generate distance map
    distance_map = apply_gaussian_distance_map_fast(labels, sigma)
    
    # Apply threshold
    distance_map[distance_map < 0.05] = 0
    
    return distance_map


# Backward compatibility alias
generate_distance_map = generate_distance_map_fast


if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("Testing Distance Map Generation")
    print("=" * 60)
    
    # Create synthetic test data
    np.random.seed(42)
    shape = (500, 500)
    
    # Create test segmentation
    test_labels = np.zeros(shape, dtype=np.uint8)
    for i in range(30):
        x, y = np.random.randint(50, 450, 2)
        size = np.random.randint(15, 40)
        tree_type = np.random.randint(1, 4)
        
        # Create circular-ish shape
        yy, xx = np.ogrid[:shape[0], :shape[1]]
        mask = ((xx - y)**2 + (yy - x)**2) < size**2
        test_labels[mask] = tree_type
    
    print(f"\nTest data: {shape}, {np.sum(test_labels > 0)} labeled pixels")
    print(f"Components: {len(np.unique(test_labels)) - 1}")
    
    # Time the optimized version
    print("\n" + "-" * 40)
    print("Testing vectorized distance map generation...")
    start = time.time()
    
    distance_map = apply_gaussian_distance_map_fast(test_labels, sigma=5)
    
    elapsed = time.time() - start
    print(f"Completed in {elapsed:.3f} seconds")
    print(f"Distance map range: [{distance_map.min():.3f}, {distance_map.max():.3f}]")
    print(f"Non-zero pixels: {np.sum(distance_map > 0)}")
    
    # Compare with original (if available)
    try:
        from generate_distance_map import apply_gaussian_distance_map
        
        print("\n" + "-" * 40)
        print("Comparing with original implementation...")
        start = time.time()
        
        distance_map_orig = apply_gaussian_distance_map(test_labels, sigma=5)
        
        elapsed_orig = time.time() - start
        print(f"Original completed in {elapsed_orig:.3f} seconds")
        
        # Check similarity
        diff = np.abs(distance_map - distance_map_orig)
        print(f"Max difference: {diff.max():.6f}")
        print(f"Mean difference: {diff.mean():.6f}")
        
        speedup = elapsed_orig / elapsed
        print(f"\nSpeedup: {speedup:.1f}x faster")
        
    except ImportError:
        print("\nOriginal implementation not available for comparison")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

