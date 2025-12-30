import os
from os.path import isfile, join, exists
from logging import getLogger
from typing import List
import numpy as np

from src.utils import check_folder, convert_to_minor_numeric_type
from src.io_operations import array2raster, get_image_metadata, read_yaml

logger = getLogger("__main__")


def delete_prediction_files(prediction_folder: str, overlaps: List[float]):
    """Delete prediction files after converting to raster.
    
    Parameters
    ----------
    prediction_folder : str
        Folder containing the prediction files
    overlaps : List[float]
        List of overlap values
    """
    for ov in overlaps:
        prediction_overlap_path = os.path.join(prediction_folder, f'prediction_{ov}.npz')
        if exists(prediction_overlap_path):
            os.remove(prediction_overlap_path)
    
    

def compute_mean_prediction(data_source: str, overlaps: List[float], prediction_folder: str):
    """Compute mean prediction across different overlap values.
    
    Parameters
    ----------
    data_source : str
        Key in the npz file ('prob_map', 'depth_map', etc.)
    overlaps : List[float]
        List of overlap values
    prediction_folder : str
        Folder containing the prediction files
    """
    for num, ov in enumerate(overlaps):

        prediction_overlap_path = join(prediction_folder, f'prediction_{ov}.npz')
        prediction_ov_data = np.load(prediction_overlap_path)
        
        if num == 0:
            prediction_test = np.float16(prediction_ov_data[data_source])
            continue

        else:
            prediction_test = np.add(prediction_test, np.float16(prediction_ov_data[data_source]))
        
        prediction_ov_data.close()

    mean_prediction = prediction_test/len(overlaps)
    
    if np.max(mean_prediction) > 2:
        mean_prediction = np.uint8(mean_prediction)
    
    return mean_prediction
    
        
def pred2raster_single_region(prediction_folder: str, output_folder: str, 
                               segmentation_path: str, overlaps: List[float]):
    """Convert predictions to raster for a single region.
    
    Parameters
    ----------
    prediction_folder : str
        Folder containing prediction .npz files
    output_folder : str
        Folder to save raster outputs
    segmentation_path : str
        Path to segmentation file (for metadata)
    overlaps : List[float]
        List of overlap values used
    """
    check_folder(output_folder)

    prediction_file = join(output_folder, f'join_class_{np.sum(overlaps)}.TIF')
    prob_file = join(output_folder, f'join_prob_{np.sum(overlaps)}.TIF')
    depth_file = join(output_folder, f'depth_{np.sum(overlaps)}.TIF')
    
    if (isfile(prediction_file) and isfile(prob_file) and isfile(depth_file)):
        return True  # Already done
    
    logger.info(f"Computing the mean between the {len(overlaps)} slices")
    
    image_metadata = get_image_metadata(segmentation_path)

    logger.info("Computing the mean of prob_map")
    prob_map_mean = compute_mean_prediction("prob_map", overlaps, prediction_folder)

    logger.info("Saving prob_map and class_map as raster files")
    array2raster(prediction_file, np.argmax(prob_map_mean, axis=-1), image_metadata, "uint8")
    array2raster(prob_file, np.amax(prob_map_mean, axis=-1), image_metadata, "uint8")
    del prob_map_mean

    logger.info("Computing the mean of depth_map")
    depth_map_mean = compute_mean_prediction("depth_map", overlaps, prediction_folder)
    
    logger.info("Saving depth_map to raster file")
    array2raster(depth_file, depth_map_mean, image_metadata, "uint8")
    del depth_map_mean

    delete_prediction_files(prediction_folder, overlaps)
    
    return True

    
def pred2raster(current_iter_folder, args):
    """Convert predictions to raster format for all regions.
    
    Supports multiple regions: processes each region's predictions
    and saves rasters in region-specific subfolders.

    Parameters
    ----------
    current_iter_folder : str
        Path to current iteration folder
    args : dict
        Arguments dictionary (normalized with multi-region support)
    """
    num_regions = getattr(args, 'num_regions', 1)
    
    logger.info(f"============ Started pred2raster for {num_regions} region(s) ============")
    
    for region_idx in range(num_regions):
        # Determine folders based on number of regions
        if num_regions > 1:
            region_folder = join(current_iter_folder, f'region_{region_idx}')
            prediction_folder = region_folder
            output_folder = join(region_folder, 'raster_prediction')
            segmentation_path = args.train_segmentation_paths[region_idx]
            logger.info(f"=== Processing Region {region_idx} ===")
        else:
            prediction_folder = current_iter_folder
            output_folder = join(current_iter_folder, 'raster_prediction')
            segmentation_path = args.train_segmentation_paths[0]
        
        pred2raster_single_region(
            prediction_folder=prediction_folder,
            output_folder=output_folder,
            segmentation_path=segmentation_path,
            overlaps=args.overlap
        )
        
        logger.info(f"Region {region_idx} pred2raster done.")


def pred2raster_for_region(current_iter_folder: str, args: dict, region_idx: int):
    """Convert predictions to raster for a specific region.
    
    Parameters
    ----------
    current_iter_folder : str
        Path to current iteration folder
    args : dict
        Arguments dictionary
    region_idx : int
        Index of the region to process
    """
    num_regions = getattr(args, 'num_regions', 1)
    
    if num_regions > 1:
        region_folder = join(current_iter_folder, f'region_{region_idx}')
        prediction_folder = region_folder
        output_folder = join(region_folder, 'raster_prediction')
    else:
        prediction_folder = current_iter_folder
        output_folder = join(current_iter_folder, 'raster_prediction')
    
    segmentation_path = args.train_segmentation_paths[region_idx]
    
    logger.info(f"=== pred2raster for Region {region_idx} ===")
    
    pred2raster_single_region(
        prediction_folder=prediction_folder,
        output_folder=output_folder,
        segmentation_path=segmentation_path,
        overlaps=args.overlap
    )

    



if __name__ == "__main__":

    args = read_yaml("args.yaml")
    # external parameters
    current_iter_folder = "/home/luiz/multi-task-fcn/MyData/iter_1"
    current_iter = int(current_iter_folder.split("_")[-1])
    current_model_folder = os.path.join(current_iter_folder, args.model_dir)

    pred2raster(current_iter_folder, args)
