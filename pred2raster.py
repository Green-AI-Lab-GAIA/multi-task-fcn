import os
from os.path import isfile, join
from logging import getLogger
from typing import List
import numpy as np

from src.utils import check_folder, convert_to_minor_numeric_type
from src.io_operations import array2raster, get_image_metadata, read_yaml

logger = getLogger("__main__")


def delete_prediction_files(current_iter_folder, overlaps):
    
    for ov in overlaps:
        prediction_overlap_path = os.path.join(current_iter_folder, f'prediction_{ov}.npz')
        os.remove(prediction_overlap_path)
    
    

def compute_mean_prediction(data_source:str, overlaps:List[float], current_iter_folder:str):
    
    for num, ov in enumerate(overlaps):

        prediction_overlap_path = join(current_iter_folder, f'prediction_{ov}.npz')
        prediction_ov_data = np.load(prediction_overlap_path)
        
        if num == 0:
            prediction_test = np.float32(prediction_ov_data[data_source])
            continue

        else:
            prediction_test = np.add(prediction_test, np.float32(prediction_ov_data[data_source]))
        
        prediction_ov_data.close()

    mean_prediction = prediction_test/len(overlaps)
    
    if np.max(mean_prediction) > 2:
        mean_prediction = np.ceil(mean_prediction)
        mean_prediction = np.uint8(mean_prediction)
    
    return mean_prediction
    
        
  

    
def pred2raster(current_iter_folder, args):

    output_folder = join(current_iter_folder, 'raster_prediction')
    check_folder(output_folder)

    prediction_file = join(output_folder,f'join_class_{np.sum(args.overlap)}.TIF')
    
    prob_file = join(output_folder,f'join_prob_{np.sum(args.overlap)}.TIF')
    
    depth_file = join(output_folder,f'depth_{np.sum(args.overlap)}.TIF')
               
    
    if (isfile(prediction_file) and isfile(prob_file) and isfile(depth_file)):
        return
    
    logger.info("============ Started pred2raster ============")
    logger.info(f"Computing the mean between the {len(args.overlap)} slices")
    
    RASTER_PATH = args.train_segmentation_path
    image_metadata = get_image_metadata(RASTER_PATH)

    logger.info("Computing the mean of prob_map")
    prob_map_mean = compute_mean_prediction("prob_map", args.overlap, current_iter_folder)

    logger.info("Saving prob_map and class_map as raster files")
    array2raster(prediction_file, np.argmax(prob_map_mean, axis = -1), image_metadata, "uint8")
    array2raster(prob_file, np.amax(prob_map_mean, axis = -1), image_metadata, "uint8")
    del prob_map_mean


    logger.info("Computing the mean of depth_map")
    depth_map_mean = compute_mean_prediction("depth_map", args.overlap, current_iter_folder)
    
    logger.info("Saving prob_map and class_map to raster files")
    array2raster(depth_file, depth_map_mean, image_metadata, "uint8")
    del depth_map_mean

    delete_prediction_files(current_iter_folder, args.overlap)

    



if __name__ == "__main__":

    args = read_yaml("args.yaml")
    # external parameters
    current_iter_folder = "/home/luiz/multi-task-fcn/MyData/iter_1"
    current_iter = int(current_iter_folder.split("_")[-1])
    current_model_folder = os.path.join(current_iter_folder, args.model_dir)

    pred2raster(current_iter_folder, args)
