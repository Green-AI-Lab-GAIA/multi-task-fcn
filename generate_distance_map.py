import os

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.measure import label
from tqdm import tqdm

from src.utils import check_folder,print_sucess
from src.io_operations import array2raster,get_image_metadata, read_tiff, read_yaml

from logging import getLogger

logger = getLogger("__main__")


def normalize_each_component(img_components:np.ndarray, distance_map:np.ndarray)->np.ndarray:
    
    # create the new image with the distance map
    save_lab = np.zeros(img_components.shape)

    for component in tqdm(np.unique(img_components[img_components>0])):

        component_mask = img_components==component

        save_lab[component_mask] = distance_map[component_mask]/np.max(distance_map[component_mask])
    
    return save_lab


def apply_gaussian_distance_map(input_img:np.ndarray, sigma=5)->np.ndarray:
    """Apply euclidean distance transform and gaussian filter to the input image

    Parameters
    ----------
    input_img : np.ndarray
        Input image matrix with the segmentation

    Returns
    -------
    np.array
        Output image matrix with the distance map and gaussian filter applied to each segmentation component
    """

    ref = input_img.copy()

    # label the image as components
    components = label(ref)
    
    # apply distance transform and gaussian filter
    distance_map = components.copy()
    distance_map = distance_transform_edt(distance_map)
    distance_map = gaussian_filter(distance_map, sigma)
    
    normalized_distance_map = normalize_each_component(components, distance_map)
    
    return normalized_distance_map
    



# %%
def generate_distance_map(input_image_path:str, output_image_path:str, sigma:int=5):
    """Generate the distance map from the input image and save it in the output path
    
    Parameters
    ----------
    input_image_path : str
        Path to the input image
    
    output_image_path : str
        Path to save the output image
    """
    
    if not os.path.exists(input_image_path):
        raise FileNotFoundError("Input image not found")

    if not os.path.exists(os.path.dirname(output_image_path)):
        raise FileNotFoundError("Output folder not found")

    img_metadata = get_image_metadata(input_image_path)

    input_img = read_tiff(input_image_path).astype('uint16')
    
    output_img = apply_gaussian_distance_map(input_img, sigma)

    
    array2raster(output_image_path, output_img, img_metadata, "float32")


if __name__ == "__main__":
    

    train_input_path = "2x_amazon_input_data/segmentation/train_set.tif"
    train_output_path = "2x_amazon_input_data/distance_map/train_distance_map.tif"
    check_folder(os.path.dirname(train_output_path))
    
    generate_distance_map(train_input_path, train_output_path)
    
    print_sucess("Distance map generated successfully")