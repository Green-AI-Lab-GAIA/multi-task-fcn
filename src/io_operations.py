import ast
import os
import sys
from os.path import dirname, join
from typing import Iterable, Tuple
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import rasterio
import torch
import yaml

ROOT_PATH = dirname(dirname(__file__))
sys.path.append(ROOT_PATH)

from src.utils import AttrDict, fix_relative_paths, get_crop_image, get_pad_width, normalize


class ParquetUpdater:
    def __init__(self, file_path):
        self.file_path = file_path

    def update(self, data):
        # Read the existing Parquet file
        try:
            existing_data = pd.read_parquet(self.file_path)
        except FileNotFoundError:
            # If the file doesn't exist, create a new DataFrame
            existing_data = pd.DataFrame(columns=data.keys())

        # Create a new DataFrame from the input data
        new_data = pd.DataFrame([data])

        # Concatenate the existing data with the new data
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)

        # Write the updated data back to the Parquet file
        updated_data.to_parquet(self.file_path, index=False, engine='pyarrow', compression='snappy', partition_cols=None)


def get_npy_filepath_from_tiff(tiff_file:str)->str:

    array_file_path = os.path.splitext(tiff_file)[0]

    array_file_path += ".npy"
    
    return array_file_path


def convert_tiff_to_npy(tiff_file:str, dtype:str=None):
    
    img_array = read_tiff(tiff_file)
    
    array_file_path = get_npy_filepath_from_tiff(tiff_file)
       

    if dtype == None:
        np.save(array_file_path, img_array)
    
    else:
        np.save(array_file_path, img_array.astype(dtype))



def read_yaml(yaml_path:str)->dict:
    """Get the yaml file and convert to dict

    Parameters
    ----------
    yaml_path : str
        Path to the yaml file

    Returns
    -------
    dict
        Dictionary with keys and values from the yaml file
    """
    with open(yaml_path, 'r') as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
            yaml_attrdict = AttrDict()
            yaml_attrdict.update(yaml_dict)
        except yaml.YAMLError as exc:
            print(exc)
            
        
    # for each value try to convert to float
    for key in yaml_attrdict.keys():
        try:
            yaml_attrdict[key] = ast.literal_eval(yaml_attrdict[key])
        except:
            pass

    return yaml_attrdict


def load_args(yaml_path:str)->dict:
    """
    1. Load arguments saved on yaml file.
    2. Convert path inside the args to absolute path
    
    Parameters
    ----------
    yaml_path : str
        Path to yaml with args
    
    Returns:
    ----------
    dict:
        Arguments as dict
    """
    
    args = read_yaml(yaml_path)
    
    fix_relative_paths(args)    
    
    return args



def save_yaml(data_dict:dict, yaml_path:str):
    
    data_to_save = data_dict.copy()

    # convert numpy metrics to python primitives
    for metric in data_to_save:
        
        if isinstance(data_to_save[metric], str):
            data_to_save[metric] = str(data_to_save[metric])

        elif isinstance(data_to_save[metric], Iterable):
            data_to_save[metric] = list(data_to_save[metric])
        
        elif isinstance(data_to_save[metric], float):
            data_to_save[metric] = float(data_to_save[metric])


    with open(yaml_path, 'w') as file:

            yaml.dump(data_to_save, file)




def load_norm(path, mask=[0], mask_indx = 0):
    """Read image from `path` divide all values by 255

    Parameters
    ----------
    path : str
        Path to load image
    mask : list, optional
        Deprecated, by default [0]
    mask_indx : int, optional
        Deprecated, by default 0

    Returns
    -------
    Image normalized
        Tensor image with the format [channels, row, cols]
    """
    image = read_tiff(path)

    if image.dtype != np.float32:
        image = np.float32(image)

    print("Image shape: ", image.shape, " Min value: ", image.min(), " Max value: ", image.max())
    if len(image.shape) < 3:
        image = np.expand_dims(image, 0)
    
    print("Before normalize, Min value: ", image.min(), " Max value: ", image.max())

    normalize(img = image)

    print("Normalize, Min value: ", image.min(), " Max value: ", image.max())

    return image



def read_tiff(tiff_file:str) -> np.ndarray:
    """Read tiff file and return a numpy array

    Parameters
    ----------
    tiff_file : str
        Path to the tiff file

    Returns
    -------
    np.ndarray
        Numpy array with the image

    Raises
    ------
    FileNotFoundError
        If the file is not found
    """
    
    # verify if file exist
    if not os.path.isfile(tiff_file):
        raise FileNotFoundError("File not found: {}".format(tiff_file))
    
    with rasterio.open(tiff_file, num_threads='all_cpus') as src:
        image_tensor = src.read()

    # if the band num is 1, reshape to (height, width)
    if image_tensor.shape[0] == 1:
        return image_tensor.squeeze()
    
    # else return in the reshape to (band, height, width)
    else:
        return image_tensor
    



def array2raster(path_to_save:str, array:np.ndarray, image_metadata:dict, dtype:str):
    """Save a NumPy array as a GeoTIFF file.

    Parameters
    ----------
    path_to_save : str
        The file path to save the array as a GeoTIFF file.
    array : np.ndarray
        The image array or tensor with the format `(band, height, width)` or `(height, width)`.
    image_metadata : dict
        Image metadata obtained from the `get_image_metadata` function.
    dtype : str
        Data type for the output GeoTIFF:
        - None: Use the same data type as specified in `image_metadata`.
        - 'byte': Use the same data type as in the input NumPy array.
        - Any other dtype compatible with the rasterio library.
    """
    
    # set data type to save.
    if dtype == None:
        RASTER_DTYPE = image_metadata['dtype']
    
    elif dtype.lower() == "byte": 
        RASTER_DTYPE = array.dtype

    else:
        RASTER_DTYPE = dtype.lower()


    # set number of band.
    if array.ndim == 2:
        BAND_NUM = 1
        HEIGHT = array.shape[0]
        WIDTH = array.shape[1]

    else:
        BAND_NUM = array.shape[0]
        HEIGHT = array.shape[1]
        WIDTH  = array.shape[2]


    with rasterio.open(
        fp = path_to_save,
        mode = "w",
        driver = image_metadata['driver'],
        height = HEIGHT,
        width = WIDTH,
        count = BAND_NUM,
        dtype = RASTER_DTYPE,
        crs = image_metadata['crs'],
        transform = image_metadata['transform'],
        compress="packbits",
        num_threads='all_cpus',
        BIGTIFF="IF_NEEDED"
    ) as writer:
        
        if BAND_NUM > 1:
            # Write each band
            for band in range(1, BAND_NUM + 1):
                writer.write(array[band - 1, :, :], band)
        
        elif BAND_NUM == 1:
            # write just on band
            writer.write(array, 1)
        

def get_image_metadata(tiff_file:str) -> dict:
    """Read a tiff file and get the meta relationed to this file

    Parameters
    ----------
    tiff_file : str
        Path to the tiff file

    Returns
    -------
    dict
        Dict with the following data:
            - driver
            - transform 
            - crs
            - dtype
            - width
            - height
            - count (bands)
    """
    with rasterio.open(tiff_file) as src:
        pass

    return src.meta


def get_image_shape(tiff_file:str) -> dict:
    
    img_metadata = get_image_metadata(tiff_file)
    
    return (img_metadata["count"], img_metadata["height"], img_metadata["width"])

def get_image_pixel_scale(tiff_file:str) -> Tuple[float, float]:
    """
    Returns
    -------
    Tuple[float, float]
        Pixel scale in the format (x, y)
    """
    with rasterio.open(tiff_file) as src:
        pixel_scale = src.res
    
    return pixel_scale


def check_file_extension(file_path:str, extension:str):

    if not file_path.endswith(extension):
        
        raise ValueError(f"The file {os.path.split(file_path)[0]} is invalid. The extension hopes for {extension} file")


def get_file_extesion(file_path:str):
        
        return os.path.splitext(file_path)[1]


def load_image(file_path:str):
    file_extension = get_file_extesion(file_path)

    if file_extension == ".npy":
        return np.load(file_path)

    elif file_extension in (".tif", ".tiff", ".TIF", ".TIFF"):
        return read_tiff(file_path)
    
    else:
        raise ValueError(f"Invalid file extension {file_extension}. The file extension must be .npy or .tif")


def get_npy_shape(npy_path:str):
    
    with open(npy_path, 'rb') as f:

        version = np.lib.format.read_magic(f)
        shape, fortran_order, dtype = np.lib.format._read_array_header(f, version)
    
    return shape



def get_npy_dtype(npy_path:str):

    with open(npy_path, 'rb') as f:

        version = np.lib.format.read_magic(f)
        shape, fortran_order, dtype = np.lib.format._read_array_header(f, version)
    
    return dtype



def load_npy_memmap(npy_path:str, mode:str = "r+"):
    
    npy_dtype = get_npy_dtype(npy_path)
    npy_shape = get_npy_shape(npy_path)
    
    return np.lib.format.open_memmap(
        npy_path,
        mode=mode,
        shape=npy_shape,
        dtype=npy_dtype
    )



def read_window_around_coord(coord:np.ndarray, crop_size:int, image_npy_path:str) -> torch.Tensor:
    
    lazy_image = load_npy_memmap(image_npy_path)
    
    image_shape = get_npy_shape(image_npy_path)
    
    image_crop = get_crop_image(lazy_image, image_shape, coord, crop_size)

    pad_width = get_pad_width(crop_size, coord, image_shape)

    # apply padding to image
    image_crop = np.pad(
        image_crop, 
        pad_width = pad_width,
        mode = "constant",
        constant_values = 0
    )
    

    if (image_crop.shape[-1] != crop_size) or (image_crop.shape[-2] != crop_size):
        raise ValueError(f"There is a bug relationed to the shape {image_crop.shape}")

    return torch.tensor(image_crop)


if __name__ == "__main__":
    
    print(
        get_file_extesion("numpy.npy")
    )

    print(
        get_file_extesion("image.tiff")
    )

    print(
        get_file_extesion("image.TIF")
    )
