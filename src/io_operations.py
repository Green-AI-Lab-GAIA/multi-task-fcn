import ast
import logging
import os
import sys
from os.path import dirname, join, exists, basename, splitext
from typing import Iterable, Tuple, Optional

import numpy as np
import pandas as pd
import rasterio
import torch
import yaml
from PIL import Image
from scipy.ndimage import binary_fill_holes
from skimage.morphology import convex_hull_image

ROOT_PATH = dirname(dirname(__file__))
sys.path.append(ROOT_PATH)

from src.utils import AttrDict, fix_relative_paths, get_crop_image, get_pad_width, normalize, check_folder

logger = logging.getLogger(__name__)


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
        except (ValueError, SyntaxError, TypeError):
            pass

    return yaml_attrdict


def load_args(yaml_path:str)->dict:
    """
    1. Load arguments saved on yaml file.
    2. Convert path inside the args to absolute path
    3. Normalize multi-region config (singular -> list)
    
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
    
    # Normalize multi-region config
    args = normalize_multi_region_args(args)
    
    return args


def normalize_multi_region_args(args: dict) -> dict:
    """Normalize args to always have list-based multi-region config.
    
    Converts singular config (ortho_image, train_segmentation_path, etc.)
    to plural/list format (ortho_images, train_segmentation_paths, etc.)
    for backwards compatibility.
    
    Parameters
    ----------
    args : dict
        Arguments dictionary
    
    Returns
    -------
    dict
        Normalized arguments with list-based paths
    """
    # Mapping from singular to plural keys
    path_mappings = {
        'ortho_image': 'ortho_images',
        'train_segmentation_path': 'train_segmentation_paths',
        'test_segmentation_path': 'test_segmentation_paths',
        'full_segmentation_path': 'full_segmentation_paths',
        'mask_path': 'mask_paths',
    }
    
    for singular_key, plural_key in path_mappings.items():
        # If plural key already exists and is a list, use it
        if plural_key in args and isinstance(args[plural_key], list):
            # Ensure singular key also exists for backwards compatibility
            if singular_key not in args:
                args[singular_key] = args[plural_key][0] if args[plural_key] else None
        # If only singular key exists, convert to list
        elif singular_key in args and args[singular_key] is not None:
            args[plural_key] = [args[singular_key]]
        # If neither exists, set empty list
        else:
            args[plural_key] = []
    
    # Add num_regions for convenience
    if 'ortho_images' in args:
        args['num_regions'] = len(args['ortho_images'])
    else:
        args['num_regions'] = 0
    
    return args


def get_region_paths(args: dict, region_idx: int) -> dict:
    """Get paths for a specific region.
    
    Parameters
    ----------
    args : dict
        Normalized arguments dictionary
    region_idx : int
        Index of the region (0-based)
    
    Returns
    -------
    dict
        Dictionary with paths for the specified region
    """
    return {
        'ortho_image': args['ortho_images'][region_idx],
        'train_segmentation_path': args['train_segmentation_paths'][region_idx],
        'test_segmentation_path': args['test_segmentation_paths'][region_idx],
        'full_segmentation_path': args['full_segmentation_paths'][region_idx] if args.get('full_segmentation_paths') else None,
        'mask_path': args['mask_paths'][region_idx] if args.get('mask_paths') else None,
    }


def generate_mask_from_orthoimage(
    ortho_image_path: str,
    output_path: str = None,
    make_convex: bool = True,
    fill_holes: bool = True,
    save_preview: bool = True,
    preview_max_size: int = 1024,
) -> np.ndarray:
    """
    Generate a mask from an orthoimage by identifying non-empty pixels.
    
    A pixel is considered empty if ALL channels are zero.
    The mask indicates valid regions (1 = valid, 0 = background/empty).
    
    Parameters
    ----------
    ortho_image_path : str
        Path to the orthoimage TIFF file
    output_path : str, optional
        Path to save the generated mask TIFF. If None, mask is not saved.
    make_convex : bool, optional
        If True, creates a convex hull of the valid region. Default True.
    fill_holes : bool, optional
        If True, fills holes in the mask. Default True.
    save_preview : bool, optional
        If True, saves a low-resolution PNG preview alongside the TIFF. Default True.
    preview_max_size : int, optional
        Maximum dimension (width or height) for the preview image. Default 1024.
        
    Returns
    -------
    np.ndarray
        Binary mask where 1 = valid region, 0 = background (uint8)
    """
    logger.info(f"Generating mask from orthoimage: {ortho_image_path}")
    
    # Read the orthoimage
    ortho = read_tiff(ortho_image_path)
    
    # Handle different shapes: (bands, height, width) or (height, width)
    if ortho.ndim == 3:
        # Multiband image: pixel is non-empty if ANY channel is non-zero
        mask = np.any(ortho != 0, axis=0)
        logger.info(f"Multiband image with shape {ortho.shape}")
    else:
        # Single band image
        mask = ortho != 0
        logger.info(f"Single band image with shape {ortho.shape}")
    
    # Count initial valid pixels
    initial_valid = np.sum(mask)
    total_pixels = mask.size
    logger.info(f"Initial valid pixels: {initial_valid:,} / {total_pixels:,} ({100*initial_valid/total_pixels:.2f}%)")
    
    # Fill holes in the mask
    if fill_holes:
        mask = binary_fill_holes(mask)
        logger.info(f"After fill_holes: {np.sum(mask):,} valid pixels")
    
    # Make the mask convex
    if make_convex:
        mask = convex_hull_image(mask)
        logger.info(f"After convex_hull: {np.sum(mask):,} valid pixels")
    
    # Convert to uint8 (0 and 1)
    mask = mask.astype(np.uint8)
    
    # Save if output path provided
    if output_path:
        # Get metadata from orthoimage
        metadata = get_image_metadata(ortho_image_path)
        
        # Ensure output directory exists
        output_dir = dirname(output_path)
        if output_dir:
            check_folder(output_dir)
        
        # Save TIFF
        array2raster(output_path, mask, metadata, dtype='uint8')
        logger.info(f"Mask TIFF saved to: {output_path}")
        
        # Save preview PNG
        if save_preview:
            _save_mask_preview(mask, output_path, preview_max_size)
    
    return mask


def _save_mask_preview(
    mask: np.ndarray, 
    tiff_path: str, 
    max_size: int = 1024
) -> str:
    """
    Save a low-resolution PNG preview of a mask.
    
    Parameters
    ----------
    mask : np.ndarray
        Binary mask array (0 and 1 values)
    tiff_path : str
        Path to the TIFF file (used to derive PNG path)
    max_size : int
        Maximum dimension for the preview
        
    Returns
    -------
    str
        Path to the saved PNG file
    """
    # Derive PNG path from TIFF path
    png_path = splitext(tiff_path)[0] + "_preview.png"
    
    # Convert mask to 0-255 range for visualization
    mask_vis = (mask * 255).astype(np.uint8)
    
    # Create PIL image
    img = Image.fromarray(mask_vis, mode='L')
    
    # Calculate resize dimensions maintaining aspect ratio
    width, height = img.size
    if width > height:
        if width > max_size:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_width, new_height = width, height
    else:
        if height > max_size:
            new_height = max_size
            new_width = int(width * max_size / height)
        else:
            new_width, new_height = width, height
    
    # Resize if needed
    if (new_width, new_height) != (width, height):
        img = img.resize((new_width, new_height), Image.Resampling.NEAREST)
    
    # Save PNG
    img.save(png_path, optimize=True)
    logger.info(f"Mask preview PNG saved to: {png_path} (size: {new_width}x{new_height})")
    
    return png_path


def get_or_generate_mask_path(
    args: dict, 
    region_idx: int, 
    data_path: str = None,
    make_convex: bool = True,
    fill_holes: bool = True,
) -> Optional[str]:
    """
    Get mask path for a region, generating it automatically if not provided.
    
    If a mask path is provided in args and the file exists, returns that path.
    Otherwise, generates a mask automatically from the orthoimage.
    
    Parameters
    ----------
    args : dict
        Arguments dictionary (should be normalized with normalize_multi_region_args)
    region_idx : int
        Index of the region (0-based)
    data_path : str, optional
        Base data path for saving generated masks. If None, uses args['data_path']
    make_convex : bool, optional
        If True, creates a convex hull of the valid region. Default True.
    fill_holes : bool, optional
        If True, fills holes in the mask. Default True.
        
    Returns
    -------
    str or None
        Path to the mask file, or None if generation fails
    """
    # Check if mask path is provided and exists
    if args.get('mask_paths') and len(args['mask_paths']) > region_idx:
        mask_path = args['mask_paths'][region_idx]
        if mask_path and exists(mask_path):
            logger.info(f"Using provided mask for region {region_idx}: {mask_path}")
            return mask_path
    
    # Need to generate mask - get orthoimage path
    if not args.get('ortho_images') or len(args['ortho_images']) <= region_idx:
        logger.warning(f"Cannot generate mask: no orthoimage path for region {region_idx}")
        return None
    
    ortho_path = args['ortho_images'][region_idx]
    if not exists(ortho_path):
        logger.warning(f"Cannot generate mask: orthoimage not found: {ortho_path}")
        return None
    
    # Determine output folder for generated masks
    if data_path is None:
        data_path = args.get('data_path', '.')
    
    masks_folder = join(data_path, "generated_masks")
    check_folder(masks_folder)
    
    # Generate mask filename based on orthoimage name
    ortho_name = splitext(basename(ortho_path))[0]
    generated_mask_path = join(masks_folder, f"{ortho_name}_mask.tif")
    
    # Generate mask if it doesn't exist
    if not exists(generated_mask_path):
        logger.info(f"Generating mask for region {region_idx} from orthoimage...")
        try:
            generate_mask_from_orthoimage(
                ortho_path, 
                output_path=generated_mask_path,
                make_convex=make_convex,
                fill_holes=fill_holes,
                save_preview=True,
            )
        except Exception as e:
            logger.error(f"Failed to generate mask for region {region_idx}: {e}")
            return None
    else:
        logger.info(f"Using existing generated mask for region {region_idx}: {generated_mask_path}")
    
    return generated_mask_path



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
