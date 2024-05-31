import sys
import argparse
import ast
import errno
import functools
import gc
import logging
import multiprocessing
import os
import random
import threading
import warnings
from collections.abc import Iterable
from logging import CRITICAL, getLogger
from os.path import dirname, isdir, isfile, join
from typing import Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.distributed as dist
import yaml

ROOT_PATH = dirname(dirname(__file__))
sys.path.append(ROOT_PATH)

plt.set_loglevel(level = 'critical')

FALSY_STRINGS = {"off", "false", "0"}
TRUTHY_STRINGS = {"on", "true", "1"}



logger = getLogger("__main__")

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_in_thread(func):
    """
    Decorator to run a function in a separate thread.
    """
    @functools.wraps(func)  # Preserve original function metadata
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.start()
        return thread

    return wrapper



def run_in_process(func):
    """
    Decorator to run a function in a separate process.
    """
    @functools.wraps(func)  # Preserve original function metadata
    def wrapper(*args, **kwargs):
        process = multiprocessing.Process(target=func, args=args, kwargs=kwargs)
        process.start()
        return process
    return wrapper


def add_padding_new(img:np.ndarray, psize:int, overl:float, const:int = 0) -> Tuple:
    """Add padding to the image based on overlap and psize(patches size)

    Parameters
    ----------
    img : np.ndarray
        The image with n bands from remote sensing
        The fomat : (row, col, bands)
    psize : int
        The patch size to cut the segment image into boxes
    overl : float
        The overlap value that will have between the patches

    const : int, optional
        Contant to fill the paddin, by default 0

    Returns
    -------
    Tuple
    - pad_img : Image with padding
    - stride : The distance between each patch center
    - step_row
    - step_col
    - absolute_overlap = overl*psize
    - k1 : Num of patches in row axis
    - k2 : Num of patches in col axis
        
    """

    try:
        bands, row, col = img.shape

    except:
        bands = 0
        row, col = img.shape
        
    # Percent of overlap between consecutive patches.
    # The overlap will be multiple of 2
    overlap = int(round(psize * overl))
    # overlap -= overlap % 2
    stride = psize - overlap

    # Add Padding to the image to match with the patch size and the overlap
    # row += overlap//2
    # col += overlap//2
    step_row = (stride - row % stride) % stride
    step_col = (stride - col % stride) % stride
    
    if bands > 0:
        npad_img = (
            (0,0), # padding to the band/channel axis
            (overlap//2, step_row+overlap), # padding to the row axis
            (overlap//2, step_col+overlap) # padding to the col axis
        )

    else:        
        npad_img = ((overlap//2, step_row+overlap), (overlap//2, step_col+overlap))  
        
    gc.collect()

    # padd with symetric (espelhado)
    pad_img = np.pad(img, npad_img, mode='constant', constant_values = const)

    gc.collect()
    
    # Number of patches: k1xk2
    k1, k2 = (row + step_row)//stride, (col+step_col)//stride
    print('Number of patches: %d' %(k1 * k2))

    return pad_img, stride, step_row, step_col, overlap, k1, k2


def extract_patches_coord(img_gt:np.ndarray, 
                          psize:int, 
                          stride:int, 
                          step_row:int, 
                          step_col:int,
                          overl:float) -> np.ndarray:
    """
    The array of poisition of patches that will be used to evaluate the model

    Parameters
    ----------
    img_gt : np.ndarray
        ground truth segmentation
    psize : int
        The patch size to cut the segment image into boxes
    stride : int
        
    step_row : int
        
    step_col : int
        
    overl : float
        Overlap rate with the overlap that will have between patches
        
    Returns
    -------
    np.ndarray
        The coordinates of the center of each patch
    """
    
    # add padding to gt raster
    img_gt, stride, step_row, step_col, overlap,_, _ = add_padding_new(img_gt, psize, overl)
    
    overlap = int(round(psize * overl))
    
    row, col = img_gt.shape
    
    unique_class = np.unique(img_gt[img_gt!=0])
    
    if stride == 1:
    
        coords = np.where(img_gt!=0)
        coords = np.array(coords)
        coords = np.rollaxis(coords, 1, 0)

    else:
        # loop over x,y coordinates and extract patches
        coord_list = list()
    
        for m in range(psize//2, row - step_row - overlap, stride): 
            for n in range(psize//2, col - step_col - overlap, stride):
                
                coord = [m,n]
                
                class_patch = np.unique(img_gt[m - psize//2: m + psize//2, n - psize//2 : n+psize//2])
                
                if len(class_patch) > 1 or class_patch[0] in unique_class:
                    
                    coord_list.append(coord)                    


        coords = np.array(coord_list)
    
    return coords



def plot_figures(img_mult:np.ndarray, ref:np.ndarray, pred:np.ndarray, depth:np.ndarray, dist, model_dir:str, epoch:int, set_name:str):
    """Plot a comparison between the reference, prediction and depth images.

    Parameters
    ----------
    img_mult : np.ndarray
        Batch of images from remote sensing.
        The shape is (batch, bands, height, width).
    ref : np.ndarray
        Batch of references segmentation images.
        The shape is (batch, height, width).
    pred : np.ndarray
        Batch of predicted segmentation images.
        The shape is (batch, classes, height, width).
    depth : np.ndarray
        Batch of depth maps.
        The shape is (batch, height, width).
        This image has reference distance map used as reference.
    dist : _type_
        Batch of predicted depth maps.
        The shape is (batch, height, width).
        This image has predicted distance map generated by the model.
    model_dir : str
        model directory to save the images.
    epoch : int
        Current epoch to set in the image name.
    set_name : str
        First name to set in images name.
    """

    if type(pred).__module__ != np.__name__:
        pred = pred.data.cpu().numpy()
        dist = dist.data.cpu().numpy()

    if type(img_mult).__module__ != np.__name__:
        img_mult = img_mult.data.cpu().numpy()
        ref = ref.data.cpu().numpy()
        depth = depth.data.cpu().numpy()
    
    # Load the first 5 images in the batch
    batch = 5

    if img_mult.shape[1] > 3:
        img_mult = img_mult[:batch,[5,3,2],:,:]
    
    else:
        img_mult = img_mult[:batch, :, :, :]

    img_mult = np.moveaxis(img_mult, 1, 3)
    

    ref = ref[:batch,:,:]
    pred_cl = np.argmax(pred[:batch,:,:,:],axis=1)+1
    pred_prob = np.amax(pred[:batch,:,:,:],axis=1)

    depth = depth[:batch,:,:]
    dist = dist[:batch,0,:,:]


    nrows = 6
    ncols = batch
    imgs = [img_mult, ref, pred_cl, pred_prob, depth, dist]
    
    getLogger('matplotlib').setLevel(level=CRITICAL)
    warnings.filterwarnings("ignore")

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(batch, nrows))
    
    cont = 0
    cont_img = 0

    for ax in axes.flat:
        ax.set_axis_off()

        if cont==0:
            # Set image from remote sensing
            ax.imshow(imgs[cont][cont_img], interpolation='nearest')

        elif cont==1 or cont==2:
            # Set image from reference and predicted segmentation
            ax.imshow(imgs[cont][cont_img], cmap='Dark2', interpolation='nearest', vmin=0, vmax=8)

        elif cont==3:
            # Set probability generated by the model
            ax.imshow(imgs[cont][cont_img], cmap='OrRd', interpolation='nearest',vmin=0, vmax=1)

        else:
            # Set the reference depth map and the predicted depth map
            ax.imshow(imgs[cont][cont_img], interpolation='nearest',vmin=0, vmax=1)
        
        cont_img+=1

        if cont_img == ncols:
            cont+=1
            cont_img=0

    
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                        wspace=0.02, hspace=0.02)
    
    axes[0,0].set_title("Real Image")
    axes[1,0].set_title("Ground Truth Segmentation")
    axes[2,0].set_title("Predicted Segmentation")
    axes[3,0].set_title("Probability Map")
    axes[4,0].set_title("Ground Truth Depth")
    axes[5,0].set_title("Predicted Depth")

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, set_name + str(epoch) + '.png'), dpi = 300, format='png', bbox_inches = 'tight')
    plt.clf()
    plt.close()



def check_folder(folder_dir:str):
    """If the folder does not exist, create it.

    Parameters
    ----------
    folder_dir : str
        Folder directory.
    """
    if not os.path.exists(folder_dir):
        try:
            os.makedirs(folder_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    
    elif s.lower() in TRUTHY_STRINGS:
        return True
    
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def init_distributed_mode(args):
    """
    Initialize the following variables:
        - world_size
        - rank
    """

    args.is_slurm_job = "SLURM_JOB_ID" in os.environ

    if args.is_slurm_job:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ["SLURM_NNODES"]) * int(
            os.environ["SLURM_TASKS_PER_NODE"][0]
        )
    else:
        # multi-GPU job (local or multi-node) - jobs started with torch.distributed.launch
        # read environment variables
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])

    # prepare distributed
    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    # set cuda device
    args.gpu_to_work_on = args.rank % torch.cuda.device_count()
    torch.cuda.set_device(args.gpu_to_work_on)
    return



def restart_from_checkpoint(ckp_paths:str, run_variables:dict=None, **kwargs):
    """Load weights and hyperparameters from a checkpoint file in ckp_paths.
    If the checkpoint is not found, the model dont change run_variables and model state_dict

    Parameters
    ----------
    ckp_paths : str
        Path to the checkpoint file.
    run_variables : dict, optional
        Hypertparameters to load from the checkpoint file, by default None

    """
    DEVICE = get_device()

    # look for a checkpoint in exp repository
    if isinstance(ckp_paths, list):
        
        for ckp_path in ckp_paths:
            
            if os.path.isfile(ckp_path):
                break


    else:
        ckp_path = ckp_paths


    if not os.path.isfile(ckp_path):
        logger.info("No checkpoint found.")
        return


    logger.info("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file and load to GPU
    checkpoint = torch.load(ckp_path, map_location=DEVICE)


    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():

        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print(msg)


            except TypeError:
                msg = value.load_state_dict(checkpoint[key])

            logger.info("=> loaded {} from checkpoint '{}'".format(key, ckp_path))

            
        else:
            logger.warning("=> failed to load {} from checkpoint '{}'".format(key, ckp_path))


    # re load variable important for the run
    if run_variables is not None:

        for var_name in run_variables:

            if var_name in checkpoint:

                run_variables[var_name] = checkpoint[var_name]


def restore_checkpoint_variables(checkpoint_path:str)->dict:
    to_restore = {"epoch": 0, 
                  "best_val":(0.), 
                  "count_early": 0, 
                  "is_iter_finished":False}
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    for key in to_restore.keys():
        to_restore[key] = checkpoint[key]
    
    return to_restore.copy()

def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Used to log loss and acc during training
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




def filter_outliers(img, bins=10000, bth=0.01, uth=0.99, mask=[0], mask_indx=0):
    """
    Apply outlier filtering to image data using histogram-based thresholds.

    Parameters
    ----------
    img : numpy.ndarray
        The input image data with shape (bands, height, width).
    bins : int, optional
        The number of bins for histogram calculation. Default is 10000.
    bth : float, optional
        Lower threshold percentage for valid values. Default is 0.01.
    uth : float, optional
        Upper threshold percentage for valid values. Default is 0.99.
    mask : list or numpy.ndarray, optional
        Binary mask indicating regions of interest. Default is [0].
    mask_indx : int, optional
        Index of mask to use for outlier filtering. Default is 0.

    Returns
    -------
    numpy.ndarray
        Image data with outliers filtered within specified thresholds.

    Notes
    -----
    - NaN values are replaced with zeros before outlier filtering.
    - The function applies outlier filtering band-wise.
    - Use the mask parameter to specify regions for outlier filtering.
    - Outliers exceeding threshold values are clipped to the respective thresholds.
    """
    img[np.isnan(img)] = 0  # Filter NaN values.

    if len(mask) == 1:
        mask = np.zeros((img.shape[1:]), dtype='int64')

    for band in range(img.shape[0]):
        hist = np.histogram(img[:, :mask.shape[0], :mask.shape[1]][band, mask==mask_indx].ravel(), bins=bins)

        cum_hist = np.cumsum(hist[0]) / hist[0].sum()

        max_value = np.ceil(100 * hist[1][len(cum_hist[cum_hist < uth])]) / 100
        min_value = np.ceil(100 * hist[1][len(cum_hist[cum_hist < bth])]) / 100

        img[band, :,:,][img[band, :,:,] > max_value] = max_value
        img[band, :,:,][img[band, :,:,] < min_value] = min_value

    return img



def normalize(img:np.ndarray):
    """Normalize image inplace.
    Apply StandardScaler to image

    Parameters
    ----------
    img : np.ndarray
        Image array to normalize
        Shape: (BANDS, ROW, COL)
    """
    # iterate through channels and standardize
    for i in range(img.shape[0]):
        
        std = np.std(img[i], ddof=0)
        mean = np.mean(img[i])

        img[i] = (img[i]-mean)/std
    


def add_padding(img, psize, val = 0):
    '''
    DEPRECATED FUNCTION!
    Function to padding image
        input:
            patches_size: psize
            stride: stride
            img: image (row,col,bands)
    '''    

    try:
        bands, row, col = img.shape
    
    except:
        bands = 0
        row, col = img.shape
    
    if bands>0:
        npad_img = ((0,0), (psize//2+1, psize//2+1), (psize//2+1, psize//2+1))
        constant_values = val

    else:        
        npad_img = ((psize//2+1, psize//2+1), (psize//2+1, psize//2+1))
        constant_values = val

    pad_img = np.pad(img, npad_img, mode='constant', constant_values=constant_values)

    return pad_img


def oversamp(coords:np.ndarray, lab:np.ndarray, under = False) -> np.ndarray:
    """Sample the data to balance the classes

    Parameters
    ----------
    coords : np.ndarray
        The coordinates of the segmentation samples, where the values are non-zero
    lab : np.ndarray
        The segmentation labels, where the values are non-zero
    under : bool, optional
        Define if the sampling is under or over.

        True: under sampling based on the median of the classes,
        False: over sampling based on the max number of samples in a class, 

        by default False

    Returns
    -------
    np.ndarray
        The coordinates of the segmentation samples, where the values are non-zero.
        Theses coordinates are balanced based on the tree-type classes.
    """

    
    uniq, count = np.unique(lab, return_counts=True)

    if under:
        max_samp = int(np.median(count))

    else:
        max_samp = np.max(count)

    
    out_coords = np.zeros( (max_samp*len(uniq), 2), dtype='int64')
    

    for j in range(len(uniq)):

        lab_ind = np.where(lab == uniq[j]) 

        # If num of samples where the class is present is less than max_samp
        # then we need to oversample
        if len(lab_ind[0]) < max_samp:
            # Randomly select samples with replacement to match max_samp
            index = np.random.choice(lab_ind[0], max_samp, replace=True)
            # Add to output array
            out_coords[j*max_samp:(j+1)*max_samp,:] = coords[index]
            
        # If the number of samples where the class is present is the same as max_samp
        # then we don't need to oversample, just add the samples randomly to the output array
        else:
            # Randomly select samples without replacement
            index = np.random.choice(lab_ind[0], max_samp, replace=False)
            # Add to output array
            out_coords[j*max_samp:(j+1)*max_samp,:] = coords[index]

            
    return out_coords


class AttrDict(dict):
    """Dictionary with attributes
    The dictionary values can be accessed as attributes

    Examples
    --------
    >>> d = AttrDict({'a':1, 'b':2})
    >>> d.a
    1
    """
    def __init__(self, *args, **kwargs):

        super(AttrDict, self).__init__(*args, **kwargs)

        self.__dict__ = self



def fix_relative_paths(args:dict):
    """Add Root Path to relative paths

    Parameters
    ----------
    args : dict
        Args with file paths
    """
    for key in args.keys():

        if type(args[key]) == str:
            
            absolute_path = join(ROOT_PATH, args[key])

            if isfile(absolute_path) or isdir(absolute_path):

                args[key] = absolute_path


def print_sucess(message:str):
    """Print success message in green color
    
    Parameters:
    ----------
    message: str
        Message to print
    """
    print("\033[92m {}\033[00m" .format(message))




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


def get_pad_width(crop_size:int, coord:np.ndarray, image_shape:tuple):
    
    image_height = image_shape[-2]
    image_width = image_shape[-1]
    
    pad_width = [[0,0], [0,0]]
    
    start_row = coord[0] - crop_size // 2
    if start_row < 0:
        
        start_row = 0
        
        pad_width[0][0] = abs(coord[0] - crop_size // 2)
        


    end_row = coord[0] + crop_size // 2
    if end_row > image_height:
        
        end_row = image_height
        
        # if end row is after the image height, add pad to the crop
        pad_width[0][1] = abs(coord[0] + crop_size//2 - end_row)
    

    # check image width
    start_column = coord[1] - crop_size // 2

    if start_column < 0:
        
        start_column = 0
        
        # if start column is before the image index 0, add pad
        pad_width[1][0] = abs(coord[1] - crop_size // 2)

    
    end_column = coord[1] + crop_size // 2
    
    if end_column > image_width:
        
        end_column = image_width
        
        # if end column is after the image width, add pad
        pad_width[1][1] = abs(coord[1] + (crop_size // 2) - end_column)

    if len(image_shape) == 3:
        pad_width = [[0,0]] + pad_width

    return pad_width

def get_slice_window(image_shape, coord, crop_size):
    
    IMAGE_HEIGHT, IMAGE_WIDTH = image_shape[-2], image_shape[-1]
    
    start_row = np.maximum(0, coord[0] - crop_size // 2)
    end_row = np.minimum(IMAGE_HEIGHT, coord[0] + crop_size // 2)    
    

    start_column = np.maximum(0, coord[1] - crop_size // 2)    
    end_column = np.minimum(IMAGE_WIDTH, coord[1] + crop_size // 2)
    
    return start_row, end_row, start_column, end_column


def get_crop_image(image, image_shape, coord, crop_size):

    start_row, end_row, start_column, end_column = get_slice_window(image_shape, coord, crop_size)

    if len(image.shape) == 3:
        image_crop = np.array(
            image[:, 
                 start_row:end_row, 
                 start_column:end_column]
        )
    

    else:
        image_crop = np.array(
            image[start_row:end_row, 
                  start_column:end_column]
        )
    
    return image_crop



def oversample(coords: np.ndarray, 
               coords_label: np.ndarray, 
               method: Literal["max", "median"]) -> np.ndarray:
    """Oversamples data to balance classes based on segmentation samples.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates of the segmentation samples where non-zero values exist.
    coords_label : np.ndarray
        Segmentation labels corresponding to non-zero values. Each value relates to the pixel label at the `coords` position.
    method : Literal["max", "median"]
        The oversampling method to balance the tree-type classes.
        - "max" for maximizing class counts.
        - "median" for achieving class counts closer to the median.

    Returns
    -------
    np.ndarray
        Balanced segmentation sample coordinates with non-zero values.
        The coordinates are adjusted to balance tree-type classes based on the chosen oversampling method.
    """

    
    uniq, count = np.unique(coords_label, return_counts=True)
    
    if method == "max":
        upper_samp_limit = np.max(count)
    
    elif method == "median":
        upper_samp_limit = int(np.median(count))
    
    else:
        raise NotImplementedError(f"Method ``{method}`` were not implemented yet.")

    out_coords = np.zeros( (upper_samp_limit*len(uniq), 2), dtype='int64')
    

    for j in range(len(uniq)):

        lab_ind = np.where(coords_label == uniq[j]) 

        # If num of samples where the class is present is less than max_samp
        # then we need to oversample
        if len(lab_ind[0]) < upper_samp_limit:
            # Randomly select samples with replacement to match max_samp
            index = np.random.choice(lab_ind[0], upper_samp_limit, replace=True)
            # Add to output array
            out_coords[j*upper_samp_limit:(j+1)*upper_samp_limit,:] = coords[index]
            
        # If the number of samples where the class is present is the same as max_samp
        # then we don't need to oversample, just add the samples randomly to the output array
        else:
            # Randomly select samples without replacement
            index = np.random.choice(lab_ind[0], upper_samp_limit, replace=False)
            # Add to output array
            out_coords[j*upper_samp_limit:(j+1)*upper_samp_limit,:] = coords[index]
    
    # shuffle out_coords order
    np.random.shuffle(out_coords)

    return out_coords


def convert_to_minor_numeric_type(array:np.ndarray)->np.ndarray:
    
    # check if all values are integers
    if np.all(np.mod(array, 1) == 0):
        is_integer = True
    
    if not is_integer:
        return array
    
    min_value_array = np.min(array)
    max_value_array = np.max(array)
    
    if min_value_array >= 0:
        if max_value_array < 255:
            return array.astype("uint8")
        
        elif max_value_array < 65_000:
            return array.astype("uint16")

    else:
        return array.astype("int")



if __name__ == "__main__":
    pass
    # array2raster("/home/luiz/multi-task-fcn/test_repo/image_float.tif", 
    #              image[:, 0:1000, 0:1000],
    #              image_metadata = meta,
    #              dtype = "Float32")
    
    # array2raster("/home/luiz/multi-task-fcn/test_repo/image_none.tif", 
    #              image[:, 0:1000, 0:1000],
    #              image_metadata = meta,
    #              dtype = "Float32")