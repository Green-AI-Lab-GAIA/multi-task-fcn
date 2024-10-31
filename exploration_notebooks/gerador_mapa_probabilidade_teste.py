#!/usr/bin/env python
# coding: utf-8

import sys
import time

from joblib import Parallel, delayed



sys.path.append("..")
sys.path.append("../src")

import os
import subprocess
import warnings
from collections.abc import Iterable
from glob import glob
from os.path import dirname, join
from statistics import mode

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import torch.optim
from IPython.display import HTML, display
from joblib import Parallel, delayed
from matplotlib import rc
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FuncFormatter
from millify import millify
from PIL import Image
from pyproj import Transformer
from skimage.color import label2rgb
from skimage.measure import find_contours, label
from sklearn import metrics
from tqdm import tqdm

sys.path.append(dirname(dirname(__file__)))

from src.io_operations import array2raster
from evaluation import evaluate_iteration, evaluate_overlap, predict_network
from pred2raster import pred2raster
from sample_selection import get_components_stats
from src.io_operations import (fix_relative_paths, get_image_metadata,
                               get_image_pixel_scale, load_args, read_tiff,
                               read_yaml)
from utils import *
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
warnings.filterwarnings('ignore')
import gc
import logging
import os
import sys
import textwrap
from logging import Logger, getLogger
from os.path import exists, join
from typing import List, Literal, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from joblib import Parallel, delayed
from seaborn import color_palette
from torch.utils.data import Dataset
from tqdm import tqdm

from src.dataset import DatasetForInference, DatasetFromCoord
from src.deepvlab3 import DeepLabv3
from src.io_operations import (check_file_extension, convert_tiff_to_npy,
                               get_file_extesion, get_image_metadata,
                               get_npy_filepath_from_tiff, get_npy_shape,
                               load_image, load_norm, read_yaml)
from src.logger import create_logger
from src.model import build_model, load_weights
from src.utils import (add_padding_new, check_folder, extract_patches_coord,
                       get_crop_image, get_device, get_pad_width, normalize,
                       oversample)

logger = getLogger("__main__")

DEBUG_MODE = False

FIG_PATH = join("figures")
os.makedirs(FIG_PATH, exist_ok=True)



# repo with model outputs
VERSION_FOLDER = "13_amazon_data"
DATA_PATH = join(dirname(dirname(__file__)), VERSION_FOLDER)

# load args from the version
args = load_args(join(DATA_PATH, "args.yaml"))
# Repo with training data
INPUT_PATH = join(dirname(dirname(__file__)), "amazon_input_data")




logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Set the logging level to DEBUG to capture all messages

# Create a StreamHandler to output to sys.stdout
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)  # Set the handler level to DEBUG

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(stream_handler)



id_tree = pd.read_csv(join(INPUT_PATH,"id_trees.csv"), index_col="label_num")["tree_name"].sort_values()


ORTHOIMAGE_PATH = args.ortho_image
OVERLAPS = args.overlap


# In[39]:


TRAIN_GT = read_tiff(args.train_segmentation_path)
# COMP_TRAIN_GT = label(TRAIN_GT)


# In[40]:


TEST_GT = read_tiff(args.test_segmentation_path)
# Data from TRAIN in TEST
# TEST_GT = np.where(TRAIN_GT>0, 0, TEST_GT)
# COMP_TEST_GT = label(TEST_GT)


# # Modificando o Dataset for Inference para considerar uma máscara de segmentação

# In[41]:


class DatasetForInference(Dataset):
    def __init__(self,
                image_path:str,
                crop_size:int,
                overlap_rate:float,
                mask:np.ndarray,
                ) -> None: 
        
        super().__init__()
        
        self.image_path = image_path
        self.crop_size = crop_size
        self.overlap_rate = overlap_rate
        
        self.image = load_image(image_path)
        self.image_shape = self.image.shape
        self.mask = mask
        self.generate_coords()


    def generate_coords(self):
        
        coords_list = []
        
        height, width = self.image_shape[-2:]
        
        self.overlap_size = int(self.crop_size * self.overlap_rate)
        self.stride_size = self.crop_size - self.overlap_size

        for m in range(0, height-self.overlap_size, self.stride_size):
            for n in range(0, width-self.overlap_size, self.stride_size):
                                
                mask_crop = self.read_window([m, n], self.mask)
                
                if mask_crop.sum() > 0:
                    coords_list.append([m, n])
                
        
        self.coords = np.array(coords_list)


    def standardize_image_channels(self):
        
        self.image = self.image.astype("float32")

        normalize(self.image)
        

    def get_slice_window(self, coord:np.ndarray) -> Tuple[int, int, int, int]:
        "Based on overlap rate and crop size, get the slice to fit the image into original image"
        
        row_start = coord[0]
        row_end = coord[0] + self.crop_size
        
        if row_end > self.image_shape[1]:
            row_start = self.image_shape[1] - self.crop_size
            row_end = self.image_shape[1]
        
        column_start = coord[1]
        column_end = coord[1] + self.crop_size
        
        if column_end > self.image_shape[2]:
            column_start = self.image_shape[2] - self.crop_size
            column_end = self.image_shape[2]
        
        return row_start, row_end, column_start, column_end

    def read_window(self, coord:np.ndarray, image:np.ndarray) -> torch.Tensor:
        
        row_start, row_end, column_start, column_end = self.get_slice_window(coord)
        
        if len(image.shape) == 2:
            image_crop = image[row_start:row_end, column_start:column_end]
        
        else:
            image_crop = image[:, row_start:row_end, column_start:column_end]
        
        if (image_crop.shape[-1] != self.crop_size) or (image_crop.shape[-2] != self.crop_size):
            raise ValueError(f"There is a bug relationed to the shape {image_crop.shape}")
        
        return torch.tensor(image_crop)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the data from the dataset
        
        Parameters
        ----------
        idx : int
            The index of the data to be loaded
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The image crop and the slice to fit the image into original image
        
        """
        current_coord = self.coords[idx].copy()
        
        image = self.read_window(
            coord=current_coord,
            image=self.image,
        )
        
        row_start, row_end, column_start, column_end = self.get_slice_window(
            current_coord,
        )
        
        return image.float(), (row_start, row_end, column_start, column_end)
    
    def __len__(self):

        return len(self.coords)




def predict_network(ortho_image_shape:Tuple, 
                    dataloader:torch.utils.data.DataLoader, 
                    model:nn.Module,
                    num_classes:int,
                    activation_aux_layer:Literal["sigmoid", "relu", "gelu"]="sigmoid",
                    debug_mode=False):

    pred_prob = np.zeros(shape = (ortho_image_shape[1], ortho_image_shape[2], num_classes),dtype='float16')
    
    pred_depth = np.zeros(shape = (ortho_image_shape[1], ortho_image_shape[2]), dtype='float16')
    
    count_image = np.zeros(shape = (ortho_image_shape[1], ortho_image_shape[2]), dtype='uint8')
    
    DEVICE = get_device()
    model.eval()
    
    if activation_aux_layer == "sigmoid":
        activation_aux_layer = nn.Sigmoid()
    elif activation_aux_layer == "relu":
        activation_aux_layer = nn.ReLU()
    elif activation_aux_layer == "gelu":
        activation_aux_layer = nn.GELU()
    else:
        raise ValueError("Activation function not recognized")

    activation_aux_layer = activation_aux_layer.to(DEVICE)
    
    activation_main_layer = nn.Softmax(dim=1).to(DEVICE)
    
    with torch.no_grad(): 
        for i, (image, slices) in enumerate(tqdm(dataloader)):      
            # ============ multi-res forward passes ... ============
            # compute model loss and output
            input_batch = image.to(DEVICE, non_blocking=True, dtype = torch.float)
            
            out_pred = model(input_batch) 
               
            out_batch = activation_main_layer(out_pred['out'])
            out_batch = out_batch.permute(0,2,3,1)
            out_batch = out_batch.data.cpu().numpy()
            
            depth_out = activation_aux_layer(out_pred['aux']).data.cpu().numpy()
            
            batch_size, output_height, output_width, cl = out_batch.shape
            
            row_start, row_end, column_start, column_end = slices
            
            for b in range(batch_size):
                                
                pred_prob[
                    row_start[b]:row_end[b],
                    column_start[b]:column_end[b]
                ] += out_batch[b]

                pred_depth[
                    row_start[b]:row_end[b],
                    column_start[b]:column_end[b]
                ] += depth_out[b][0]
                
                count_image[
                    row_start[b]:row_end[b],
                    column_start[b]:column_end[b]
                ] += 1

            if debug_mode and i > 2:
                break
                
        # avoid zero division
        count_image[count_image == 0] = 1
        mask_division = count_image > 1
        
        logger.info("Dividing prob_map and depth_map by the number of times the pixel was predicted")
        pred_prob[mask_division] = pred_prob[mask_division]/count_image[mask_division][:, None]
        pred_depth[mask_division] = pred_depth[mask_division]/count_image[mask_division]
        
        del count_image
        return pred_prob, pred_depth


def evaluate_overlap(overlap:float,
                     current_iter_folder:str,
                     ortho_image_shape:tuple,
                     args
                     ):
    DEVICE = get_device()

    current_model_folder = join(current_iter_folder, args.model_dir)

    test_dataset = DatasetForInference(
        args.ortho_image,
        args.size_crops,
        overlap,
        mask=(TEST_GT > 0)
    )

    test_dataset.standardize_image_channels()

    test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size*2,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
    )

    logger.info("Building data done with {} patches loaded.".format(test_dataset.coords.shape[0]))
    
        
    model = DeepLabv3(
        in_channels = ortho_image_shape[0],
        num_classes = args.nb_class, 
        pretrained = args.is_pretrained, 
        dropout_rate = args.dropout_rate,
        batch_norm = args.batch_norm,
        downsampling_factor = args.downsampling_factor,
    )


    last_checkpoint = join(current_model_folder, args.checkpoint_file)
    model = load_weights(model, last_checkpoint)
    logger.info("Model loaded from {}".format(last_checkpoint))

    # Load model to GPU
    model = model.to(DEVICE)

    cudnn.benchmark = True

    prob_map, depth_map = predict_network(
        ortho_image_shape = ortho_image_shape,
        dataloader = test_loader,
        model = model,
        num_classes = args.nb_class,
        debug_mode=DEBUG_MODE
    )
    del depth_map, test_dataset, test_loader
    assert prob_map.max() > 0
    gc.collect()
    
    logger.info(f"Saving prediction outputs..")

    return prob_map




def evaluate_iteration(current_iter_folder:str, args:dict):

    ortho_image_metadata = get_image_metadata(args.ortho_image)
    
    ortho_image_shape = (ortho_image_metadata["count"], ortho_image_metadata["height"], ortho_image_metadata["width"])
    
    logger.info("============ Initialized Evaluation ============")
    path_to_save = join(current_iter_folder, "prob_map_test.tif")
    # change to npy file
    path_to_save = path_to_save.replace(".tif", ".npy")
    logger.info(f"Output will be saved to {path_to_save}")
    
    if exists(path_to_save):
        logger.info("Prediction already done. Skipping...")
        return
    
    for num, overlap in enumerate(args.overlap):
                
        logger.info(f"Overlap {overlap} is not done. Starting...")
        if num == 0:
            prob_map = evaluate_overlap(
                overlap, 
                current_iter_folder, 
                ortho_image_shape,
                args=args)
        
        else:
            prob_map_temp = evaluate_overlap(
                overlap, 
                current_iter_folder, 
                ortho_image_shape,
                args=args)
            
            mask_sum = (prob_map_temp > 0)
            prob_map[mask_sum] += prob_map_temp[mask_sum]
    
        gc.collect()
        torch.cuda.empty_cache()
    
    logger.info("Dividing prob_map by the number of overlaps")
    mask = (prob_map > 0)
    prob_map[mask] = (prob_map[mask] / len(args.overlap))*255
    prob_map = prob_map.astype("uint8")
    
    # assert prob_map.max() > 0
    
    logger.info("Summed all predictions")
    
    logger.info("Saving prediction outputs on disk")
    # array2raster(
    #     path_to_save=path_to_save,
    #     array=prob_map,
    #     image_metadata=ortho_image_metadata,
    #     dtype="uint8"
    # )
    # save a npy file
    np.save(path_to_save, prob_map)
    logger.info(f"Saved to {path_to_save}")



def get_iter_folders(output_folder):
    # load data from all iterations
    iter_folders = os.listdir(output_folder)

    iter_folders = [join(output_folder, folder) for folder in iter_folders if folder.startswith("iter_")]

    iter_folders.sort()
    iter_folders = iter_folders[1:-1].copy()
    
    return iter_folders.copy()



# In[45]:


iter_folders = get_iter_folders(DATA_PATH)


# In[14]:
important_folders = ["/home/luiz.luz/multi-task-fcn/13_amazon_data/iter_020",
                     "/home/luiz.luz/multi-task-fcn/13_amazon_data/iter_008",
                     "/home/luiz.luz/multi-task-fcn/13_amazon_data/iter_001"]

# set the important folders as priority in iter_folders
for folder in important_folders:
    iter_folders.remove(folder)
    iter_folders.insert(0, folder)



def evaluate_with_delay(iter_folder, args, num):
    # Wait for 10 minutes
    time.sleep(600*num)
    evaluate_iteration(iter_folder, args)
    

for iter_folder in reversed(iter_folders):
    evaluate_iteration(iter_folder, args)
    if DEBUG_MODE:
        break

# with ThreadPoolExecutor(max_workers=10) as executor:
#     for num, iter_folder in enumerate(iter_folders):
#         executor.submit(evaluate_with_delay, iter_folder, args, num)

      
      
# from concurrent.futures import ProcessPoolExecutor
# with ProcessPoolExecutor(max_workers=3) as executor:
#     for num, iter_folder in enumerate(iter_folders):
#         executor.submit(evaluate_with_delay, iter_folder, args, num)

# from concurrent.futures import ProcessPoolExecutor
# with ProcessPoolExecutor(max_workers=3) as executor:
#     for num, iter_folder in enumerate(iter_folders):
#         executor.submit(evaluate_with_delay, iter_folder, args, num)

print("All tasks submitted")



