import gc
import os
from os.path import join, exists
from logging import Logger
from logging import getLogger
from typing import Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from tqdm import tqdm

from src.deepvlab3 import DeepLabv3
from src.logger import create_logger
from src.model import build_model, load_weights
from src.dataset import DatasetFromCoord, DatasetForInference
from src.utils import (add_padding_new, check_folder,
                       extract_patches_coord, get_device)

from src.io_operations import get_image_metadata, load_norm, read_yaml,  convert_tiff_to_npy, check_file_extension, get_npy_filepath_from_tiff


ROOT_PATH = os.path.dirname(__file__)
args = read_yaml(join(ROOT_PATH, "args.yaml"))

logger = getLogger("__main__")
        
def define_test_loader(ortho_image:str, size_crops:int, overlap_rate:float)->Tuple:
    """Define the PyTorch loader for evaluation.\\
    This loader is different from the trainning loader.\\
    Here, the loader gets patches from the entire image map.\\
    On the other hand, the training loader just loads patches with some segmentation

    Parameters
    ----------
    ortho_image : str
        Path to the ortho_image. The image from remote sensing
    size_crops : int
        - The size of each patch    
    overlap_rate : float
        - The overlap rate between each patch

    Returns
    -------
    Tuple
        - image
            The normalized image from remote sensing
        - coords
            The coordinate of the center of each patch
        - stride
            The size of each step between each patch center
        - step_row
        - step_col
        - overlap
            The real overlap in pixels
    """

    
    
    image = load_norm(ortho_image)

    lab = np.ones(image.shape[1:])
    lab[np.sum(image, axis=0) == (11*image.shape[0]) ] = 0
    
    image, stride, step_row, step_col, overlap, _, _ = add_padding_new(image, size_crops, overlap_rate)
    
    coords = extract_patches_coord(
        img_gt = lab, 
        psize = size_crops,
        stride = stride, 
        step_row = step_row,
        step_col = step_col,
        overl = overlap_rate
    )

    return image, coords, stride, overlap


def predict_network(ortho_image_shape:Tuple, 
                    dataloader:torch.utils.data.DataLoader, 
                    model:nn.Module,
                    num_classes:int):
    """
    It runs the inference of the entire image map.\\
    Get depth values and the probability of each class

    Parameters
    ----------
    ortho_image_shape : Tuple
        The shape of the orthoimage
    dataloader : torch.utils.data.DataLoader
        Torch dataloader with all the patches from the image to be evaluated
    model : nn.Module
        The model with the weight of the current iteration
    num_classes : int
        The number of classes in the segmentation

    Returns
    -------
    Tuple
        - pred_prob :  Probability of each class
        - The class with the highest probability value
        - pred_depth :  The depth_map generated by the model
    """
    
    pred_prob = np.zeros(shape = (ortho_image_shape[1], ortho_image_shape[2], num_classes),dtype='float16')
    
    pred_depth = np.zeros(shape = (ortho_image_shape[1], ortho_image_shape[2]), dtype='float16')
    
    count_image = np.zeros(shape = (ortho_image_shape[1], ortho_image_shape[2]), dtype='uint8')
    
    DEVICE = get_device()
    model.eval()
    
    soft = nn.Softmax(dim=1).to(DEVICE)
    sig = nn.Sigmoid().to(DEVICE)
    

    with torch.no_grad(): 
        for i, (image, slices) in enumerate(tqdm(dataloader)):      
            # ============ multi-res forward passes ... ============
            # compute model loss and output
            input_batch = image.to(DEVICE, non_blocking=True, dtype = torch.float)
            
            out_pred = model(input_batch) 
               
            out_batch = soft(out_pred['out'])
            out_batch = out_batch.permute(0,2,3,1)
            out_batch = out_batch.data.cpu().numpy()
            
            depth_out = sig(out_pred['aux']).data.cpu().numpy()
            
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

        # avoid zero division
        count_image[count_image == 0] = 1
        
        pred_prob = pred_prob/count_image[...,np.newaxis]
        pred_depth = pred_depth/count_image
        
        return pred_prob, np.argmax(pred_prob,axis=-1).astype("uint8"), pred_depth


def evaluate_overlap(prediction_path:float,
                     overlap:float,
                     current_iter_folder:str,
                     ortho_image_shape:tuple,
                     save_compressed:bool = True
):
    """This function runs an evaluation on the entire image.\\
    The image is divided into patches, that will be the inputs of the model.\\
    The overlap parameter sets overlap rate between each patch cut.

    Parameters
    ----------
    prediction_path : str
        Path to save predictions
    overlap : float
        Overlap rate between each patch
    current_iter_folder : str
        The path to the current iteration folder
    ortho_image_shape : tuple
        The shape the ortho image - Image from remote sensing
    """
    
    DEVICE = get_device()

    current_model_folder = join(current_iter_folder, args.model_dir)

    test_dataset = DatasetForInference(
        args.ortho_image,
        args.size_crops,
        overlap
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

    prob_map, pred_class, depth_map = predict_network(
        ortho_image_shape = ortho_image_shape,
        dataloader = test_loader,
        model = model,
        num_classes = args.nb_class
    )

    gc.collect()
    
    logger.info(f"Saving prediction outputs..")

    if save_compressed:
        np.savez_compressed(
            prediction_path,
            prob_map=prob_map,
            pred_class=pred_class,
            depth_map=depth_map
        )

    else:
        np.savez(
            prediction_path,
            prob_map=prob_map,
            pred_class=pred_class,
            depth_map=depth_map
        )
    
    logger.info(f"Predictions saved on {prediction_path}")
    gc.collect()




def evaluate_iteration(current_iter_folder:str, args:dict):
    """Evaluate the entire image to predict the segmentation map.
    Evaluate for many overlap values and save the results in the current_iter_folder/prediction folder.

    Parameters
    ----------
    current_iter_folder : str
        Path to the current iteration folder.
    args : dict
        Dictionary of arguments.
    """

    logger.info("============ Initialized Evaluation ============")

    ortho_image_metadata = get_image_metadata(args.ortho_image)
    
    ortho_image_shape = (ortho_image_metadata["count"], ortho_image_metadata["height"], ortho_image_metadata["width"])
    
    # check if raster_prediction is done
    raster_depth = join(current_iter_folder, 'raster_prediction', f'depth_{sum(args.overlap)}.TIF') 
    raster_class_pred = join(current_iter_folder, 'raster_prediction', f'join_class_{sum(args.overlap)}.TIF') 
    raster_prob = join(current_iter_folder, 'raster_prediction', f'join_prob_{sum(args.overlap)}.TIF') 

    if exists(raster_depth) and exists(raster_class_pred) and exists(raster_prob):
        return
    

    for overlap in args.overlap:
        
        prediction_path = join(current_iter_folder, f'prediction_{overlap}.npz')
        
        is_prediction_overlap_done = exists(prediction_path)

        if is_prediction_overlap_done:
        
            logger.info(f"Overlap {overlap} is already done. Skipping...")

            continue
        
        logger.info(f"Overlap {overlap} is not done. Starting...")
        evaluate_overlap(
            prediction_path,
            overlap, 
            current_iter_folder, 
            ortho_image_shape,
            save_compressed=False)
        
        logger.info(f"Overlap {overlap} done.")

        gc.collect()
        torch.cuda.empty_cache()


##############################
#### E V A L U A T I O N #####
##############################
if __name__ == "__main__":
    from src.io_operations import load_args, get_image_metadata
    import matplotlib.pyplot as plt
    
    prediction_path = r"0.0_test_data\iter_001\prediction.npz"

    args = load_args(r"0.0_test_data\args.yaml")
    ortho_image_metadata = get_image_metadata(args.ortho_image)
    
    ortho_image_shape = (ortho_image_metadata["count"], ortho_image_metadata["height"], ortho_image_metadata["width"])
    
    evaluate_overlap(
        prediction_path = prediction_path,
        overlap = 0.3,
        current_iter_folder = r"0.0_test_data\iter_001",
        ortho_image_shape = ortho_image_shape
    )


