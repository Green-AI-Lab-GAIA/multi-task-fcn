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

from src.logger import create_logger
from src.model import build_model, load_weights
from src.dataset import DatasetFromCoord
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
                    batch_size:int, 
                    coords:np.ndarray, 
                    pred_prob:np.ndarray,
                    pred_depth:np.ndarray,
                    stride:int, 
                    overlap:int):
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
    batch_size : int
        The batch size at each prediction
    coords : np.ndarray
        The array with the coordinates of the patch centers
    pred_prob : np.ndarray
        An empty tensor to be filled with the probability values generated by the model
    pred_depth : np.ndarray
        An empty matrice to be filled with the depth values generated by the model
    stride : int
        The step between each patch
        The model use this to fill the pred_prob and pred_depth
    overlap : int
        The overlap between each patch in pixels

    Returns
    -------
    Tuple
        - pred_prob :  Probability of each class
        - The class with the highest probability value
        - pred_depth :  The depth_map generated by the model
    """
    DEVICE = get_device()
    model.eval()
    
    soft = nn.Softmax(dim=1).to(DEVICE)
    sig = nn.Sigmoid().to(DEVICE)
    
    st = stride//2
    ovr = overlap//2
    
    j = 0
    with torch.no_grad(): 
        for i, (inputs, coords) in enumerate(tqdm(dataloader)):      
            # ============ multi-res forward passes ... ============
            # compute model loss and output
            input_batch = inputs.to(DEVICE, non_blocking=True, dtype = torch.float)
            
            out_pred = model(input_batch) 
               
            out_batch = soft(out_pred['out'])
            out_batch = out_batch.permute(0,2,3,1)
                
            out_batch = out_batch.data.cpu().numpy()
            
            depth_out = sig(out_pred['aux']).data.cpu().numpy()
            
            batch_size, output_height, output_width, cl = out_batch.shape

            for b in range(batch_size):
                
                # The slice from image to fill with the prediction
                row_start = coords[b][0].item() - st
                row_end = coords[b][0].item() + st + stride % 2
                # Condition to avoid surpass the image shape
                row_end = np.minimum(row_end, ortho_image_shape[-2])
                
                
                # the total image height
                height = row_end - row_start


                col_start = coords[b][1].item() - st
                col_end = coords[b][1].item() + st + stride % 2
                col_end = np.minimum(col_end, ortho_image_shape[-1])
                

                width = col_end - col_start
                
                
                prob_output = out_batch[b, 
                                        output_height//2 - st : (output_height//2 - st) + height, 
                                        output_width//2 - st : (output_width//2 - st) + width]
                
                

                depth_output =  depth_out[b,
                                          :,
                                          output_height//2 - st : (output_height//2 - st) + height, 
                                          output_width//2 - st : (output_width//2 - st) + width]
                
                
                bbox = ((row_start, row_end), (col_start,  col_end))

                
                pred_prob[
                    bbox[0][0]:bbox[0][1],
                    bbox[1][0]:bbox[1][1]
                ] = prob_output

                
                pred_depth[
                    bbox[0][0]:bbox[0][1],
                    bbox[1][0]:bbox[1][1]
                ] = depth_output
                
        

        return pred_prob, np.argmax(pred_prob,axis=-1).astype("uint8"), pred_depth


def evaluate_overlap(prediction_path:float,
                     overlap:float,
                     current_iter_folder:str,
                     current_model_folder:str, 
                     ortho_image_shape:tuple,
                     size_crops:int = args.size_crops, 
                     num_classes:int = args.nb_class,
                     ortho_image:str = args.ortho_image, 
                     batch_size:int = args.batch_size, 
                     workers:int = args.workers, 
                     checkpoint_file:str = args.checkpoint_file, 
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
    current_model_folder : str
        The path to the current model folder
    ortho_image_shape : tuple
        The shape the ortho image - Image from remote sensing
    logger : Logger
        The logger that tracks the model evaluation
    size_crops : int, optional
        The size of the patches, by default args.size_crops
    num_classes : int, optional
        Num of tree_types, by default args.nb_class
    ortho_image : str, optional
        The path to the image from remote sensing, by default args.ortho_image
    batch_size : int, optional
        The batch size of the stochastic trainnig, by default args.batch_size
    workers : int, optional
        Num of parallel workers, by default args.workers
    checkpoint_file : str, optional
        The filename of the checkpoint, by default args.checkpoint_file
    """
    
    DEVICE = get_device()

    
    test_segmentation_npy_path = get_npy_filepath_from_tiff(args.test_segmentation_path) 
    if not exists(test_segmentation_npy_path):
        convert_tiff_to_npy(args.test_segmentation_path)
    

    ortho_image_npy_path = get_npy_filepath_from_tiff(ortho_image)
    if not exists(ortho_image_npy_path):
        convert_tiff_to_npy(ortho_image)
    
    
    test_dataset = DatasetFromCoord(
        ortho_image_npy_path,
        dataset_type="test",
        distance_map_path=None,
        segmentation_path = test_segmentation_npy_path,
        overlap_rate=overlap,
        crop_size=size_crops,
    )

    test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
    )

    logger.info("Building data done with {} patches loaded.".format(test_dataset.coords.shape[0]))
    
    model = build_model(
        ortho_image_shape,
        args.nb_class,
        args.arch, 
        args.filters, 
        args.is_pretrained,
        psize = args.size_crops,
        dropout_rate = args.dropout_rate
    )


    last_checkpoint = join(current_model_folder, checkpoint_file)
    model = load_weights(model, last_checkpoint)
    logger.info("Model loaded from {}".format(last_checkpoint))

    # Load model to GPU
    model = model.to(DEVICE)

    cudnn.benchmark = True


    pred_prob = np.zeros(shape = (ortho_image_shape[1], ortho_image_shape[2], num_classes), dtype='float16')
    pred_depth = np.zeros(shape = (ortho_image_shape[1], ortho_image_shape[2]), dtype='float16')

    prob_map, pred_class, depth_map = predict_network(
        ortho_image_shape = ortho_image_shape,
        dataloader = test_loader,
        model = model,
        batch_size = batch_size,
        coords = test_dataset.coords,
        pred_prob = pred_prob,
        pred_depth = pred_depth,
        stride = test_dataset.stride_size,
        overlap = test_dataset.overlap_size,
    )

    gc.collect()
    logger.info(f"Saving prediction outputs..")
    np.savez_compressed(
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

    current_model_folder = join(current_iter_folder, args.model_dir)

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
            current_model_folder,
            ortho_image_shape)
        
        logger.info(f"Overlap {overlap} done.")

        gc.collect()
        torch.cuda.empty_cache()


##############################
#### E V A L U A T I O N #####
##############################
if __name__ == "__main__":
    ## arguments
    args = read_yaml("args.yaml")
    # external parameters
    current_iter_folder = join(args.data_path, "iter_001")
    current_iter = int(current_iter_folder.split("_")[-1])
    current_model_folder = join(current_iter_folder, args.model_dir)

    evaluate_iteration(current_iter_folder, args)

    print("ok")
    


