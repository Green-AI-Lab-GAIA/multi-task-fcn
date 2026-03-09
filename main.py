import argparse
import gc
import math
import os
import shutil
from multiprocessing import Process
from threading import Thread
from os.path import abspath, dirname, exists, isfile, join

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from skimage.measure import label
from tqdm import tqdm

import wandb
from evaluation import evaluate_iteration
from generate_distance_map import generate_distance_map
from generate_distance_map_vector import generate_distance_map_fast
from pred2raster import pred2raster
from sample_selection import get_components_stats, get_new_segmentation_sample
from sample_selection_vector import (
    get_new_segmentation_sample_vector,
    save_labels_vector,
    load_or_convert_labels,
    get_components_stats_vector,
)
from src.dataset import DatasetFromCoord, MultiRegionDatasetFromCoord
from src.deepvlab3 import DeepLabv3
from src.io_operations import (ParquetUpdater, array2raster,
                               get_image_metadata,
                               load_args,
                               read_tiff, save_yaml,
                               get_or_generate_mask_path)
from src.logger import create_logger
from src.metrics import (
    evaluate_component_metrics, 
    evaluate_f1_by_component, 
    evaluate_metrics,
    evaluate_miou_per_polygon,
    get_miou_polygon_data,
)
from src.model import eval, load_weights, save_checkpoint, train, build_model
from src.utils import (check_folder, fix_random_seeds, from_255_to_1,
                       get_device, print_sucess,
                       restart_from_checkpoint, restore_checkpoint_variables,
                       wrap_model_for_gpu, unwrap_model)
from src.vector_operations import (
    compute_reference_stats,
    geodataframe_to_raster,
    labels_to_geodataframe_with_stats,
    load_labels_from_geopackage,
    save_labels_as_geopackage,
    select_n_labels_by_class_vector,
    select_n_labels_by_feature_distance,
)
from visualization import generate_labels_view

# Flag to use vector operations (set to True for faster processing)
USE_VECTOR_OPERATIONS = True

gc.set_threshold(0)

plt.set_loglevel(level = 'info')



def delete_useless_files(current_iter_folder:str):
    
    folder_to_remove = join(current_iter_folder,"prediction")

    if exists(folder_to_remove):
        shutil.rmtree(folder_to_remove)
    
    else:
        pass




def is_iter_0_done(data_path: str, num_regions: int = 1) -> bool:
    """Verify if the distance map from the ground truth segmentation is done for all regions.

    Parameters
    ----------
    data_path : str
        Root data path
    num_regions : int
        Number of regions to check

    Returns
    -------
    bool
    """
    for region_idx in range(num_regions):
        if num_regions > 1:
            path_distance_map = join(data_path, "iter_000", f"region_{region_idx}", "distance_map")
        else:
            path_distance_map = join(data_path, "iter_000", "distance_map")
        
        is_test_map_done = exists(join(path_distance_map, "test_distance_map.tif"))
        is_train_map_done = exists(join(path_distance_map, "train_distance_map.tif"))
        
        if not (is_test_map_done and is_train_map_done):
            return False
    
    return True


def get_current_iter_folder(data_path, overlap, num_regions: int = 1):
    """Get the current iteration folder
    This function verify which iteration folder isn't finished yet and return the path to it.
    
    For multi-region: checks that ALL regions have completed predictions and distance maps.

    Parameters
    ----------
    data_path : str
        Path to the data folder 
    overlap : float
        Parameter used for create the output file name
    num_regions : int
        Number of regions (default 1 for backwards compatibility)

    Returns
    -------
    str
        The path to the current iteration folder
    """

    iter_0_path = join(data_path, "iter_000")

    folders = pd.Series(os.listdir(data_path))
    
    if folders.shape[0] == 0:
        return iter_0_path

    folders = folders[folders.str.contains("iter_")]

    iter_folders = folders.sort_values(ascending=False)

    for idx, iter_folder_name in enumerate(iter_folders):
        
        iter_path = join(data_path, iter_folder_name)
        
        # Check if all regions are done
        all_regions_done = True
        for region_idx in range(num_regions):
            if num_regions > 1:
                region_folder = join(iter_path, f"region_{region_idx}")
                prediction_path = join(region_folder, "raster_prediction")
                distance_map_path = join(region_folder, "distance_map")
            else:
                prediction_path = join(iter_path, "raster_prediction")
                distance_map_path = join(iter_path, "distance_map")
            
            is_folder_generated = exists(prediction_path)
            is_depth_done = isfile(join(prediction_path, f"depth_{np.sum(overlap)}.TIF"))
            is_pred_done = isfile(join(prediction_path, f"join_class_{np.sum(overlap)}.TIF"))
            is_prob_done = isfile(join(prediction_path, f"join_prob_{np.sum(overlap)}.TIF"))
            is_distance_map_done = isfile(join(distance_map_path, "selected_distance_map.tif"))
            
            if not (is_folder_generated and is_depth_done and is_pred_done and is_prob_done and is_distance_map_done):
                all_regions_done = False
                break

        if all_regions_done:
            next_iter = int(iter_folder_name.split("_")[-1]) + 1
            next_iter_path = join(data_path, f"iter_{next_iter:03d}")
            
            check_folder(next_iter_path)

            return next_iter_path
    
    if is_iter_0_done(data_path, num_regions):
        return join(data_path, f"iter_{1:03d}")


    # create iter_0 folder if not exists
    check_folder(iter_0_path)

    return iter_0_path
    

def get_last_segmentation_path(train_segmentation_path, current_iter_folder, region_idx: int = None, num_regions: int = 1):
    """Get the path to the last segmentation file.
    
    Now supports both TIFF and GeoPackage formats and multiple regions.
    Prioritizes GeoPackage if available.
    
    Parameters
    ----------
    train_segmentation_path : str
        Path to the initial train segmentation
    current_iter_folder : str
        Current iteration folder path
    region_idx : int, optional
        Region index for multi-region support
    num_regions : int
        Total number of regions
    
    Returns
    -------
    str
        Path to the segmentation file
    """
    current_iter = int(current_iter_folder.split("_")[-1])
    data_path = dirname(current_iter_folder)

    if current_iter == 1:
        image_path = train_segmentation_path

    else:
        # Build path based on region
        # For training dataset, we need TIFF format (raster) since load_image only supports .tif/.npy
        # GeoPackage is used for vector operations but TIFF is always saved alongside
        if num_regions > 1 and region_idx is not None:
            region_folder = f"region_{region_idx}"
            tiff_path = join(data_path, f"iter_{current_iter-1:03d}", region_folder, "new_labels", "selected_labels_set.tif")
        else:
            tiff_path = join(data_path, f"iter_{current_iter-1:03d}", "new_labels", "selected_labels_set.tif")
        
        image_path = tiff_path

    return image_path


def get_last_segmentation_paths_all_regions(args: dict, current_iter_folder: str) -> list:
    """Get segmentation paths for all regions.
    
    Parameters
    ----------
    args : dict
        Arguments dictionary with train_segmentation_paths
    current_iter_folder : str
        Current iteration folder
    
    Returns
    -------
    list
        List of paths for all regions
    """
    num_regions = getattr(args, 'num_regions', 1)
    paths = []
    for region_idx in range(num_regions):
        path = get_last_segmentation_path(
            args.train_segmentation_paths[region_idx],
            current_iter_folder,
            region_idx=region_idx,
            num_regions=num_regions
        )
        paths.append(path)
    return paths


def load_segmentation_from_path(image_path: str, reference_tiff: str = None) -> np.ndarray:
    """Load segmentation from either TIFF or GeoPackage format.
    
    Parameters
    ----------
    image_path : str
        Path to the segmentation file (.tif or .gpkg)
    reference_tiff : str, optional
        Reference TIFF for shape when loading GeoPackage
    
    Returns
    -------
    np.ndarray
        Segmentation array
    """
    if image_path.endswith('.gpkg'):
        import geopandas as gpd
        gdf = gpd.read_file(image_path)
        
        if reference_tiff is None:
            raise ValueError("reference_tiff required when loading from GeoPackage")
        
        meta = get_image_metadata(reference_tiff)
        shape = (meta['height'], meta['width'])
        transform = meta.get('transform')
        
        return geodataframe_to_raster(gdf, shape, transform)
    else:
        return read_tiff(image_path)


def read_last_segmentation(current_iter_folder:str, train_segmentation_path:str)-> np.ndarray:
    """Read the segmentation labels from the last iteration.
    If is the first iteration, the function reads the ground_truth_segmentation
    If is not, the function reads the output from the last iteration in the folder `new_labels/`

    Parameters
    ----------
    current_iter_folder : str
        The folder of the current iteration
    train_segmentation_path : str
        The path to the ground truth segmentation

    Returns
    -------
    np.ndarray
        A image array with the segmentation set
    """

    image_path = get_last_segmentation_path(train_segmentation_path, current_iter_folder)
    image = read_tiff(image_path)

    return image



def read_val_segmentation():
    return read_tiff(args.train_segmentation_path)

def get_last_distance_map_path(current_iter_folder: str, region_idx: int = None, num_regions: int = 1):
    """Get the path to the last distance map file.
    
    Parameters
    ----------
    current_iter_folder : str
        Current iteration folder path
    region_idx : int, optional
        Region index for multi-region support
    num_regions : int
        Total number of regions
    
    Returns
    -------
    str
        Path to the distance map file
    """
    current_iter = int(current_iter_folder.split("_")[-1])
    data_path = dirname(current_iter_folder)

    if current_iter == 1:
        distance_map_filename = "train_distance_map.tif"
    else:
        distance_map_filename = "selected_distance_map.tif"

    # Build path based on region
    if num_regions > 1 and region_idx is not None:
        region_folder = f"region_{region_idx}"
        image_path = join(data_path, f"iter_{current_iter-1:03d}", region_folder, "distance_map", distance_map_filename)
    else:
        image_path = join(data_path, f"iter_{current_iter-1:03d}", "distance_map", distance_map_filename)
    
    return image_path


def get_last_distance_map_paths_all_regions(args: dict, current_iter_folder: str) -> list:
    """Get distance map paths for all regions.
    
    Parameters
    ----------
    args : dict
        Arguments dictionary
    current_iter_folder : str
        Current iteration folder
    
    Returns
    -------
    list
        List of paths for all regions
    """
    num_regions = getattr(args, 'num_regions', 1)
    paths = []
    for region_idx in range(num_regions):
        path = get_last_distance_map_path(
            current_iter_folder,
            region_idx=region_idx,
            num_regions=num_regions
        )
        paths.append(path)
    return paths


def read_last_distance_map(current_iter_folder:str)->np.ndarray:
    """Read the last segmentation file with gaussian filter and distance map applied.
    If the current iter is 1, the distance map from the ground truth segmentation is loaded

    Parameters
    ----------
    current_iter_folder : str
        Current iteration folde

    Returns
    -------
    np.ndarray
        Image with the application of distance map.
    """
    
    image_path = get_last_distance_map_path(current_iter_folder)
        
    image = read_tiff(image_path)

    return image


def read_val_distance_map():
    
    IMAGE_PATH = join(args.data_path, f"iter_000", "distance_map", "train_distance_map.tif")

    return read_tiff(IMAGE_PATH)


def get_learning_rate_schedule(train_loader: torch.utils.data.DataLoader, 
                               base_lr:float,
                               final_lr:float, 
                               epochs:int, 
                               warmup_epochs:int, 
                               start_warmup:float,
                               schedule_type:str="cosine_warmup",
                               step_decay_rate:float=0.1,
                               step_decay_every:int=5)->np.ndarray:
    """Get the learning rate schedule.

    Supports two schedule types:
    - "cosine_warmup": linear warmup followed by cosine annealing to final_lr.
    - "step_decay": multiplies lr by step_decay_rate every step_decay_every epochs
      (inverse time decay as in Cué La Rosa et al., 2021).

    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        Model train loader
    base_lr : float
        base learning rate
    final_lr : float
        final learning rate (only used by cosine_warmup)
    epochs : int
        number of total epochs to run
    warmup_epochs : int
        number of warmup epochs (only used by cosine_warmup)
    start_warmup : float
        initial warmup learning rate (only used by cosine_warmup)
    schedule_type : str
        "cosine_warmup" or "step_decay"
    step_decay_rate : float
        multiplicative factor applied every step_decay_every epochs (only for step_decay)
    step_decay_every : int
        number of epochs between each decay step (only for step_decay)

    Returns
    -------
    np.array
        learning rate schedule (one value per training iteration)
    """
    steps_per_epoch = len(train_loader)
    total_iters = steps_per_epoch * epochs

    if schedule_type == "cosine_warmup":
        warmup_lr_schedule = np.linspace(start_warmup, base_lr, steps_per_epoch * warmup_epochs)

        iters = np.arange(steps_per_epoch * (epochs - warmup_epochs))
        cosine_lr_schedule = np.array([
            final_lr + 0.5 * (base_lr - final_lr) * (1 + math.cos(math.pi * t / (steps_per_epoch * (epochs - warmup_epochs))))
            for t in iters
        ])

        lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

    elif schedule_type == "step_decay":
        lr_schedule = np.empty(total_iters)
        current_lr = base_lr
        for epoch in range(epochs):
            if epoch > 0 and epoch % step_decay_every == 0:
                current_lr *= step_decay_rate
            start = epoch * steps_per_epoch
            end = start + steps_per_epoch
            lr_schedule[start:end] = current_lr

    else:
        raise ValueError(f"Unknown lr schedule type: '{schedule_type}'. "
                         f"Supported: 'cosine_warmup', 'step_decay'.")

    return lr_schedule



def train_epochs(last_checkpoint:str, 
                 start_epoch:str, 
                 num_epochs:int, 
                 best_val:float, 
                 train_loader:torch.utils.data.DataLoader, 
                 model:nn.Module, 
                 optimizer:torch.optim.Optimizer, 
                 lr_schedule:np.ndarray, 
                 count_early:int,
                 val_loader:torch.utils.data.DataLoader,
                 current_iter_folder:str,
                 lambda_weight:float,
                 patience:int=5,
                 early_stopping_threshold:float=0.001,
                 ):
    """Train the model with the specified epochs numbers

    Parameters
    ----------
    last_checkpoint : str
        last checkpoint file path to load
    start_epoch : str
        Epoch num to start from
    num_epochs : int
        Total number fo epochs to execute
    best_val : float
        Best value got in this iteration
    train_loader : torch.utils.data.DataLoader
        The train dataload
    model : nn.Module
        Pytorch model
    optimizer : torch.optim.optimizer
        Pytorch optimizer
    lr_schedule : np.ndarray
        Learning rate schedule to use at each iteration        
    count_early : int
        The counting how many epochs we didnt have improving in the loss
    val_loader : torch.utils.data.DataLoader
        Dataloader for evaluation
    patience : int, optional
        The limit of the count early variable, by default 5
    """
    current_iter = int(current_iter_folder.split("iter_")[-1])
    data_path = abspath(dirname(current_iter_folder))
    
    training_stats = ParquetUpdater(join(data_path, "training_stats.parquet"))

    # Create figures folder to save training figures every epochsaved.
    figures_path = join(dirname(last_checkpoint), 'figures')
    check_folder(figures_path)

    for epoch in tqdm(range(start_epoch, num_epochs)):

        
        if count_early == patience:
            logger.info("============ Early Stop at epoch %i ... ============" % epoch)
            break
        
        np.random.shuffle(train_loader.dataset.coords)
        # NOTE: val_loader.coords should NOT be shuffled - validation set must remain fixed
        # to ensure consistent evaluation across epochs and prevent overfitting

        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # train the network
        logger.info("Training the model...")
        epoch, scores_tr = train(train_loader=train_loader, 
                                 model=model, 
                                 optimizer=optimizer, 
                                 epoch=epoch, 
                                 lr_schedule=lr_schedule, 
                                 figures_path=figures_path, 
                                 lambda_weight=lambda_weight,
                                 gradient_accumulation_steps=getattr(args, 'gradient_accumulation_steps', 1),
                                 nb_class=args.nb_class)
        
        logger.info("Evaluating the model...")
        f1_avg, f1_by_class_avg = eval(val_loader, model, args.nb_class)
        
        ### Save training stats ####
        eval_data = {f"f1_class_{num+1}": f1_score for num, f1_score in enumerate(f1_by_class_avg)}
        eval_data["train_loss"] = scores_tr
        eval_data["epoch"] = epoch
        eval_data["iter"] = current_iter
        
        wandb.log({"train_loss": scores_tr,
                   "f1_avg": f1_avg,
                   "f1_by_class_avg": f1_by_class_avg
        })
        training_stats.update(eval_data)

        
        logger.info("scores_tr: {}".format(f1_avg))

        # Check if model improved based on early_stopping_threshold
        is_best = (f1_avg - best_val) > early_stopping_threshold

        # save checkpoints        
        if is_best: 
            logger.info("============ Saving best models at epoch %i ... ============" % epoch)
            
            best_val = f1_avg
            
            save_checkpoint(last_checkpoint, model, optimizer, epoch+1, best_val, count_early)
            
            count_early = 0

        else:
            count_early += 1

            
    print_sucess("Training done !")



def train_iteration(current_iter_folder: str, args: dict):
    """Train the model in the current iteration.
    Load the output and the model trained from the last iteration, and train the model again.
    
    Supports multiple regions: combines data from all regions into a single dataset.

    Parameters
    ----------
    current_iter_folder : str
        The current iteration folder
    args : dict
        The dict with the parameters for the model.
        The parameters are defined in the args.yaml file
    """
    DEVICE = get_device()
    num_regions = getattr(args, 'num_regions', 1)

    current_model_folder = join(current_iter_folder, args.model_dir)
    current_iter = int(current_iter_folder.split("_")[-1])

    current_iter_checkpoint = join(current_model_folder, args.checkpoint_file)
    
    if isfile(current_iter_checkpoint):
        last_checkpoint = current_iter_checkpoint
        loaded_from_last_iteration = False
    
    elif not isfile(current_iter_checkpoint):
        if current_iter > 1:
            last_checkpoint = join(args.data_path, f"iter_{current_iter-1:03d}", args.model_dir, args.checkpoint_file)
            loaded_from_last_iteration = True
            print_sucess("Loaded_from_last_checkpoint")
        elif current_iter == 1:
            last_checkpoint = None
            loaded_from_last_iteration = False    

    if last_checkpoint is None:
        pass
    else:
        to_restore = restore_checkpoint_variables(checkpoint_path=last_checkpoint)
        if to_restore["is_iter_finished"] and not loaded_from_last_iteration:
            return

    logger.info(f"============ Initialized Training with {num_regions} region(s) ============")
    
    # Get input_dimension from args, default to size_crops if not specified
    input_dimension = getattr(args, 'input_dimension', args.size_crops)
    logger.info(f"Crop size: {args.size_crops}, Input dimension to model: {input_dimension}")
    
    # Get multi-scale crop settings
    min_crop_size = getattr(args, 'min_crop_size', None)
    max_crop_size = getattr(args, 'max_crop_size', None)
    eval_crop_size = getattr(args, 'eval_crop_size', args.size_crops)
    random_interpolation = getattr(args, 'random_interpolation', True)
    
    if min_crop_size is not None and max_crop_size is not None:
        logger.info(f"Multi-scale training enabled: crop size range [{min_crop_size}, {max_crop_size}]")
        logger.info(f"Evaluation crop size: {eval_crop_size}")

    # Get paths for all regions
    segmentation_paths = get_last_segmentation_paths_all_regions(args, current_iter_folder)
    distance_map_paths = get_last_distance_map_paths_all_regions(args, current_iter_folder)
    
    # Build validation paths
    if args.validation_set == "train":
        val_segmentation_paths = segmentation_paths
        val_distance_map_paths = distance_map_paths
    elif args.validation_set == "test":
        val_segmentation_paths = args.test_segmentation_paths
        val_distance_map_paths = []
        for region_idx in range(num_regions):
            if num_regions > 1:
                path = join(args.data_path, "iter_000", f"region_{region_idx}", "distance_map", "test_distance_map.tif")
            else:
                path = join(args.data_path, "iter_000", "distance_map", "test_distance_map.tif")
            val_distance_map_paths.append(path)
    elif args.validation_set == "full":
        val_segmentation_paths = args.full_segmentation_paths
        val_distance_map_paths = []
        for region_idx in range(num_regions):
            if num_regions > 1:
                path = join(args.data_path, "iter_000", f"region_{region_idx}", "distance_map", "full_distance_map.tif")
            else:
                path = join(args.data_path, "iter_000", "distance_map", "full_distance_map.tif")
            val_distance_map_paths.append(path)
    else:
        raise ValueError("validation_set must be 'train', 'test' or 'full'")

    # Use MultiRegionDatasetFromCoord for multiple regions, or single region dataset
    if num_regions > 1:
        logger.info(f"Creating multi-region dataset with {num_regions} regions")
        train_dataset = MultiRegionDatasetFromCoord(
            image_paths=args.ortho_images,
            segmentation_paths=segmentation_paths,
            distance_map_paths=distance_map_paths,
            samples=args.samples,
            augment=args.augment,
            crop_size=args.size_crops,
            input_dimension=input_dimension,
            copy_paste_augmentation=args.copy_and_paste_augmentation,
            balance_regions=True,
            min_crop_size=min_crop_size,
            max_crop_size=max_crop_size,
            random_interpolation=random_interpolation
        )
        
        # Validation uses fixed crop size (eval_crop_size), no multi-scale
        # Use same number of samples as training
        val_dataset = MultiRegionDatasetFromCoord(
            image_paths=args.ortho_images,
            segmentation_paths=val_segmentation_paths,
            distance_map_paths=val_distance_map_paths,
            samples=args.samples,  # Same as training
            augment=False,
            crop_size=eval_crop_size,
            input_dimension=input_dimension,
            copy_paste_augmentation=False,
            balance_regions=True,
            min_crop_size=None,  # No multi-scale for validation
            max_crop_size=None,
            random_interpolation=False
        )
        
        # Log sample distribution
        train_counts = train_dataset.get_region_sample_counts()
        logger.info(f"Training samples per region: {train_counts}")
        logger.info(f"[DEBUG] Train dataset length: {len(train_dataset)}")
        logger.info(f"[DEBUG] Val dataset length: {len(val_dataset)}")
    else:
        # Single region - use original dataset
        train_dataset = DatasetFromCoord(
            image_path=args.ortho_images[0],
            segmentation_path=segmentation_paths[0],
            distance_map_path=distance_map_paths[0],
            samples=args.samples,
            augment=args.augment,
            crop_size=args.size_crops,
            input_dimension=input_dimension,
            copy_paste_augmentation=args.copy_and_paste_augmentation,
            min_crop_size=min_crop_size,
            max_crop_size=max_crop_size,
            random_interpolation=random_interpolation
        )
        
        # Validation uses fixed crop size (eval_crop_size), no multi-scale
        # Use same number of samples as training
        val_dataset = DatasetFromCoord(
            image_path=args.ortho_images[0],
            segmentation_path=val_segmentation_paths[0],
            distance_map_path=val_distance_map_paths[0],
            samples=args.samples,  # Same as training
            augment=False,
            crop_size=eval_crop_size,
            input_dimension=input_dimension,
            copy_paste_augmentation=False,
            min_crop_size=None,  # No multi-scale for validation
            max_crop_size=None,
            random_interpolation=False
        )
    
    # CRITICAL FIX: Lazy normalization - compute statistics only, normalize crops on-the-fly
    # This avoids keeping normalized full images in memory (saves ~75 GB)
    logger.info("[DEBUG] Computing normalization statistics (lazy normalization)...")
    
    logger.info("[DEBUG] Computing statistics for val_dataset...")
    val_dataset.standardize_image_channels()
    
    logger.info("[DEBUG] Computing statistics for train_dataset...")
    train_dataset.standardize_image_channels()
    
    # Free memory after computing statistics
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info(f"[DEBUG] Creating train_loader with batch_size={args.batch_size}, workers={args.workers}")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )
    logger.info(f"[DEBUG] train_loader created with {len(train_loader)} batches")

    logger.info(f"[DEBUG] Creating val_loader with batch_size={args.batch_size}, workers={args.workers}")
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        shuffle=False,  # Validation set should not be shuffled to ensure consistent evaluation
    )
    logger.info(f"[DEBUG] val_loader created with {len(val_loader)} batches")

    logger.info("Building data done with {} batches loaded.".format(len(train_loader)))

    # Get image metadata from first region (all should have same number of channels)
    orthoimage_meta = get_image_metadata(args.ortho_images[0])
    
    model = build_model(
        in_channels=orthoimage_meta["count"],
        num_classes=args.nb_class,
        arch=args.arch,
        dropout_rate=args.dropout_rate,
        batch_norm=args.batch_norm,
        pretrained=args.is_pretrained,
        psize=input_dimension,
    )

    logger.info("Building model done.")

    logger.info("[DEBUG] Creating optimizer...")
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.weight_decay
    )
    logger.info("[DEBUG] Optimizer created")

    schedule_type = getattr(args, 'lr_schedule_type', 'cosine_warmup')
    step_decay_rate = getattr(args, 'step_decay_rate', 0.1)
    step_decay_every = getattr(args, 'step_decay_every', 5)
    logger.info(f"[DEBUG] Creating learning rate schedule (type={schedule_type})...")
    lr_schedule = get_learning_rate_schedule(
        train_loader, 
        args.base_lr, 
        args.final_lr, 
        args.epochs, 
        args.warmup_epochs, 
        args.start_warmup,
        schedule_type=schedule_type,
        step_decay_rate=step_decay_rate,
        step_decay_every=step_decay_every,
    )
    logger.info("Building optimizer done.")
    
    if last_checkpoint is None:
        logger.info("[DEBUG] No checkpoint found, creating initial checkpoint...")
        save_checkpoint(current_iter_checkpoint, model, optimizer, 0, 0.0, 0, is_iter_finished=False)
        last_checkpoint = current_iter_checkpoint
        logger.info("[DEBUG] Initial checkpoint saved")

    logger.info(f"[DEBUG] Loading weights from {last_checkpoint}...")
    model = load_weights(model, last_checkpoint)
    logger.info("[DEBUG] Weights loaded")
    
    logger.info(f"[DEBUG] Moving model to {DEVICE}...")
    model = model.to(DEVICE)
    logger.info("[DEBUG] Model moved to device")
    
    to_restore = {"epoch": 0, "best_val": 0., "count_early": 0, "is_iter_finished": False}
    logger.info("[DEBUG] Restarting from checkpoint...")
    restart_from_checkpoint(
        last_checkpoint,
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
    )
    logger.info(f"[DEBUG] Checkpoint restored: epoch={to_restore['epoch']}, best_val={to_restore['best_val']}, is_iter_finished={to_restore['is_iter_finished']}")

    # Wrap with DataParallel after loading weights (multi_gpu from YAML)
    multi_gpu = getattr(args, 'multi_gpu', False)
    model = wrap_model_for_gpu(model, multi_gpu=multi_gpu)

    if loaded_from_last_iteration:
        to_restore["epoch"] = 0
        to_restore["best_val"] = 0.0
        to_restore["count_early"] = 0
        to_restore["is_iter_finished"] = False
        logger.info("[DEBUG] Reset to_restore for new iteration")

    current_checkpoint = join(current_model_folder, args.checkpoint_file)
    
    logger.info("[DEBUG] Setting cudnn.benchmark = True and running gc.collect()...")
    cudnn.benchmark = True
    gc.collect()
    logger.info("[DEBUG] Ready to start training")

    if not to_restore["is_iter_finished"]:
        logger.info("[DEBUG] Starting train_epochs...")
        train_epochs(current_checkpoint, 
                     to_restore["epoch"], 
                     args.epochs, 
                     to_restore["best_val"], 
                     train_loader, 
                     model, 
                     optimizer, 
                     lr_schedule, 
                     to_restore["count_early"],
                     val_loader=val_loader,
                     current_iter_folder=current_iter_folder,
                     lambda_weight=args.lambda_weight,
                     patience=getattr(args, 'patience', 5),
                     early_stopping_threshold=getattr(args, 'early_stopping_threshold', 0.001))
    gc.collect()

    model = load_weights(model, current_checkpoint)

    to_restore = {"epoch": 0, "count_early": 0, "is_iter_finished": False, "best_val": 0.}
    restart_from_checkpoint(
        current_checkpoint,
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
    )
    
    save_checkpoint(current_checkpoint, 
                    model, 
                    optimizer, 
                    to_restore["epoch"], 
                    to_restore["best_val"], 
                    to_restore["count_early"], 
                    is_iter_finished=True)

    with torch.no_grad():
        torch.cuda.empty_cache()
    
    gc.collect()


def load_old_labels_all_regions(current_iter_folder: str, args: dict) -> gpd.GeoDataFrame:
    """
    Load old labels from all regions and concatenate into a single GeoDataFrame.
    
    This is used to compute global reference statistics across all regions.
    
    Parameters
    ----------
    current_iter_folder : str
        Current iteration folder
    args : dict
        Arguments dictionary
    
    Returns
    -------
    gpd.GeoDataFrame
        Concatenated GeoDataFrame with labels from all regions
    """
    num_regions = getattr(args, 'num_regions', 1)
    current_iter = int(current_iter_folder.split("iter_")[-1])
    
    all_labels_gdfs = []
    
    for region_idx in range(num_regions):
        train_seg_path = args.train_segmentation_paths[region_idx]
        
        # Determine old labels path
        if current_iter == 1:
            OLD_ALL_LABELS_FILE = train_seg_path
        else:
            if num_regions > 1:
                prev_region_folder = join(args.data_path, f"iter_{current_iter-1:03d}", f"region_{region_idx}")
            else:
                prev_region_folder = join(args.data_path, f"iter_{current_iter-1:03d}")
            
            old_gpkg_all = join(prev_region_folder, "new_labels", 'all_labels_set.gpkg')
            old_tiff_all = join(prev_region_folder, "new_labels", 'all_labels_set.tif')
            
            OLD_ALL_LABELS_FILE = old_gpkg_all if exists(old_gpkg_all) else old_tiff_all
        
        # Load labels
        reference_tiff = train_seg_path
        
        if OLD_ALL_LABELS_FILE.endswith('.gpkg'):
            old_all_gdf = gpd.read_file(OLD_ALL_LABELS_FILE)
        else:
            old_all_labels = read_tiff(OLD_ALL_LABELS_FILE)
            meta = get_image_metadata(reference_tiff)
            transform = meta.get('transform')
            crs = meta.get('crs')
            old_all_gdf = labels_to_geodataframe_with_stats(old_all_labels, transform, crs)
        
        # Add region index
        old_all_gdf['region_idx'] = region_idx
        all_labels_gdfs.append(old_all_gdf)
    
    # Concatenate all regions
    if all_labels_gdfs:
        combined_gdf = pd.concat(all_labels_gdfs, ignore_index=True)
        return gpd.GeoDataFrame(combined_gdf, crs=all_labels_gdfs[0].crs if all_labels_gdfs[0].crs else None)
    
    return gpd.GeoDataFrame(columns=['geometry', 'tree_type', 'area', 'region_idx'])


def load_selected_labels_all_regions(current_iter_folder: str, args: dict) -> gpd.GeoDataFrame:
    """
    Load selected labels (used for training) from all regions and concatenate.
    
    This loads the samples that were actually used to train the current iteration's model.
    For iter_001, uses train_segmentation_paths. For later iterations, uses 
    selected_labels_set from the previous iteration.
    
    Parameters
    ----------
    current_iter_folder : str
        Current iteration folder
    args : dict
        Arguments dictionary
    
    Returns
    -------
    gpd.GeoDataFrame
        Concatenated GeoDataFrame with selected training labels from all regions
    """
    from src.vector_operations import labels_to_geodataframe_with_stats
    
    num_regions = getattr(args, 'num_regions', 1)
    current_iter = int(current_iter_folder.split("iter_")[-1])
    
    all_labels_gdfs = []
    
    for region_idx in range(num_regions):
        train_seg_path = args.train_segmentation_paths[region_idx]
        
        # Determine selected labels path
        if current_iter == 1:
            # For first iteration, use initial training data
            SELECTED_LABELS_FILE = train_seg_path
        else:
            # For later iterations, use selected_labels_set from previous iteration
            if num_regions > 1:
                prev_region_folder = join(args.data_path, f"iter_{current_iter-1:03d}", f"region_{region_idx}")
            else:
                prev_region_folder = join(args.data_path, f"iter_{current_iter-1:03d}")
            
            selected_gpkg = join(prev_region_folder, "new_labels", 'selected_labels_set.gpkg')
            selected_tiff = join(prev_region_folder, "new_labels", 'selected_labels_set.tif')
            
            SELECTED_LABELS_FILE = selected_gpkg if exists(selected_gpkg) else selected_tiff
        
        # Load labels
        reference_tiff = train_seg_path
        
        if SELECTED_LABELS_FILE.endswith('.gpkg'):
            selected_gdf = gpd.read_file(SELECTED_LABELS_FILE)
        else:
            selected_labels = read_tiff(SELECTED_LABELS_FILE)
            meta = get_image_metadata(reference_tiff)
            transform = meta.get('transform')
            crs = meta.get('crs')
            selected_gdf = labels_to_geodataframe_with_stats(selected_labels, transform, crs)
        
        # Handle 'label' column (from train_segmentation_paths) - rename to 'tree_type'
        if 'label' in selected_gdf.columns and 'tree_type' not in selected_gdf.columns:
            selected_gdf['tree_type'] = selected_gdf['label']
        
        # Add region index
        selected_gdf['region_idx'] = region_idx
        all_labels_gdfs.append(selected_gdf)
    
    # Concatenate all regions
    if all_labels_gdfs:
        combined_gdf = pd.concat(all_labels_gdfs, ignore_index=True)
        return gpd.GeoDataFrame(combined_gdf, crs=all_labels_gdfs[0].crs if all_labels_gdfs[0].crs else None)
    
    return gpd.GeoDataFrame(columns=['geometry', 'tree_type', 'area', 'region_idx'])


def generate_labels_for_next_iteration(current_iter_folder: str, args: dict):
    """
    Generate labels and distance map for the next iteration for all regions.
    
    Now uses vector operations for faster processing and saves as GeoPackage.
    Also saves TIFF for backwards compatibility with training dataset.
    Supports multiple regions.
    
    Uses global reference statistics computed from all regions and selects
    samples globally (5 per tree_type across all regions, not per region).
    """
    num_regions = getattr(args, 'num_regions', 1)
    current_iter = int(current_iter_folder.split("iter_")[-1])
    
    logger.info(f"============ Generating New Samples for {num_regions} region(s) ============")
    
    # Check if all regions are already done
    all_done = True
    for region_idx in range(num_regions):
        if num_regions > 1:
            region_folder = join(current_iter_folder, f"region_{region_idx}")
        else:
            region_folder = current_iter_folder
        
        output_folder = join(region_folder, "new_labels")
        ALL_LABELS_GPKG_PATH = join(output_folder, 'all_labels_set.gpkg')
        SELECTED_LABELS_GPKG_PATH = join(output_folder, 'selected_labels_set.gpkg')
        ALL_LABELS_TIFF_PATH = join(output_folder, 'all_labels_set.tif')
        SELECTED_LABELS_TIFF_PATH = join(output_folder, 'selected_labels_set.tif')
        
        gpkg_done = exists(ALL_LABELS_GPKG_PATH) and exists(SELECTED_LABELS_GPKG_PATH)
        tiff_done = exists(ALL_LABELS_TIFF_PATH) and exists(SELECTED_LABELS_TIFF_PATH)
        
        if not (gpkg_done or tiff_done):
            all_done = False
            break
    
    if all_done:
        logger.info("All regions already processed. Skipping...")
        return
    
    # For single region or non-vector operations, use the old per-region approach
    if num_regions == 1 or not USE_VECTOR_OPERATIONS:
        for region_idx in range(num_regions):
            generate_labels_for_region(current_iter_folder, args, region_idx)
        return
    
    # ========== PHASE 1: Load old labels from all regions and compute global stats ==========
    logger.info("Phase 1: Loading old labels from all regions...")
    all_regions_old_labels = load_old_labels_all_regions(current_iter_folder, args)
    
    logger.info("Computing global reference statistics...")
    global_ref_stats = compute_reference_stats(all_regions_old_labels)
    logger.info(f"Global reference stats computed for {len(global_ref_stats)} tree types")
    
    # ========== PHASE 1.5: Calculate global Otsu thresholds if using otsu_per_class method ==========
    global_otsu_thresholds = None
    filter_method = getattr(args, 'filter_method', 'fixed')
    if filter_method == "otsu_per_class":
        from sample_selection_vector import collect_global_otsu_samples, calculate_global_otsu_thresholds
        
        logger.info("Phase 1.5: Calculating global Otsu thresholds from all regions...")
        otsu_sample_size = getattr(args, 'otsu_sample_size', 10000)
        otsu_random_seed = getattr(args, 'otsu_random_seed', 42)
        sigma = getattr(args, 'sigma', 9)
        
        # Collect samples from all regions
        global_samples = collect_global_otsu_samples(
            current_iter_folder=current_iter_folder,
            args=args,
            num_regions=num_regions,
            otsu_sample_size=otsu_sample_size,
            otsu_random_seed=otsu_random_seed,
            sigma=sigma,
        )
        
        # Calculate global thresholds
        global_otsu_thresholds = calculate_global_otsu_thresholds(
            global_samples=global_samples,
            prob_thr=getattr(args, 'prob_thr', None),
            depth_thr=getattr(args, 'depth_thr', None),
        )
        
        logger.info(f"Global Otsu thresholds calculated for {len(global_otsu_thresholds)} class(es)")
        for class_id, threshold in global_otsu_thresholds.items():
            logger.info(f"  Class {class_id}: threshold = {threshold:.4f}")
    
    # ========== PHASE 2: Process each region and collect deltas ==========
    logger.info("Phase 2: Processing each region...")
    all_deltas = []
    all_unbalanced_deltas = []
    region_data = {}  # Store data needed for Phase 3
    
    for region_idx in range(num_regions):
        logger.info(f"=== Processing Region {region_idx} ===")
        
        delta_gdf, unbalanced_delta_gdf, data = process_region_for_global_selection(
            current_iter_folder, args, region_idx, global_ref_stats, global_otsu_thresholds
        )
        
        if delta_gdf is not None and not delta_gdf.empty:
            delta_gdf['region_idx'] = region_idx
            all_deltas.append(delta_gdf)
        
        if unbalanced_delta_gdf is not None and not unbalanced_delta_gdf.empty:
            unbalanced_delta_gdf['region_idx'] = region_idx
            all_unbalanced_deltas.append(unbalanced_delta_gdf)
        
        region_data[region_idx] = data
    
    # ========== PHASE 3: Global selection and save per region ==========
    logger.info("Phase 3: Global sample selection...")
    
    if all_deltas:
        global_delta = pd.concat(all_deltas, ignore_index=True)
        global_delta = gpd.GeoDataFrame(global_delta, crs=all_deltas[0].crs if all_deltas[0].crs else None)
        
        logger.info(f"Total delta components across all regions: {len(global_delta)}")
        
        # Load model for feature extraction
        logger.info("Loading model for feature-based selection...")
        current_model_folder = join(current_iter_folder, args.model_dir)
        checkpoint_path = join(current_model_folder, args.checkpoint_file)
        
        # Get image metadata to determine number of channels
        sample_ortho_metadata = get_image_metadata(args.ortho_images[0])
        in_channels = sample_ortho_metadata['count']
        
        # Build and load model
        model = build_model(
            in_channels=in_channels,
            num_classes=args.nb_class,
            arch=args.arch,
            pretrained=False,
            psize=getattr(args, 'input_dimension', args.size_crops),
            dropout_rate=args.dropout_rate,
            batch_norm=args.batch_norm,
        )
        model = load_weights(model, checkpoint_path)
        device = get_device()
        model = model.to(device)
        multi_gpu = getattr(args, 'multi_gpu', False)
        model = wrap_model_for_gpu(model, multi_gpu=multi_gpu)
        model.eval()
        
        # Load training samples (selected labels used to train current model)
        logger.info("Loading training samples for feature comparison...")
        train_gdf = load_selected_labels_all_regions(current_iter_folder, args)
        
        # Prepare ortho_info_list
        logger.info("Preparing orthoimage metadata...")
        ortho_info_list = []
        for region_idx, ortho_path in enumerate(args.ortho_images):
            with rasterio.open(ortho_path) as src:
                ortho_info_list.append({
                    'region_idx': region_idx,
                    'path': ortho_path,
                    'bounds': src.bounds,
                    'crs': src.crs,
                    'transform': src.transform,
                    'width': src.width,
                    'height': src.height,
                })
        
        # Select 5 samples per tree_type globally based on feature distance
        distance_method = getattr(args, 'distance_aggregation_method', 'mean')
        logger.info(f"Selecting samples based on feature distance (aggregation method: {distance_method})...")
        try:
            selected_delta = select_n_labels_by_feature_distance(
                gdf=global_delta,
                model=model,
                train_gdf=train_gdf,
                ortho_info_list=ortho_info_list,
                args=args,
                samples_by_class=5,
                crop_size=args.size_crops,
                input_dimension=getattr(args, 'input_dimension', args.size_crops),
                device=device,
                distance_aggregation_method=distance_method,
            )
            logger.info(f"Selected {len(selected_delta)} samples globally (5 per tree_type) based on feature distance")
        except Exception as e:
            logger.warning(f"Feature-based selection failed: {e}. Falling back to random selection.")
            selected_delta = select_n_labels_by_class_vector(global_delta, samples_by_class=5)
            logger.info(f"Selected {len(selected_delta)} samples globally (5 per tree_type) using random selection")
        
        # Clean up model
        del model
        torch.cuda.empty_cache()
        gc.collect()
    else:
        selected_delta = gpd.GeoDataFrame(columns=['geometry', 'tree_type', 'region_idx'])
    
    if all_unbalanced_deltas:
        global_unbalanced_delta = pd.concat(all_unbalanced_deltas, ignore_index=True)
        global_unbalanced_delta = gpd.GeoDataFrame(global_unbalanced_delta, crs=all_unbalanced_deltas[0].crs if all_unbalanced_deltas[0].crs else None)
    else:
        global_unbalanced_delta = gpd.GeoDataFrame(columns=['geometry', 'tree_type', 'region_idx'])
    
    # Save labels for each region
    for region_idx in range(num_regions):
        logger.info(f"=== Saving labels for Region {region_idx} ===")
        
        # Get selected samples for this region
        region_selected_delta = selected_delta[selected_delta['region_idx'] == region_idx].copy()
        region_unbalanced_delta = global_unbalanced_delta[global_unbalanced_delta['region_idx'] == region_idx].copy()
        
        save_region_labels_from_global_selection(
            current_iter_folder, args, region_idx, 
            region_selected_delta, region_unbalanced_delta, region_data[region_idx]
        )
    
    # Clean up
    del all_regions_old_labels, global_ref_stats, all_deltas, all_unbalanced_deltas
    gc.collect()


def process_region_for_global_selection(
    current_iter_folder: str, 
    args: dict, 
    region_idx: int,
    global_ref_stats: pd.DataFrame,
    global_otsu_thresholds: dict = None,
) -> tuple:
    """
    Process a single region for global sample selection.
    
    This function filters predictions and computes deltas but does NOT perform
    the final sample selection (which is done globally across all regions).
    
    Parameters
    ----------
    current_iter_folder : str
        Current iteration folder
    args : dict
        Arguments dictionary
    region_idx : int
        Index of the region to process
    global_ref_stats : pd.DataFrame
        Global reference statistics computed from all regions
    global_otsu_thresholds : dict, optional
        Pre-calculated global Otsu thresholds per class. If provided, these will be used
        instead of calculating thresholds per region.
    
    Returns
    -------
    tuple
        (delta_gdf, unbalanced_delta_gdf, region_data_dict)
    """
    from sample_selection_vector import filter_map_by_depth_prob_to_gdf
    from src.vector_operations import (
        filter_by_geometric_properties_vector,
        filter_by_mask_vector,
        filter_by_reference_stats,
        get_labels_delta_vector,
        get_label_intersection_vector,
        join_labels_vector,
    )
    
    num_regions = getattr(args, 'num_regions', 1)
    current_iter = int(current_iter_folder.split("iter_")[-1])
    
    # Determine region folder
    region_folder = join(current_iter_folder, f"region_{region_idx}")
    
    # Get paths for this region
    train_seg_path = args.train_segmentation_paths[region_idx]
    raster_pred_folder = join(region_folder, "raster_prediction")
    reference_tiff = train_seg_path
    
    # Load model predictions for this region
    NEW_PRED_FILE = join(raster_pred_folder, f'join_class_{np.sum(args.overlap)}.TIF')
    new_pred_map = read_tiff(NEW_PRED_FILE)
    
    NEW_PROB_FILE = join(raster_pred_folder, f'join_prob_{np.sum(args.overlap)}.TIF')
    new_prob_map = read_tiff(NEW_PROB_FILE)
    new_prob_map = from_255_to_1(new_prob_map)
    
    NEW_DEPTH_FILE = join(raster_pred_folder, f'depth_{np.sum(args.overlap)}.TIF')
    new_depth_map = read_tiff(NEW_DEPTH_FILE)
    new_depth_map = from_255_to_1(new_depth_map)
    
    # Determine old labels paths
    if current_iter == 1:
        OLD_SELECTED_LABELS_FILE = train_seg_path
        OLD_ALL_LABELS_FILE = train_seg_path
    else:
        prev_region_folder = join(args.data_path, f"iter_{current_iter-1:03d}", f"region_{region_idx}")
        
        old_gpkg_selected = join(prev_region_folder, "new_labels", 'selected_labels_set.gpkg')
        old_gpkg_all = join(prev_region_folder, "new_labels", 'all_labels_set.gpkg')
        old_tiff_selected = join(prev_region_folder, "new_labels", 'selected_labels_set.tif')
        old_tiff_all = join(prev_region_folder, "new_labels", 'all_labels_set.tif')
        
        OLD_SELECTED_LABELS_FILE = old_gpkg_selected if exists(old_gpkg_selected) else old_tiff_selected
        OLD_ALL_LABELS_FILE = old_gpkg_all if exists(old_gpkg_all) else old_tiff_all
    
    # Get metadata
    meta = get_image_metadata(reference_tiff)
    transform = meta.get('transform')
    crs = meta.get('crs')
    shape = (meta['height'], meta['width'])
    
    # Load old labels as GeoDataFrames
    if OLD_SELECTED_LABELS_FILE.endswith('.gpkg'):
        old_selected_gdf = gpd.read_file(OLD_SELECTED_LABELS_FILE)
    else:
        old_selected_labels = read_tiff(OLD_SELECTED_LABELS_FILE)
        old_selected_gdf = labels_to_geodataframe_with_stats(old_selected_labels, transform, crs)
    
    if OLD_ALL_LABELS_FILE.endswith('.gpkg'):
        old_all_gdf = gpd.read_file(OLD_ALL_LABELS_FILE)
    else:
        old_all_labels = read_tiff(OLD_ALL_LABELS_FILE)
        old_all_gdf = labels_to_geodataframe_with_stats(old_all_labels, transform, crs)
    
    ground_truth_segmentation = read_tiff(train_seg_path)
    ground_truth_gdf = labels_to_geodataframe_with_stats(ground_truth_segmentation, transform, crs)
    
    # Shift predictions to match ground truth scale
    new_pred_map = new_pred_map.copy()
    new_pred_map += 1
    
    # Get filter method parameters
    filter_method = getattr(args, 'filter_method', 'fixed')
    otsu_sample_size = getattr(args, 'otsu_sample_size', 10000)
    otsu_random_seed = getattr(args, 'otsu_random_seed', 42)
    
    # Filter and convert to GeoDataFrame
    if filter_method == "fixed":
        logger.info(f"Region {region_idx}: Filtering components with depth+prob >= {args.depth_thr + args.prob_thr}")
    else:
        if global_otsu_thresholds is not None:
            logger.info(f"Region {region_idx}: Filtering components using global Otsu thresholds per class (method: {filter_method})")
        else:
            logger.info(f"Region {region_idx}: Filtering components using Otsu thresholds per class (method: {filter_method})")
    
    new_pred_gdf = filter_map_by_depth_prob_to_gdf(
        new_pred_map, new_prob_map, new_depth_map,
        args.prob_thr, args.depth_thr, args.sigma, reference_tiff,
        filter_method=filter_method,
        otsu_sample_size=otsu_sample_size,
        otsu_random_seed=otsu_random_seed,
        global_otsu_thresholds=global_otsu_thresholds,
    )
    logger.info(f"Region {region_idx}: After threshold filter: {len(new_pred_gdf)} components")
    
    # Apply geometric filters using GLOBAL reference stats
    logger.info(f"Region {region_idx}: Filtering by global reference statistics")
    new_pred_gdf = filter_by_reference_stats(new_pred_gdf, global_ref_stats, args)
    logger.info(f"Region {region_idx}: After quality filter: {len(new_pred_gdf)} components")
    
    # Filter by mask (auto-generate if not provided)
    def _get_arg(key, default=None):
        if hasattr(args, 'get'):
            return args.get(key, default)
        return getattr(args, key, default)
    
    # Get or generate mask path for this region
    data_path = _get_arg('data_path', '.')
    mask_path = get_or_generate_mask_path(args, region_idx, data_path)
    
    if mask_path and exists(mask_path):
        logger.info(f"Region {region_idx}: Filtering by mask: {mask_path}")
        new_pred_gdf = filter_by_mask_vector(new_pred_gdf, mask_path)
        logger.info(f"Region {region_idx}: After mask filter: {len(new_pred_gdf)} components")
    
    # Area filter
    lower_limit_area = _get_arg('lower_limit_area')
    upper_limit_area = _get_arg('upper_limit_area')
    if lower_limit_area is not None and upper_limit_area is not None:
        logger.info(f"Region {region_idx}: Filtering by area limits: {lower_limit_area} m² - {upper_limit_area} m²")
        new_pred_gdf = filter_by_geometric_properties_vector(
            new_pred_gdf,
            min_area=float(lower_limit_area),
            max_area=float(upper_limit_area),
            area_in_meters=True,
        )
        logger.info(f"Region {region_idx}: After area filter: {len(new_pred_gdf)} components")
    
    # Join all labels with new predictions
    logger.info(f"Region {region_idx}: Joining old and new components")
    new_labels_gdf = join_labels_vector(new_pred_gdf, old_all_gdf, overlap_limit=0.05)
    
    # Get delta (new components not in old selected)
    logger.info(f"Region {region_idx}: Getting new components (delta)")
    delta_gdf = get_labels_delta_vector(old_selected_gdf, new_labels_gdf, overlap_threshold=0.10)
    unbalanced_delta_gdf = delta_gdf.copy()
    
    logger.info(f"Region {region_idx}: Delta components: {len(delta_gdf)}")
    
    # Get intersection (updated shapes for old components)
    intersection_gdf = get_label_intersection_vector(old_selected_gdf, new_labels_gdf, overlap_threshold=0.10)
    
    # Update old selected with new shapes
    old_selected_updated_gdf = join_labels_vector(intersection_gdf, old_selected_gdf, overlap_limit=0.10)
    
    # Store data needed for Phase 3
    region_data = {
        'ground_truth_gdf': ground_truth_gdf,
        'old_selected_updated_gdf': old_selected_updated_gdf,
        'reference_tiff': reference_tiff,
        'shape': shape,
        'transform': transform,
        'crs': crs,
        'OLD_SELECTED_LABELS_FILE': OLD_SELECTED_LABELS_FILE,
    }
    
    # Clean up
    del new_pred_map, new_prob_map, new_depth_map, ground_truth_segmentation
    gc.collect()
    
    return delta_gdf, unbalanced_delta_gdf, region_data


def save_region_labels_from_global_selection(
    current_iter_folder: str,
    args: dict,
    region_idx: int,
    region_selected_delta: 'gpd.GeoDataFrame',
    region_unbalanced_delta: 'gpd.GeoDataFrame',
    region_data: dict,
):
    """
    Save labels for a region after global sample selection.
    
    Parameters
    ----------
    current_iter_folder : str
        Current iteration folder
    args : dict
        Arguments dictionary
    region_idx : int
        Index of the region
    region_selected_delta : gpd.GeoDataFrame
        Selected delta samples for this region (from global selection)
    region_unbalanced_delta : gpd.GeoDataFrame
        All delta samples for this region (unbalanced)
    region_data : dict
        Dictionary with region-specific data from Phase 2
    """
    from src.vector_operations import join_labels_vector
    
    # Unpack region data
    ground_truth_gdf = region_data['ground_truth_gdf']
    old_selected_updated_gdf = region_data['old_selected_updated_gdf']
    reference_tiff = region_data['reference_tiff']
    shape = region_data['shape']
    transform = region_data['transform']
    OLD_SELECTED_LABELS_FILE = region_data['OLD_SELECTED_LABELS_FILE']
    
    # Determine region folder and output paths
    region_folder = join(current_iter_folder, f"region_{region_idx}")
    output_folder = join(region_folder, "new_labels")
    
    ALL_LABELS_GPKG_PATH = join(output_folder, 'all_labels_set.gpkg')
    SELECTED_LABELS_GPKG_PATH = join(output_folder, 'selected_labels_set.gpkg')
    ALL_LABELS_TIFF_PATH = join(output_folder, 'all_labels_set.tif')
    SELECTED_LABELS_TIFF_PATH = join(output_folder, 'selected_labels_set.tif')
    
    # Create selected labels set
    logger.info(f"Region {region_idx}: Creating selected labels set with {len(region_selected_delta)} new samples")
    selected_labels_gdf = join_labels_vector(region_selected_delta, old_selected_updated_gdf, overlap_limit=0.10)
    selected_labels_gdf = join_labels_vector(ground_truth_gdf, selected_labels_gdf, overlap_limit=0.01)
    
    # Create all labels set
    logger.info(f"Region {region_idx}: Creating all labels set")
    all_labels_gdf = join_labels_vector(region_unbalanced_delta, old_selected_updated_gdf, overlap_limit=0.10)
    all_labels_gdf = join_labels_vector(ground_truth_gdf, all_labels_gdf, overlap_limit=0.01)
    
    # Apply mask filter to final sets to ensure no component touches area outside study area
    def _get_arg(key, default=None):
        if hasattr(args, 'get'):
            return args.get(key, default)
        return getattr(args, key, default)
    
    data_path = _get_arg('data_path', '.')
    mask_path = get_or_generate_mask_path(args, region_idx, data_path)
    
    if mask_path and exists(mask_path):
        logger.info(f"Region {region_idx}: Filtering final sets by mask to ensure all components are within study area")
        from src.vector_operations import filter_by_mask_vector
        selected_labels_gdf = filter_by_mask_vector(selected_labels_gdf, mask_path, max_outside_ratio=0.0)
        all_labels_gdf = filter_by_mask_vector(all_labels_gdf, mask_path, max_outside_ratio=0.0)
        logger.info(f"Region {region_idx}: After final mask filter: {len(all_labels_gdf)} all labels, {len(selected_labels_gdf)} selected labels")
    
    logger.info(f"Region {region_idx}: Final: {len(all_labels_gdf)} all labels, {len(selected_labels_gdf)} selected labels")
    
    # Convert to raster
    all_labels_set = geodataframe_to_raster(all_labels_gdf, shape, transform)
    selected_labels_set = geodataframe_to_raster(selected_labels_gdf, shape, transform)
    
    # Get metadata for saving
    if OLD_SELECTED_LABELS_FILE.endswith('.gpkg'):
        image_metadata = get_image_metadata(reference_tiff)
    else:
        image_metadata = get_image_metadata(OLD_SELECTED_LABELS_FILE)
    
    check_folder(output_folder)
    
    # Save as GeoPackage
    logger.info(f"Region {region_idx}: Saving labels as GeoPackage...")
    save_labels_as_geopackage(all_labels_set, ALL_LABELS_GPKG_PATH, reference_tiff)
    save_labels_as_geopackage(selected_labels_set, SELECTED_LABELS_GPKG_PATH, reference_tiff)
    
    # Save as TIFF
    logger.info(f"Region {region_idx}: Saving labels as TIFF (for training)...")
    array2raster(ALL_LABELS_TIFF_PATH, all_labels_set, image_metadata, "Byte")
    array2raster(SELECTED_LABELS_TIFF_PATH, selected_labels_set, image_metadata, "Byte")
    
    logger.info(f"Region {region_idx}: Labels saved to {output_folder}")
    
    # Clean up
    del all_labels_set, selected_labels_set, all_labels_gdf, selected_labels_gdf
    gc.collect()


def generate_labels_for_region(current_iter_folder: str, args: dict, region_idx: int):
    """
    Generate labels for a single region.
    
    Parameters
    ----------
    current_iter_folder : str
        Current iteration folder
    args : dict
        Arguments dictionary
    region_idx : int
        Index of the region to process
    """
    num_regions = getattr(args, 'num_regions', 1)
    current_iter = int(current_iter_folder.split("iter_")[-1])
    
    # Determine region folder
    if num_regions > 1:
        region_folder = join(current_iter_folder, f"region_{region_idx}")
        logger.info(f"=== Generating labels for Region {region_idx} ===")
    else:
        region_folder = current_iter_folder
    
    # Output paths
    output_folder = join(region_folder, "new_labels")
    ALL_LABELS_GPKG_PATH = join(output_folder, 'all_labels_set.gpkg')
    SELECTED_LABELS_GPKG_PATH = join(output_folder, 'selected_labels_set.gpkg')
    ALL_LABELS_TIFF_PATH = join(output_folder, 'all_labels_set.tif')
    SELECTED_LABELS_TIFF_PATH = join(output_folder, 'selected_labels_set.tif')

    # Check if already done
    gpkg_done = exists(ALL_LABELS_GPKG_PATH) and exists(SELECTED_LABELS_GPKG_PATH)
    tiff_done = exists(ALL_LABELS_TIFF_PATH) and exists(SELECTED_LABELS_TIFF_PATH)
    
    if gpkg_done or tiff_done:
        logger.info(f"Region {region_idx} labels already exist. Skipping...")
        return
    
    # Get paths for this region
    train_seg_path = args.train_segmentation_paths[region_idx]
    raster_pred_folder = join(region_folder, "raster_prediction")

    # Load model predictions for this region
    NEW_PRED_FILE = join(raster_pred_folder, f'join_class_{np.sum(args.overlap)}.TIF')
    new_pred_map = read_tiff(NEW_PRED_FILE)

    NEW_PROB_FILE = join(raster_pred_folder, f'join_prob_{np.sum(args.overlap)}.TIF')
    new_prob_map = read_tiff(NEW_PROB_FILE)
    new_prob_map = from_255_to_1(new_prob_map)

    NEW_DEPTH_FILE = join(raster_pred_folder, f'depth_{np.sum(args.overlap)}.TIF')
    new_depth_map = read_tiff(NEW_DEPTH_FILE)
    new_depth_map = from_255_to_1(new_depth_map)

    # Determine old labels paths
    if current_iter == 1:
        OLD_SELECTED_LABELS_FILE = train_seg_path
        OLD_ALL_LABELS_FILE = train_seg_path
    else:
        # Build paths for previous iteration
        if num_regions > 1:
            prev_region_folder = join(args.data_path, f"iter_{current_iter-1:03d}", f"region_{region_idx}")
        else:
            prev_region_folder = join(args.data_path, f"iter_{current_iter-1:03d}")
        
        old_gpkg_selected = join(prev_region_folder, "new_labels", 'selected_labels_set.gpkg')
        old_gpkg_all = join(prev_region_folder, "new_labels", 'all_labels_set.gpkg')
        old_tiff_selected = join(prev_region_folder, "new_labels", 'selected_labels_set.tif')
        old_tiff_all = join(prev_region_folder, "new_labels", 'all_labels_set.tif')
        
        OLD_SELECTED_LABELS_FILE = old_gpkg_selected if exists(old_gpkg_selected) else old_tiff_selected
        OLD_ALL_LABELS_FILE = old_gpkg_all if exists(old_gpkg_all) else old_tiff_all

    # Load labels
    reference_tiff = train_seg_path
    
    if OLD_SELECTED_LABELS_FILE.endswith('.gpkg'):
        import geopandas as gpd
        meta = get_image_metadata(reference_tiff)
        shape = (meta['height'], meta['width'])
        transform = meta.get('transform')
        
        old_selected_gdf = gpd.read_file(OLD_SELECTED_LABELS_FILE)
        old_selected_labels = geodataframe_to_raster(old_selected_gdf, shape, transform)
    else:
        old_selected_labels = read_tiff(OLD_SELECTED_LABELS_FILE)
    
    if OLD_ALL_LABELS_FILE.endswith('.gpkg'):
        import geopandas as gpd
        meta = get_image_metadata(reference_tiff)
        shape = (meta['height'], meta['width'])
        transform = meta.get('transform')
        
        old_all_gdf = gpd.read_file(OLD_ALL_LABELS_FILE)
        old_all_labels = geodataframe_to_raster(old_all_gdf, shape, transform)
    else:
        old_all_labels = read_tiff(OLD_ALL_LABELS_FILE)

    ground_truth_segmentation = read_tiff(train_seg_path)

    # Use vector operations for faster sample selection
    if USE_VECTOR_OPERATIONS:
        logger.info("Using vector-based sample selection (optimized)")
        all_labels_set, selected_labels_set = get_new_segmentation_sample_vector(
            ground_truth_map=ground_truth_segmentation,
            old_all_labels=old_all_labels,
            old_selected_labels=old_selected_labels,
            new_pred_map=new_pred_map, 
            new_prob_map=new_prob_map, 
            new_depth_map=new_depth_map,
            prob_thr=args.prob_thr,
            depth_thr=args.depth_thr,
            sigma=args.sigma,
            args=args,
            reference_tiff=reference_tiff,
            output_as_raster=True,
        )
    else:
        logger.info("Using raster-based sample selection (original)")
        all_labels_set, selected_labels_set = get_new_segmentation_sample(
            ground_truth_map=ground_truth_segmentation,
            old_all_labels=old_all_labels,
            old_selected_labels=old_selected_labels,
            new_pred_map=new_pred_map, 
            new_prob_map=new_prob_map, 
            new_depth_map=new_depth_map,
            prob_thr=args.prob_thr,
            depth_thr=args.depth_thr,
            sigma=args.sigma,
            args=args
        )

    # Save new labels
    if OLD_SELECTED_LABELS_FILE.endswith('.gpkg'):
        image_metadata = get_image_metadata(reference_tiff)
    else:
        image_metadata = get_image_metadata(OLD_SELECTED_LABELS_FILE)
    
    check_folder(output_folder)
    
    logger.info("Saving labels as GeoPackage...")
    save_labels_as_geopackage(all_labels_set, ALL_LABELS_GPKG_PATH, reference_tiff)
    save_labels_as_geopackage(selected_labels_set, SELECTED_LABELS_GPKG_PATH, reference_tiff)

    logger.info("Saving labels as TIFF (for training)...")
    array2raster(ALL_LABELS_TIFF_PATH, all_labels_set, image_metadata, "Byte")
    array2raster(SELECTED_LABELS_TIFF_PATH, selected_labels_set, image_metadata, "Byte")
    
    logger.info(f"Region {region_idx} labels saved to {output_folder}")
    
    # Clean up memory
    del all_labels_set, selected_labels_set, new_pred_map, new_prob_map, new_depth_map
    gc.collect()
    
def generate_distance_map_for_first_iteration(current_iter_folder: str, args: dict):
    """
    Generate distance maps for the first iteration (from ground truth) for all regions.
    
    Now uses the optimized vectorized distance map generation.
    Supports multiple regions.
    """
    num_regions = getattr(args, 'num_regions', 1)
    logger.info(f"============ Generating Distance Map for {num_regions} region(s) (Optimized) ============")

    # Use the optimized distance map generation
    distance_map_func = generate_distance_map_fast if USE_VECTOR_OPERATIONS else generate_distance_map
    
    processes = []
    
    for region_idx in range(num_regions):
        train_seg_path = args.train_segmentation_paths[region_idx]
        test_seg_path = args.test_segmentation_paths[region_idx]
        
        if num_regions > 1:
            region_folder = join(current_iter_folder, f"region_{region_idx}")
            logger.info(f"=== Generating distance maps for Region {region_idx} ===")
        else:
            region_folder = current_iter_folder
        
        distance_map_folder = join(region_folder, "distance_map")
        check_folder(distance_map_folder)
        
        train_output = join(distance_map_folder, "train_distance_map.tif")
        test_output = join(distance_map_folder, "test_distance_map.tif")
        full_output = join(distance_map_folder, "full_distance_map.tif")
        
        # Skip if already done
        if exists(train_output) and exists(test_output):
            logger.info(f"Region {region_idx} distance maps already exist. Skipping generation...")
            continue
        
        # Create processes for parallel distance map generation
        if not exists(test_output):
            test_process = Process(target=distance_map_func, args=(test_seg_path, test_output, args.sigma))
            processes.append(test_process)
            test_process.start()
        
        if not exists(train_output):
            train_process = Process(target=distance_map_func, args=(train_seg_path, train_output, args.sigma))
            processes.append(train_process)
            train_process.start()
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    # Generate full distance maps for each region
    for region_idx in range(num_regions):
        train_seg_path = args.train_segmentation_paths[region_idx]
        
        if num_regions > 1:
            region_folder = join(current_iter_folder, f"region_{region_idx}")
        else:
            region_folder = current_iter_folder
        
        distance_map_folder = join(region_folder, "distance_map")
        train_output = join(distance_map_folder, "train_distance_map.tif")
        test_output = join(distance_map_folder, "test_distance_map.tif")
        full_output = join(distance_map_folder, "full_distance_map.tif")
        
        if not exists(full_output) and exists(train_output) and exists(test_output):
            test_distance_map = read_tiff(test_output)
            train_distance_map = read_tiff(train_output)
            
            train_metadata = get_image_metadata(train_seg_path)
            
            full_distance_map = np.maximum(test_distance_map, train_distance_map)
            array2raster(full_output, full_distance_map, train_metadata, "float32")
            
            del test_distance_map, train_distance_map, full_distance_map

            

def generate_distance_map_for_next_iteration(current_iter_folder: str, args: dict):
    """
    Generate distance maps for the next iteration from new labels for all regions.
    
    Now supports both GeoPackage and TIFF inputs with optimized generation.
    Supports multiple regions.
    """
    num_regions = getattr(args, 'num_regions', 1)
    
    logger.info(f"============ Generating Distance Maps for {num_regions} region(s) ============")
    
    for region_idx in range(num_regions):
        generate_distance_map_for_region(current_iter_folder, args, region_idx)


def generate_distance_map_for_region(current_iter_folder: str, args: dict, region_idx: int):
    """
    Generate distance maps for a single region.
    
    Parameters
    ----------
    current_iter_folder : str
        Current iteration folder
    args : dict
        Arguments dictionary
    region_idx : int
        Index of the region to process
    """
    num_regions = getattr(args, 'num_regions', 1)
    
    # Determine region folder
    if num_regions > 1:
        region_folder = join(current_iter_folder, f"region_{region_idx}")
        train_seg_path = args.train_segmentation_paths[region_idx]
        logger.info(f"=== Generating distance maps for Region {region_idx} ===")
    else:
        region_folder = current_iter_folder
        train_seg_path = args.train_segmentation_paths[0]
    
    # Check for GeoPackage first, fallback to TIFF
    ALL_LABELS_GPKG = join(region_folder, "new_labels", 'all_labels_set.gpkg')
    SELECTED_LABELS_GPKG = join(region_folder, "new_labels", 'selected_labels_set.gpkg')
    ALL_LABELS_TIFF = join(region_folder, "new_labels", 'all_labels_set.tif')
    SELECTED_LABELS_TIFF = join(region_folder, "new_labels", 'selected_labels_set.tif')
    
    ALL_LABELS_PATH = ALL_LABELS_GPKG if exists(ALL_LABELS_GPKG) else ALL_LABELS_TIFF
    SELECTED_LABELS_PATH = SELECTED_LABELS_GPKG if exists(SELECTED_LABELS_GPKG) else SELECTED_LABELS_TIFF
    
    distance_map_folder = join(region_folder, "distance_map")
    ALL_LABELS_DISTANCE_MAP_OUTPUT_PATH = join(distance_map_folder, 'all_labels_distance_map.tif')
    SELECTED_LABELS_DISTANCE_MAP_OUTPUT_PATH = join(distance_map_folder, 'selected_distance_map.tif')

    # Use the optimized or original function based on flag
    distance_map_func = generate_distance_map_fast if USE_VECTOR_OPERATIONS else generate_distance_map

    if not exists(ALL_LABELS_DISTANCE_MAP_OUTPUT_PATH):
        logger.info(f"Generating distance map at {ALL_LABELS_DISTANCE_MAP_OUTPUT_PATH}")
        check_folder(distance_map_folder)

        if ALL_LABELS_PATH.endswith('.gpkg'):
            from generate_distance_map_vector import generate_distance_map_from_geopackage
            generate_distance_map_from_geopackage(
                ALL_LABELS_PATH, 
                ALL_LABELS_DISTANCE_MAP_OUTPUT_PATH,
                reference_tiff=train_seg_path,
                sigma=args.sigma
            )
        else:
            distance_map_func(ALL_LABELS_PATH, ALL_LABELS_DISTANCE_MAP_OUTPUT_PATH, args.sigma)

    if not exists(SELECTED_LABELS_DISTANCE_MAP_OUTPUT_PATH):
        logger.info(f"Generating distance map at {SELECTED_LABELS_DISTANCE_MAP_OUTPUT_PATH}")
        check_folder(distance_map_folder)

        if SELECTED_LABELS_PATH.endswith('.gpkg'):
            from generate_distance_map_vector import generate_distance_map_from_geopackage
            generate_distance_map_from_geopackage(
                SELECTED_LABELS_PATH, 
                SELECTED_LABELS_DISTANCE_MAP_OUTPUT_PATH,
                reference_tiff=train_seg_path,
                sigma=args.sigma
            )
        else:
            distance_map_func(SELECTED_LABELS_PATH, SELECTED_LABELS_DISTANCE_MAP_OUTPUT_PATH, args.sigma)
    


def compile_metrics(current_iter_folder: str, args: dict):
    """Compile metrics for all regions.
    
    Supports multiple regions and creates both per-region and aggregated metrics.
    Also computes GLOBAL metrics combining all regions together.
    """
    num_regions = getattr(args, 'num_regions', 1)
    
    logger.info(f"============ Compiling Metrics for {num_regions} region(s) ============")
    
    all_metrics_test = {}
    all_metrics_train = {}
    
    # Lists to accumulate data for global metrics calculation
    all_pred_test_pixels = []
    all_gt_test_pixels = []
    all_pred_train_pixels = []
    all_gt_train_pixels = []
    
    for region_idx in range(num_regions):
        if num_regions > 1:
            region_folder = join(current_iter_folder, f"region_{region_idx}")
            test_seg_path = args.test_segmentation_paths[region_idx]
            train_seg_path = args.train_segmentation_paths[region_idx]
            logger.info(f"=== Compiling metrics for Region {region_idx} ===")
        else:
            region_folder = current_iter_folder
            test_seg_path = args.test_segmentation_paths[0]
            train_seg_path = args.train_segmentation_paths[0]
        
        METRICS_TEST_PATH = join(region_folder, "test_metrics.yaml")
        METRICS_TRAIN_PATH = join(region_folder, "train_metrics.yaml")

        ground_truth_test = read_tiff(test_seg_path)
        ground_truth_train = read_tiff(train_seg_path)

        PRED_PATH = join(region_folder, "raster_prediction", f"join_class_{np.sum(args.overlap)}.TIF")
        predicted_seg = read_tiff(PRED_PATH)

        # Accumulate masked pixels for global metrics
        if num_regions > 1:
            # Test pixels
            test_mask = ground_truth_test > 0
            all_gt_test_pixels.append(ground_truth_test[test_mask].flatten())
            all_pred_test_pixels.append((predicted_seg[test_mask] + 1).flatten())
            
            # Train pixels
            train_mask = ground_truth_train > 0
            all_gt_train_pixels.append(ground_truth_train[train_mask].flatten())
            all_pred_train_pixels.append((predicted_seg[train_mask] + 1).flatten())
        
        # Skip per-region metrics if already computed
        if exists(METRICS_TEST_PATH) and exists(METRICS_TRAIN_PATH):
            logger.info(f"Region {region_idx} metrics already exist. Skipping per-region calculation...")
            del ground_truth_test, ground_truth_train, predicted_seg
            continue

        # Test metrics
        metrics_test = evaluate_metrics(predicted_seg, ground_truth_test)
        if num_regions > 1:
            metrics_test = {f"region_{region_idx}/test/{key}": value for key, value in metrics_test.items()}
        else:
            metrics_test = {f"test/{key}": value for key, value in metrics_test.items()}
        
        wandb.log(metrics_test)
        save_yaml(metrics_test, METRICS_TEST_PATH)
        all_metrics_test.update(metrics_test)
        
        # Train metrics
        metrics_train = evaluate_metrics(predicted_seg, ground_truth_train, args.nb_class)
        if num_regions > 1:
            metrics_train = {f"region_{region_idx}/train/{key}": value for key, value in metrics_train.items()}
        else:
            metrics_train = {f"train/{key}": value for key, value in metrics_train.items()}
        
        wandb.log(metrics_train)
        save_yaml(metrics_train, METRICS_TRAIN_PATH)
        all_metrics_train.update(metrics_train)
        
        del ground_truth_test, ground_truth_train, predicted_seg
    
    # Save aggregated metrics and compute GLOBAL metrics if multiple regions
    if num_regions > 1:
        aggregated_metrics_path = join(current_iter_folder, "aggregated_metrics.yaml")
        global_metrics_path = join(current_iter_folder, "global_metrics.yaml")
        
        # Compute GLOBAL metrics combining all regions
        if not exists(global_metrics_path) and len(all_gt_test_pixels) > 0:
            logger.info("=== Computing GLOBAL metrics for all regions combined ===")
            
            from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, cohen_kappa_score
            
            # Concatenate all pixels from all regions
            global_gt_test = np.concatenate(all_gt_test_pixels)
            global_pred_test = np.concatenate(all_pred_test_pixels)
            global_gt_train = np.concatenate(all_gt_train_pixels)
            global_pred_train = np.concatenate(all_pred_train_pixels)
            
            # Use only classes that EXIST in the ground truth (avoid inflated metrics from non-existent classes)
            test_labels = np.unique(global_gt_test).tolist()
            train_labels = np.unique(global_gt_train).tolist()
            logger.info(f"Classes in test GT: {test_labels}")
            logger.info(f"Classes in train GT: {train_labels}")
            
            # Global TEST metrics (using only existing classes, zero_division=0 to avoid inflation)
            global_test_metrics = {
                "global/test/Accuracy": float(np.round(accuracy_score(global_gt_test, global_pred_test) * 100, 2)),
                "global/test/avgF1": float(f1_score(global_gt_test, global_pred_test, average="macro", zero_division=0, labels=test_labels)) * 100,
                "global/test/avgF1_weighted": float(f1_score(global_gt_test, global_pred_test, average="weighted", zero_division=0)) * 100,
                "global/test/avgPre": float(precision_score(global_gt_test, global_pred_test, average="macro", zero_division=0, labels=test_labels)) * 100,
                "global/test/avgRec": float(recall_score(global_gt_test, global_pred_test, average="macro", zero_division=0, labels=test_labels)) * 100,
                "global/test/F1": (f1_score(global_gt_test, global_pred_test, average=None, zero_division=0, labels=test_labels) * 100).tolist(),
                "global/test/Pre": (precision_score(global_gt_test, global_pred_test, average=None, zero_division=0, labels=test_labels) * 100).tolist(),
                "global/test/Rec": (recall_score(global_gt_test, global_pred_test, average=None, zero_division=0, labels=test_labels) * 100).tolist(),
                "global/test/classes": test_labels,
                "global/test/KappaScore": float(cohen_kappa_score(global_gt_test, global_pred_test, labels=test_labels)) * 100,
            }
            
            # Global TRAIN metrics (using only existing classes)
            global_train_metrics = {
                "global/train/Accuracy": float(np.round(accuracy_score(global_gt_train, global_pred_train) * 100, 2)),
                "global/train/avgF1": float(f1_score(global_gt_train, global_pred_train, average="macro", zero_division=0, labels=train_labels)) * 100,
                "global/train/avgF1_weighted": float(f1_score(global_gt_train, global_pred_train, average="weighted", zero_division=0)) * 100,
                "global/train/avgPre": float(precision_score(global_gt_train, global_pred_train, average="macro", zero_division=0, labels=train_labels)) * 100,
                "global/train/avgRec": float(recall_score(global_gt_train, global_pred_train, average="macro", zero_division=0, labels=train_labels)) * 100,
                "global/train/F1": (f1_score(global_gt_train, global_pred_train, average=None, zero_division=0, labels=train_labels) * 100).tolist(),
                "global/train/Pre": (precision_score(global_gt_train, global_pred_train, average=None, zero_division=0, labels=train_labels) * 100).tolist(),
                "global/train/Rec": (recall_score(global_gt_train, global_pred_train, average=None, zero_division=0, labels=train_labels) * 100).tolist(),
                "global/train/classes": train_labels,
                "global/train/KappaScore": float(cohen_kappa_score(global_gt_train, global_pred_train, labels=train_labels)) * 100,
            }
            
            global_metrics = {**global_test_metrics, **global_train_metrics}
            
            wandb.log(global_metrics)
            save_yaml(global_metrics, global_metrics_path)
            
            logger.info(f"Global Test F1-score: {global_test_metrics['global/test/avgF1']:.2f}%")
            logger.info(f"Global Train F1-score: {global_train_metrics['global/train/avgF1']:.2f}%")
            
            # Add global metrics to all_metrics
            all_metrics_test.update(global_test_metrics)
            all_metrics_train.update(global_train_metrics)
            
            del global_gt_test, global_pred_test, global_gt_train, global_pred_train
        
        # Save aggregated metrics (per-region + global)
        if not exists(aggregated_metrics_path):
            all_metrics = {**all_metrics_test, **all_metrics_train}
            save_yaml(all_metrics, aggregated_metrics_path)

 

def _get_component_labels(pred: np.ndarray, gt: np.ndarray, num_class: int) -> tuple:
    """
    Extract ground truth and predicted labels per component.
    
    For each annotated component in the ground truth, determines the predicted class
    by majority vote within that component.
    
    Parameters
    ----------
    pred : np.ndarray
        Predicted segmentation map
    gt : np.ndarray
        Ground truth segmentation map
    num_class : int
        Number of classes
    
    Returns
    -------
    tuple
        (gt_labels_per_component, pred_labels_per_component) as lists
    """
    # Ensure pred is 1-indexed to match gt (model outputs 0..num_class-1)
    pred = pred.copy()
    if (gt.min() == 0) and (pred.min() == 0) and (pred.max() <= num_class - 1):
        pred = pred + 1
    
    # Get connected components from ground truth
    gt_components = label(gt > 0)
    unique_components = np.unique(gt_components)
    unique_components = unique_components[unique_components > 0]
    
    gt_labels_per_component = []
    pred_labels_per_component = []
    
    for comp_id in unique_components:
        comp_mask = gt_components == comp_id
        
        # True class
        gt_class = gt[comp_mask].max()
        
        # Predicted class (majority vote)
        pred_in_comp = pred[comp_mask]
        pred_nonzero = pred_in_comp[pred_in_comp > 0]
        
        if len(pred_nonzero) == 0:
            pred_class = 0
        else:
            values, counts = np.unique(pred_nonzero, return_counts=True)
            pred_class = values[np.argmax(counts)]
        
        gt_labels_per_component.append(gt_class)
        pred_labels_per_component.append(pred_class)
    
    return gt_labels_per_component, pred_labels_per_component


def compile_component_metrics(current_iter_folder: str, args: dict):
    """
    Compile component-level metrics for all regions.
    
    Now supports loading from GeoPackage for faster vector-based statistics.
    Supports multiple regions.
    Also computes global F1-score by component across all regions.
    """
    num_regions = getattr(args, 'num_regions', 1)
    current_iter_num = int(current_iter_folder.split("_")[-1])
    
    logger.info(f"============ Compiling Component Metrics for {num_regions} region(s) ============")
    
    all_stats_list = []
    
    # Lists to accumulate data for global F1 by component calculation
    all_gt_labels_per_component = []
    all_pred_labels_per_component = []
    
    # Lists to accumulate data for global mIoU calculation
    all_iou_per_polygon = []
    all_gt_classes_per_polygon = []
    
    for region_idx in range(num_regions):
        if num_regions > 1:
            region_folder = join(current_iter_folder, f"region_{region_idx}")
            test_seg_path = args.test_segmentation_paths[region_idx]
            logger.info(f"=== Compiling component metrics for Region {region_idx} ===")
        else:
            region_folder = current_iter_folder
            test_seg_path = args.test_segmentation_paths[0]
        
        COMPONENTS_PRECISION_METRICS_PATH = join(region_folder, 'all_labels_test_metrics.yaml')
        COMPONENTS_STATS_PATH = join(region_folder, "all_labels_stats.parquet")
        
        # Load data for global metrics even if per-region metrics exist
        ground_truth_test = read_tiff(test_seg_path)
        PRED_PATH = join(region_folder, "raster_prediction", f"join_class_{np.sum(args.overlap)}.TIF")
        
        # Accumulate component-level labels for global metrics
        if num_regions > 1 and exists(PRED_PATH):
            predicted_seg = read_tiff(PRED_PATH)
            gt_labels, pred_labels = _get_component_labels(predicted_seg, ground_truth_test, args.nb_class)
            all_gt_labels_per_component.extend(gt_labels)
            all_pred_labels_per_component.extend(pred_labels)
            
            # Accumulate mIoU data for global calculation
            iou_data, gt_classes = get_miou_polygon_data(predicted_seg, ground_truth_test, args.nb_class)
            all_iou_per_polygon.extend(iou_data)
            all_gt_classes_per_polygon.extend(gt_classes)
        
        if exists(COMPONENTS_PRECISION_METRICS_PATH) and exists(COMPONENTS_STATS_PATH):
            logger.info(f"Region {region_idx} component metrics already exist. Skipping per-region calculation...")
            del ground_truth_test
            if 'predicted_seg' in dir():
                del predicted_seg
            continue
        
        # Try GeoPackage first, fallback to TIFF
        ALL_LABELS_GPKG = join(region_folder, "new_labels", "all_labels_set.gpkg")
        ALL_LABELS_TIFF = join(region_folder, "new_labels", "all_labels_set.tif")
        
        if exists(ALL_LABELS_GPKG) and USE_VECTOR_OPERATIONS:
            import geopandas as gpd
            logger.info("Loading labels from GeoPackage for fast statistics")
            all_labels_gdf = gpd.read_file(ALL_LABELS_GPKG)
            
            meta = get_image_metadata(test_seg_path)
            shape = (meta['height'], meta['width'])
            transform = meta.get('transform')
            all_labels = geodataframe_to_raster(all_labels_gdf, shape, transform)
            
            all_labels_stats = get_components_stats_vector(all_labels_gdf)
        else:
            all_labels = read_tiff(ALL_LABELS_TIFF)
            all_labels_stats = get_components_stats(label(all_labels), all_labels).reset_index()

        # Evaluate component precision metrics (pixel-based)
        all_labels_metrics = evaluate_component_metrics(ground_truth_test, all_labels, args.nb_class)
        
        # Evaluate F1-score by component (majority vote per component)
        if exists(PRED_PATH):
            if 'predicted_seg' not in dir():
                predicted_seg = read_tiff(PRED_PATH)
            f1_by_component_metrics = evaluate_f1_by_component(
                pred=predicted_seg, 
                gt=ground_truth_test, 
                num_class=args.nb_class
            )
            all_labels_metrics.update(f1_by_component_metrics)
            logger.info(f"Region {region_idx}: F1 by component = {f1_by_component_metrics['avgF1_component']:.2f}%")
            
            # Evaluate mIoU per polygon
            miou_metrics = evaluate_miou_per_polygon(
                pred=predicted_seg,
                gt=ground_truth_test,
                num_class=args.nb_class
            )
            all_labels_metrics["avgMIoU"] = miou_metrics["avgMIoU"]
            all_labels_metrics["MIoU_per_class"] = miou_metrics["MIoU_per_class"]
            all_labels_metrics["n_polygons_evaluated"] = miou_metrics["n_polygons_evaluated"]
            all_labels_metrics["n_polygons_matched"] = miou_metrics["n_polygons_matched"]
            logger.info(f"Region {region_idx}: mIoU per polygon = {miou_metrics['avgMIoU']:.2f}%")
            
            del predicted_seg
        
        if num_regions > 1:
            all_labels_metrics = {f"region_{region_idx}/all_labels_{key}": value for key, value in all_labels_metrics.items()}
        else:
            all_labels_metrics = {f"all_labels_{key}": value for key, value in all_labels_metrics.items()}
        
        all_labels_stats["iter"] = f"iter_{current_iter_num:03d}"
        all_labels_stats["iter_num"] = current_iter_num
        all_labels_stats["region"] = region_idx
        
        # Save metrics
        save_yaml(all_labels_metrics, COMPONENTS_PRECISION_METRICS_PATH)
        all_labels_stats.to_parquet(COMPONENTS_STATS_PATH)
        
        all_stats_list.append(all_labels_stats)
        
        del ground_truth_test, all_labels
    
    # Save aggregated stats if multiple regions
    if num_regions > 1 and all_stats_list:
        aggregated_stats_path = join(current_iter_folder, "all_regions_stats.parquet")
        if not exists(aggregated_stats_path):
            aggregated_stats = pd.concat(all_stats_list, ignore_index=True)
            aggregated_stats.to_parquet(aggregated_stats_path)
    
    # Compute and save global F1-score by component
    if num_regions > 1 and len(all_gt_labels_per_component) > 0:
        global_component_metrics_path = join(current_iter_folder, "global_component_metrics.yaml")
        
        if not exists(global_component_metrics_path):
            logger.info("=== Computing GLOBAL F1-score by component ===")
            
            from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
            
            gt_array = np.array(all_gt_labels_per_component)
            pred_array = np.array(all_pred_labels_per_component)
            
            # Use only classes that exist in the ground truth
            labels = list(range(1, args.nb_class + 1))
            
            global_component_metrics = {
                "global/component/n_components": len(gt_array),
                "global/component/Accuracy": float(accuracy_score(gt_array, pred_array)) * 100,
                "global/component/avgF1": float(f1_score(gt_array, pred_array, average="macro", zero_division=0, labels=labels)) * 100,
                "global/component/F1": (f1_score(gt_array, pred_array, average=None, zero_division=0, labels=labels) * 100).tolist(),
                "global/component/avgPrec": float(precision_score(gt_array, pred_array, average="macro", zero_division=0, labels=labels)) * 100,
                "global/component/avgRec": float(recall_score(gt_array, pred_array, average="macro", zero_division=0, labels=labels)) * 100,
            }
            
            # Add global mIoU metrics
            if all_iou_per_polygon:
                avg_miou = float(np.mean(all_iou_per_polygon) * 100)
                n_matched = sum(1 for iou in all_iou_per_polygon if iou > 0)
                
                # Compute mIoU per class
                miou_per_class = []
                for c in range(1, args.nb_class + 1):
                    class_ious = [iou for iou, cls in zip(all_iou_per_polygon, all_gt_classes_per_polygon) if cls == c]
                    if class_ious:
                        miou_per_class.append(float(np.mean(class_ious) * 100))
                    else:
                        miou_per_class.append(0.0)
                
                global_component_metrics["global/miou/avgMIoU"] = avg_miou
                global_component_metrics["global/miou/MIoU_per_class"] = miou_per_class
                global_component_metrics["global/miou/n_polygons_evaluated"] = len(all_iou_per_polygon)
                global_component_metrics["global/miou/n_polygons_matched"] = n_matched
                
                logger.info(f"Global mIoU per polygon: {avg_miou:.2f}%")
            
            save_yaml(global_component_metrics, global_component_metrics_path)
            logger.info(f"Global F1 by component: {global_component_metrics['global/component/avgF1']:.2f}%")
            logger.info(f"Total components evaluated: {global_component_metrics['global/component/n_components']}")



#############
### SETUP ###
#############


if __name__ == "__main__":
    ROOT_PATH = dirname(__file__)
    
    parser = argparse.ArgumentParser(description="Get the arguments file")
    parser.add_argument('file_path', nargs='?', default='args.yaml', help='Path to the file (default: args.yaml)')
    args_path = parser.parse_args().file_path
    
    args = load_args(args_path)
    print(f"Args loaded from: {args_path}")
    print(f"Model directory: {args.model_dir}")
    print(f"Architecture: {args.arch}")
    print(f"Input dimension: {args.input_dimension}")
    print(f"Dropout rate: {args.dropout_rate}")
    print(f"Number of iterations: {args.num_iter}")

    # Use model_dir name for version/experiment name instead of data_path
    if hasattr(args, 'model_dir') and args.model_dir:
        version_name = os.path.split(args.model_dir)[-1]
        # If split returns empty (e.g. path ends with /), try dirname
        if not version_name:
            version_name = os.path.split(os.path.dirname(args.model_dir))[-1]
    else:
        version_name = os.path.split(args.data_path)[-1]

    logger = create_logger(module_name=__name__, filename=version_name)

    logger.info(f"################### {version_name.upper()} ###################")

    # create output path
    check_folder(args.data_path)

    # Save args state into data_path
    save_yaml(args, join(args.data_path, "args.yaml"))

    run = wandb.init(mode="disabled")

    ##### LOOP #####

    # Set random seed
    fix_random_seeds(args.seed)

    
    num_regions = getattr(args, 'num_regions', 1)
    logger.info(f"Running with {num_regions} region(s)")
    
    # List to track background threads across all iterations
    background_threads = []
    
    while True:

        print_sucess("Working ON:")
        print_sucess(get_device()) 
        
        # get current iteration folder (checks all regions)
        current_iter_folder = get_current_iter_folder(args.data_path, args.overlap, num_regions)
        current_iter = int(current_iter_folder.split("_")[-1])

        if current_iter > args.num_iter:
            break
        
        logger.info(f"##################### ITERATION {current_iter} ##################### ")
        logger.info(f"Current iteration folder: {current_iter_folder}")
        
        # if the iteration 0 applies distance map to ground truth segmentation
        if current_iter == 0:
            generate_distance_map_for_first_iteration(current_iter_folder, args)
            
            logger.info("Generating labels view for iter 0")
            
            # Generate visualization for each region in background threads
            for region_idx in range(num_regions):
                if num_regions > 1:
                    region_folder = join(current_iter_folder, f"region_{region_idx}")
                    check_folder(region_folder)
                else:
                    region_folder = current_iter_folder
                labels_view_thread = Thread(
                    target=generate_labels_view,
                    args=(region_folder, args.ortho_images[region_idx], args.train_segmentation_paths[region_idx])
                )
                labels_view_thread.start()
                background_threads.append(labels_view_thread)
    
            logger.info("Done!")
            continue
        
        with torch.no_grad():
            torch.cuda.empty_cache()
        
        # Get current model folder
        current_model_folder = join(current_iter_folder, args.model_dir)
        check_folder(current_model_folder)

        train_iteration(current_iter_folder, args)

        evaluate_iteration(current_iter_folder, args)

        pred2raster(current_iter_folder, args)
        
        # Start background threads for non-critical visualization/metrics tasks
        # compile_metrics in background thread
        compile_metrics_thread = Thread(target=compile_metrics, args=(current_iter_folder, args))
        compile_metrics_thread.start()
        background_threads.append(compile_metrics_thread)

        generate_labels_for_next_iteration(current_iter_folder, args)
        
        # compile_component_metrics in background thread
        compile_component_metrics_thread = Thread(target=compile_component_metrics, args=(current_iter_folder, args))
        compile_component_metrics_thread.start()
        background_threads.append(compile_component_metrics_thread)
        
        # generate_labels_view for each region in background threads
        for region_idx in range(num_regions):
            if num_regions > 1:
                region_folder = join(current_iter_folder, f"region_{region_idx}")
            else:
                region_folder = current_iter_folder
            labels_view_thread = Thread(
                target=generate_labels_view, 
                args=(region_folder, args.ortho_images[region_idx], args.train_segmentation_paths[region_idx])
            )
            labels_view_thread.start()
            background_threads.append(labels_view_thread)
            
        generate_distance_map_for_next_iteration(current_iter_folder, args)

        #############################################

        # Delete useless files for each region
        for region_idx in range(num_regions):
            if num_regions > 1:
                region_folder = join(current_iter_folder, f"region_{region_idx}")
            else:
                region_folder = current_iter_folder
            delete_useless_files(current_iter_folder=region_folder)

        print_sucess("Distance map generated")
    
    # Wait for all background threads to complete after all iterations
    logger.info(f"Waiting for {len(background_threads)} background threads to complete...")
    for thread in background_threads:
        thread.join()
    logger.info("All background threads completed.")
 

if __name__ != "__main__":
    from logging import getLogger
    logger = getLogger("__main__")
