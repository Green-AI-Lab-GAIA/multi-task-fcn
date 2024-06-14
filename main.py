import argparse
import gc
import math
import os
import shutil
from multiprocessing import Process
from os.path import dirname, exists, isfile, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from tqdm import tqdm

from scipy.ndimage import label

from evaluation import evaluate_iteration
from generate_distance_map import generate_distance_map
from pred2raster import pred2raster
from sample_selection import get_new_segmentation_sample, get_components_stats
from src.dataset import DatasetFromCoord
from src.io_operations import (ParquetUpdater, array2raster,
                               convert_tiff_to_npy, get_image_metadata,
                               get_npy_filepath_from_tiff, load_args,
                               read_tiff, save_yaml)
from src.lazy_dataset import LazyDatasetFromCoord
from src.logger import create_logger
from src.metrics import evaluate_component_metrics, evaluate_metrics
from src.model import (build_model, define_loader, eval, load_weights,
                       save_checkpoint, train)
from src.utils import (check_folder, fix_random_seeds, get_device, oversamp,
                       print_sucess, restart_from_checkpoint,
                       restore_checkpoint_variables)
from visualization import generate_labels_view
import wandb
gc.set_threshold(0)

plt.set_loglevel(level = 'info')



def delete_useless_files(current_iter_folder:str):
    
    folder_to_remove = join(current_iter_folder,"prediction")

    if exists(folder_to_remove):
        shutil.rmtree(folder_to_remove)
    
    else:
        pass




def is_iter_0_done(data_path:str):
    """Verify if the distance map from the ground truth segmentation is done

    Parameters
    ----------
    data_path : str
        Root data path

    Returns
    -------
    bool
    """
    path_distance_map = join(data_path, "iter_000", "distance_map")
    is_test_map_done = exists(join(path_distance_map, "test_distance_map.tif"))
    is_train_map_done = exists(join(path_distance_map, "train_distance_map.tif"))
    
    if is_test_map_done and is_train_map_done:
        return True
    
    else:
        return False


def get_current_iter_folder(data_path, overlap):
    """Get the current iteration folder
    This function verify which iteration folder isn't finished yet and return the path to it.

    Parameters
    ----------
    data_path : str
        Path to the data folder 
    overlap : float
        Parameter used for create the output file name

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

        prediction_path = join(iter_path, "raster_prediction")
        
        is_folder_generated = exists(prediction_path)
        is_depth_done = isfile(join(prediction_path, f"depth_{np.sum(overlap)}.TIF"))
        is_pred_done = isfile(join(prediction_path, f"join_class_{np.sum(overlap)}.TIF"))
        is_prob_done = isfile(join(prediction_path, f"join_prob_{np.sum(overlap)}.TIF"))
        
        distance_map_path = join(iter_path, "distance_map")
        is_distance_map_done = isfile(join(distance_map_path, "selected_distance_map.tif"))

        if is_folder_generated and is_depth_done and is_pred_done and is_prob_done and is_distance_map_done:
            
            next_iter = int(iter_folder_name.split("_")[-1]) + 1
            next_iter_path = join(data_path, f"iter_{next_iter:03d}")
            
            check_folder(next_iter_path)

            return next_iter_path
    
    if is_iter_0_done(args.data_path):
        return join(data_path, f"iter_{1:03d}")


    # create iter_0 folder if not exists
    check_folder(iter_0_path)

    return iter_0_path
    

def get_last_segmentation_path(train_segmentation_path, current_iter_folder):
    current_iter = int(current_iter_folder.split("_")[-1])

    data_path = dirname(current_iter_folder)

    if current_iter == 1:
        image_path = train_segmentation_path
        

    else:
        image_path = join(data_path, f"iter_{current_iter-1:03d}", "new_labels", "selected_labels_set.tif")

    return image_path


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

def get_last_distance_map_path(current_iter_folder:str):
    current_iter = int(current_iter_folder.split("_")[-1])
    data_path = dirname(current_iter_folder)

    if current_iter == 1:
        distance_map_filename = "train_distance_map.tif"
    else:
        distance_map_filename = "selected_distance_map.tif"

    image_path = join(data_path, f"iter_{current_iter-1:03d}", "distance_map", distance_map_filename)
    
    return image_path


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
                               start_warmup:float)->np.ndarray:
    """Get the learning rate schedule using cosine annealing with warmup
    
    This schedule start with a warmup learning rate and then decrease to the final learning rate

    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        Model train loader
    base_lr : float
        base learning rate
    final_lr : float
        final learning rate
    epochs : int
        number of total epochs to run
    warmup_epochs : int
        number of warmup epochs
    start_warmup : float
        initial warmup learning rate

    Returns
    -------
    np.array
        learning rate schedule
    """

    # define a linear distribuition from start_warmup to base_lr
    warmup_lr_schedule = np.linspace(start_warmup, base_lr, len(train_loader) * warmup_epochs)
    
    # iteration numbers
    iters = np.arange(len(train_loader) * (epochs - warmup_epochs))

    cosine_lr_schedule = []

    for t in iters:

        lr = final_lr + 0.5 * (base_lr - final_lr) * (1 + math.cos(math.pi * t / (len(train_loader) * (epochs - warmup_epochs))))

        cosine_lr_schedule.append(lr)

    cosine_lr_schedule = np.array(cosine_lr_schedule)

    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

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
                 patience:int=5,
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

    training_stats = ParquetUpdater(join(args.data_path, "training_stats.parquet"))

    # Create figures folder to save training figures every epochsaved.
    figures_path = join(dirname(last_checkpoint), 'figures')
    check_folder(figures_path)

    for epoch in tqdm(range(start_epoch, num_epochs)):

        
        if count_early == patience:
            logger.info("============ Early Stop at epoch %i ... ============" % epoch)
            break
        
        np.random.shuffle(train_loader.dataset.coords)
        np.random.shuffle(val_loader.dataset.coords)

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
                                 lambda_weight=args.lambda_weight)
        
        logger.info("Evaluating the model...")
        f1_avg, f1_by_class_avg = eval(val_loader, model)
        
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

        is_best = (f1_avg - best_val) > 0.0009

        # save checkpoints        
        if is_best: 
            logger.info("============ Saving best models at epoch %i ... ============" % epoch)
            
            best_val = f1_avg
            
            save_checkpoint(last_checkpoint, model, optimizer, epoch+1, best_val, count_early)
            
            count_early = 0

        else:
            count_early += 1

            
    print_sucess("Training done !")



def train_iteration(current_iter_folder:str, args:dict):
    """Train the model in the current iteration.
    Load the output and the model trained from the last iteration, and train the model again

    Parameters
    ----------
    current_iter_folder : str
        The current iteration folder
    args : dict
        The dict with the parameters for the model.
        The parameters are defined in the args.yaml file
    """
    DEVICE = get_device()

    current_model_folder = join(current_iter_folder, args.model_dir)

    current_iter = int(current_iter_folder.split("_")[-1])


    current_iter_checkpoint = join(current_model_folder, args.checkpoint_file)
    
    if isfile(current_iter_checkpoint):
        last_checkpoint = current_iter_checkpoint
        loaded_from_last_iteration = False
    
    # If the weights from the current iteration, doenst exist.
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
    

    logger.info("============ Initialized Training ============")
    

    segmentation_path = get_last_segmentation_path(args.train_segmentation_path, current_iter_folder)
    distance_map_path = get_last_distance_map_path(current_iter_folder)
            

    train_dataset = DatasetFromCoord(
        image_path=args.ortho_image,
        segmentation_path=segmentation_path,
        distance_map_path=distance_map_path,
        samples=args.samples,
        augment=args.augment,
        crop_size=args.size_crops,
        copy_paste_augmentation=args.copy_and_paste_augmentation
    )
    
    train_dataset.standardize_image_channels()
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )


    # LOAD VALIDATION SET
    val_dataset = DatasetFromCoord(
        image_path=args.ortho_image,
        segmentation_path=segmentation_path,
        distance_map_path=distance_map_path,
        samples=args.samples//3,
        augment=args.augment,
        crop_size=args.size_crops,
        copy_paste_augmentation=False
    )
    val_dataset.standardize_image_channels()

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = args.batch_size,
        num_workers = args.workers,
        pin_memory = True,
        drop_last = True,
        shuffle = True,
    )

    logger.info("Building data done with {} images loaded.".format(len(train_loader)))

    orthoimage_meta = get_image_metadata(args.ortho_image)

    model  = build_model(
        (orthoimage_meta["count"], orthoimage_meta["height"], orthoimage_meta["width"]),
        args.nb_class,  
        args.arch, 
        args.is_pretrained,
        psize = args.size_crops,
        dropout_rate = args.dropout_rate
    )


    logger.info("Building model done.")

    ###### BULD OPTMIZER #######
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.weight_decay
    )

    # define how the learning rate will be changed in the training process.
    lr_schedule = get_learning_rate_schedule(
        train_loader, 
        args.base_lr, 
        args.final_lr, 
        args.epochs, 
        args.warmup_epochs, 
        args.start_warmup
    )
    logger.info("Building optimizer done.")


    
    ########## LOAD MODEL WEIGHTS FROM THE LAST CHECKPOINT ##########
    if last_checkpoint is None:
        # create empty checkpoint
        save_checkpoint(current_iter_checkpoint, model, optimizer, 0, 0.0, 0, is_iter_finished=False)
        
        last_checkpoint = current_iter_checkpoint
    

    model = load_weights(model, last_checkpoint)
    # Load model to GPU
    model = model.to(DEVICE)
    
    # restore variables
    to_restore = {"epoch": 0, "best_val":(0.), "count_early": 0, "is_iter_finished":False}
    restart_from_checkpoint(
        last_checkpoint,
        run_variables = to_restore,
        state_dict = model,
        optimizer = optimizer,
    )


    # If the metrics are from the model from the last iteration, 
    # the model reset the metrics
    if loaded_from_last_iteration:
        to_restore["epoch"] = 0
        to_restore["best_val"] = 0.0
        to_restore["count_early"] = 0
        to_restore["is_iter_finished"] = False
    

    ######## TRAIN MODEL #########
    current_checkpoint = join(current_model_folder, args.checkpoint_file)
    
    cudnn.benchmark = True
    
    gc.collect()

    # If the model isnt finished yet, train!
    if not to_restore["is_iter_finished"]:
        train_epochs(current_checkpoint, 
                     to_restore["epoch"], 
                     args.epochs, 
                     to_restore["best_val"], 
                     train_loader, 
                     model, 
                     optimizer, 
                     lr_schedule, 
                     to_restore["count_early"],
                     val_loader = val_loader,
                     current_iter_folder = current_iter_folder)
    gc.collect()


    #### CHANGE MODEL STATUS TO FINISHED ####
    # load models weights again to change status to is_iter_finished=True
    model = load_weights(model, current_checkpoint)

    to_restore = {"epoch": 0, "count_early": 0, "is_iter_finished":False, "best_val":(0.)}
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
    

    # FREE UP MEMORY
    with torch.no_grad():
        torch.cuda.empty_cache()
    
    gc.collect()


def generate_labels_for_next_iteration(current_iter_folder:str, args:dict):
    """
    Generate labels and distance map for the next iteration
    """
    
    ALL_LABELS_OUPUT_PATH = join(current_iter_folder, "new_labels", f'all_labels_set.tif')
    SELECTED_LABELS_OUTPUT_PATH = join(current_iter_folder, "new_labels", f'selected_labels_set.tif')

    if exists(ALL_LABELS_OUPUT_PATH) and exists(SELECTED_LABELS_OUTPUT_PATH):
        return
    
    logger.info("============ Generating New Samples ============")

    current_iter = int(current_iter_folder.split("iter_")[-1])

    NEW_PRED_FILE = join(current_iter_folder, "raster_prediction", f'join_class_{np.sum(args.overlap)}.TIF')
    new_pred_map = read_tiff(NEW_PRED_FILE)

    NEW_PROB_FILE = join(current_iter_folder, "raster_prediction", f'join_prob_{np.sum(args.overlap)}.TIF')
    new_prob_map = read_tiff(NEW_PROB_FILE)

    NEW_DEPTH_FILE = join(current_iter_folder, "raster_prediction", f'depth_{np.sum(args.overlap)}.TIF')
    new_depth_map = read_tiff(NEW_DEPTH_FILE)


    if current_iter == 1:
        OLD_SELECTED_LABELS_FILE = args.train_segmentation_path
        OLD_ALL_LABELS_FILE = args.train_segmentation_path

    else:
        OLD_SELECTED_LABELS_FILE = join(args.data_path, f"iter_{current_iter-1:03d}", "new_labels", f'selected_labels_set.tif')
        OLD_ALL_LABELS_FILE = join(args.data_path, f"iter_{current_iter-1:03d}", "new_labels", f'all_labels_set.tif')


    old_selected_labels = read_tiff(OLD_SELECTED_LABELS_FILE)
    old_all_labels = read_tiff(OLD_ALL_LABELS_FILE)


    ground_truth_segmentation = read_tiff(args.train_segmentation_path)

    all_labels_set, selected_labels_set = get_new_segmentation_sample(
        ground_truth_map = ground_truth_segmentation,
        old_all_labels = old_all_labels,
        old_selected_labels = old_selected_labels,
        new_pred_map = new_pred_map, 
        new_prob_map = new_prob_map, 
        new_depth_map = new_depth_map,
        prob_thr = args.prob_thr,
        depth_thr = args.depth_thr
    )

    ##### SAVE NEW LABELS ####
    image_metadata = get_image_metadata(OLD_SELECTED_LABELS_FILE)
    
    check_folder(dirname(ALL_LABELS_OUPUT_PATH))

    array2raster(ALL_LABELS_OUPUT_PATH, all_labels_set, image_metadata, "Byte")


    check_folder(dirname(SELECTED_LABELS_OUTPUT_PATH))

    array2raster(SELECTED_LABELS_OUTPUT_PATH, selected_labels_set, image_metadata, "Byte")
    


def generate_distance_map_for_next_iteration(current_iter_folder):

    ALL_LABELS_PATH = join(current_iter_folder, "new_labels", f'all_labels_set.tif')
    SELECTED_LABELS_PATH = join(current_iter_folder, "new_labels", f'selected_labels_set.tif')
    
    
    ALL_LABELS_DISTANCE_MAP_OUTPUT_PATH = join(current_iter_folder, "distance_map", f'all_labels_distance_map.tif')
    
    SELECTED_LABELS_DISTANCE_MAP_OUPTPUT_PATH  = join(current_iter_folder, "distance_map", f'selected_distance_map.tif')

    #######################################
    ######## GENERATE DISTANCE MAP ########
    if not exists(ALL_LABELS_DISTANCE_MAP_OUTPUT_PATH):
        logger.info(f"Generating distance map and saving at {ALL_LABELS_DISTANCE_MAP_OUTPUT_PATH}")

        check_folder(dirname(ALL_LABELS_DISTANCE_MAP_OUTPUT_PATH))

        generate_distance_map(ALL_LABELS_PATH, ALL_LABELS_DISTANCE_MAP_OUTPUT_PATH, args.sigma)
    

    if not exists(SELECTED_LABELS_DISTANCE_MAP_OUPTPUT_PATH):
        logger.info(f"Generating distance map and saving at {SELECTED_LABELS_DISTANCE_MAP_OUPTPUT_PATH}")

        check_folder(dirname(SELECTED_LABELS_DISTANCE_MAP_OUPTPUT_PATH))

        generate_distance_map(SELECTED_LABELS_PATH, SELECTED_LABELS_DISTANCE_MAP_OUPTPUT_PATH, args.sigma)
    


def compile_metrics(current_iter_folder, args):

    GROUND_TRUTH_TEST_PATH = args.test_segmentation_path
    ground_truth_test = read_tiff(GROUND_TRUTH_TEST_PATH)

    GROUND_TRUTH_TRAIN_PATH = args.train_segmentation_path
    ground_truth_train = read_tiff(GROUND_TRUTH_TRAIN_PATH)

    PRED_PATH = join(current_iter_folder, "raster_prediction", f"join_class_{np.sum(args.overlap)}.TIF")
    predicted_seg = read_tiff(PRED_PATH)

    ### Save test metrics ###
    metrics_test = evaluate_metrics(predicted_seg, ground_truth_test)
    metrics_test = {f"test/{key}": value for key, value in metrics_test.items()}.copy()
    
    wandb.log(metrics_test)
    
    save_yaml(metrics_test, join(current_iter_folder,'test_metrics.yaml'))    

    
    ### Save train metrics ###
    metrics_train = evaluate_metrics(predicted_seg, ground_truth_train, args.nb_class)
    metrics_train = {f"train/{key}": value for key, value in metrics_train.items()}.copy()
    
    wandb.log(metrics_train)
    
    save_yaml(metrics_train, join(current_iter_folder,'train_metrics.yaml'))

 

def compile_component_metrics(current_iter_folder, args):
    
    GROUND_TRUTH_TEST_PATH = args.test_segmentation_path
    ground_truth_test = read_tiff(GROUND_TRUTH_TEST_PATH)
    
    ### Save test component metrics ###
    HIGH_PROB_COMPONENTS_PATH = join(current_iter_folder,'all_labels_test_metrics.yaml')
    
    all_labels = read_tiff(join(current_iter_folder, "new_labels", "all_labels_set.tif"))

    all_labels_metrics = evaluate_component_metrics(ground_truth_test, all_labels, args.nb_class)
    
    all_labels_metrics = {f"all_labels_{key}": value for key, value in all_labels_metrics.items()}.copy()
    
    wandb.log(all_labels_metrics)
    
    all_labels_stats = get_components_stats(label(all_labels), all_labels)
    num_trees_by_class = all_labels_stats.groupby("tree_type").nunique()
    
    # create a dict in pair of this data frame
    num_trees_by_class = num_trees_by_class.to_dict()
    
    wandb.log(num_trees_by_class)
    
    save_yaml(all_labels_metrics, HIGH_PROB_COMPONENTS_PATH)



#############
### SETUP ###
#############


if __name__ == "__main__":
    ROOT_PATH = dirname(__file__)
    
    parser = argparse.ArgumentParser(description="Get the arguments file")
    parser.add_argument('file_path', nargs='?', default='args.yaml', help='Path to the file (default: args.yaml)')
    args_path = parser.parse_args().file_path
    
    args = load_args(args_path)
    print("Args loaded from",  )

    version_name = os.path.split(args.data_path)[-1]

    logger = create_logger(module_name=__name__, filename=version_name)

    logger.info(f"################### {version_name.upper()} ###################")

    # create output path
    check_folder(args.data_path)

    # Save args state into data_path
    save_yaml(args, join(args.data_path, "args.yaml"))

    run = wandb.init(project="segmentation", name=version_name, config=args)

    ##### LOOP #####

    # Set random seed
    fix_random_seeds(args.seed)

    
    while True:

        print_sucess("Working ON:")
        print_sucess(get_device()) 
        
        # get current iteration folder
        current_iter_folder = get_current_iter_folder(args.data_path, args.overlap)
        current_iter = int(current_iter_folder.split("_")[-1])

        if current_iter > args.num_iter:
            break
        
        logger.info(f"##################### ITERATION {current_iter} ##################### ")
        logger.info(f"Current iteration folder: {current_iter_folder}")
        
        # if the iteration 0 applies distance map to ground truth segmentation
        if current_iter == 0:
            logger.info(f"Generating test distance map")

            TEST_SEGMENTATION_PATH = args.test_segmentation_path
            TEST_DISTANCE_MAP_OUTPUT = join(current_iter_folder, "distance_map", "test_distance_map.tif")
            check_folder(dirname(TEST_DISTANCE_MAP_OUTPUT))

            logger.info(f"Generating train distance map")
            TRAIN_SEGMENTATION_PATH  = args.train_segmentation_path
            TRAIN_DISTANCE_MAP_OUTPUT = join(current_iter_folder, "distance_map", "train_distance_map.tif")
            check_folder(dirname(TRAIN_DISTANCE_MAP_OUTPUT))

            # Create processes for test and train distance map generation
            test_process = Process(target=generate_distance_map, args=(TEST_SEGMENTATION_PATH, TEST_DISTANCE_MAP_OUTPUT, args.sigma))
            train_process = Process(target=generate_distance_map, args=(TRAIN_SEGMENTATION_PATH, TRAIN_DISTANCE_MAP_OUTPUT, args.sigma))
            
            # Start the processes
            test_process.start()
            train_process.start()
            
            # Wait for both processes to finish
            test_process.join()
            train_process.join()

            logger.info("Both test and train distance maps have been generated in parallel using multiprocessing.")

            logger.info("Generating labels view for iter 0")

            generate_labels_view(current_iter_folder, args.ortho_image, args.train_segmentation_path)
            
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
        
        compile_metrics(current_iter_folder, args)

        generate_labels_for_next_iteration(current_iter_folder, args)
        
        compile_component_metrics(current_iter_folder, args)
        
        generate_distance_map_for_next_iteration(current_iter_folder)

        #############################################

        delete_useless_files(current_iter_folder = current_iter_folder)
        
        generate_labels_view(current_iter_folder, args.ortho_image, args.train_segmentation_path)

        print_sucess("Distance map generated")
 


