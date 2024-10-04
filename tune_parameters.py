import numpy as np
from tqdm import tqdm
import wandb
import torch
import time

from os.path import join, abspath, dirname

from evaluation import predict_network
from main import get_learning_rate_schedule, train_epochs
from src.deepvlab3 import DeepLabv3
from src.io_operations import get_image_metadata, read_tiff, read_yaml, save_yaml
from src.metrics import evaluate_metrics
from src.model import build_model, train, eval
from src.utils import check_folder, get_device, AttrDict
from src.dataset import DatasetForInference, DatasetFromCoord
from src.logger import create_logger
import argparse
import torch.backends.cudnn as cudnn
import yaml
import gc

logger = create_logger("tune_parameters", "tune_parameters.log")

DEVICE = get_device()

def train_epochs(config):

    SEGMENTATION_PATH = abspath(config.segmentation_path)
    DISTANCE_MAP_PATH = abspath(config.distance_map_path)
    ORTHOIMAGE_PATH = abspath(config.orthoimage_path)
    GROUND_TRUTH_TEST_PATH = abspath(config.ground_truth_test_path)
    
    GROUND_TRUTH_TEST = read_tiff(GROUND_TRUTH_TEST_PATH)
    
    current_time_seconds = time.time()
    
    wandb.log({"time": current_time_seconds})
    
    train_dataset = DatasetFromCoord(
        image_path=config.orthoimage_path,
        segmentation_path=config.segmentation_path,
        distance_map_path=config.distance_map_path,
        samples=config.samples,
        augment=config.augment,
        crop_size=config.size_crops,
        copy_paste_augmentation=config.copy_and_paste_augmentation
    )
    
    if config.standardize:
        train_dataset.standardize_image_channels()
    

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )

    # LOAD VALIDATION SET
    val_dataset = DatasetFromCoord(
        image_path=ORTHOIMAGE_PATH,
        segmentation_path=SEGMENTATION_PATH,
        distance_map_path=DISTANCE_MAP_PATH,
        samples=config.samples//3,
        augment=config.augment,
        crop_size=config.size_crops,
        copy_paste_augmentation=False
    )
    
    if config.standardize:
        val_dataset.standardize_image_channels()

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = config.batch_size,
        num_workers = 8,
        pin_memory = True,
        drop_last = True,
        shuffle = True,
    )
    logger.info("Data loaded")

    orthoimage_meta = get_image_metadata(ORTHOIMAGE_PATH)
    ortho_image_shape = (orthoimage_meta["count"], orthoimage_meta["height"], orthoimage_meta["width"])
    
    ###### BUILD MODEL ########    
    model = DeepLabv3(
        in_channels = ortho_image_shape[0],
        num_classes = config.nb_class,
        pretrained = config.pretrained,
        dropout_rate = config.dropout_rate,
        batch_norm = config.batch_norm_layer,
    )
    logger.info("Model built")
    print(config.weight_decay)
    ###### BULD OPTMIZER #######
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.base_lr,
        momentum=0.9,
        weight_decay=config.weight_decay,
    )

    # define how the learning rate will be changed in the training process.
    lr_schedule = get_learning_rate_schedule(
        train_loader, 
        config.base_lr, 
        config.final_lr, 
        config.epochs, 
        config.warmup_epochs, 
        config.start_warmup
    )
    
    ######## TRAIN MODEL #########
    logger.info("Start training")
    cudnn.benchmark = True
    count_early = 0
    best_val = 0
    for epoch in range(0, config.epochs):
        logger.info(f"Epoch {epoch}")

        if count_early == config.patience:
            logger.info("Early stopping")
            break
        
        np.random.shuffle(train_loader.dataset.coords)
        np.random.shuffle(val_loader.dataset.coords)


        epoch, scores_tr = train(train_loader=train_loader, 
                                 model=model, 
                                 optimizer=optimizer, 
                                 epoch=epoch, 
                                 lr_schedule=lr_schedule, 
                                 figures_path=None, 
                                 lambda_weight=config.lambda_weight,
                                 )
        
        f1_avg, f1_by_class_avg = eval(val_loader, model)
        
        wandb.log({"f1_train_score": f1_avg})
        
        if (f1_avg - best_val) > config.early_stopping_threshold:
            
            best_val = f1_avg
            count_early = 0

        else:
            count_early += 1
    
    # free up cuda memory
    cudnn.benchmark = False
    torch.cuda.empty_cache()
    
    # free up memory
    del train_dataset, train_loader, val_dataset, val_loader
    gc.collect()
    
    ######### INFERENCE ##########
    logger.info("Start inference")
    for num, overlap in enumerate(config.overlap):
        
        logger.info(f"Inference with: {overlap}")
        
        test_dataset = DatasetForInference(
            ORTHOIMAGE_PATH,
            config.size_crops,
            overlap,
        )
        if config.standardize:
            test_dataset.standardize_image_channels()

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.batch_size*4,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
        )
        
        prob_map, pred_class, depth_map = predict_network(
            ortho_image_shape = ortho_image_shape,
            dataloader = test_loader,
            model = model,
            num_classes = config.nb_class,
            activation_aux_layer = config.activation_aux_layer,    
        )
        
        del pred_class
        
        torch.cuda.empty_cache()
        gc.collect()
        
        logger.info("Predictions done")
        
        if num == 0:
            prob_map_final = np.uint8(prob_map * np.float16(255)) // np.uint8(len(config.overlap))
            del prob_map, depth_map
            
        else:
            prob_map_final += np.uint8(prob_map * np.float16(255)) // np.uint8(len(config.overlap))
            del prob_map, depth_map
        
        gc.collect()
    
    
    logger.info("Computing class with highest probability")
    pred_class_final = np.argmax(prob_map_final, axis=-1)
    
    logger.info("Metrics evaluation")
    
    metrics_test = evaluate_metrics(pred_class_final, GROUND_TRUTH_TEST)

    wandb.log({"f1_score": metrics_test["avgF1"]})
    wandb.log({"precision": metrics_test["avgPre"]})
    wandb.log({"recall": metrics_test["avgRec"]})
    wandb.log({"accuracy": metrics_test["Accuracy"]})
    
    logger.info("Time spent")
    wandb.log({"time_spent": time.time() - current_time_seconds})
    logger.info(f"Time spent: {(time.time() - current_time_seconds)/60} minutes")
    
    logger.info("Save model")
    
    folder_to_save = join(dirname(__file__), "results", f"model_{metrics_test['avgF1']:06.1f}")
    check_folder(folder_to_save)
    
    torch.save(model.state_dict(), join(folder_to_save, "model.pth"))
    
    save_yaml(dict(config), join(folder_to_save, "config.yaml"))
               
    # free up cuda memory
    cudnn.benchmark = False
    torch.cuda.empty_cache()
    
    
def tune():

    with wandb.init():
        config = wandb.config
        train_epochs(config)
        

def sweep():
    
    logger.info(f"Sweep id: {SWEEP_ID}")
    
    wandb.agent(SWEEP_ID, function=tune, count=50, project="tune_parameters")


def test_code(sweep_config):
    # init wandb offline
    wandb.init(mode="disabled")
    
    config = AttrDict()
    for param, values in sweep_config['parameters'].items():
        if "value" in values:
            config[param] = values["value"]
        else:
            config[param] = values["values"][0]

    config["epochs"] = 1
    config["samples"] = 100
    config["size_crops"] = 256
    config["overlap"] = [0.1, 0.5]
    config["num_workers"] = 8
    
    config["orthoimage_path"] = "amazon_input_data/orthoimage/orthoimage.tif"
    config["distance_map_path"] = "amazon_input_data/distance_map/train_distance_map.tif"
    config["segmentation_path"] = "amazon_input_data/segmentation/train_set.tif"
    config["ground_truth_test_path"] = "amazon_input_data/segmentation/test_set.tif"
    
    train_epochs(config)
    

if __name__ == "__main__":  
    
    SWEEP_ID_FILE = "sweep_id.yaml"
    SWEEP_FILE = "tune_parameters.yaml"
    
    sweep_config = read_yaml(SWEEP_FILE)
    
    with open(SWEEP_ID_FILE, "r") as file:
        SWEEP_ID = yaml.safe_load(file)["SWEEP_ID"]
    
    try:
        wandb.agent(SWEEP_ID, function=tune, count=50, project="tune_parameters")
    
    
    except wandb.Error as e:
        
        SWEEP_ID = wandb.sweep(sweep_config, project="tune_parameters")
        
        # save sweep id
        with open(SWEEP_ID_FILE, "w") as file:
            yaml.dump({"SWEEP_ID": SWEEP_ID}, file)
            
        wandb.agent(SWEEP_ID, function=tune, count=50, project="tune_parameters")
