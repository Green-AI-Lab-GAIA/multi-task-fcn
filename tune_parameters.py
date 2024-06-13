import numpy as np
from tqdm import tqdm
import wandb
import torch
import time
from os.path import join

from evaluation import predict_network
from main import get_learning_rate_schedule, train_epochs
from src.io_operations import get_image_metadata, read_tiff
from src.metrics import evaluate_metrics
from src.model import build_model, train, eval
from src.utils import check_folder, get_device, AttrDict
from src.dataset import DatasetForInference, DatasetFromCoord
from src.logger import create_logger
import torch.backends.cudnn as cudnn

logger = create_logger("tune_parameters", "tune_parameters.log")

DEVICE = get_device()

def train_epochs(config):
    SEGMENTATION_PATH = "/home/luiz.luz/multi-task-fcn/amazon_input_data/segmentation/train_set.tif"
    DISTANCE_MAP_PATH = "/home/luiz.luz/multi-task-fcn/6_amazon_data/iter_000/distance_map/train_distance_map.tif"
    ORTHOIMAGE_PATH = "/home/luiz.luz/multi-task-fcn/amazon_input_data/orthoimage/NOV_2017_FINAL_004.tif"
    
    current_time_seconds = time.time()
    
    wandb.log({"time": current_time_seconds})
    
    train_dataset = DatasetFromCoord(
        image_path=ORTHOIMAGE_PATH,
        segmentation_path=SEGMENTATION_PATH,
        distance_map_path=DISTANCE_MAP_PATH,
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
    
    model = build_model(
        ortho_image_shape,
        config.nb_class,  
        config.arch, 
        config.pretrained,
        psize = config.size_crops,
        dropout_rate = config.dropout_rate,
    )
    logger.info("Model built")
    
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
                                 batch_norm_layer=config.batch_norm_layer,
                                 )
        
        f1_avg, f1_by_class_avg = eval(val_loader, model)
        
        wandb.log({"f1_train_score": f1_avg})
        
        if (f1_avg - best_val) > 0.0009: 
            
            best_val = f1_avg
            count_early = 0

        else:
            count_early += 1
    
    # free up cuda memory
    cudnn.benchmark = False
    torch.cuda.empty_cache()
    
    ######### INFERENCE ##########
    logger.info("Start inference")
    test_dataset = DatasetForInference(
        ORTHOIMAGE_PATH,
        config.size_crops,
        config.overlap,
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
    )
    
    logger.info("Metrics evaluation")
    GROUND_TRUTH_TEST = read_tiff("/home/luiz.luz/multi-task-fcn/amazon_input_data/segmentation/test_set.tif")
    metrics_test = evaluate_metrics(pred_class, GROUND_TRUTH_TEST)

    wandb.log({
        "f1_score": metrics_test["avgF1"],
        "precision": metrics_test["avgPrec"],
        "recall": metrics_test["avgRec"],
        "accuracy": metrics_test["Accuracy"],
    })
    
    logger.info("Time spent")
    wandb.log({"time_spent": time.time() - current_time_seconds})
    logger.info(f"Time spent: {(time.time() - current_time_seconds)/60} minutes")
    
    logger.info("Save model")
    torch.save(model.state_dict(), 
               join(wandb.run.dir, "model.pth"))
    
    # free up cuda memory
    cudnn.benchmark = False
    torch.cuda.empty_cache()
    
    
def tune():

    with wandb.init():
        config = wandb.config
        train_epochs(config)
        

sweep_config = {
    'method': 'random',
    "name": "tune_parameters",
    'metric': {
        "name":"f1_score",
        "goal":"maximize"
    },
    'parameters':{
        "overlap":{
            "values":[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        },
        "copy_and_paste_augmentation":{
            "values":[True, False]
        },
        "dropout_rate":{
            "values":[0.1,0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        },
        "size_crops":{
            "values":[512]
        },
        "batch_norm_layer":{
            "values":[True, False]
        },
        "lambda_weight":{
            "values":[0.5,1,2,3,4,5]
        },
        "pretrained":{
            "values":[True, False]
        },
        "standardize":{
            "values":[True, False]
        },
        "augment":{
            "value":True
        },
        "samples":{
            "values":[2500, 5000, 10000]
        },
        "batch_size":{
            "value":16
        },
        "nb_class":{
            "value":17
        },
        "arch":{
            "value":"deeplabv3_resnet50"
        },
        "base_lr":{
            "value":0.01
        },
        "final_lr":{
            "value":0.0001
        },
        "weight_decay":{
            "value":1e-06
        },
        "warmup_epochs":{
            "value":5
        },
        "start_warmup":{
            "value":0
        },
        "epochs":{
            "value":30
        },
        "patience":{
            "value":5
        },
        "num_workers":{
            "value":8
        }
    
    }
}

def sweep():
    
    sweep_id = "z8k7yte5"
    
    logger.info(f"Sweep id: {sweep_id}")
    
    wandb.agent(sweep_id, function=tune, count=50)


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
    config["size_crops"] = 512
    
    train_epochs(config)
    
if __name__ == "__main__":
    sweep()