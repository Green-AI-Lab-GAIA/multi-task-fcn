import gc
import logging
import os
import sys
from logging import Logger, getLogger
from os.path import dirname, join
from typing import Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torcheval.metrics import MulticlassF1Score
from tqdm import tqdm

ROOT_PATH = dirname(dirname(__file__))
sys.path.append(ROOT_PATH)
from src.deepvlab3 import DeepLabv3
from src.deepvlab3plus import DeepLabv3_plus
from src.deepvlab3plus_resnet9 import DeepLabv3Plus_resnet9
from src.metrics import evaluate_f1, evaluate_metrics
from src.utils import (AverageMeter, check_folder, get_device, plot_figures)
from src.io_operations import load_norm, read_yaml
import wandb
args = read_yaml(join(ROOT_PATH, "args.yaml"))

logger = getLogger("__main__")


def build_model(in_channels:list, 
                num_classes:int, 
                arch:Literal["deeplabv3_resnet50","deeplabv3_resnet101","deeplabv3+_resnet34","deeplabv3+_resnet18","deeplabv3+_resnet10", "deeplabv3+_resnet9"], 
                pretrained:bool, 
                psize:int,
                dropout_rate:float,
                batch_norm:bool,
                **kwargs)->nn.Module:


    # build model
    if arch == "deeplabv3_resnet50":
        model = DeepLabv3(
            in_channels = in_channels,
            num_classes = num_classes, 
            pretrained = pretrained, 
            dropout_rate = dropout_rate,
            batch_norm = batch_norm
        )
    
    elif arch == "deeplabv3_resnet101":
        model = DeepLabv3(
            in_channels = in_channels,
            num_classes = num_classes, 
            pretrained = pretrained, 
            dropout_rate = dropout_rate,
            batch_norm = batch_norm,
            resnet_arch = "resnet101"
        )

    elif arch.startswith("deeplabv3+"):
        # get resnet_depth
        resnet_depth = int(arch.split("resnet")[-1])
        model = DeepLabv3_plus(
            model_depth = resnet_depth,
            nb_class = num_classes,
            num_ch_1 = in_channels,
            psize = psize
        )


    elif arch == "deeplabv3+_resnet9":
        model = DeepLabv3Plus_resnet9(
            num_ch = in_channels,
            num_class = num_classes,
            psize = psize
        )
    
    else:
        raise ValueError(f"Unknown architecture {arch}.\nPlease choose among 'resunet' or 'deeplabv3_resnet50'")

    return model


def load_weights(model: nn.Module, checkpoint_file_path:str)-> nn.Module:
    """Load weights for model from checkpoint file
    If the checkpoint file doesnt exist, the model is loaded with random weights

    Parameters
    ----------
    model : nn.Module
        Pytorch builded model
    checkpoint_file_path : str
        Path to checkpoint file
    Returns
    -------
    nn.Module
        Pytorch model with loaded weights
    """
    DEVICE = get_device()

    # load weights
    if os.path.isfile(checkpoint_file_path):

        state_dict = torch.load(checkpoint_file_path, map_location=DEVICE, weights_only=False)

        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        
        # remove prefixe "module."
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}


        # Execute some verifications about the stat dict loaded from checkpoint
        for k, v in model.state_dict().items():
            
            if k not in list(state_dict):
                logger.info(f'key "{k}" could not be found in provided state dict')

            elif state_dict[k].shape != v.shape:
                logger.info(f'key "{k}" is of different shape in model and provided state dict')
                state_dict[k] = v
        

        # Set the model weights
        msg = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Load pretrained model with msg: {msg}")

    
    else:
        logger.info("No pretrained weights found => training with random weights")
    
    return model



def categorical_focal_loss(input:torch.Tensor, target:torch.Tensor, gamma = 2) -> torch.Tensor:
    """Partial Categorical Focal Loss Implementation based on the paper 
    "Multi-task fully convolutional network for tree species
    mapping in dense forests using small training
    hyperspectral data"



    Parameters
    ----------
    input : torch.Tensor
        The output from ResNet Model without the classification layer 
        shape: (batch, class, image_height, image_width)

    target : torch.Tensor
        The ground_truth segmentation with index for each class
        shape: (batch, image_height, image_width)
    
    Returns
    -------
    torch.Tensor
        The loss for each pixel in image
        shape : (batch, image_height, image_width)
    """

    prob = F.softmax(input, dim = 1)
    log_prob = F.log_softmax(input, dim = 1)

    return F.nll_loss(
        ((1 - prob) ** gamma) * log_prob, 
        target=target,
        reduction = "none"
    )




def train(train_loader:torch.utils.data.DataLoader, 
          model:nn.Module, 
          optimizer:torch.optim.Optimizer, 
          epoch:int, 
          lr_schedule:np.ndarray, 
          lambda_weight:float,
          activation_aux_layer:Literal["sigmoid", "relu", "gelu"] = "sigmoid",
          figures_path:str=None):
    """Train model for one epoch

    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        Data loader for training
    model : nn.Module
        Pytorch model
    optimizer : torch.optim.Optimizer
        Pytorch optimizer
    current_epoch : int
        Current epoch to update current learning rate 
    lr_schedule : np.array
        Learning rate schedule to update at each iteratio
    lambda_weight : float
        Weight for the auxiliary task
    figures_path : str
        Path to save sample figures 

    Returns
    -------
    tuple[int, float]
        Tuple with epoch and average loss
    """
    DEVICE = get_device()

    model.train()
    model.to(DEVICE)
    
    loss_avg = AverageMeter()
    
    if activation_aux_layer == "sigmoid":
        activation_aux_module = nn.Sigmoid().to(DEVICE)
    
    elif activation_aux_layer == "relu":
        activation_aux_module = nn.ReLU().to(DEVICE)
        
    elif activation_aux_layer == "gelu":
        activation_aux_module = nn.GELU().to(DEVICE)
    
    else:
        raise ValueError(f"Unknown activation function {activation_aux_layer}")
    
    # define functions
    soft = nn.Softmax(dim=1).to(DEVICE)
    
    # define losses
    # criterion = nn.NLLLoss(reduction='none').cuda()
    aux_criterion = nn.MSELoss(reduction='none').to(DEVICE)

    for it, (inp_img, depth, ref) in enumerate(tqdm(train_loader)):      

        # update learning rate
        iteration = epoch * len(train_loader) + it
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[iteration]

        # ============ forward pass and loss ... ============
        # compute model loss and output
        inp_img = inp_img.to(DEVICE, non_blocking=True)
        depth = depth.to(DEVICE, non_blocking=True)
        ref = ref.to(DEVICE, non_blocking=True)


        # create mask for the unknown pixels
        mask = torch.where(ref == 0, torch.tensor(0.0), torch.tensor(1.0))
        mask = mask.to(DEVICE, non_blocking=True)

        # ref data with the class id
        ref_copy = torch.where(mask > 0, torch.sub(ref, 1), 0)
        
        # Foward Passs
        out_batch = model(inp_img)
        depht_out = activation_aux_module(out_batch['aux'][:,0,:,:])
        
        loss1 = mask*categorical_focal_loss(out_batch["out"], ref_copy)

        loss2 = mask*aux_criterion(depht_out, depth)
        
        loss = (loss1 + lambda_weight*loss2)/2 
        loss = torch.sum(loss)/torch.sum(ref>0)

        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # performs updates using calculated gradients
        optimizer.step()
        
        # update the average loss
        loss_avg.update(loss)

        wandb.log({"train/loss": loss})
        
        gc.collect()

        # Evaluate summaries only once in a while
        if it % 50 == 0:
            with torch.no_grad():
                summary_batch = evaluate_metrics(soft(out_batch['out']), ref)
            
            logger.info(
                "Epoch: [{0}][{1}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Lr: {lr:.4f}".format(
                    epoch,
                    it,
                    loss=loss_avg,
                    lr=optimizer.param_groups[0]["lr"],
                )
            )
            logger.info(f"Accuracy:{summary_batch['Accuracy']}, avgF1:{summary_batch['avgF1']}")
            
        if it == 0 and figures_path is not None:
            # plot samples results for visual inspection
            with torch.no_grad():
                plot_figures(inp_img, 
                            ref, 
                            soft(out_batch['out']),
                            depth,
                            depht_out,
                            figures_path,
                            epoch,
                            'train')

            
    return (epoch, float(loss_avg.avg))


def eval(val_loader:torch.utils.data.DataLoader, 
          model:nn.Module, 
        ) -> Tuple[float, float]:
    """Function to evaluate model based on f1 score

    Parameters
    ----------
    val_loader : torch.utils.data.DataLoader
        Dataloader with validation set
    model : nn.Module
        Model to evaluate

    Returns
    -------
    float
        Average f1 score of the evaluation
    """
    
    # Validation
    model.eval()

    DEVICE = get_device()

    f1_avg = MulticlassF1Score(num_classes=args.nb_class, average="macro", device=DEVICE)
    f1_by_class_avg = MulticlassF1Score(num_classes=args.nb_class, average=None, device=DEVICE)
    
    soft = nn.Softmax(dim=1).to(DEVICE)

    with torch.no_grad():

        for (inp_img, depth, ref) in tqdm(val_loader):

            inp_img = inp_img.to(DEVICE, non_blocking=True)
            
            out_batch = model(inp_img)
            
            out_prob = soft(out_batch['out'])
            
            mask = (ref > 0)
            
            ref_masked = ref[mask]
            ref_masked = ref_masked - 1
            
            pred_class = torch.argmax(out_prob, dim=1)
            pred_class_masked = pred_class[mask]
            
            f1_avg.update(pred_class_masked, ref_masked)
            f1_by_class_avg.update(pred_class_masked, ref_masked)
         

    return f1_avg.compute().cpu().item(), f1_by_class_avg.compute().cpu().numpy()



def save_checkpoint(last_checkpoint_path:str, model:nn.Module, optimizer:torch.optim.Optimizer, epoch:int, best_acc:float, count_early:int, is_iter_finished=False):
    """Save model checkpoint at last_checkpoint_path

    Parameters
    ----------
    last_checkpoint_path : str
        File path to save checkpoint
    model : nn.Module
        Pytorch model at the end of epoch
    optimizer : torch.optim.Optimizer
        Pytorch optimizer at the end of epoch
    epoch : int
        Current epoch
    best_val : float
        Best accuracy achieved so far
    """
    save_dict = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "is_iter_finished": is_iter_finished,
        "best_val": best_acc,
        "count_early": count_early,
    }
    torch.save(save_dict, last_checkpoint_path)