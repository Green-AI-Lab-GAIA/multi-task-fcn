import sys
from os.path import dirname, join
from typing import Literal, Union

import numpy as np
import torch
from skimage.measure import label
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                             jaccard_score, precision_score, recall_score)

ROOT_PATH = dirname(dirname(__file__))
sys.path.append(ROOT_PATH)

from src.io_operations import read_yaml

args = None # Removed hardcoded loading of args.yaml


def evaluate_metrics(pred:Union[np.ndarray, torch.Tensor], gt:Union[np.ndarray, torch.Tensor], num_class:int = args.nb_class) -> dict:
    """Calculte the metrics:
    Accuracy, F1 Score, Precision, and Recall.
    Calculate based on the pred with highest probability and the gt - ground truth segmentation.

    Parameters
    ----------
    pred : Union[np.ndarray, torch.Tensor]
        Tensor with shape [row, cols, num_class]
        This tensor has one matrix of confidence for each label class.
        The function uses the class with highest probability/confidence to evaluate metric.

    gt : Union[np.ndarray, torch.Tensor]
        Tensor with shape [row, cols]
        This tensor has the ground truth segmentation matrix
    val : int, optional
        , by default 0

    Returns
    -------
    dict
        Return the metrics:
        - Accuracy, as Accuracy
        - F1 Score, as avgF1
        - Precision, as avgPre
        - Recall, as avgRec
    """

    accu_criteria = dict()

    if type(pred).__module__ != np.__name__:
        pred = pred.data.cpu().numpy()
    
    if type(gt).__module__ != np.__name__:
        gt = gt.data.cpu().numpy()


    if  len(pred.shape) >= 3 and pred.shape[-1] > 1:
        # Get the class with highest probability
        pred = np.argmax(pred, axis=1)

    else:
        pass

    # Create to just the place where the ground_truth_segmentation is non zero
    mask = np.where(gt>0)


    comp_pred = label(pred)
    
    comp_pred_in_test = comp_pred[gt > 0]
    comp_pred_in_test = comp_pred_in_test[comp_pred_in_test!=0].copy()

    pred_in_test = np.where(np.isin(comp_pred, comp_pred_in_test), pred+1, 0)

    list_of_labels = list(range(1, num_class + 1) )
    
    iou_score = jaccard_score(gt.flatten(),
                              pred_in_test.flatten(),
                              average = "macro",
                              labels=list_of_labels,
                              )

  
    accu_criteria["avgIOU"] = float(iou_score)*100

    # Apply Mask
    gt = gt[mask][:]

    pred = pred[mask][:]+1

    #### CALCULATE METRICS WITH SKLEARN ####
    accuracy = accuracy_score(gt, pred)*100
    accu_criteria["Accuracy"] = float(np.round(accuracy,2))
    

    accu_criteria["avgF1"] = float(f1_score(gt, pred, average="macro", zero_division=True, labels =  list_of_labels))*100
    accu_criteria["avgPre"] = float(precision_score(gt, pred, average="macro", zero_division=True, labels = list_of_labels))*100
    accu_criteria["avgRec"] = float(recall_score(gt, pred, average="macro", zero_division=True, labels = list_of_labels))*100
    
    accu_criteria["F1"] = (f1_score(gt, pred, average=None, zero_division=True, labels = list_of_labels)*100).tolist()
    accu_criteria["Pre"] = (precision_score(gt, pred, average=None, zero_division=True, labels = list_of_labels)*100).tolist()
    accu_criteria["Rec"] = (recall_score(gt, pred, average=None, zero_division=True, labels = list_of_labels)*100).tolist()

    accu_criteria["KappaScore"] = float(cohen_kappa_score(gt, pred, labels=list_of_labels))*100

    return accu_criteria

def evaluate_f1(pred:Union[np.ndarray, torch.Tensor], 
                gt:Union[np.ndarray, torch.Tensor], 
                num_class:int = args.nb_class,
                average:Literal[None, "micro", "macro", "weighted"] = "macro") -> float:
    """Calculte the F1 Score macro average

    Parameters
    ----------
    pred : Union[np.ndarray, torch.Tensor]
        Tensor with shape [row, cols, num_class]
        This tensor has one matrix of confidence for each label class.
        The function uses the class with highest probability/confidence to evaluate metric.

    gt : Union[np.ndarray, torch.Tensor]
        Tensor with shape [row, cols]
        This tensor has the ground truth segmentation matrix
    
    average : [None, "micro", "macro", "weighted"]
        The average type to run calculate f1 score

    Returns
    -------
    float
        F1 Score
    """

    if type(pred).__module__ != np.__name__:
        pred = pred.data.cpu().numpy()
    
    if type(gt).__module__ != np.__name__:
        gt = gt.data.cpu().numpy()


    if  len(pred.shape) >= 3 and pred.shape[-1] > 1:
        # Get the class with highest probability
        pred = np.argmax(pred, axis=1)

    else:
        pass

    # Create to just the place where the ground_truth_segmentation is non zero
    mask = np.where(gt>0)

    # Apply Mask
    gt = gt[mask][:]

    pred = pred[mask][:]+1

    #### CALCULATE METRICS WITH SKLEARN ####
    f1 = f1_score(gt, pred, average=average, zero_division=True, labels = list(range(1, num_class + 1) ))

    return f1

def evaluate_component_metrics(ground_truth_labels:np.ndarray, predicted_labels:np.ndarray, num_class:int = None, average:str = "macro")->dict:
    """Evaluate the metrics of the non zero labels in ground_truth labels

    Parameters
    ----------
    ground_truth_labels : np.ndarray
        The true label class
    predicted_labels : np.ndarray
        The predicted label class
    num_class : int, optional
        The number of non zero classes, by default None
    average : str, optional
        The method to compute the metrics, by default "macro"

    Returns
    -------
    dict
        Accuracy, F1-Score, Precision, and Recall Score
    """
    # compute metrics ignoring 0 class
    if num_class != None:
        labels = list(range(1, num_class+1))

    else:
        labels = np.unique(ground_truth_labels[np.nonzero(ground_truth_labels)])

    # mask for non zero ground_truth_labels
    mask = ground_truth_labels > 0     

    gt_labels = ground_truth_labels[mask]

    pred_labels = predicted_labels[mask]

    # Compute metrics
    metrics = dict()

    metrics["Accuracy"] = accuracy_score(gt_labels, pred_labels)*100

    metrics['avgF1'] = f1_score(gt_labels, 
                                pred_labels, 
                                average = average, 
                                zero_division = True, 
                                labels = labels)*100
    
    metrics["avgPrec"] = precision_score(gt_labels, 
                                        pred_labels, 
                                        average = average, 
                                        zero_division = True, 
                                        labels = labels)*100
    
    metrics["avgRec"] = recall_score(gt_labels, 
                                        pred_labels, 
                                        average = average, 
                                        zero_division = True, 
                                        labels = labels)*100

    return metrics


def evaluate_f1_by_component(
    pred: Union[np.ndarray, torch.Tensor],
    gt: Union[np.ndarray, torch.Tensor],
    num_class: int = None,
    average: Literal[None, "micro", "macro", "weighted"] = "macro"
) -> dict:
    """
    Calculate F1-score based on components (not pixels).
    
    For each annotated component in the ground truth, the predicted class
    is determined by the most common (majority vote) class within that component.
    
    Parameters
    ----------
    pred : Union[np.ndarray, torch.Tensor]
        Predicted segmentation map with shape [rows, cols].
        Values should be class indices (0 = background, 1+ = classes).
    gt : Union[np.ndarray, torch.Tensor]
        Ground truth segmentation map with shape [rows, cols].
        Each unique non-zero value represents a different class.
    num_class : int, optional
        Number of classes (excluding background). If None, inferred from gt.
    average : Literal[None, "micro", "macro", "weighted"], optional
        Averaging method for F1-score. Default is "macro".
    
    Returns
    -------
    dict
        Dictionary containing:
        - avgF1_component: Average F1-score by component (float)
        - F1_component: F1-score per class (list, if average=None)
        - avgPrec_component: Average Precision by component (float)
        - avgRec_component: Average Recall by component (float)
        - Accuracy_component: Accuracy by component (float)
        - n_components: Number of components evaluated (int)
    """
    # Convert to numpy if needed
    if type(pred).__module__ != np.__name__:
        pred = pred.data.cpu().numpy()
    if type(gt).__module__ != np.__name__:
        gt = gt.data.cpu().numpy()
    
    # Ensure pred is class labels (shift if needed based on model output)
    # Model outputs are 0-indexed (0..num_class-1) while gt uses 1-indexed classes
    pred = pred.copy()
    if num_class is None:
        # infer classes from gt ignoring background
        num_class = int(gt.max())
    # Shift only when predictions look 0-indexed and gt has background
    if (gt.min() == 0) and (pred.min() == 0) and (pred.max() <= num_class - 1):
        pred = pred + 1
    
    # Get connected components from ground truth
    gt_components = label(gt > 0)
    unique_components = np.unique(gt_components)
    unique_components = unique_components[unique_components > 0]  # Remove background
    
    # Define labels for metrics calculation
    if num_class is not None:
        labels = list(range(1, num_class + 1))
    else:
        labels = np.unique(gt[gt > 0]).tolist()
    
    if len(unique_components) == 0:
        return {
            "avgF1_component": 0.0,
            "F1_component": [0.0] * len(labels),
            "avgPrec_component": 0.0,
            "avgRec_component": 0.0,
            "Accuracy_component": 0.0,
            "n_components": 0
        }
    
    # For each component, get true class and predicted class (majority vote)
    gt_labels_per_component = []
    pred_labels_per_component = []
    
    for comp_id in unique_components:
        comp_mask = gt_components == comp_id
        
        # True class: the class value in ground truth for this component
        # All pixels in a component should have the same gt class
        gt_class = gt[comp_mask].max()
        
        # Predicted class: most common non-zero class in prediction within this component
        pred_in_comp = pred[comp_mask]
        pred_nonzero = pred_in_comp[pred_in_comp > 0]
        
        if len(pred_nonzero) == 0:
            # No prediction in this component - assign 0 (will be counted as error)
            pred_class = 0
        else:
            values, counts = np.unique(pred_nonzero, return_counts=True)
            pred_class = values[np.argmax(counts)]
        
        gt_labels_per_component.append(gt_class)
        pred_labels_per_component.append(pred_class)
    
    gt_labels_per_component = np.array(gt_labels_per_component)
    pred_labels_per_component = np.array(pred_labels_per_component)
    
    # Compute metrics
    metrics = dict()
    
    metrics["n_components"] = len(unique_components)
    
    metrics["Accuracy_component"] = float(accuracy_score(
        gt_labels_per_component, 
        pred_labels_per_component
    )) * 100
    
    metrics["avgF1_component"] = float(f1_score(
        gt_labels_per_component,
        pred_labels_per_component,
        average=average,
        zero_division=0,
        labels=labels
    )) * 100
    
    metrics["F1_component"] = (f1_score(
        gt_labels_per_component,
        pred_labels_per_component,
        average=None,
        zero_division=0,
        labels=labels
    ) * 100).tolist()
    
    metrics["avgPrec_component"] = float(precision_score(
        gt_labels_per_component,
        pred_labels_per_component,
        average=average,
        zero_division=0,
        labels=labels
    )) * 100
    
    metrics["avgRec_component"] = float(recall_score(
        gt_labels_per_component,
        pred_labels_per_component,
        average=average,
        zero_division=0,
        labels=labels
    )) * 100
    
    return metrics


def evaluate_miou_per_polygon(
    pred: Union[np.ndarray, torch.Tensor],
    gt: Union[np.ndarray, torch.Tensor],
    num_class: int = None,
) -> dict:
    """
    Calculate Mean IoU per polygon (instance-level, class-aware, masked).
    
    For each connected component ("polygon") in the ground truth for a given
    class, find the predicted component of the SAME class with the highest
    pixel intersection. Then compute IoU between that GT polygon and the matched
    prediction polygon, but ONLY over pixels where the ground truth is annotated
    (gt > 0). This avoids penalizing predictions in unknown/unannotated areas
    (where gt == 0 means "unknown").
    
    Parameters
    ----------
    pred : Union[np.ndarray, torch.Tensor]
        Predicted segmentation map with shape [rows, cols].
        Values should be class indices (0 = background, 1+ = classes).
    gt : Union[np.ndarray, torch.Tensor]
        Ground truth segmentation map with shape [rows, cols].
        0 means "unknown / not annotated". Non-zero values are classes (1..K).
    num_class : int, optional
        Number of classes (excluding background). If None, inferred from gt.
    
    Returns
    -------
    dict
        Dictionary containing:
        - avgMIoU: Mean IoU across all polygons (float, percentage)
        - MIoU_per_class: Mean IoU per class (list)
        - IoU_per_polygon: IoU for each polygon (list)
        - n_polygons_evaluated: Total number of GT polygons (int)
        - n_polygons_matched: Number of GT polygons with at least one overlapping prediction (int)
    """
    # Convert to numpy if needed
    if type(pred).__module__ != np.__name__:
        pred = pred.data.cpu().numpy()
    if type(gt).__module__ != np.__name__:
        gt = gt.data.cpu().numpy()
    
    # Ensure pred is class labels (shift if needed based on model output)
    pred = pred.copy()
    if num_class is None:
        num_class = int(gt.max())
    
    # Shift only when predictions look 0-indexed and gt has background
    if (gt.min() == 0) and (pred.min() == 0) and (pred.max() <= num_class - 1):
        pred = pred + 1
    
    valid_mask = gt > 0  # only annotated pixels are valid for union/intersection

    # Early exit if there are no annotated polygons at all
    if not np.any(valid_mask):
        return {
            "avgMIoU": 0.0,
            "MIoU_per_class": [0.0] * num_class,
            "IoU_per_polygon": [],
            "n_polygons_evaluated": 0,
            "n_polygons_matched": 0,
        }
    
    # Store IoU per polygon and per class
    iou_per_polygon = []
    iou_per_class = {c: [] for c in range(1, num_class + 1)}
    n_matched = 0

    # Process per class to ensure class-aware polygons and matching
    for c in range(1, num_class + 1):
        # Connected components for this class only
        gt_components_c = label(gt == c)
        pred_components_c = label(pred == c)

        unique_gt_components_c = np.unique(gt_components_c)
        unique_gt_components_c = unique_gt_components_c[unique_gt_components_c > 0]

        if len(unique_gt_components_c) == 0:
            continue

        for gt_comp_id in unique_gt_components_c:
            gt_mask = gt_components_c == gt_comp_id  # subset of valid_mask by definition

            # Candidate predicted components of the same class that overlap this GT polygon
            overlapping_pred_ids = np.unique(pred_components_c[gt_mask])
            overlapping_pred_ids = overlapping_pred_ids[overlapping_pred_ids > 0]

            if len(overlapping_pred_ids) == 0:
                iou_per_polygon.append(0.0)
                iou_per_class[c].append(0.0)
                continue

            # Pick the predicted polygon with the largest intersection area
            best_pred_id = None
            best_intersection = -1
            for pred_comp_id in overlapping_pred_ids:
                pred_mask = pred_components_c == pred_comp_id
                inter = int(np.sum(gt_mask & pred_mask))
                if inter > best_intersection:
                    best_intersection = inter
                    best_pred_id = int(pred_comp_id)

            pred_mask_best = pred_components_c == best_pred_id

            # IoU computed only on annotated pixels (valid_mask)
            intersection = float(np.sum(gt_mask & pred_mask_best))  # already within valid
            union = float(np.sum((gt_mask | pred_mask_best) & valid_mask))
            iou = (intersection / union) if union > 0 else 0.0

            iou_per_polygon.append(iou)
            iou_per_class[c].append(iou)

            if best_intersection > 0:
                n_matched += 1
    
    # Calculate mean IoU
    avg_miou = np.mean(iou_per_polygon) * 100 if iou_per_polygon else 0.0
    
    # Calculate mean IoU per class
    miou_per_class = []
    for c in range(1, num_class + 1):
        if iou_per_class[c]:
            miou_per_class.append(float(np.mean(iou_per_class[c]) * 100))
        else:
            miou_per_class.append(0.0)
    
    return {
        "avgMIoU": float(avg_miou),
        "MIoU_per_class": miou_per_class,
        "IoU_per_polygon": [float(x * 100) for x in iou_per_polygon],
        "n_polygons_evaluated": len(iou_per_polygon),
        "n_polygons_matched": n_matched,
    }


def get_miou_polygon_data(
    pred: Union[np.ndarray, torch.Tensor],
    gt: Union[np.ndarray, torch.Tensor],
    num_class: int = None,
) -> tuple:
    """
    Extract per-polygon IoU data for global aggregation.
    
    This function returns the raw data needed to compute global mIoU
    across multiple regions, using the same definition as
    `evaluate_miou_per_polygon` (class-aware matching by max intersection,
    IoU masked to annotated pixels where gt > 0).
    
    Parameters
    ----------
    pred : Union[np.ndarray, torch.Tensor]
        Predicted segmentation map
    gt : Union[np.ndarray, torch.Tensor]
        Ground truth segmentation map
    num_class : int, optional
        Number of classes
    
    Returns
    -------
    tuple
        (iou_per_polygon, gt_classes_per_polygon) where:
        - iou_per_polygon: list of IoU values for each GT polygon
        - gt_classes_per_polygon: list of GT class for each polygon
    """
    # Convert to numpy if needed
    if type(pred).__module__ != np.__name__:
        pred = pred.data.cpu().numpy()
    if type(gt).__module__ != np.__name__:
        gt = gt.data.cpu().numpy()
    
    pred = pred.copy()
    if num_class is None:
        num_class = int(gt.max())
    
    if (gt.min() == 0) and (pred.min() == 0) and (pred.max() <= num_class - 1):
        pred = pred + 1

    valid_mask = gt > 0
    
    iou_per_polygon = []
    gt_classes_per_polygon = []

    if not np.any(valid_mask):
        return iou_per_polygon, gt_classes_per_polygon

    for c in range(1, num_class + 1):
        gt_components_c = label(gt == c)
        pred_components_c = label(pred == c)

        unique_gt_components_c = np.unique(gt_components_c)
        unique_gt_components_c = unique_gt_components_c[unique_gt_components_c > 0]

        for gt_comp_id in unique_gt_components_c:
            gt_mask = gt_components_c == gt_comp_id
            gt_classes_per_polygon.append(int(c))

            overlapping_pred_ids = np.unique(pred_components_c[gt_mask])
            overlapping_pred_ids = overlapping_pred_ids[overlapping_pred_ids > 0]

            if len(overlapping_pred_ids) == 0:
                iou_per_polygon.append(0.0)
                continue

            best_pred_id = None
            best_intersection = -1
            for pred_comp_id in overlapping_pred_ids:
                pred_mask = pred_components_c == pred_comp_id
                inter = int(np.sum(gt_mask & pred_mask))
                if inter > best_intersection:
                    best_intersection = inter
                    best_pred_id = int(pred_comp_id)

            pred_mask_best = pred_components_c == best_pred_id

            intersection = float(np.sum(gt_mask & pred_mask_best))
            union = float(np.sum((gt_mask | pred_mask_best) & valid_mask))
            iou = (intersection / union) if union > 0 else 0.0

            iou_per_polygon.append(iou)
    
    return iou_per_polygon, gt_classes_per_polygon


if __name__ == "__main___":
    import os

    import yaml

    from utils import read_tiff, read_yaml

    args = read_yaml("../args.yaml")

    current_iter_folder = "/home/luiz/multi-task-fcn/4.3_version_data"

    GROUND_TRUTH_PATH = os.path.join(args.data_path, args.test_segmentation_path)
    ground_truth_test = read_tiff(GROUND_TRUTH_PATH)

    PRED_PATH = os.path.join(current_iter_folder, "raster_prediction", f"join_class_{np.sum(args.overlap)}.TIF")
    predicted_seg = read_tiff(PRED_PATH)

    
    metrics = evaluate_metrics(predicted_seg, ground_truth_test)


    with open(os.path.join(current_iter_folder,'store_file.yaml'), 'w') as file:

        documents = yaml.dump(metrics, file)
        