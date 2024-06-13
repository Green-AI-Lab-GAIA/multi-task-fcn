from logging import getLogger
from os.path import dirname, join
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops_table, regionprops
from shapely.geometry import Polygon

from tqdm import tqdm

from src.io_operations import (fix_relative_paths, load_args, read_tiff,
                               read_yaml, get_image_metadata)

from src.utils import convert_to_minor_numeric_type

args = load_args(join(dirname(__file__), "args.yaml"))

logger = getLogger("__main__")

def set_same_class_at_component(label_img:np.ndarray):

    components_img = label(np.where(label_img>0, 1, 0))
    
    # remove component 0
    component_ids = np.unique(components_img)
    component_ids = component_ids[np.nonzero(component_ids)]

    for component_id in tqdm(component_ids):
        filter_component = components_img == component_id

        ids, counts = np.unique(label_img[filter_component], return_counts=True)

        most_commom_id = ids[np.argmax(counts)]

        label_img[filter_component] = most_commom_id
    


def get_components_stats(components_img: np.ndarray, label_img: np.ndarray):
    """
    Calculate various geometric and intensity statistics for connected components in an image.

    This function computes statistics such as area, convex area, bounding box area, extent, solidity, eccentricity,
    orientation, centroid, bounding box, label, and mean intensity (tree_type) for each connected component in the
    input image.

    Parameters
    ----------
    components_img : np.ndarray
        Image containing connected components represented by unique integer labels.
    label_img : np.ndarray
        Labeled image corresponding to the connected components in components_img.

    Returns
    -------
    pd.DataFrame
        DataFrame containing computed statistics for each connected component. The component labels are set as
        the index, and the "intensity_mean" column is renamed to "tree_type".

    Notes
    -----
    The input images are expected to have components labeled with unique integers. The resulting DataFrame
    includes geometric and intensity statistics for each labeled component.
    """

    properties = [
        "area",
        "convex_area",
        "bbox_area",
        "extent",
        "solidity",
        "eccentricity",
        "orientation",
        "centroid",
        "bbox",
        "label",
        "intensity_mean",
    ]

    components_stats = pd.DataFrame(
        regionprops_table(
            components_img, properties=properties, intensity_image=label_img
        )
    )
    # set component id as index
    components_stats.set_index("label", inplace=True)
    # tree_type
    components_stats.rename(columns={"intensity_mean": "tree_type"}, inplace=True)
    components_stats["tree_type"] = components_stats["tree_type"].astype(int)

    return components_stats



def remove_components_by_index(
    component_ids_to_remove: np.ndarray,
    components_img: np.ndarray,
    label_img: np.ndarray,
):
    """
    Remove components from labeled and component images by their indices.

    Given an array of component indices to be removed, this function sets the corresponding pixels in both the
    components_img and label_img arrays to zero, effectively removing the specified components.

    Parameters
    ----------
    component_ids_to_remove : np.ndarray
        Array of component indices to be removed.
    components_img : np.ndarray
        Image containing connected components represented by unique integer labels.
    label_img : np.ndarray
        Labeled image corresponding to the connected components in components_img.

    Returns
    -------
    None

    Notes
    -----
    This function modifies the input arrays components_img and label_img in place by setting the pixels
    corresponding to the specified component indices to zero.
    """
    
    id_selection = np.argwhere(np.isin(components_img, component_ids_to_remove))
    
    components_img[id_selection[:, 0], id_selection[:, 1]] = 0
    label_img[id_selection[:, 0], id_selection[:, 1]] = 0
    


def get_labels_delta(
    old_label_img: np.ndarray, new_label_img: np.ndarray
) -> np.ndarray:
    """Get the components labels that are in the new image but not in the old image

    Parameters
    ----------
    old_label_img : np.array
        New image map with the label classes (tree type)

    new_label_img : np.array
        New image map with the label classes (tree type)

    """
    
    # Give numbers to each individual component on image
    new_components_img = label(new_label_img)
    old_components_img = label(old_label_img)

    # Select the components in new_img that have some shared pixel with the old_label_img
    new_components_in_old  = np.unique( 
        new_components_img[(old_components_img > 0)]
    )
    new_components_in_old = new_components_in_old[np.nonzero(new_components_in_old)]
    # Select the components in new_img that doesnt have any share pixel with old_label
    new_components_only_in_new = np.unique(
        new_components_img[~np.isin(new_components_img, new_components_in_old) & (new_components_img!=0)]
    )

    # Create a label delta img with components that doenst have any pixel in old_segmentation
    label_delta = np.where(np.isin(new_components_img, new_components_only_in_new), new_label_img, 0)
    
    # Select the components in new_label_img that share only 10% of the pixels with the old_segmentation
    for idx in tqdm(new_components_in_old):
        component_mask = new_components_img == idx
        # if more than 90% of the area is empty it will be added to the new predicted sample
        if np.mean(old_components_img[component_mask] == 0) > 0.9:
            
            # count labels
            unique_labels, count_labels = np.unique(
                new_label_img[component_mask &  (new_label_img > 0)], return_counts=True
            )

            # get value of class with higher count
            class_common_idx = np.argmax(count_labels)
            class_common = unique_labels[class_common_idx]

            label_delta[component_mask] = class_common

    return label_delta


def get_label_intersection(
    old_label_img: np.ndarray, new_label_img: np.ndarray
)-> np.ndarray:
    """Get the labels from the new_label_img that are in the old_components_img

    Parameters
    ----------
    old_label_img : np.ndarray
        Segmentation predicted by the previous iteration

    new_label_img : np.ndarray
        Segmentation predicted

    Returns
    ---------
        The labels from the new segmentation that are in the old_components_img
    """
    
    old_components_img = label(old_label_img)
    new_components_img = label(new_label_img)

    label_intersection = np.zeros_like(new_components_img)

    components_to_iter = np.unique(new_components_img)
    components_to_iter = components_to_iter[np.nonzero(components_to_iter)]

    print("Getting the intersection between the old and new segmentation")
    for idx in tqdm(components_to_iter):
        component_mask = new_components_img == idx
        
        # if more than 90% of the area is filled it will be added to the intersection sample
        if np.mean(old_components_img[component_mask] == 0) < 0.9:
            
            # count labels
            unique_labels, count_labels = np.unique(
                new_label_img[(component_mask) & (new_label_img > 0)], 
                return_counts=True
            )

            # get value of class with higher count
            class_common = unique_labels[np.argmax(count_labels)]

            label_intersection[component_mask] = class_common

    return label_intersection



def filter_components_by_geometric_property(label_img:np.ndarray, low_limit:float, high_limit:float, property = "area"):
    """
    Filter components in a labeled image based on geometric properties.

    This function selects components whose geometric property (such as area) falls outside the specified
    range defined by the low_limit and high_limit parameters.

    Parameters
    ----------
    label_img : np.ndarray
        Labeled image containing connected components.
    low_limit : float
        Lower limit for the geometric property. Components with property values below this limit will be removed.
    high_limit : float
        Upper limit for the geometric property. Components with property values above this limit will be removed.
    property : str, optional
        The geometric property to use for filtering (default is "area").

    Returns
    -------
    None

    Notes
    -----
    This function modifies the input labeled image in place by removing components that do not meet
    the specified geometric property criteria.
    """

    components_img = label(label_img)

    stats_label_data = get_components_stats(components_img, label_img).reset_index()

    filter_property = (stats_label_data[property] < low_limit) | (stats_label_data[property] > high_limit)

    component_ids_to_remove = stats_label_data[filter_property]['label'].astype(int).values

    remove_components_by_index(
        component_ids_to_remove = component_ids_to_remove,
        components_img = components_img,
        label_img = label_img
    )



def select_n_labels_by_class(pred_labels:np.ndarray, samples_by_class:int = 5):

    components_pred_map = label(pred_labels)
    
    delta_stats = get_components_stats(components_pred_map, pred_labels)
    
    # random shufle dataframe
    delta_stats = delta_stats.sample(frac=1, random_state=0)
    
    # sample components
    selected_samples = delta_stats.groupby("tree_type").head(samples_by_class)

    # component_ids_to_remove = stats_pred_data[filter_area]["label"]

    # remove selected components
    delta_stats.drop(index=selected_samples.index, inplace=True)

    # remove inplace
    remove_components_by_index(
        component_ids_to_remove = delta_stats.index.astype(int),
        components_img = components_pred_map,
        label_img = pred_labels
    )




def filter_components_by_mask(pred_map:np.ndarray):
    """Remove labels and components out of the mask.tif area

    Parameters
    ----------
    data_path : str
        Path to the data folder
    pred_map : np.ndarray
        The labels map from the current iteration.
    """
    
    mask = read_tiff(args["mask_path"])
    
    if not mask.dtype == "bool":
        mask = np.where(mask > 0, False, True)

    # Skip the mask filter if the mask array is entirely False
    if np.sum(mask) == 0:
        return

    components_pred_map = label(pred_map)
    
    # Components with some peace out the mask
    components_to_check = np.unique(components_pred_map[mask])
    components_to_check = components_to_check[np.nonzero(components_to_check)]
    
    print("Filtering components out of the area of the experiment")
    for component in tqdm(np.unique(components_to_check)):

        component_filter = components_pred_map==component
        
        area_out_mask = np.mean( mask[component_filter])
        
        # if more than 20% of the component is out, remove it
        if area_out_mask >= 0.20:
            pred_map[component_filter] = 0
            components_pred_map[component_filter] = 0



def join_labels_set(high_priority_labels:np.ndarray, low_priority_labels:np.ndarray, overlap_limit:int=0.05) -> np.ndarray:
    """Join the high priority labels with the low priority labels based on the components area
    The join doesnt overlap the component or cut any component by the other\n
    If there is an overlap, the component from the high priority labels is kept

    Parameters
    ----------
    high_priority_labels : np.ndarray
        The image arrary with the segmentation labels with high priority.

    low_priority_labels : np.ndarray
        The image arrary with the segmentation labels with low priority.
    
    overlap_limit : float
        The limit of the overlap between the components
        If there is an overlap between the components lower than the limit
        the component from low priority labels is added to the high priority labels

    Returns
    -------
    np.ndarray
        The image arrary with the segmentation labels with high and low priority.
    """

    labels_union = high_priority_labels.copy()
    
    # get low priority components
    low_priority_comp = label(low_priority_labels)
    
    low_components_in_high = np.unique(low_priority_comp[(low_priority_comp>0) & (high_priority_labels>0) ])
    
    low_components_not_in_high = np.unique(low_priority_comp[~np.isin(low_priority_comp,low_components_in_high)])
    low_components_not_in_high = low_components_not_in_high[np.nonzero(low_components_not_in_high)]
    
    # Joining the components that doesnt have intersection
    labels_union = np.where(
        np.isin(low_priority_comp, low_components_not_in_high), 
        low_priority_labels, 
        high_priority_labels
    )

    # Adding the components with intersection
    for component in tqdm(np.unique(low_components_in_high)):
        
        component_mask = low_priority_comp==component

        overlap = np.mean(high_priority_labels[component_mask]>0)

        # If the overlap is lower than the limit, add the component to the labels union
        if overlap < overlap_limit:
            low_id = low_priority_labels[component_mask]
            
            low_id = np.unique(low_id[np.nonzero(low_id)])[0]
            
            labels_union[component_mask] = low_id

    
    return labels_union


def filter_map_by_depth_prob(pred_map:np.ndarray, prob_map:np.ndarray, depth_map:np.ndarray,  prob_thr:str, depth_thr:float,)->np.ndarray:
    """
    Filter a prediction map by the probability map.

    Parameters
    ----------
    pred_map:np.ndarray
        A 2D matrix with the class prediction, based on the class with highest confidence.
    prob_map : np.ndarray
        A 2D Probability map with the probability of the class with the highest model confidence
    depth_map : np.ndarray
        A 2D distance map with model the prediction
    prob_thr : str
        Prob threshold to select high confidence objects
    depth_thr : float
        Depth threshold to define the tree contours

    Returns
    -------
    np.ndarray
        Filtered image
    """
    # create a local copy for array
    pred_map = pred_map.copy()
    
    # Smothing the contours of depth_map
    depth_gauss = gaussian_filter(depth_map, sigma = 9)

    # Smothing the contours of prob_map
    prob_gauss = gaussian_filter(prob_map, sigma = 9)

    # Selection the image
    pred_map = np.where((depth_gauss > depth_thr) & (prob_gauss > prob_thr), pred_map, 0)

    return pred_map



def select_good_samples(old_pred_map:np.ndarray,
                        new_pred_map:np.ndarray
                        ) -> np.ndarray:
    """
    Selects high-quality samples based on model outputs.

    Parameters
    ----------
    old_pred_map : np.ndarray
        Segmentation map from the previous iteration with tree type labels.
    new_pred_map : np.ndarray
        New segmentation map with tree type labels.
    new_prob_map : np.ndarray
        Confidence/probability map corresponding to the new segmentation.
    new_depth_map : np.ndarray
        Depth map predicted by the auxiliary task of the model.

    Returns
    -------
    np.ndarray
        New segmentation map with the selected high-quality samples.
    """

    new_pred_map = new_pred_map.copy()
    
    # filter components too small or too large
    filter_components_by_geometric_property(new_pred_map, 
                                            low_limit = np.float32(args.lower_limit_area), 
                                            high_limit = np.float32(args.upper_limit_area), # high limit area
                                            property = "area")
    
    filter_components_by_mask(new_pred_map)
    
    # Calculate main metrics of each tree
    comp_old_pred = label(old_pred_map)
    comp_old_stats = get_components_stats(comp_old_pred, old_pred_map).reset_index()

    comp_old_stats = comp_old_stats.groupby("tree_type").agg(
        {"extent":"median", 
        "solidity":"median", 
        "eccentricity":"median", 
        "area":"median"
        }
    )

    comp_old_stats.columns  = "ref_" + comp_old_stats.columns 

    # Get metrics about the new labels
    comp_new_pred = label(new_pred_map)
    comp_new_stats =  get_components_stats(comp_new_pred, new_pred_map).reset_index()
    
    # Join data from the last with the new one
    comp_new_stats = comp_new_stats.merge(comp_old_stats, on = "tree_type", how = "left")

    comp_new_stats["dist_area"] =  np.abs(comp_new_stats["area"] - comp_new_stats["ref_area"])/comp_new_stats["ref_area"]
    comp_new_stats["diff_area"] =  (comp_new_stats["area"] - comp_new_stats["ref_area"])/comp_new_stats["ref_area"]

    comp_new_stats["diff_soli"] =  (comp_new_stats["solidity"] - comp_new_stats["ref_solidity"])
    
    if args.scale_area == "log":
        comp_new_stats["diff_area"] = np.log(comp_new_stats["diff_area"])

    median_filter = (
        (comp_new_stats["diff_area"] <= np.float32(args.upper_limit_area_rlted_to_tree_type)) & 
        (comp_new_stats["diff_area"] >= np.float32(args.lower_limit_area_rlted_to_tree_type)) & 
        (comp_new_stats["diff_soli"] >= np.float32(args.lower_limit_solidity_rlted_to_tree_type))
    )

    # Select componentes based on some metrics
    selected_comp = comp_new_stats[median_filter].copy()

    new_pred_map =  np.where(np.isin(comp_new_pred, selected_comp["label"].unique()), new_pred_map, 0)

    return new_pred_map



def get_new_segmentation_sample(ground_truth_map:np.ndarray, 
                                old_selected_labels:np.ndarray,
                                old_all_labels:np.ndarray,
                                new_pred_map:np.ndarray, 
                                new_prob_map:np.ndarray, 
                                new_depth_map:np.ndarray, 
                                prob_thr:float,
                                depth_thr:float)->Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Get the new segmentation sample based on the segmentation from the last iteration and the new segmentation prediction set
    
    Parameters
    ----------
    old_pred_map : np.ndarray
        Segmentation map from the last iteration with tree two labels
    new_pred_map : np.ndarray
        New segmentation map with tree type labels
    new_prob_map : np.ndarray
        New segmentation map with confidence/probability at each pixel
    new_depth_map : np.ndarray
        New depth map predicted by the auxiliar task of the model
    Returns
    -------
    all_labes_set : np.array
        New segmentation map with tree type labels with all components with confidence higher than 0.90

    selected_labels_set : np.array
        The same set as all_labels_set but with some filters applied to minimize unbalanced classes problem
    """
    
    # set labels at the same scale as ground truth labels
    new_pred_map += 1
    
    logger.info(f"Filtering the components with distance map >= {depth_thr} and prob >= {prob_thr}")

    new_pred_map = filter_map_by_depth_prob(new_pred_map, 
                                            new_prob_map, 
                                            new_depth_map, 
                                            prob_thr,
                                            depth_thr)
    
    logger.info("Selecting the samples with good aspects")
    new_pred_map = select_good_samples(
        old_all_labels,
        new_pred_map,
    )
    new_pred_map = convert_to_minor_numeric_type(new_pred_map)
    
    # Join all labels set with new prediction
    logger.info("Joining the old and new components")
    new_labels_set = join_labels_set(new_pred_map, old_all_labels)
    new_labels_set = convert_to_minor_numeric_type(new_labels_set)


    logger.info("Getting the new components")
    # Select components that are in new but not in old
    delta_label_map = get_labels_delta(old_label_img = old_selected_labels, 
                                       new_label_img = new_labels_set)
    delta_label_map = convert_to_minor_numeric_type(delta_label_map)


    unbalanced_delta = delta_label_map.copy()
    unbalanced_delta = convert_to_minor_numeric_type(unbalanced_delta)

    logger.info("Selecting 5 components by tree_type")
    select_n_labels_by_class(
        delta_label_map,
        samples_by_class = 5
    )


    logger.info("Getting the components that are in the old and new segmentation")
    # get the new predicted shapes for components from old segmentation
    intersection_label_map = get_label_intersection(old_label_img = old_selected_labels, 
                                                    new_label_img = new_labels_set)
    intersection_label_map = convert_to_minor_numeric_type(intersection_label_map)


    logger.info("Getting the old components but with the updated shape")
    # join updated shapes with the old ones that were not updated
    old_selected_labels_updated = join_labels_set(intersection_label_map, old_selected_labels, 0.10 )
    old_selected_labels_updated = convert_to_minor_numeric_type(old_selected_labels_updated)

    logger.info("Joining the old components updated shape with the new selected components")
    # join the old labels set with the new labels. balanced sample addition
    selected_labels_set = join_labels_set(delta_label_map, old_selected_labels_updated, 0.10 )
    selected_labels_set = convert_to_minor_numeric_type(selected_labels_set)
    
    logger.info('Joining the selected components with the original groud_truth train set')
    # Adding the ground truth segmentation
    selected_labels_set = join_labels_set(ground_truth_map, selected_labels_set, 0.01 )
    selected_labels_set = convert_to_minor_numeric_type(selected_labels_set)


    logger.info("Joining the old components updated shape with the new components")
    # join the old labels set with the new labels. unbalanced sample addition
    all_labels_set = join_labels_set(unbalanced_delta, old_selected_labels_updated, 0.10)
    all_labels_set = convert_to_minor_numeric_type(all_labels_set)

    logger.info('Joining the new components with the original groud_truth train set')
    # Adding the ground truth segmentation
    all_labels_set = join_labels_set(ground_truth_map, all_labels_set, 0.01)
    all_labels_set = convert_to_minor_numeric_type(all_labels_set)

    return all_labels_set, selected_labels_set



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    args = read_yaml("args.yaml")
    ROOT_PATH = dirname(__file__)
        
    version_folder = "/media/dariossh/Extreme SSD/Arquivos SSH/2.8.2_version_data"
    input_data_folder = join(ROOT_PATH, "amazon_input_data")

    gt_map = read_tiff(f"{input_data_folder}/segmentation/train_set.tif")

    test_gt_map = read_tiff(f"{input_data_folder}/segmentation/train_set.tif")
    
    old_all_labels = read_tiff(f"{version_folder}/iter_001/new_labels/all_labels_set.tif")

    old_selected_labels = read_tiff(f"{version_folder}/iter_001/new_labels/selected_labels_set.tif")
                               
    new_pred_map = read_tiff(f"{version_folder}/iter_002/raster_prediction/join_class_0.6.TIF")

    new_prob_map = read_tiff(f"{version_folder}/iter_002/raster_prediction/join_prob_0.6.TIF")

    depth_predicted = read_tiff(f"{version_folder}/iter_001/raster_prediction/depth_0.6.TIF")
    
    all_labels_set, selected_labels_set =  get_new_segmentation_sample(old_selected_labels = old_selected_labels,
                                                                       old_all_labels = old_all_labels,

                                                                       new_pred_map = new_pred_map,
                                                                       new_prob_map = new_prob_map,
                                                                       new_depth_map = depth_predicted,
                                                                       
                                                                       ground_truth_map = gt_map,
                                                                       
                                                                       prob_thr=0.7,
                                                                       depth_thr=0.1
                                                                       )
    
    print("Ok")