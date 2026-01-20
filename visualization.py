import gc
from os.path import dirname, join
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from skimage.color import label2rgb
from skimage.measure import find_contours

from src.io_operations import read_tiff
from src.utils import check_folder, run_in_process, run_in_thread


# generate view only for sythentic labels
def generate_view_for_sythentic_label(current_iter_folder: str, train_segmentation_path: str, orthoimage_path: str, region_suffix: str = ""):
    """Generate visualization for synthetic labels (predictions not in ground truth).
    
    Parameters
    ----------
    current_iter_folder : str
        Path to iteration folder (or region folder for multi-region)
    train_segmentation_path : str
        Path to ground truth segmentation
    orthoimage_path : str
        Path to orthoimage
    region_suffix : str
        Suffix for multi-region support
    """
    # Handle both iter folder and region folder formats
    folder_name = current_iter_folder.split("/")[-1].split("\\")[-1]
    if folder_name.startswith("region_"):
        parent_folder = dirname(current_iter_folder)
        current_iter = int(parent_folder.split("iter_")[-1].split("/")[0].split("\\")[0])
        region_suffix = f"_{folder_name}"
    else:
        current_iter = int(current_iter_folder.split("iter_")[-1].split("/")[0].split("\\")[0])
    
    if current_iter == 0:
        ALL_LABELS_PATH = join(train_segmentation_path)
        
    else:
        ALL_LABELS_PATH = join(current_iter_folder, "new_labels", "all_labels_set.tif")

    ALL_LABELS_MAP = read_tiff(ALL_LABELS_PATH)

    TRAIN_GT_MAP = read_tiff(train_segmentation_path)

    synthetic_labels = np.where(TRAIN_GT_MAP > 0, 0, ALL_LABELS_MAP)
    
    # folder to save view - go up from region folder if needed
    if folder_name.startswith("region_"):
        base_folder = dirname(dirname(current_iter_folder))
    else:
        base_folder = dirname(current_iter_folder)
    
    OUTPUT_MAP_FOLDER = join(base_folder, "visualization", "synthetic_all_labels")
    check_folder(OUTPUT_MAP_FOLDER)


    fig, ax = plt.subplots(dpi=500)

    ORTHOIMAGE = read_tiff(orthoimage_path)
    ORTHOIMAGE = np.moveaxis(ORTHOIMAGE, 0, 2)
    
    # Select appropriate bands based on number of channels
    num_channels = ORTHOIMAGE.shape[-1]
    if num_channels == 25:
        # Hyperspectral - use bands 5, 3, 2
        ORTHOIMAGE = ORTHOIMAGE[..., [5,3,2]].copy()
    elif num_channels == 4:
        # 4-band image (e.g., RGB + NIR) - use first 3 bands (RGB)
        ORTHOIMAGE = ORTHOIMAGE[..., [0,1,2]].copy()
    elif num_channels > 3:
        # Multi-band - use first 3 bands
        ORTHOIMAGE = ORTHOIMAGE[..., [0,1,2]].copy()
    # If 3 or less bands, keep as is
    
    if ORTHOIMAGE.max() > 255:
        ORTHOIMAGE = np.divide(ORTHOIMAGE, np.quantile(ORTHOIMAGE, 0.99, axis=(0,1)))

    ax.imshow(ORTHOIMAGE)
    del ORTHOIMAGE

    # plot contours
    for contour in find_contours(synthetic_labels):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=0.3, color = "red", label = "Prediction")

    ax.axis('off')
    fig.savefig(join(OUTPUT_MAP_FOLDER, f"{current_iter:03d}{region_suffix}_segmentation.png"), bbox_inches='tight', pad_inches=0)
    plt.close(fig)


# generate view only for synthetic selected labels
def generate_view_for_synthetic_selected_label(current_iter_folder: str, train_segmentation_path: str, orthoimage_path: str, region_suffix: str = ""):
    """Generate visualization for synthetic selected labels (selected predictions not in ground truth).
    
    Parameters
    ----------
    current_iter_folder : str
        Path to iteration folder (or region folder for multi-region)
    train_segmentation_path : str
        Path to ground truth segmentation
    orthoimage_path : str
        Path to orthoimage
    region_suffix : str
        Suffix for multi-region support
    """
    # Handle both iter folder and region folder formats
    folder_name = current_iter_folder.split("/")[-1].split("\\")[-1]
    if folder_name.startswith("region_"):
        parent_folder = dirname(current_iter_folder)
        current_iter = int(parent_folder.split("iter_")[-1].split("/")[0].split("\\")[0])
        region_suffix = f"_{folder_name}"
    else:
        current_iter = int(current_iter_folder.split("iter_")[-1].split("/")[0].split("\\")[0])
    
    if current_iter == 0:
        SELECTED_LABELS_PATH = join(train_segmentation_path)
        
    else:
        SELECTED_LABELS_PATH = join(current_iter_folder, "new_labels", "selected_labels_set.tif")

    SELECTED_LABELS_MAP = read_tiff(SELECTED_LABELS_PATH)

    TRAIN_GT_MAP = read_tiff(train_segmentation_path)

    synthetic_selected_labels = np.where(TRAIN_GT_MAP > 0, 0, SELECTED_LABELS_MAP)
    
    # folder to save view - go up from region folder if needed
    if folder_name.startswith("region_"):
        base_folder = dirname(dirname(current_iter_folder))
    else:
        base_folder = dirname(current_iter_folder)
    
    OUTPUT_MAP_FOLDER = join(base_folder, "visualization", "synthetic_selected_labels")
    check_folder(OUTPUT_MAP_FOLDER)


    fig, ax = plt.subplots(dpi=500)

    ORTHOIMAGE = read_tiff(orthoimage_path)
    ORTHOIMAGE = np.moveaxis(ORTHOIMAGE, 0, 2)
    
    # Select appropriate bands based on number of channels
    num_channels = ORTHOIMAGE.shape[-1]
    if num_channels == 25:
        # Hyperspectral - use bands 5, 3, 2
        ORTHOIMAGE = ORTHOIMAGE[..., [5,3,2]].copy()
    elif num_channels == 4:
        # 4-band image (e.g., RGB + NIR) - use first 3 bands (RGB)
        ORTHOIMAGE = ORTHOIMAGE[..., [0,1,2]].copy()
    elif num_channels > 3:
        # Multi-band - use first 3 bands
        ORTHOIMAGE = ORTHOIMAGE[..., [0,1,2]].copy()
    # If 3 or less bands, keep as is
    
    # Normalize image values
    if ORTHOIMAGE.max() > 255:
        quantiles = np.quantile(ORTHOIMAGE, 0.99, axis=(0,1), keepdims=True)
        # Avoid division by zero
        quantiles = np.where(quantiles == 0, 1, quantiles)
        ORTHOIMAGE = np.divide(ORTHOIMAGE, quantiles)
        # Clip to valid range [0, 1] for imshow
        ORTHOIMAGE = np.clip(ORTHOIMAGE, 0, 1)
    
    # Ensure valid shape and values
    if ORTHOIMAGE.ndim == 2:
        # Single channel - convert to 3D
        ORTHOIMAGE = ORTHOIMAGE[..., np.newaxis]
    if ORTHOIMAGE.shape[-1] == 1:
        # Single channel - replicate to RGB
        ORTHOIMAGE = np.repeat(ORTHOIMAGE, 3, axis=-1)
    
    # Ensure no NaN or Inf values
    ORTHOIMAGE = np.nan_to_num(ORTHOIMAGE, nan=0.0, posinf=1.0, neginf=0.0)
    
    ax.imshow(ORTHOIMAGE)
    del ORTHOIMAGE

    # plot contours
    for contour in find_contours(synthetic_selected_labels):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=0.3, color = "red", label = "Prediction")

    ax.axis('off')
    fig.savefig(join(OUTPUT_MAP_FOLDER, f"{current_iter:03d}{region_suffix}_segmentation.png"), bbox_inches='tight', pad_inches=0)
    plt.close(fig)



def generate_labels_view(current_iter_folder: str, orthoimage_path: str, train_segmentation_path: str, region_suffix: str = ""):
    """Function to generate images for qualitative evaluation.
    These images are not used for any kind of numeric evaluation.

    Parameters
    ----------
    current_iter_folder : str
        Path to iteration folder (or region folder for multi-region)
    
    orthoimage_path : str
        Path to remote sensing orthoimage
    
    train_segmentation_path : str
        Path to ground_truth segmentation used for train
    
    region_suffix : str, optional
        Suffix to add to filenames for multi-region support (e.g., "_region_0")
    """
    # Handle both iter folder and region folder formats
    folder_name = current_iter_folder.split("/")[-1].split("\\")[-1]
    if folder_name.startswith("region_"):
        # This is a region folder, get iter from parent
        parent_folder = dirname(current_iter_folder)
        current_iter = int(parent_folder.split("iter_")[-1].split("/")[0].split("\\")[0])
        region_suffix = f"_{folder_name}"
    else:
        current_iter = int(current_iter_folder.split("iter_")[-1].split("/")[0].split("\\")[0])

    if current_iter == 0:
        ALL_LABELS_PATH = train_segmentation_path
        SELECTED_LABELS_PATH = train_segmentation_path

    else:
        ALL_LABELS_PATH = join(current_iter_folder, "new_labels", "all_labels_set.tif")
        SELECTED_LABELS_PATH = join(current_iter_folder, "new_labels", "selected_labels_set.tif")


    ALL_LABELS_MAP = read_tiff(ALL_LABELS_PATH)
    SELECTED_LABELS_MAP = read_tiff(SELECTED_LABELS_PATH)
    

    num_classes = np.unique(ALL_LABELS_MAP[ALL_LABELS_MAP != 0]).shape[0]

    if num_classes == 14:
        DEFAULT_COLORS = ("black",'silver', 'blue', 'yellow', 'magenta', 'green', 
                        'indigo', 'darkorange', 'cyan', 'pink', 'yellowgreen', 
                        'red', 'darkgreen', 'gold', 'teal')

    else:
        from seaborn import color_palette
        DEFAULT_COLORS = color_palette("tab20", num_classes)
        DEFAULT_COLORS.insert(0, (0,0,0))

    DEFAULT_COLORS = ListedColormap(DEFAULT_COLORS)

    OUTPUT_MAP_FOLDER = join(dirname(current_iter_folder), "visualization")
    
    # create output folder
    check_folder(OUTPUT_MAP_FOLDER)

    # PLOT ALL LABELS
    ALL_LABELS_OUT_FOLDER = join(OUTPUT_MAP_FOLDER, "all_labels",)
    check_folder(ALL_LABELS_OUT_FOLDER)
    
    fig, ax = plt.subplots(dpi = 500)
    ax.imshow(ALL_LABELS_MAP, cmap=DEFAULT_COLORS, interpolation="bilinear")
    ax.axis('off')
    fig.savefig(join(ALL_LABELS_OUT_FOLDER, f"{current_iter:03d}{region_suffix}_segmentation.png"), bbox_inches='tight', pad_inches=0)
    plt.close(fig)


    # PLOT SELECTED LABELS
    SELECTED_LABELS_OUT_FOLDER = join(OUTPUT_MAP_FOLDER, "selected_labels")
    check_folder(SELECTED_LABELS_OUT_FOLDER)

    fig, ax = plt.subplots(dpi = 500)
    ax.imshow(SELECTED_LABELS_MAP, cmap=DEFAULT_COLORS)
    ax.axis('off')
    fig.savefig(join(SELECTED_LABELS_OUT_FOLDER, f"{current_iter:03d}{region_suffix}_segmentation.png"), bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    gc.collect()


    generate_view_for_sythentic_label(
        current_iter_folder=current_iter_folder,
        train_segmentation_path=train_segmentation_path,
        orthoimage_path=orthoimage_path
    )

    generate_view_for_synthetic_selected_label(
        current_iter_folder=current_iter_folder,
        train_segmentation_path=train_segmentation_path,
        orthoimage_path=orthoimage_path,
        region_suffix=region_suffix
    )


    

if __name__ == "__main__":

    from src.io_operations import load_args

    # parameters
    VERSION_FOLDER = r"C:\Users\ziull\OneDrive\Documentos\Estudos\TCC\multi-task-fcn\19.2_version_data"
    
    
    ROOT_PATH = dirname(__file__)

    # run it in parallel
    import multiprocessing as mp
    
    args = load_args(join(ROOT_PATH, VERSION_FOLDER, "args.yaml"))
    
    # with mp.Pool(mp.cpu_count()//4) as pool:
    #     pool.starmap(
    #         generate_labels_view,
    #         [(join(VERSION_FOLDER, f"iter_{iter_num:03d}"), args.ortho_image, args.train_segmentation_path) for iter_num in range(20,0,-1)]
    #     )
    for iter_num in range(20,0,-1):
        iter_path = join(VERSION_FOLDER, f"iter_{iter_num:03d}")
    
        generate_view_for_sythentic_label(
            iter_path, args.train_segmentation_path, args.ortho_image
        )