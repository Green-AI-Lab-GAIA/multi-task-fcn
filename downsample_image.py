from src.io_operations import read_tiff, get_image_metadata, array2raster
from src.utils import check_folder
from glob import glob
from os.path import join, dirname, abspath
from os import cpu_count
import numpy as np
from skimage.transform import resize
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm




def downsample_tiff_files(input_folder_path, downsample_factor, max_workers=cpu_count()):
    
    OUTPUT_FOLDER = join(dirname(abspath(input_folder_path)), f"{downsample_factor}x_{input_folder_path}")
    
    check_folder(OUTPUT_FOLDER)
    check_folder(join(OUTPUT_FOLDER, "segmentation"))
    check_folder(join(OUTPUT_FOLDER, "orthoimage"))


    orthoimage_path = glob(join(input_folder_path, "orthoimage", "*.tif"))[0]
    orthoimage = read_tiff(orthoimage_path)
    orthoimage_metadata = get_image_metadata(orthoimage_path)
    
    resized_orthoimage = resize(
        orthoimage,
        (
            orthoimage.shape[0],
            orthoimage.shape[1] // np.sqrt(downsample_factor),
            orthoimage.shape[2] // np.sqrt(downsample_factor),
        ),
        anti_aliasing=True,
    )
    
    if resized_orthoimage.max() <= 1:
            
        resized_orthoimage = (resized_orthoimage * 255).astype(np.uint8)
    
    del orthoimage


    test_path = glob(join(input_folder_path, "segmentation", "*test*.tif"))[0]
    test_set = read_tiff(test_path)
    test_set_metadata = get_image_metadata(test_path)

    resized_test_set = resize(
        test_set.astype("float32"),
        (
            test_set.shape[0] // np.sqrt(downsample_factor), 
            test_set.shape[1] // np.sqrt(downsample_factor)
        ),
        anti_aliasing=True,
        order=0,
    ).round().astype("uint8")
    del test_set
    
    
    train_path = glob(join(input_folder_path, "segmentation", "*train*.tif"))[0]
    train_set = read_tiff(train_path)
    train_set_metadata = get_image_metadata(train_path)

    resized_train_set = resize(
        train_set.astype("float32"),
        (
            train_set.shape[0] // np.sqrt(downsample_factor), 
            train_set.shape[1] // np.sqrt(downsample_factor)
        ),
        anti_aliasing=False,
        order=0,
    ).round().astype("uint8")
    del train_set
    

    mask_path = join(input_folder_path, "mask.tif")
    mask = read_tiff(mask_path)
    mask_metadata = get_image_metadata(mask_path)
    
    resized_mask = resize(
        mask.astype("float32"),
        (
            mask.shape[0] // np.sqrt(downsample_factor), 
            mask.shape[1] // np.sqrt(downsample_factor)
        ),
        anti_aliasing=False,
        order=0,
    ).round().astype("uint8")
    del mask

    array2raster(
        join(OUTPUT_FOLDER, "orthoimage", "orthoimage.tif"),
        resized_orthoimage,
        orthoimage_metadata,
        "uint8"
    )
    
    array2raster(
        join(OUTPUT_FOLDER, "segmentation", "test_set.tif"),
        resized_test_set,
        test_set_metadata,
        "uint8"
    )
    
    array2raster(
        join(OUTPUT_FOLDER, "segmentation", "train_set.tif"),
        resized_train_set,
        train_set_metadata,
        "uint8"
    )
    
    array2raster(
        join(OUTPUT_FOLDER, "mask.tif"),
        resized_mask,
        mask_metadata,
        "uint8"
    )
    
    print("Downsampling completed") 



if __name__ == "__main__":
    input_path = "amazon_input_data"
    DOWN_SAMPLE_FACTOR = 4
    downsample_tiff_files(input_path, DOWN_SAMPLE_FACTOR)
       