from glob import glob
from os.path import join
from pathlib import Path


import wandb
from main import *
from src.io_operations import load_args, read_tiff

# disable wandb
wandb.init(mode="disabled", project="test")

args = load_args("args.yaml")

# change for a small dataset
args.data_path = "00_test_data"
args.note = "Debugging code"
args.ortho_image = glob("16x_amazon_input_data/orthoimage/*.tif")[0]
args.train_segmentation_path = glob("16x_amazon_input_data/segmentation/*train*.tif")[0]
args.test_segmentation_path = glob("16x_amazon_input_data/segmentation/*test*.tif")[0]
args.mask_path = "16x_amazon_input_data/mask.tif"

# change for less epochs and samples
args.epochs = 3
args.samples = 100
args.batch_size = 32
args.size_crops = 256
args.num_iter = 2
args.overlap = [0.1, 0.5]
args.num_workers = 0

def get_tiff_file_paths(data_path):
    
    extensions = [".tif", ".tiff", ".TIF", ".TIFF"]
    
    directory = Path(data_path)
    
    tiff_files = [str(file) for ext in extensions for file in directory.rglob(f'*{ext}')]
    
    return tiff_files.copy()
        
def convert_every_tiff_to_png(data_path):
    
    tiff_files = get_tiff_file_paths(data_path)
    
    for file in tiff_files:
        image = read_tiff(file)
        
        if image.ndim == 3:
            image = image.transpose(1,2,0)
            
        plt.imshow(image, interpolation="nearest")
        plt.axis("off")
        # replace extension with .png
        new_file_name = file.replace(".tif", ".png")\
                            .replace(".tiff", ".png")\
                            .replace(".TIF", ".png")\
                            .replace(".TIFF", ".png")
        plt.savefig(new_file_name, bbox_inches="tight", pad_inches=0)
        plt.close()
    
        
if __name__ == "__main__":
    version_name = "test_code"
    logger = create_logger(module_name=__name__, filename=version_name)

    logger.info(f"################### {version_name.upper()} ###################")

    # create output path
    check_folder(args.data_path)

    # Save args state into data_path
    save_yaml(args, join(args.data_path, "args.yaml"))

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
            generate_distance_map_for_first_iteration(current_iter_folder, args=args)
            
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
        
        generate_distance_map_for_next_iteration(current_iter_folder, args)

        #############################################

        delete_useless_files(current_iter_folder = current_iter_folder)
        
        generate_labels_view(current_iter_folder, args.ortho_image, args.train_segmentation_path)

        print_sucess("Distance map generated")
    
        convert_every_tiff_to_png(args.data_path)



