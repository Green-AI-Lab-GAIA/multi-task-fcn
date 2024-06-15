import sys
from typing import Literal,Tuple

import numpy as np
import torch

from os.path import dirname, join
from torch.utils.data import Dataset
from torchvision import transforms

ROOT_PATH = dirname(dirname(__file__))
sys.path.append(ROOT_PATH)

from src.utils import get_crop_image, get_pad_width, oversample, normalize
from src.io_operations import check_file_extension, get_file_extesion, get_npy_shape, load_image



class DatasetFromCoord(Dataset):
    def __init__(self,
                image_path:str,
                segmentation_path:str,
                distance_map_path:str,
                crop_size:int,
                samples:int = None,
                augment:bool = False,
                copy_paste_augmentation:bool = False
                ) -> None: 
        
        super().__init__()
        
        self.image_path = image_path
        self.segmentation_path = segmentation_path
        self.distance_map_path = distance_map_path
        
        self.samples = samples
        self.crop_size = crop_size
        self.augment = augment
        
        self.copy_paste_augmentation = copy_paste_augmentation
        
        self.img_segmentation = load_image(segmentation_path)
        self.img_depth = load_image(distance_map_path)
        self.image = load_image(image_path)
        
        self.image_shape = self.image.shape
        
        self.generate_coords()


    def generate_coords(self):
            
        coords = np.where(self.img_segmentation!=0)
        coords = np.array(coords)
        coords = np.rollaxis(coords, 1, 0)
        
        coords_label = self.img_segmentation[np.nonzero(self.img_segmentation)]

        coords = oversample(coords, coords_label, "min")   

        self.coords = np.array(coords)


    def standardize_image_channels(self):
        
        self.image = self.image.astype("float32")

        normalize(self.image)
        


    def read_window_around_coord(self, coord:np.ndarray, image:np.ndarray) -> torch.Tensor:
        
        image_crop = get_crop_image(image, image.shape, coord, self.crop_size)

        pad_width = get_pad_width(self.crop_size, coord, image.shape)

        # apply padding to image
        image_crop = np.pad(
            image_crop, 
            pad_width = pad_width,
            mode = "constant",
            constant_values = 0
        )
        

        if (image_crop.shape[-1] != self.crop_size) or (image_crop.shape[-2] != self.crop_size):
            raise ValueError(f"There is a bug relationed to the shape {image_crop.shape}")

        return torch.tensor(image_crop)


    def copy_and_paste_augmentation(self, image:torch.Tensor, segmentation:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # random select a crop from the image
        random_row = np.random.randint(self.crop_size, self.image_shape[1] - self.crop_size)
        random_column = np.random.randint(self.crop_size, self.image_shape[2] - self.crop_size)
        
        image_crop = self.read_window_around_coord(
            coord=[random_row, random_column],
            image=self.image,
        )
        
        # paste the crop into the image
        return torch.where(segmentation > 0, image, image_crop)
        
        
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the data from the dataset
        
        Parameters
        ----------
        idx : int
            The index of the data to be loaded
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            The image crop, depth map crop, and label ref crop
        
        """
        current_coord = self.coords[idx].copy()
        
        
        if self.augment:
            
            # Run random shift
            uniform_dist_range = (-0.99, 0.99)
            
            random_row_prop = np.random.uniform(*uniform_dist_range)
            random_column_prop = np.random.uniform(*uniform_dist_range)

            current_coord[0] += int(random_row_prop * (self.crop_size//2))
            current_coord[1] += int(random_column_prop * (self.crop_size//2))


        image = self.read_window_around_coord(
            coord=current_coord,
            image=self.image,
        )
        

        segmentation = self.read_window_around_coord(
            coord=current_coord,
            image=self.img_segmentation
        )

        distance_map = self.read_window_around_coord(
            coord=current_coord,
            image=self.img_depth
        )


        if self.augment:
            if np.random.random() < 0.5 and self.copy_paste_augmentation:
                image = self.copy_and_paste_augmentation(image, segmentation)
                    
            # Run Horizontal Flip
            if np.random.random() > 0.5:
                image = transforms.functional.hflip(image)
                segmentation = transforms.functional.hflip(segmentation)
                distance_map = transforms.functional.hflip(distance_map)

            # Run Vertical Flip
            if np.random.random() > 0.5:
                image = transforms.functional.vflip(image)
                segmentation = transforms.functional.vflip(segmentation)
                distance_map = transforms.functional.vflip(distance_map)
            
            # Run random rotation
            angle = int(np.random.choice([0, 90, 180, 270]))
            
            image = transforms.functional.rotate(image.unsqueeze(0), angle).squeeze(0)
            segmentation = transforms.functional.rotate(segmentation.unsqueeze(0), angle).squeeze(0)
            distance_map = transforms.functional.rotate(distance_map.unsqueeze(0), angle).squeeze(0)


        return image.float(), distance_map.float(), segmentation.long()          

    
    def __len__(self):

        if (self.samples is None):
            return len(self.coords)

        if (self.samples > len(self.coords)):
            return len(self.coords)
        
        
        return self.samples


class DatasetForInference(Dataset):
    def __init__(self,
                image_path:str,
                crop_size:int,
                overlap_rate:float
                ) -> None: 
        
        super().__init__()
        
        self.image_path = image_path
        self.crop_size = crop_size
        self.overlap_rate = overlap_rate
        
        self.image = load_image(image_path)
        self.image_shape = self.image.shape
        
        self.generate_coords()


    def generate_coords(self):
        
        coords_list = []
        
        height, width = self.image_shape[-2:]
        
        self.overlap_size = int(self.crop_size * self.overlap_rate)
        self.stride_size = self.crop_size - self.overlap_size

        for m in range(0, height-self.overlap_size, self.stride_size):
            for n in range(0, width-self.overlap_size, self.stride_size):
                
                coords_list.append([m, n])
                
        
        self.coords = np.array(coords_list)


    def standardize_image_channels(self):
        
        self.image = self.image.astype("float32")

        normalize(self.image)
        

    def get_slice_window(self, coord:np.ndarray) -> Tuple[int, int, int, int]:
        "Based on overlap rate and crop size, get the slice to fit the image into original image"
        
        row_start = coord[0]
        row_end = coord[0] + self.crop_size
        
        if row_end > self.image_shape[1]:
            row_start = self.image_shape[1] - self.crop_size
            row_end = self.image_shape[1]
        
        column_start = coord[1]
        column_end = coord[1] + self.crop_size
        
        if column_end > self.image_shape[2]:
            column_start = self.image_shape[2] - self.crop_size
            column_end = self.image_shape[2]
        
        return row_start, row_end, column_start, column_end

    def read_window(self, coord:np.ndarray, image:np.ndarray) -> torch.Tensor:
        
        row_start, row_end, column_start, column_end = self.get_slice_window(coord)
        
        image_crop = image[:, row_start:row_end, column_start:column_end]
        
        if (image_crop.shape[-1] != self.crop_size) or (image_crop.shape[-2] != self.crop_size):
            raise ValueError(f"There is a bug relationed to the shape {image_crop.shape}")
        
        return torch.tensor(image_crop)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the data from the dataset
        
        Parameters
        ----------
        idx : int
            The index of the data to be loaded
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The image crop and the slice to fit the image into original image
        
        """
        current_coord = self.coords[idx].copy()
        
        image = self.read_window(
            coord=current_coord,
            image=self.image,
        )
        
        row_start, row_end, column_start, column_end = self.get_slice_window(
            current_coord,
        )
        
        return image.float(), (row_start, row_end, column_start, column_end)
    
    def __len__(self):

        return len(self.coords)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    overlap_rates_to_test = [0.1, 0.3, 0.5]
    crop_size_to_test = [128, 256, 512]
    
    for overlap_rate in overlap_rates_to_test:
        for crop_size in crop_size_to_test:
            
            inference_dataset = DatasetForInference(
                image_path = r"amazon_mc_input_data\orthoimage\NOV_2017_FINAL_004.tif",
                crop_size = crop_size,
                overlap_rate = overlap_rate
            )
            
            inference_dataset.standardize_image_channels()
            
            inference_dataloader = torch.utils.data.DataLoader(
                inference_dataset,
                batch_size = 10,
                shuffle = False,
                num_workers = 0
            )
            
            # rebuild the own image
            output_image = np.zeros_like(inference_dataset.image)
            count_image = np.zeros_like(inference_dataset.image)
            
            for i, (image, slice) in enumerate(inference_dataloader):
                
                row_start, row_end, column_start, column_end = slice
                
                for j in range(image.shape[0]):
                    
                    output_image[:, row_start[j]:row_end[j], column_start[j]:column_end[j]] += image[j].numpy()
                    count_image[:, row_start[j]:row_end[j], column_start[j]:column_end[j]] += 1
            
            
            count_image = np.where(count_image == 0, 1, count_image)
            output_image = output_image / count_image
            
            plt.imshow(np.moveaxis(output_image, 0,2))
            plt.savefig(f"test_data/inference_image_{overlap_rate}_{crop_size}.png")
            plt.close()