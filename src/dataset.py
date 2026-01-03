import sys
from typing import Literal, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

from os.path import dirname, join
from torch.utils.data import Dataset
from torchvision import transforms

ROOT_PATH = dirname(dirname(__file__))
sys.path.append(ROOT_PATH)

from src.utils import get_crop_image, get_pad_width, oversample, normalize
from src.io_operations import check_file_extension, get_file_extesion, get_npy_shape, load_image



class DatasetFromCoord(Dataset):
    """Dataset for training that extracts crops around labeled coordinates.
    
    Supports automatic resizing: crops are extracted at `crop_size` and 
    optionally resized to `input_dimension` for the model.
    
    Supports multi-scale cropping: if min_crop_size and max_crop_size are provided,
    the crop size is sampled uniformly between them for each sample.
    
    Parameters
    ----------
    image_path : str
        Path to the orthoimage
    segmentation_path : str
        Path to the segmentation labels
    distance_map_path : str
        Path to the distance map
    crop_size : int
        Size of crops to extract from the image (used when multi-scale is disabled)
    input_dimension : int, optional
        Size to resize crops before feeding to model. If None, uses crop_size.
    samples : int, optional
        Number of samples per epoch
    augment : bool
        Whether to apply data augmentation
    copy_paste_augmentation : bool
        Whether to use copy-paste augmentation
    min_crop_size : int, optional
        Minimum crop size for multi-scale cropping. If None, multi-scale is disabled.
    max_crop_size : int, optional
        Maximum crop size for multi-scale cropping. If None, multi-scale is disabled.
    """
    def __init__(self,
                image_path: str,
                segmentation_path: str,
                distance_map_path: str,
                crop_size: int,
                input_dimension: Optional[int] = None,
                samples: int = None,
                augment: bool = False,
                copy_paste_augmentation: bool = False,
                min_crop_size: Optional[int] = None,
                max_crop_size: Optional[int] = None
                ) -> None: 
        
        super().__init__()
        
        self.image_path = image_path
        self.segmentation_path = segmentation_path
        self.distance_map_path = distance_map_path
        
        self.samples = samples
        self.crop_size = crop_size
        self.input_dimension = input_dimension if input_dimension is not None else crop_size
        self.augment = augment
        
        self.copy_paste_augmentation = copy_paste_augmentation
        
        # Multi-scale crop settings
        self.min_crop_size = min_crop_size
        self.max_crop_size = max_crop_size
        self.multi_scale = (min_crop_size is not None and max_crop_size is not None 
                           and min_crop_size != max_crop_size)
        
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

    def _resize_tensor(self, tensor: torch.Tensor, target_size: int, mode: str = 'bilinear') -> torch.Tensor:
        """Resize a tensor to target_size.
        
        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor of shape (C, H, W) or (H, W)
        target_size : int
            Target size for height and width
        mode : str
            Interpolation mode ('bilinear' for images, 'nearest' for labels)
        
        Returns
        -------
        torch.Tensor
            Resized tensor
        """
        if tensor.shape[-1] == target_size and tensor.shape[-2] == target_size:
            return tensor
        
        # Add batch dimension if needed
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            squeeze_dims = 2
        elif tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)  # (1, C, H, W)
            squeeze_dims = 1
        else:
            squeeze_dims = 0
        
        # Resize
        resized = F.interpolate(
            tensor.float(),
            size=(target_size, target_size),
            mode=mode,
            align_corners=False if mode != 'nearest' else None
        )
        
        # Remove added dimensions
        if squeeze_dims == 2:
            resized = resized.squeeze(0).squeeze(0)
        elif squeeze_dims == 1:
            resized = resized.squeeze(0)
        
        return resized

    def read_window_around_coord(self, coord:np.ndarray, image:np.ndarray, crop_size: Optional[int] = None) -> torch.Tensor:
        
        # Use provided crop_size or default to self.crop_size
        current_crop_size = crop_size if crop_size is not None else self.crop_size

        image_crop = get_crop_image(image, image.shape, coord, current_crop_size)

        pad_width = get_pad_width(current_crop_size, coord, image.shape)

        # apply padding to image
        image_crop = np.pad(
            image_crop, 
            pad_width = pad_width,
            mode = "constant",
            constant_values = 0
        )
        

        if (image_crop.shape[-1] != current_crop_size) or (image_crop.shape[-2] != current_crop_size):
            raise ValueError(f"There is a bug relationed to the shape {image_crop.shape}, expected {current_crop_size}")

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
        
        # Determine crop size for this sample
        if self.multi_scale:
            current_crop_size = np.random.randint(self.min_crop_size, self.max_crop_size + 1)
            # Ensure crop size is even (required by get_crop_image/get_pad_width functions)
            if current_crop_size % 2 != 0:
                current_crop_size += 1
        else:
            current_crop_size = self.crop_size
        
        if self.augment:
            
            # Run random shift
            uniform_dist_range = (-0.99, 0.99)
            
            random_row_prop = np.random.uniform(*uniform_dist_range)
            random_column_prop = np.random.uniform(*uniform_dist_range)

            current_coord[0] += int(random_row_prop * (current_crop_size//2))
            current_coord[1] += int(random_column_prop * (current_crop_size//2))


        image = self.read_window_around_coord(
            coord=current_coord,
            image=self.image,
            crop_size=current_crop_size,
        )
        

        segmentation = self.read_window_around_coord(
            coord=current_coord,
            image=self.img_segmentation,
            crop_size=current_crop_size,
        )

        distance_map = self.read_window_around_coord(
            coord=current_coord,
            image=self.img_depth,
            crop_size=current_crop_size,
        )


        if self.augment:
            if np.random.random() < 0.3 and self.copy_paste_augmentation:
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

        # Resize to input_dimension (always needed for multi-scale, or when crop_size != input_dimension)
        if current_crop_size != self.input_dimension:
            image = self._resize_tensor(image, self.input_dimension, mode='bilinear')
            distance_map = self._resize_tensor(distance_map, self.input_dimension, mode='bilinear')
            segmentation = self._resize_tensor(segmentation, self.input_dimension, mode='nearest')

        return image.float(), distance_map.float(), segmentation.long()          

    
    def __len__(self):

        if (self.samples is None):
            return len(self.coords)

        if (self.samples > len(self.coords)):
            return len(self.coords)
        
        
        return self.samples

    def get_patches(self):
        
        np.random.shuffle(self.coords)
        
        image_patches = []

        for i in range(len(self)):
            image, depth, label = self[i]
            
            image_patches.append(image)
            
        return image_patches


class DatasetForInference(Dataset):
    """Dataset for inference that extracts overlapping patches from the image.
    
    Supports automatic resizing: crops are extracted at `crop_size` and 
    optionally resized to `input_dimension` for the model.
    
    Parameters
    ----------
    image_path : str
        Path to the orthoimage
    crop_size : int
        Size of crops to extract from the image
    overlap_rate : float
        Overlap rate between consecutive patches (0-1)
    input_dimension : int, optional
        Size to resize crops before feeding to model. If None, uses crop_size.
    """
    def __init__(self,
                image_path: str,
                crop_size: int,
                overlap_rate: float,
                input_dimension: Optional[int] = None
                ) -> None: 
        
        super().__init__()
        
        self.image_path = image_path
        self.crop_size = crop_size
        self.input_dimension = input_dimension if input_dimension is not None else crop_size
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

    def _resize_tensor(self, tensor: torch.Tensor, target_size: int, mode: str = 'bilinear') -> torch.Tensor:
        """Resize a tensor to target_size.
        
        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor of shape (C, H, W) or (H, W)
        target_size : int
            Target size for height and width
        mode : str
            Interpolation mode
        
        Returns
        -------
        torch.Tensor
            Resized tensor
        """
        if tensor.shape[-1] == target_size and tensor.shape[-2] == target_size:
            return tensor
        
        # Add batch dimension if needed
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
            squeeze_dims = 2
        elif tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
            squeeze_dims = 1
        else:
            squeeze_dims = 0
        
        resized = F.interpolate(
            tensor.float(),
            size=(target_size, target_size),
            mode=mode,
            align_corners=False if mode != 'nearest' else None
        )
        
        if squeeze_dims == 2:
            resized = resized.squeeze(0).squeeze(0)
        elif squeeze_dims == 1:
            resized = resized.squeeze(0)
        
        return resized

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

    def __getitem__(self, idx) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        """Get the data from the dataset
        
        Parameters
        ----------
        idx : int
            The index of the data to be loaded
        
        Returns
        -------
        Tuple[torch.Tensor, Tuple[int, int, int, int]]
            The image crop (resized to input_dimension) and the slice coordinates
            for placing output back into the original image (at crop_size)
        
        """
        current_coord = self.coords[idx].copy()
        
        image = self.read_window(
            coord=current_coord,
            image=self.image,
        )
        
        row_start, row_end, column_start, column_end = self.get_slice_window(
            current_coord,
        )
        
        # Resize to input_dimension if different from crop_size
        if self.input_dimension != self.crop_size:
            image = self._resize_tensor(image, self.input_dimension, mode='bilinear')
        
        return image.float(), (row_start, row_end, column_start, column_end)
    
    def __len__(self):

        return len(self.coords)


class MultiRegionDatasetFromCoord(Dataset):
    """Dataset for training that combines samples from multiple regions.
    
    Combines data from multiple orthoimages with their corresponding segmentations
    and distance maps. Samples are balanced across regions.
    
    Supports multi-scale cropping: if min_crop_size and max_crop_size are provided,
    the crop size is sampled uniformly between them for each sample.
    
    Parameters
    ----------
    image_paths : list of str
        Paths to the orthoimages for each region
    segmentation_paths : list of str
        Paths to the segmentation labels for each region
    distance_map_paths : list of str
        Paths to the distance maps for each region
    crop_size : int
        Size of crops to extract from the images (used when multi-scale is disabled)
    input_dimension : int, optional
        Size to resize crops before feeding to model. If None, uses crop_size.
    samples : int, optional
        Total number of samples per epoch (distributed across regions)
    augment : bool
        Whether to apply data augmentation
    copy_paste_augmentation : bool
        Whether to use copy-paste augmentation
    balance_regions : bool
        Whether to balance samples across regions (default True)
    min_crop_size : int, optional
        Minimum crop size for multi-scale cropping. If None, multi-scale is disabled.
    max_crop_size : int, optional
        Maximum crop size for multi-scale cropping. If None, multi-scale is disabled.
    """
    def __init__(self,
                image_paths: list,
                segmentation_paths: list,
                distance_map_paths: list,
                crop_size: int,
                input_dimension: Optional[int] = None,
                samples: int = None,
                augment: bool = False,
                copy_paste_augmentation: bool = False,
                balance_regions: bool = True,
                min_crop_size: Optional[int] = None,
                max_crop_size: Optional[int] = None
                ) -> None: 
        
        super().__init__()
        
        self.num_regions = len(image_paths)
        assert len(segmentation_paths) == self.num_regions, "Mismatch in number of segmentation paths"
        assert len(distance_map_paths) == self.num_regions, "Mismatch in number of distance map paths"
        
        self.image_paths = image_paths
        self.segmentation_paths = segmentation_paths
        self.distance_map_paths = distance_map_paths
        
        self.samples = samples
        self.crop_size = crop_size
        self.input_dimension = input_dimension if input_dimension is not None else crop_size
        self.augment = augment
        self.copy_paste_augmentation = copy_paste_augmentation
        self.balance_regions = balance_regions
        
        # Multi-scale crop settings
        self.min_crop_size = min_crop_size
        self.max_crop_size = max_crop_size
        self.multi_scale = (min_crop_size is not None and max_crop_size is not None 
                           and min_crop_size != max_crop_size)
        
        # Load all images into memory
        self.images = []
        self.img_segmentations = []
        self.img_depths = []
        self.image_shapes = []
        
        for i in range(self.num_regions):
            img = load_image(image_paths[i])
            seg = load_image(segmentation_paths[i])
            depth = load_image(distance_map_paths[i])
            
            self.images.append(img)
            self.img_segmentations.append(seg)
            self.img_depths.append(depth)
            self.image_shapes.append(img.shape)
        
        self.generate_coords()

    def generate_coords(self):
        """Generate coordinates for all regions with region index."""
        all_coords = []
        
        for region_idx in range(self.num_regions):
            seg = self.img_segmentations[region_idx]
            
            coords = np.where(seg != 0)
            coords = np.array(coords)
            coords = np.rollaxis(coords, 1, 0)
            
            coords_label = seg[np.nonzero(seg)]
            coords = oversample(coords, coords_label, "min")
            
            # Add region index as third column
            region_indices = np.full((len(coords), 1), region_idx)
            coords_with_region = np.hstack([coords, region_indices])
            
            all_coords.append(coords_with_region)
        
        # Combine all coordinates
        all_coords = np.vstack(all_coords)
        
        if self.balance_regions:
            # Balance samples across regions
            all_coords = self._balance_coords(all_coords)
        
        self.coords = all_coords

    def _balance_coords(self, coords: np.ndarray) -> np.ndarray:
        """Balance coordinates across regions by oversampling smaller regions."""
        region_coords = {}
        for region_idx in range(self.num_regions):
            mask = coords[:, 2] == region_idx
            region_coords[region_idx] = coords[mask]
        
        max_samples = max(len(c) for c in region_coords.values())
        
        balanced_coords = []
        for region_idx in range(self.num_regions):
            region_data = region_coords[region_idx]
            if len(region_data) < max_samples:
                # Oversample to match max
                indices = np.random.choice(len(region_data), max_samples, replace=True)
                region_data = region_data[indices]
            balanced_coords.append(region_data)
        
        return np.vstack(balanced_coords)

    def standardize_image_channels(self):
        """Normalize all images in place."""
        for i in range(self.num_regions):
            self.images[i] = self.images[i].astype("float32")
            normalize(self.images[i])

    def _resize_tensor(self, tensor: torch.Tensor, target_size: int, mode: str = 'bilinear') -> torch.Tensor:
        """Resize a tensor to target_size."""
        if tensor.shape[-1] == target_size and tensor.shape[-2] == target_size:
            return tensor
        
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
            squeeze_dims = 2
        elif tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
            squeeze_dims = 1
        else:
            squeeze_dims = 0
        
        resized = F.interpolate(
            tensor.float(),
            size=(target_size, target_size),
            mode=mode,
            align_corners=False if mode != 'nearest' else None
        )
        
        if squeeze_dims == 2:
            resized = resized.squeeze(0).squeeze(0)
        elif squeeze_dims == 1:
            resized = resized.squeeze(0)
        
        return resized

    def read_window_around_coord(self, coord: np.ndarray, image: np.ndarray, crop_size: Optional[int] = None) -> torch.Tensor:
        """Read a crop window around the coordinate."""
        # Use provided crop_size or default to self.crop_size
        current_crop_size = crop_size if crop_size is not None else self.crop_size
        
        image_crop = get_crop_image(image, image.shape, coord, current_crop_size)
        pad_width = get_pad_width(current_crop_size, coord, image.shape)
        
        image_crop = np.pad(
            image_crop, 
            pad_width=pad_width,
            mode="constant",
            constant_values=0
        )
        
        if (image_crop.shape[-1] != current_crop_size) or (image_crop.shape[-2] != current_crop_size):
            raise ValueError(f"Shape mismatch: {image_crop.shape}, expected {current_crop_size}")
        
        return torch.tensor(image_crop)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset.
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            The image crop, depth map crop, and label crop
        """
        coord_data = self.coords[idx].copy()
        current_coord = coord_data[:2].astype(int)
        region_idx = int(coord_data[2])
        
        # Determine crop size for this sample
        if self.multi_scale:
            current_crop_size = np.random.randint(self.min_crop_size, self.max_crop_size + 1)
            # Ensure crop size is even (required by get_crop_image/get_pad_width functions)
            if current_crop_size % 2 != 0:
                current_crop_size += 1
        else:
            current_crop_size = self.crop_size
        
        # Get the correct images for this region
        image_data = self.images[region_idx]
        seg_data = self.img_segmentations[region_idx]
        depth_data = self.img_depths[region_idx]
        
        if self.augment:
            uniform_dist_range = (-0.99, 0.99)
            random_row_prop = np.random.uniform(*uniform_dist_range)
            random_column_prop = np.random.uniform(*uniform_dist_range)
            current_coord[0] += int(random_row_prop * (current_crop_size // 2))
            current_coord[1] += int(random_column_prop * (current_crop_size // 2))
        
        image = self.read_window_around_coord(coord=current_coord, image=image_data, crop_size=current_crop_size)
        segmentation = self.read_window_around_coord(coord=current_coord, image=seg_data, crop_size=current_crop_size)
        distance_map = self.read_window_around_coord(coord=current_coord, image=depth_data, crop_size=current_crop_size)
        
        if self.augment:
            # Horizontal Flip
            if np.random.random() > 0.5:
                image = transforms.functional.hflip(image)
                segmentation = transforms.functional.hflip(segmentation)
                distance_map = transforms.functional.hflip(distance_map)
            
            # Vertical Flip
            if np.random.random() > 0.5:
                image = transforms.functional.vflip(image)
                segmentation = transforms.functional.vflip(segmentation)
                distance_map = transforms.functional.vflip(distance_map)
            
            # Random rotation
            angle = int(np.random.choice([0, 90, 180, 270]))
            image = transforms.functional.rotate(image.unsqueeze(0), angle).squeeze(0)
            segmentation = transforms.functional.rotate(segmentation.unsqueeze(0), angle).squeeze(0)
            distance_map = transforms.functional.rotate(distance_map.unsqueeze(0), angle).squeeze(0)
        
        # Resize to input_dimension (always needed for multi-scale, or when crop_size != input_dimension)
        if current_crop_size != self.input_dimension:
            image = self._resize_tensor(image, self.input_dimension, mode='bilinear')
            distance_map = self._resize_tensor(distance_map, self.input_dimension, mode='bilinear')
            segmentation = self._resize_tensor(segmentation, self.input_dimension, mode='nearest')
        
        return image.float(), distance_map.float(), segmentation.long()

    def __len__(self):
        if self.samples is None:
            return len(self.coords)
        if self.samples > len(self.coords):
            return len(self.coords)
        return self.samples

    def get_region_sample_counts(self) -> dict:
        """Get the number of samples per region."""
        counts = {}
        for region_idx in range(self.num_regions):
            counts[region_idx] = np.sum(self.coords[:, 2] == region_idx)
        return counts


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    INPUT_IMAGE = r"4x_amazon_input_data/orthoimage/orthoimage.tif"
    overlap_rates_to_test = [0.1, 0.3, 0.5]
    crop_size_to_test = [128, 256, 512]
    for batch_size in [8,16,32,64]:
        for overlap_rate in overlap_rates_to_test:
            for crop_size in crop_size_to_test:
                
                inference_dataset = DatasetForInference(
                    image_path = INPUT_IMAGE,
                    crop_size = crop_size,
                    overlap_rate = overlap_rate
                )
                
                inference_dataset.standardize_image_channels()
                
                inference_dataloader = torch.utils.data.DataLoader(
                    inference_dataset,
                    batch_size = batch_size,
                    shuffle = False,
                    num_workers = 0
                )
                
                # rebuild the own image
                output_image = np.zeros_like(inference_dataset.image)
                count_image = np.zeros_like(inference_dataset.image)
                
                for i, (image, slice) in enumerate(inference_dataloader):
                    
                    row_start, row_end, column_start, column_end = slice
                    
                    for j in range(image.shape[0]):
                        
                        output_image[:, 
                            row_start[j]:row_end[j], 
                            column_start[j]:column_end[j]
                        ] += image[j].numpy()
                        
                        count_image[:, 
                            row_start[j]:row_end[j], 
                            column_start[j]:column_end[j]
                        ] += 1
                
                
                count_image = np.where(count_image == 0, 1, count_image)
                output_image = output_image / count_image
                
                plt.imshow(np.moveaxis(output_image, 0,2))
                plt.savefig(f"test_data/inference_image_{overlap_rate}_{crop_size}_{batch_size}.png")
                plt.close()