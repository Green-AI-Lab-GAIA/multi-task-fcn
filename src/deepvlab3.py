import sys
import os
from os.path import join, abspath, dirname, isdir

ROOT_PATH = dirname(dirname(abspath(__file__)))
sys.path.append(ROOT_PATH)

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_mobilenet_v3_large

from src.utils import check_folder

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.downsample = nn.MaxPool2d(kernel_size=scale_factor)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.downsample(x)
        return x

class DeepLabv3(nn.Module):
    def __init__(self, 
                 in_channels:int, 
                 num_classes:int, 
                 pretrained:bool, 
                 dropout_rate:float, 
                 batch_norm:bool=False,
                 downsampling_factor:int=None
                 ):
        super(DeepLabv3, self).__init__()
        
        self.in_channels = in_channels
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.downsampling_factor = downsampling_factor
        
        self.load_model()
         
        self.model.backbone.conv1 = nn.Conv2d(
            in_channels = in_channels,
            out_channels= 64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), 
            bias=False
        )
        
        self.model.aux_classifier[4] = nn.Conv2d(
            in_channels = 256, 
            out_channels = 1, 
            kernel_size=(1, 1), 
            stride=(1, 1)
        )
        
        self.model.classifier[4] = nn.Sequential(
            nn.Dropout(p = dropout_rate),
            nn.Conv2d(
                in_channels = 256, 
                out_channels = num_classes, 
                kernel_size=(1, 1), 
                stride=(1, 1)
            )
        )
        
        if downsampling_factor is not None:
            self.downsample_layer = DownsampleBlock(
                in_channels = in_channels,
                out_channels = in_channels,
                scale_factor = downsampling_factor
            )
            
            # Upsample layer for output
            self.upsample_layer = UpsampleBlock(
                in_channels = num_classes,
                out_channels = num_classes,
                scale_factor = downsampling_factor
            )
            
            self.upsample_layer_aux = UpsampleBlock(
                in_channels = 1,
                out_channels = 1,
                scale_factor = downsampling_factor
            )
            
            self.shortcut1= nn.Conv2d(
                in_channels = in_channels,
                out_channels = num_classes,
                kernel_size = (1,1),
                stride = (1,1)
            )
            self.shortcut2= nn.Conv2d(
                in_channels = in_channels,
                out_channels = 1,
                kernel_size = (1,1),
                stride = (1,1)
            )
            
        if batch_norm:
            self.batch_norm_layer = nn.BatchNorm2d(in_channels)
        
    def load_model(self):
        if self.pretrained:
            model_path = join(ROOT_PATH, 'pretrained_models', 'deeplabv3_resnet50')
        else:
            model_path = join(ROOT_PATH, 'random_w_models', 'deeplabv3_resnet50')
        
        if isdir(model_path):
            model_file = os.listdir(model_path)
            self.model = torch.load(join(model_path, model_file[0]),
                                    weights_only=False)
        
        else:
            check_folder(model_path)
            self.model = deeplabv3_resnet50(pretrained=self.pretrained,
                                            aux_loss=True)
            torch.save(self.model, join(model_path, 'model'))
    
    def forward(self, x):
        
        if self.batch_norm:
            x = self.batch_norm_layer(x)
        
        skip_connection = x.clone()
        
        if self.downsampling_factor is not None:
            x = self.downsample_layer(x)
        
        x = self.model(x)
        
        if self.downsampling_factor is not None:
            x['out'] = self.upsample_layer(x['out']) + self.shortcut1(skip_connection)
            x['aux'] = self.upsample_layer_aux(x['aux']) + self.shortcut2(skip_connection)
        
        return x
        

if __name__ == "__main__":

    model = DeepLabv3(
        in_channels = 3,
        num_classes = 17,
        pretrained = True,
        dropout_rate = 0.5,
        batch_norm = False,
        downsampling_factor = 2
    )
    
    # Test the model
    input_tensor = torch.randn(16, 3, 224, 224)  # Example input tensor
    output = model(input_tensor)
    
    
    downsampling_factor = 2
    downsample_layer = nn.Conv2d(
        in_channels = 3,
        out_channels = 3,
        kernel_size = (3,3),
        stride = (downsampling_factor, downsampling_factor),
        padding = (1,1)
    )
    
    print(input_tensor.shape)
    output = downsample_layer(input_tensor)
    print(output.shape)