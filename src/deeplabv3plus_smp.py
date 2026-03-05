import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class DeepLabV3Plus_SMP(nn.Module):
    """
    Wrapper for DeepLabV3+ from segmentation_models_pytorch with ResNet18 encoder
    and custom auxiliary head for depth map (multitask).
    
    This wrapper maintains compatibility with the existing multitask pipeline
    by returning dict(out=segmentation, aux=depth_map).
    """
    
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 pretrained: bool = True,
                 dropout_rate: float = 0.5,
                 batch_norm: bool = False,
                 psize: int = 256):
        """
        Parameters
        ----------
        in_channels : int
            Number of input channels
        num_classes : int
            Number of segmentation classes
        pretrained : bool
            Whether to use pretrained encoder weights
        dropout_rate : float
            Dropout rate for segmentation head
        batch_norm : bool
            Whether to apply batch normalization (not used in SMP, kept for compatibility)
        psize : int
            Input size (used for determining ASPP rates, kept for compatibility)
        """
        super(DeepLabV3Plus_SMP, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        self.psize = psize
        
        # Build DeepLabV3+ model with ResNet18 encoder
        encoder_weights = 'imagenet' if pretrained else None
        
        self.segmentation_model = smp.DeepLabV3Plus(
            encoder_name='resnet18',
            encoder_depth=5,
            encoder_weights=encoder_weights,
            encoder_output_stride=16,
            decoder_channels=256,
            decoder_atrous_rates=(12, 24, 36),
            decoder_aspp_separable=True,
            decoder_aspp_dropout=dropout_rate,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,  # Return logits
            upsampling=4,
            aux_params=None,  # We'll create our own aux head
        )
        
        # Get encoder to access features for auxiliary head
        self.encoder = self.segmentation_model.encoder
        self.decoder = self.segmentation_model.decoder
        self.segmentation_head = self.segmentation_model.segmentation_head
        
        # Determine decoder output channels (usually 256 for DeepLabV3+)
        decoder_output_channels = 256
        
        # Build auxiliary head for depth map
        # Similar structure to __build_features_depth in deepvlab3plus.py
        depth = 128  # Intermediate depth for aux head
        
        # Get low-level features channel count dynamically from encoder
        # For ResNet18, encoder stages output: [64, 64, 128, 256, 512]
        # DeepLabV3+ decoder uses stage 1 (index 1) for low-level features
        # We'll determine this dynamically by checking encoder output
        # For now, using known value for ResNet18: stage 1 outputs 64 channels
        low_level_channels = 64  # From stage 1 (layer1 output)
        
        # Auxiliary head: process decoder features and combine with low-level features
        self.conv1depth = nn.Sequential(
            nn.Conv2d(decoder_output_channels, depth,
                     kernel_size=(1, 1),
                     stride=1,
                     padding=(0, 0),
                     bias=False),
            nn.BatchNorm2d(depth),
            nn.ReLU()
        )
        
        self.conv2depth = nn.Sequential(
            nn.Conv2d(depth + low_level_channels, depth,
                     kernel_size=(3, 3),
                     stride=1,
                     padding=(1, 1),
                     bias=False),
            nn.BatchNorm2d(depth),
            nn.ReLU()
        )
        
        self.outdepth = nn.Conv2d(depth, 1,
                                  kernel_size=(3, 3),
                                  stride=1,
                                  padding=(1, 1),
                                  bias=False)
    
    def forward(self, x):
        """
        Forward pass returning segmentation and depth map.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, in_channels, H, W)
        
        Returns
        -------
        dict
            Dictionary with keys:
            - 'out': segmentation logits (B, num_classes, H, W)
            - 'aux': depth map (B, 1, H, W)
        """
        input_size = x.size()[2:]
        
        # Get encoder features
        # SMP encoder returns a list of features from different stages
        features = self.encoder(x)
        
        # Get decoder output (segmentation features)
        # DeepLabV3+ decoder uses features[0] (high-level) and features[1] (low-level)
        decoder_output = self.decoder(*features)
        
        # Get segmentation logits
        segmentation_logits = self.segmentation_head(decoder_output)
        
        # Build depth map using auxiliary head
        # Use high-level features from decoder and low-level features from encoder
        # decoder_output is already processed by decoder, shape: (B, decoder_channels, H/4, W/4)
        high_level_features = decoder_output  # (B, 256, H/4, W/4)
        # features[1] is low-level features from encoder stage 1 (before decoder processing)
        # For ResNet18: features[1] shape is (B, 64, H/4, W/4)
        low_level_features = features[1]  # Stage 1 features (B, 64, H/4, W/4)
        
        # Process high-level features
        depth_features = self.conv1depth(high_level_features)
        
        # Upsample to match low-level features size (they should already match, but ensure)
        if depth_features.size()[2:] != low_level_features.size()[2:]:
            depth_features = F.interpolate(
                depth_features,
                size=low_level_features.size()[2:],
                mode='bilinear',
                align_corners=True
            )
        
        # Concatenate with low-level features
        combined_features = torch.cat([depth_features, low_level_features], dim=1)
        
        # Process combined features
        depth_features = self.conv2depth(combined_features)
        
        # Upsample to input resolution
        depth_map = F.interpolate(
            depth_features,
            size=input_size,
            mode='bilinear',
            align_corners=True
        )
        
        # Final output layer
        depth_map = self.outdepth(depth_map)
        
        return dict(out=segmentation_logits, aux=depth_map)


if __name__ == "__main__":
    # Test the model
    model = DeepLabV3Plus_SMP(
        in_channels=25,
        num_classes=4,
        pretrained=False,
        dropout_rate=0.5,
        batch_norm=False,
        psize=256
    )
    
    model.eval()
    
    # Test input
    image = torch.randn(1, 25, 256, 256)
    
    with torch.no_grad():
        output = model(image)
    
    print(f"Segmentation output shape: {output['out'].shape}")
    print(f"Depth map output shape: {output['aux'].shape}")
    print(f"Expected segmentation: (1, 4, 256, 256)")
    print(f"Expected depth map: (1, 1, 256, 256)")
