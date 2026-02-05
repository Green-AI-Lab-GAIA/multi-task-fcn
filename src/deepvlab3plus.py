import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_inplanes():
    return [64, 128, 256, 256]


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(
            self,
            block,
            layers,
            seq=3,
            zero_init_residual=False,
            groups=1,
            widen=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
    ):
        super(ResNet, self).__init__()

        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.depth = len(layers)

        self.padding = nn.ConstantPad2d(1, 0.0)

        self.inplanes = width_per_group * widen
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # change padding 3 -> 2 compared to original torchvision code because added a padding layer
        num_out_filters = width_per_group * widen
        self.conv1 = nn.Conv2d(
            seq, num_out_filters, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = norm_layer(num_out_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, num_out_filters, layers[0])
        num_out_filters *= 2
        self.layer2 = self._make_layer(
            block, num_out_filters, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        if self.depth > 2:
            self.layer3 = self._make_layer(
                block, num_out_filters, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
            )
            if self.depth > 3:
                self.layer4 = self._make_layer(
                    block, num_out_filters, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
                )

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        low_level_feat = x
        if self.depth>2:
            x = self.layer3(x)
            if self.depth>3:
                x = self.layer4(x)
            
        return x, low_level_feat
    
    
class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = (1,1)
            padding = (0,0)
        else:
            kernel_size = (3, 3)
            padding = (rate, rate)
        self.atrous_convolution = nn.Conv2d(inplanes, planes, 
                                            kernel_size=kernel_size,
                                            stride=1, 
                                            padding=padding, 
                                            dilation=(rate,rate), 
                                            bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

class DeepLabv3_plus(nn.Module):
    def __init__(self, 
                 model_depth, 
                 num_ch_1 = 8,
                 psize = 64,
                 nb_class = None):
        super(DeepLabv3_plus, self).__init__()
        self.num_ch_1 = num_ch_1
        self.psize = psize
        self.nb_class = nb_class
        
        ##### processing layer for classification
        
        
        # Atrous Conv
        assert model_depth in [10, 18, 34]
    
        if model_depth == 10:
            self.resnet_features_1 = ResNet(BasicBlock, 
                                            [1, 1, 1, 1], 
                                            seq=self.num_ch_1)
        elif model_depth == 18:
            self.resnet_features_1 = ResNet(BasicBlock, 
                                            [2, 2, 2, 2], 
                                            seq=self.num_ch_1)
        elif model_depth == 34:
            self.resnet_features_1 = ResNet(BasicBlock, 
                                            [3, 4, 6, 3], 
                                            seq=self.num_ch_1)
            

        # ASPP
        if self.psize >= 256:
            rates = [1, 6, 12, 18]
        else:
            rates = [1, 3, 6, 9]
        
        depth = 128

        self.aspp1 = ASPP_module(128, depth, rate=rates[0])
        self.aspp2 = ASPP_module(128, depth, rate=rates[1])
        self.aspp3 = ASPP_module(128, depth, rate=rates[2])
        self.aspp4 = ASPP_module(128, depth, rate=rates[3])

        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(128, depth, 
                                                       kernel_size=(1,1),
                                                       stride=1, 
                                                       padding=(0,0), 
                                                       bias=False),
                                             nn.BatchNorm2d(depth),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(depth*5, depth, 
                               kernel_size=(3,3),
                               stride=1, 
                               padding=(1,1), 
                               bias=False)
        self.bn1 = nn.BatchNorm2d(depth)


        self.conv2 = nn.Conv2d(128, depth, 
                               kernel_size=(3,3),
                               stride=1, 
                               padding=(1,1), 
                               bias=False)
        self.bn2 = nn.BatchNorm2d(depth)
        
        self.conv3 = nn.Conv2d(depth*2, depth, 
                               kernel_size=(3,3),
                               stride=1, 
                               padding=(1,1), 
                               bias=False)
        self.bn3 = nn.BatchNorm2d(depth)
        
        
        self.drop = nn.Dropout2d(p=0.65)
        
        if not self.nb_class:
            self.out_layer = nn.Conv2d(depth, 1, 
                                        kernel_size=(3,3), 
                                        stride=1, 
                                        padding=(1,1), bias=False)
        else:
            self.out_layer = nn.Conv2d(depth, self.nb_class, 
                                        kernel_size=(3,3), 
                                        stride=1, 
                                        padding=(1,1), bias=False)
            
            
        ##### porcessing layer for distance map
        self.conv1depth = nn.Sequential(nn.Conv2d(128, depth, 
                                      kernel_size=(1,1),
                                      stride=1, 
                                      padding=(0,0), 
                                      bias=False),
                                   nn.BatchNorm2d(depth),
                                   nn.ReLU())
        
        self.conv2depth = nn.Sequential(nn.Conv2d(depth*2, depth, 
                                      kernel_size=(3,3),
                                      stride=1, 
                                      padding=(1,1), 
                                      bias=False),
                                   nn.BatchNorm2d(depth),
                                   nn.ReLU())
        
        self.outdepth = nn.Sequential(nn.Conv2d(depth, 1, 
                                      kernel_size=(3,3),
                                      stride=1, 
                                      padding=(1,1), 
                                      bias=False))
            

    def __build_features_class(self, x, low_level_features, size):

        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = nn.functional.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = nn.functional.interpolate(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=True)
        x = self.drop(x)
        x = self.out_layer(x)
       
        return x
    
    
    def __build_features_depth(self, x, low_level_features, size):
        x = self.conv1depth(x)
        
        x = nn.functional.interpolate(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat((x, low_level_features), dim=1)
        
        x = self.conv2depth(x)
        
        x = nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=True)
        
        x = self.outdepth(x)
       
        return x
    
    def forward(self, x1):
        size = x1.size()[2:]
        x, low_level_features = self.resnet_features_1(x1)

        logits_class = self.__build_features_class(x, low_level_features, size)
        
        logits_distance = self.__build_features_depth(x, low_level_features, size)

        return dict(out = logits_class, aux = logits_distance)
            

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.eval()
                
if __name__ == "__main__":
    model = DeepLabv3_plus(10, 
                 num_ch_1=25,
                 psize = 128,
                 nb_class = 14)
    
    model.eval()

    image = torch.randn(1, 25, 128, 128)
    
    with torch.no_grad():
        output_cl, output_depth = model.forward(image)
    print(output_cl.size())
    print(output_depth.size())
    