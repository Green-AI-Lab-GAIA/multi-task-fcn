import torch
from torch import nn


def conv_padding_same(kernel_size , **kwargs):
    if "dilation" in kwargs.keys():
        padding = ((kernel_size - 1) * kwargs["dilation"]) // 2

    else:
        padding = (kernel_size - 1) // 2

    return nn.Conv2d(**kwargs, padding = padding, kernel_size = kernel_size)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: bool,
    ) -> None:
        super(ResidualBlock, self).__init__()
        

        
        self.downsample = downsample

        self.batch_norm1 = nn.BatchNorm2d(
            num_features=in_channels,
            momentum=0.9,
            eps=1e-05,
            affine=True,
            track_running_stats=True,
        )
        self.elu = nn.ELU(inplace=True)

        if downsample:
  
            self.conv_down1 = conv_padding_same(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2
            )

            self.conv_down2 = conv_padding_same(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2,
            )
            in_channels = out_channels


        self.conv1 = conv_padding_same(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
        )

        self.batch_norm2 = nn.BatchNorm2d(
            num_features=in_channels,
            momentum=0.9,
            eps=1e-05,
            affine=True,
            track_running_stats=True,
        )

        self.conv2 = conv_padding_same(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
        )

    def forward(self, x):
        x_init = x

        x = self.batch_norm1(x)
        x = self.elu(x)

        if self.downsample:
            x = self.conv_down1(x)
            x_init = self.conv_down2(x_init)

        x = self.batch_norm2(x)
        x = self.conv2(x)

        return x + x_init


class ASPPModule(nn.Module):
    def __init__(self, pool_height, pool_width, in_channels, out_channels = 128):
        super(ASPPModule, self).__init__()
        

        self.aspp1x1 = conv_padding_same(in_channels = in_channels, out_channels=out_channels, kernel_size=1)
        self.aspp3x3_1 = conv_padding_same(in_channels = in_channels, out_channels=out_channels, kernel_size=3, dilation=6)
        self.aspp3x3_2 = conv_padding_same(in_channels = in_channels, out_channels=out_channels, kernel_size=3, dilation=12)
        self.aspp3x3_3 = conv_padding_same(in_channels = in_channels, out_channels=out_channels, kernel_size=3, dilation=18)

        self.image_feature = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            conv_padding_same(in_channels = in_channels, out_channels=out_channels, kernel_size=1),
            nn.Upsample(size=(pool_height, pool_width), mode='bilinear', align_corners=True)
        )

        self.batch_norm = nn.BatchNorm2d(out_channels*5, 
                                         momentum=0.9,
                                            eps=1e-05,
                                            affine=True,
                                            track_running_stats=True,)
        
        self.relu = nn.ELU(inplace=True)


    def forward(self, x):

        aspp1x1 = self.aspp1x1(x)
        aspp3x3_1 = self.aspp3x3_1(x)
        aspp3x3_2 = self.aspp3x3_2(x)
        aspp3x3_3 = self.aspp3x3_3(x)

        image_feature = self.image_feature(x)

        outputs = torch.cat([aspp1x1, aspp3x3_1, aspp3x3_2, aspp3x3_3, image_feature], dim=1)
        outputs = self.batch_norm(outputs)
        outputs = self.relu(outputs)

        return outputs



class DeepLabv3Plus_resnet9(nn.Module):
    def __init__(self, num_ch, psize, num_class, dropout_rate=0.65):
        super(DeepLabv3Plus_resnet9, self).__init__()
        
        if psize != 128:
            raise NotImplementedError("The model does not support patch sizes other than 128x128. Please use a patch size of 128x128 for this model.")
        
        self.num_ch = num_ch
        self.psize = psize
        self.nb_class = num_class
        self.dropout_rate = dropout_rate

        self.conv1 = conv_padding_same(in_channels = self.num_ch,
                                       out_channels = 64,
                                       kernel_size = 3, 
                                       stride = 1)
        
        self.res_block1 = ResidualBlock(in_channels=64,
                                        out_channels=64,
                                        downsample=False)
        
        self.res_block2 = ResidualBlock(in_channels=64,
                                        out_channels=128,
                                        downsample=True)
        
        self.res_block3 = ResidualBlock(in_channels=128,
                                        out_channels=128,
                                        downsample=False)
    
        self.res_block4 = ResidualBlock(in_channels=128,
                                        out_channels=256,
                                        downsample=True)
        
        self.res_block5 = ResidualBlock(in_channels=256,
                                        out_channels=256,
                                        downsample=False)

        self.batch_norm1 = nn.BatchNorm2d(num_features=256, 
                                          momentum=0.9, 
                                          eps=1e-05, 
                                          affine=True, 
                                          track_running_stats=True)

        self.elu1 = nn.ELU(inplace=True)

        ####### CLASSIFICATION #######
        self.atrous_pyramid_pooling = ASPPModule(pool_height=32, 
                                                 pool_width=32, 
                                                 in_channels=256, 
                                                 out_channels=128)

        self.conv2 = conv_padding_same(kernel_size = 1,
                                       in_channels = 640,
                                       out_channels = 128
                                       )

        self.batch_norm2 = nn.BatchNorm2d(num_features=128, 
                                          momentum=0.9, 
                                          eps=1e-05, 
                                          affine=True, 
                                          track_running_stats=True)
        self.elu1 = nn.ELU(inplace=True)
        # UpSampling based in skip connection shape
        self.upsample1 = nn.Upsample(size = (64, 64), 
                                     mode='bilinear', 
                                     align_corners=True)

        self.conv3 = conv_padding_same(kernel_size = 3, 
                                       in_channels = 256, 
                                       out_channels = 128)
        self.elu2 = nn.ELU(inplace=True)

        self.upsample2 = nn.Upsample(size = (psize, psize),
                                     mode = "bilinear",
                                     align_corners=True,
                                     )
        self.dropout = nn.Dropout(p=self.dropout_rate)

        
        self.conv_class = conv_padding_same(in_channels=128,
                                            out_channels=self.nb_class,
                                            kernel_size=1,
                                            stride=1,
                                            dilation=1)

        ####### REG #######
        self.conv4 = conv_padding_same(in_channels = 256,
                                       out_channels = 128,
                                       kernel_size = 1)
        
        self.pool1 = nn.MaxPool2d(kernel_size=2,
                                  stride=2)
        
        self.upsample3 = nn.Upsample(size = (64, 64),
                                     mode = "bilinear",
                                     align_corners=True
                                     )
        
        self.conv5 = conv_padding_same(in_channels = 256,
                                       out_channels = 128,
                                       kernel_size=3)
        
        self.batch_norm3 = nn.BatchNorm2d(num_features=128, 
                                          momentum=0.9, 
                                          eps=1e-05, 
                                          affine=True, 
                                          track_running_stats=True)
        self.elu3 = nn.ELU(inplace=True)

        self.upsample4 = nn.Upsample(size = (psize, psize),
                                     mode = "bilinear",
                                     align_corners = True
                                     )

        self.conv_reg = conv_padding_same(in_channels=128,
                                          out_channels=1,
                                          kernel_size=3,
                                          stride=1,
                                          dilation=1)


    def enconder(self, x):
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        x_skip = x

        x = self.res_block4(x)
        x = self.res_block5(x)

        x = self.batch_norm1(x)
        x = self.elu1(x)
                
        return x, x_skip
    
    def decoder_class(self, x, x_skip):
        x = self.atrous_pyramid_pooling(x)
        
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.elu1(x)

        x = self.upsample1(x)

        x = torch.cat([x_skip, x], axis = 1)
    
        x = self.conv3(x)
        x = self.elu2(x)

        x = self.upsample2(x)
        x = self.dropout(x)
        
        x = self.conv_class(x)
       
        return x


    def decoder_aux(self, x, x_skip):
        x = self.conv4(x)
        x = self.pool1(x)
        x = self.upsample3(x)
        
        x = torch.cat([x_skip, x], axis = 1)

        x = self.conv5(x)
        x = self.batch_norm3(x)
        x = self.elu3(x)
        x = self.upsample4(x)

        x = self.conv_reg(x)
        
        return x
        


    def forward(self, x):
        output = dict()
        x, x_skip = self.enconder(x)

        output["out"] = self.decoder_class(x, x_skip)
        
        output["aux"] = self.decoder_aux(x, x_skip)

        return output



if __name__ == "__main__":

    image = torch.randn(1, 25, 128, 128)

    model = DeepLabv3Plus_resnet9(
        num_ch = 25,
        num_class = 14,
        psize = 128,
    )

    output = model(image)
    print(output["aux"])

    image = torch.randn(1, 128, 32, 32)

    model = ASPPModule(image.size(2), 
                       image.size(3), 
                       image.size(1), 
                       128)
    model.eval()
    
    with torch.no_grad():
        output = model(image)

    # image = torch.randn(1, 25, 128, 128)

    # model = ResidualBlock(in_channels=64, out_channels=64, downsample=False)

    # with torch.no_grad():
    #     output = model(image)

    # print(output.size())

    # model = ResidualBlock(in_channels = 64, out_channels=64, downsample=True)

    # with torch.no_grad():
    #     output = model(image)
    
    # print(output.size())

    

    model.eval()

    image = torch.randn(1, 25, 128, 128)

    with torch.no_grad():
        x, x_skip = model.enconder(image)
    
    print(x.size())
    print(x.size())
