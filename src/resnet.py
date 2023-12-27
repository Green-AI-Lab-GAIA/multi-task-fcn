import copy

import torch
import torch.nn as nn


class new_model(nn.Module):
    def __init__(self, model, channels, nb_classes,
                 output_layer = 'backbone'):
        super().__init__()
        self.encoder = copy.deepcopy(model)
        self.dec_class = copy.deepcopy(model)
        self.dec_dist = copy.deepcopy(model)
        self.layers = list(self.model._modules.keys())
        self.output_layer = output_layer
        self.layer_count = 0
        
        self.encoder.backbone.conv1 = nn.Conv2d(channels, 64, 
                                                kernel_size=(7, 7), 
                                                stride=(2, 2), 
                                                padding=(3, 3), 
                                                bias=False)

        
        for l in self.layers:
            if l != self.output_layer:
                self.layer_count += 1
            else:
                break
        for i in range(1,len(self.layers)-self.layer_count):
            self.dummy_var = self.self.encoder._modules.pop(self.layers[-i])
            
        for i in range(1,len(self.layers)-self.layer_count):
            self.dummy_var = self.self.dec_class._modules.pop(self.layers[i])
            
        for i in range(1,len(self.layers)-self.layer_count):
            self.dummy_var = self.self.dec_dist._modules.pop(self.layers[i])
        
        self.feat_extract = nn.Sequential(self.encoder._modules)
        self.decoder_cl = nn.Sequential(self.dec_class._modules)
        self.decoder_dist = nn.Sequential(self.dec_dist._modules)
        
        self.decoder_cl.aux_classifier[4] = nn.Conv2d(256, nb_classes, kernel_size=(1, 1), stride=(1, 1))
        self.decoder_cl.classifier[4] = nn.Conv2d(256, nb_classes, kernel_size=(1, 1), stride=(1, 1))

        self.decoder_dist.aux_classifier[4] = nn.Conv2d(256, nb_classes, kernel_size=(1, 1), stride=(1, 1))
        self.decoder_dist.classifier[4] = nn.Conv2d(256, nb_classes, kernel_size=(1, 1), stride=(1, 1))



    def forward(self,x):
        x = self.feat_extract(x)
        out1 = self.decoder_cl(x)
        out2 = self.decoder_dist(x)
        return out1, out2


class ResUnet(nn.Module):
    def __init__(self, channel, nb_classes, filters=[64, 128, 256, 512]):
        super(ResUnet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )
        
        self.drop = nn.Dropout2d(p=0.4)

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        # self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)
        self.up_residual_conv1 = ResidualConv(filters[3], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        # self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)
        self.up_residual_conv2 = ResidualConv(filters[2], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)
        self.up_residual_conv3 = ResidualConv(filters[1], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], nb_classes, 1, 1),
        )
        
        # self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        # x1 = self.drop(x1)
        x2 = self.residual_conv_1(x1)
        # x2 = self.drop(x2)
        x3 = self.residual_conv_2(x2)
        # x3 = self.drop(x3)
        # Bridge
        x4 = self.bridge(x3)
        # Decode
        x4 = nn.functional.interpolate(x4, size=x3.size()[2:], mode='bilinear', align_corners=True)
        # x4 = self.upsample_1(x4)
        # x5 = torch.cat([x4, x3], dim=1)
        x5 = x4 + x3
        # x5 = self.drop(x5)
        x6 = self.up_residual_conv1(x5)

        x6 = nn.functional.interpolate(x6, size=x2.size()[2:], mode='bilinear', align_corners=True)
        # x6 = self.upsample_2(x6)
        # x7 = torch.cat([x6, x2], dim=1)
        x7 = x6 + x2
        # x7 = self.drop(x7)
        x8 = self.up_residual_conv2(x7)

        x8 = nn.functional.interpolate(x8, size=x1.size()[2:], mode='bilinear', align_corners=True)
        # x8 = self.upsample_3(x8)
        # x9 = torch.cat([x8, x1], dim=1)
        x9 = x8 + x1
        # x9 = self.drop(x9)
        x10 = self.up_residual_conv3(x9)
        # x10 = self.drop(x10)
        output = self.output_layer(x10)

        return output
    
    
class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()
        
        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


class Squeeze_Excite_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Squeeze_Excite_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ASPP(nn.Module):
    def __init__(self, in_dims, out_dims, rate=[6, 12, 18]):
        super(ASPP, self).__init__()

        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )

        self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
        self._init_weights()

    def forward(self, x):
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Upsample_(nn.Module):
    def __init__(self, scale=2):
        super(Upsample_, self).__init__()

        self.upsample = nn.Upsample(mode="bilinear", scale_factor=scale)

    def forward(self, x):
        return self.upsample(x)


class AttentionBlock(nn.Module):
    def __init__(self, input_encoder, input_decoder, output_dim):
        super(AttentionBlock, self).__init__()

        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(input_encoder),
            nn.ReLU(),
            nn.Conv2d(input_encoder, output_dim, 3, padding=1),
            nn.MaxPool2d(2, 2),
        )

        self.conv_decoder = nn.Sequential(
            nn.BatchNorm2d(input_decoder),
            nn.ReLU(),
            nn.Conv2d(input_decoder, output_dim, 3, padding=1),
        )

        self.conv_attn = nn.Sequential(
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, 1, 1),
        )

    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2
    
if __name__ == "__main__":
    # model = ResUnet(channel=8, nb_classes = 8, filters=[3, 3, 3, 3])
    pret = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    model = new_model(pret, 8, 8,
                     output_layer = 'backbone')
    model.eval()
    image = torch.randn(5, 8, 128, 128)
    with torch.no_grad():
        output = model.forward(image)
    print(output.size())
    