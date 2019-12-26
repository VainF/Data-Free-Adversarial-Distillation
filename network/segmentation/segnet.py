import torch.nn as nn

from .utils import segnetDown2, segnetDown3, segnetDown4, segnetUp2, segnetUp3,segnetUp4
from torchvision.models import vgg
import torch

class SegNetVgg16(nn.Module):
    def __init__(self, num_classes=21, in_channels=3, is_unpooling=True, pretrained_backbone=False):
        super(SegNetVgg16, self).__init__()
        #self.num_classes = num_classes
        self.in_channels = in_channels
        self.is_unpooling = is_unpooling

        self.down1 = segnetDown2(self.in_channels, 64)
        self.down2 = segnetDown2(64, 128)
        self.down3 = segnetDown3(128, 256)
        self.down4 = segnetDown3(256, 512)
        self.down5 = segnetDown3(512, 512)

        self.up5 = segnetUp3(512, 512)
        self.up4 = segnetUp3(512, 256)
        self.up3 = segnetUp3(256, 128)
        self.up2 = segnetUp2(128, 64)
        self.up1 = segnetUp2(64, num_classes)

        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        nn.init.kaiming_normal_(m.weight)
        #        nn.init.constant_(m.bias, 0)
        #    elif isinstance(m, nn.BatchNorm2d):
        #        nn.init.constant_(m.weight, 1)
        #        nn.init.constant_(m.bias, 0)
        #self.init_vgg16_params()

        if pretrained_backbone==True:
            self.init_vgg16_params()


    def forward(self, inputs, positive_out=False):

        down1, indices_1, unpool_shape1 = self.down1(inputs)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        down4, indices_4, unpool_shape4 = self.down4(down3)
        down5, indices_5, unpool_shape5 = self.down5(down4)

        up5 = self.up5(down5, indices_5, unpool_shape5)
        up4 = self.up4(up5, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        up1 = self.up1(up2, indices_1, unpool_shape1)
        if positive_out:
            return torch.relu( up1 )
        return up1

    def init_vgg16_params(self):

        vgg16 = vgg.vgg16_bn(pretrained=True)

        blocks = [self.down1, self.down2, self.down3, self.down4, self.down5]

        features = list(vgg16.features.children())

        vgg_layers = []
        for _layer in features:
            if isinstance(_layer, nn.Conv2d):
                vgg_layers.append(_layer)
            elif isinstance(_layer, nn.BatchNorm2d):
                vgg_layers.append(_layer)
        
        merged_layers = []
        for idx, conv_block in enumerate(blocks):
            if idx < 2:
                units = [conv_block.conv1.cbr_unit, conv_block.conv2.cbr_unit]
            else:
                units = [
                    conv_block.conv1.cbr_unit,
                    conv_block.conv2.cbr_unit,
                    conv_block.conv3.cbr_unit,
                ]
            for _unit in units:
                for _layer in _unit:
                    if isinstance(_layer, nn.Conv2d):
                        merged_layers.append(_layer)
                    elif isinstance(_layer, nn.BatchNorm2d):
                        merged_layers.append(_layer)

        assert len(vgg_layers) == len(merged_layers)

        for l1, l2 in zip(vgg_layers, merged_layers):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
            elif isinstance(l1, nn.BatchNorm2d) and isinstance(l2, nn.BatchNorm2d):
                l2.running_mean.data = l1.running_mean.data
                l2.running_var.data = l1.running_var.data
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data

class SegNetVgg19(nn.Module):
    def __init__(self, num_classes=21, in_channels=3, is_unpooling=True, pretrained_backbone=False):
        super(SegNetVgg19, self).__init__()
        #self.num_classes = num_classes
        self.in_channels = in_channels
        self.is_unpooling = is_unpooling

        self.down1 = segnetDown2(self.in_channels, 64)
        self.down2 = segnetDown2(64, 128)
        self.down3 = segnetDown4(128, 256)
        self.down4 = segnetDown4(256, 512)
        self.down5 = segnetDown4(512, 512)

        self.up5 = segnetUp4(512, 512)
        self.up4 = segnetUp4(512, 256)
        self.up3 = segnetUp4(256, 128)
        self.up2 = segnetUp2(128, 64)
        self.up1 = segnetUp2(64, num_classes)

        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        nn.init.kaiming_normal_(m.weight)
        #        nn.init.constant_(m.bias, 0)
        #    elif isinstance(m, nn.BatchNorm2d):
        #        nn.init.constant_(m.weight, 1)
        #        nn.init.constant_(m.bias, 0)
        if pretrained_backbone==True:
            self.init_vgg19_params()
        

    def forward(self, inputs, positive_out=False):

        down1, indices_1, unpool_shape1 = self.down1(inputs)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        down4, indices_4, unpool_shape4 = self.down4(down3)
        down5, indices_5, unpool_shape5 = self.down5(down4)

        up5 = self.up5(down5, indices_5, unpool_shape5)
        up4 = self.up4(up5, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        up1 = self.up1(up2, indices_1, unpool_shape1)
        if positive_out:
            return torch.relu( up1 )
        return up1

    def init_vgg19_params(self):

        vgg19 = vgg.vgg19_bn(pretrained=True)

        blocks = [self.down1, self.down2, self.down3, self.down4, self.down5]

        features = list(vgg19.features.children())

        vgg_layers = []
        for _layer in features:
            if isinstance(_layer, nn.Conv2d):
                vgg_layers.append(_layer)
            elif isinstance(_layer, nn.BatchNorm2d):
                vgg_layers.append(_layer)
        
        merged_layers = []
        for idx, conv_block in enumerate(blocks):
            if idx < 2:
                units = [conv_block.conv1.cbr_unit, conv_block.conv2.cbr_unit]
            else:
                units = [
                    conv_block.conv1.cbr_unit,
                    conv_block.conv2.cbr_unit,
                    conv_block.conv3.cbr_unit,
                    conv_block.conv4.cbr_unit,
                ]
            for _unit in units:
                for _layer in _unit:
                    if isinstance(_layer, nn.Conv2d):
                        merged_layers.append(_layer)
                    elif isinstance(_layer, nn.BatchNorm2d):
                        merged_layers.append(_layer)

        assert len(vgg_layers) == len(merged_layers)

        for l1, l2 in zip(vgg_layers, merged_layers):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
            elif isinstance(l1, nn.BatchNorm2d) and isinstance(l2, nn.BatchNorm2d):
                l2.running_mean.data = l1.running_mean.data
                l2.running_var.data = l1.running_var.data
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data


class SegNetVgg13(nn.Module):
    def __init__(self, num_classes=21, in_channels=3, is_unpooling=True, pretrained_backbone=False):
        super(SegNetVgg13, self).__init__()
        #self.num_classes = num_classes
        self.in_channels = in_channels
        self.is_unpooling = is_unpooling

        self.down1 = segnetDown2(self.in_channels, 64)
        self.down2 = segnetDown2(64, 128)
        self.down3 = segnetDown2(128, 256)
        self.down4 = segnetDown2(256, 512)
        self.down5 = segnetDown2(512, 512)

        self.up5 = segnetUp2(512, 512)
        self.up4 = segnetUp2(512, 256)
        self.up3 = segnetUp2(256, 128)
        self.up2 = segnetUp2(128, 64)
        self.up1 = segnetUp2(64, num_classes)

        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        nn.init.kaiming_normal_(m.weight)
        #        nn.init.constant_(m.bias, 0)
        #    elif isinstance(m, nn.BatchNorm2d):
        #        nn.init.constant_(m.weight, 1)
        #        nn.init.constant_(m.bias, 0)
        if pretrained_backbone==True:
            self.init_vgg13_params()
        

    def forward(self, inputs, positive_out=False):

        down1, indices_1, unpool_shape1 = self.down1(inputs)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        down4, indices_4, unpool_shape4 = self.down4(down3)
        down5, indices_5, unpool_shape5 = self.down5(down4)

        up5 = self.up5(down5, indices_5, unpool_shape5)
        up4 = self.up4(up5, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        up1 = self.up1(up2, indices_1, unpool_shape1)
        if positive_out:
            return torch.relu( up1 )
        return up1

    def init_vgg13_params(self):

        vgg13 = vgg.vgg13_bn(pretrained=True)

        blocks = [self.down1, self.down2, self.down3, self.down4, self.down5]

        features = list(vgg13.features.children())

        vgg_layers = []
        for _layer in features:
            if isinstance(_layer, nn.Conv2d):
                vgg_layers.append(_layer)
            elif isinstance(_layer, nn.BatchNorm2d):
                vgg_layers.append(_layer)
        
        merged_layers = []
        for idx, conv_block in enumerate(blocks):
            if idx < 2:
                units = [conv_block.conv1.cbr_unit, conv_block.conv2.cbr_unit]
            else:
                units = [
                    conv_block.conv1.cbr_unit,
                    conv_block.conv2.cbr_unit,
                    #conv_block.conv3.cbr_unit,
                    #conv_block.conv4.cbr_unit,
                ]
            for _unit in units:
                for _layer in _unit:
                    if isinstance(_layer, nn.Conv2d):
                        merged_layers.append(_layer)
                    elif isinstance(_layer, nn.BatchNorm2d):
                        merged_layers.append(_layer)

        assert len(vgg_layers) == len(merged_layers)

        for l1, l2 in zip(vgg_layers, merged_layers):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
            elif isinstance(l1, nn.BatchNorm2d) and isinstance(l2, nn.BatchNorm2d):
                l2.running_mean.data = l1.running_mean.data
                l2.running_var.data = l1.running_var.data
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data