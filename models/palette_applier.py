from abc import ABC

import torch
from torch import nn
from torchvision import models


class PaletteApplier(nn.Module, ABC):
    def __init__(self,
        use_pretrained_resnet, conditioning_length, dropout, 
        decode_mode, output_mode='linear', normalization='batch'
    ):
        super(PaletteApplier, self).__init__()
        self.dropout = dropout
        self.decode_mode = decode_mode
        self.output_mode = output_mode
        self.normalization = normalization

        if self.normalization == 'batch':
            normalizer = nn.BatchNorm2d
        elif self.normalization == 'instance':
            normalizer = nn.InstanceNorm2d
        else:
            raise Exception("Unsupported normalization: %s" % self.normalization)

        self.dropout = nn.Dropout(dropout)
        base_resnet = models.resnet50(
            pretrained=use_pretrained_resnet, norm_layer=normalizer
        )
        # output: b, 256, 56, 56
        self.conv1 = nn.Sequential(
            base_resnet.conv1,
            base_resnet.bn1,
            base_resnet.relu,
            base_resnet.maxpool,
            base_resnet.layer1
        )
        # output: b, 512, 28, 28
        self.conv2 = base_resnet.layer2
        # output: b, 1024, 14, 14
        self.conv3 = base_resnet.layer3
        # output: b, 2048, 7, 7
        self.conv4 = base_resnet.layer4

        self.conditioning_length = conditioning_length

        if self.decode_mode == 'deconv':
            # input: b, 2048 + 5, 7, 7, output: b, 1024, 14, 14
            self.decode1 = nn.Sequential(
                nn.ConvTranspose2d(2048 + self.conditioning_length, 1024, 2, stride=2),
                normalizer(1024),
                nn.LeakyReLU()
            )
            # input: b, 2048 + 5, 14, 14, output: b, 512, 28, 28
            self.decode2 = nn.Sequential(
                nn.ConvTranspose2d(2048 + self.conditioning_length, 512, 2, stride=2),
                normalizer(512),
                nn.LeakyReLU()
            )
            # input: b, 1024 + 5, 28, 28, output: b, 256, 56, 56
            self.decode3 = nn.Sequential(
                nn.ConvTranspose2d(1024 + self.conditioning_length, 256, 2, stride=2),
                normalizer(256),
                nn.LeakyReLU()
            )
            # input: b, 512 + 5, 56, 56, output: b, 256, 112, 112
            self.decode4 = nn.Sequential(
                nn.ConvTranspose2d(512 + self.conditioning_length, 256, 2, stride=2),
                normalizer(256),
                nn.LeakyReLU()
            )
            # input: b, 256 + 5, 112, 122, output: b, 3, 224, 224
            self.decode5 = nn.ConvTranspose2d(256 + self.conditioning_length, 2, 2, stride=2)
        else:
            # input: b, 2048 + 5, 7, 7, output: b, 128, 14, 14
            self.decode1 = nn.Sequential(
                nn.Conv2d(2048 + self.conditioning_length, 1024, kernel_size=3, stride=1, padding=1),
                normalizer(1024),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2)
            )
            self.decode2 = nn.Sequential(
                nn.Conv2d(2048 + self.conditioning_length, 512, kernel_size=3, stride=1, padding=1),
                normalizer(512),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2)
            )
            self.decode3 = nn.Sequential(
                nn.Conv2d(1024 + self.conditioning_length, 256, kernel_size=3, stride=1, padding=1),
                normalizer(256),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2)
            )
            self.decode4 = nn.Sequential(
                nn.Conv2d(512 + self.conditioning_length, 128, kernel_size=3, stride=1, padding=1),
                normalizer(128),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2)
            )
            self.decode5 = nn.Sequential(
                nn.Conv2d(128 + self.conditioning_length, 2, kernel_size=3, stride=1, padding=1),
                nn.Upsample(scale_factor=2)
            )

    def forward(self, images, conditioning):
        encoded_conv1 = self.dropout(self.conv1(images))
        encoded_conv2 = self.dropout(self.conv2(encoded_conv1))
        encoded_conv3 = self.dropout(self.conv3(encoded_conv2))
        encoded_conv4 = self.dropout(self.conv4(encoded_conv3))

        conditioning_deconv1 = conditioning.repeat(7, 7, 1, 1).permute(2, 3, 0, 1).float()
        output = self.decode1(torch.cat((encoded_conv4, conditioning_deconv1), dim=1))
        conditioning_deconv2 = conditioning.repeat(14, 14, 1, 1).permute(2, 3, 0, 1).float()
        output = self.decode2(torch.cat((encoded_conv3, conditioning_deconv2, output), dim=1))
        conditioning_deconv3 = conditioning.repeat(28, 28, 1, 1).permute(2, 3, 0, 1).float()
        output = self.decode3(torch.cat((encoded_conv2, conditioning_deconv3, output), dim=1))
        conditioning_deconv4 = conditioning.repeat(56, 56, 1, 1).permute(2, 3, 0, 1).float()
        output = self.decode4(torch.cat((encoded_conv1, conditioning_deconv4, output), dim=1))
        conditioning_deconv5 = conditioning.repeat(112, 112, 1, 1).permute(2, 3, 0, 1).float()
        output = self.decode5(torch.cat((conditioning_deconv5, output), dim=1))

        if self.output_mode == 'activation':
            output = torch.tanh(output)
        else:
            assert self.output_mode == 'linear'

        return output