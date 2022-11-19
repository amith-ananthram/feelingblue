from abc import ABC

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

import constants
from utils import convert_activations

RESNET_OUTPUT_SIZE = 2048


class EmotionClassifier(nn.Module, ABC):
    def __init__(self, device, use_pretrained_resnet, resnet_size, normalization, dropout):
        super(EmotionClassifier, self).__init__()
        self.device = device
        self.normalization = normalization

        resnet_args = {
            'pretrained': use_pretrained_resnet
        }
        if resnet_size == 50:
            base_resnet = models.resnet50(**resnet_args)
        elif resnet_size == 101:
            base_resnet = models.resnet101(**resnet_args)
        else:
            raise Exception("Unsupported ResNet size: %s" % (resnet_size))
        # we convert the activations to allow us to backpropagate
        # better when we use this to train our image transformer
        convert_activations(base_resnet, nn.ReLU, nn.LeakyReLU)
        self.resnet = nn.Sequential(
            *(list(base_resnet.children())[:-1])
        )
        assert RESNET_OUTPUT_SIZE == base_resnet.fc.in_features
        self.dropout = nn.Dropout(dropout)

        if self.normalization == 'none':
            normalizer = (lambda dim: (lambda image: image))
        elif self.normalization == 'batch':
            normalizer = nn.BatchNorm1d
        elif self.normalization == 'instance':
            normalizer = nn.InstanceNorm1d
        else:
            raise Exception("Unsupported normalization: %s" % self.normalization)

        self.fc1 = nn.Linear(
            2 * RESNET_OUTPUT_SIZE + len(constants.EMOTIONS),
            RESNET_OUTPUT_SIZE
        )
        self.ln1 = normalizer(RESNET_OUTPUT_SIZE)
        self.fc2 = nn.Linear(
            RESNET_OUTPUT_SIZE + len(constants.EMOTIONS),
            RESNET_OUTPUT_SIZE // 2
        )
        self.ln2 = normalizer(RESNET_OUTPUT_SIZE // 2)
        self.fc3 = nn.Linear(
            RESNET_OUTPUT_SIZE // 2 + len(constants.EMOTIONS),
            RESNET_OUTPUT_SIZE // 4,
        )
        self.ln3 = normalizer(RESNET_OUTPUT_SIZE // 4)
        self.fc4 = nn.Linear(
            RESNET_OUTPUT_SIZE // 4 + len(constants.EMOTIONS),
            RESNET_OUTPUT_SIZE // 8
        )
        self.ln4 = normalizer(RESNET_OUTPUT_SIZE // 8)
        self.fc5 = nn.Linear(
            RESNET_OUTPUT_SIZE // 8 + len(constants.EMOTIONS),
            RESNET_OUTPUT_SIZE // 16
        )
        self.ln5 = normalizer(RESNET_OUTPUT_SIZE // 16)
        self.fc6 = nn.Linear(
            RESNET_OUTPUT_SIZE // 16 + len(constants.EMOTIONS),
            2
        )

    def forward(self, image1, image2, emotion):
        image1 = self.dropout(
            self.resnet(image1).view((image1.shape[0], RESNET_OUTPUT_SIZE))
        )
        image2 = self.dropout(
            self.resnet(image2).view((image2.shape[0], RESNET_OUTPUT_SIZE))
        )

        out = self.dropout(
            F.leaky_relu(
                self.ln1(
                    self.fc1(torch.cat((image1, image2, emotion), dim=1))
                )
            )
        )
        out = self.dropout(
            F.leaky_relu(
                self.ln2(
                    self.fc2(torch.cat((out, emotion), dim=1))
                )
            )
        )
        out = self.dropout(
            F.leaky_relu(
                self.ln3(
                    self.fc3(torch.cat((out, emotion), dim=1))
                )
            )
        )
        out = F.leaky_relu(
            self.ln4(
                self.fc4(torch.cat((out, emotion), dim=1))
            )
        )
        out = F.leaky_relu(
            self.ln5(
                self.fc5(torch.cat((out, emotion), dim=1))
            )
        )
        return self.fc6(torch.cat((out, emotion), dim=1))
