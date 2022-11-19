import os
import json
import torch
import pickle
import colorgram
import numpy as np
from abc import ABC
import pandas as pd
import skimage.color

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.nn import functional as F
import torchvision.transforms.functional as TF

import utils
import constants
from utils.colors import normalize_lab_image, normalize_lab_colors
from constants import NUM_HUE_SHIFTS, HUE_SHIFT_NUM_COLORS_IN_PALETTE


def get_default_transformation(convert_to_lab=False, normalize_inputs=True):
    transformations = [utils.transform_and_pad]

    if convert_to_lab:
        transformations.append(skimage.color.rgb2lab)

    # handles re-arranging dimensions
    # and normalizing (if recognized as RGB)
    transformations.append(transforms.ToTensor())

    if convert_to_lab and normalize_inputs:
        transformations.append(normalize_lab_image)
    elif not convert_to_lab and not normalize_inputs:
        transformations.append(lambda image: image * 255)

    return transforms.Compose(transformations)


CHOICE_COLUMN = 'selected_image_id'
MIN_MAX_COLUMN = 'min/max'
EMOTION_COLUMN = 'emotion'


class EmotionClassificationDataset(Dataset):
    def __init__(self, subset, convert_to_lab=False, normalize_inputs=True):
        assert subset in {'train', 'valid'}

        self.convert_to_lab = convert_to_lab
        self.normalize_inputs = normalize_inputs
        self.transform = get_default_transformation(
            convert_to_lab, normalize_inputs
        )

        with open(constants.FEELING_BLUE_SPLITS_FILE, 'r') as f:
            split_filenames = json.load(f)

        self.rows = []
        for _, annotation in pd.read_csv(constants.FEELING_BLUE_ANNOTATIONS_FILE).iterrows():
            image_ids_to_filenames = {
                annotation['image%s_id' % i]: annotation['image%s_filename' % i] for i in range(1, 5)
            }

            selected_image_id = annotation[CHOICE_COLUMN]
            if image_ids_to_filenames[selected_image_id] not in split_filenames[subset]:
                continue

            for other_image_id in sorted(image_ids_to_filenames.keys()):
                if other_image_id == selected_image_id:
                    continue

                if image_ids_to_filenames[other_image_id] not in split_filenames[subset]:
                    continue

                if annotation[MIN_MAX_COLUMN] == 'MIN':  # selected image is less emotional
                    selected_image_first_label = 1
                    selected_image_second_label = 0
                elif annotation[MIN_MAX_COLUMN] == 'MAX':  # selected image is more emotional
                    selected_image_first_label = 0
                    selected_image_second_label = 1
                else:
                    raise Exception("Unsupported direction: %s" % (annotation[MIN_MAX_COLUMN]))

                self.rows.append((
                    image_ids_to_filenames[selected_image_id],
                    image_ids_to_filenames[other_image_id],
                    annotation[EMOTION_COLUMN],
                    selected_image_first_label
                ))
                self.rows.append((
                    image_ids_to_filenames[other_image_id],
                    image_ids_to_filenames[selected_image_id],
                    annotation[EMOTION_COLUMN],
                    selected_image_second_label
                ))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        task_id, image1_filename, image2_filename, emotion, label = self.rows[idx]

        # io.imread returns H x W x C
        # transform handles rearranging, normalizing, etc
        return (
            image1_filename,
            image2_filename,
            self.transform(Image.open(
                os.path.join(constants.WIKIART_DIRECTORY, image1_filename)
            ).convert('RGB')).float(),
            self.transform(Image.open(
                os.path.join(constants.WIKIART_DIRECTORY, image2_filename)
            ).convert('RGB')).float(),
            F.one_hot(
                torch.tensor(constants.EMOTIONS.index(emotion)),
                num_classes=len(constants.EMOTIONS)
            ).float(),
            torch.tensor([label])
        )


def get_palette(convert_to_lab, image, colors, num_colors=6, normalize_inputs=True):
    if image not in colors:
        colors[image] = colorgram.extract(image, num_colors)

    palette = []
    for color in colors[image]:
        rgb = [channel / 255 for channel in [color.rgb.r, color.rgb.g, color.rgb.b]]
        if convert_to_lab:
            palette.append(skimage.color.rgb2lab(rgb))
        else:
            palette.append(rgb)

    while len(palette) < num_colors:
        palette.append([0.0, 0.0, 0.0])

    palette = torch.tensor(np.array(palette))

    if convert_to_lab and normalize_inputs:
        palette = normalize_lab_colors(palette, 1)

    return palette



HUE_SHIFT_DIRECTORY = os.path.join(constants.WIKIART_DIRECTORY, 'hue_shifts')
HUE_SHIFT_PALETTE_DIRECTORY = os.path.join(HUE_SHIFT_DIRECTORY, 'palettes')

VALID_AUGMENTATIONS = {'horizontal', 'vertical', 'rotate90', 'rotate180', 'rotate270'}


class PaletteApplicationDataset(Dataset, ABC):
    def __init__(self,
                 subset,
                 num_hue_shifts=NUM_HUE_SHIFTS,
                 add_noise=True,
                 normalize_inputs=True,
                 augmentations=[]
                 ):
        assert NUM_HUE_SHIFTS % num_hue_shifts == 0
        assert len(set(augmentations) - VALID_AUGMENTATIONS) == 0
        assert len(augmentations) == 0 or subset == 'train'

        self.subset = subset
        self.num_hue_shifts = num_hue_shifts
        self.add_noise = add_noise
        self.normalize_inputs = normalize_inputs
        self.augmentations = augmentations

        self.src_image_transform = get_default_transformation(
            convert_to_lab=True, normalize_inputs=self.normalize_inputs
        )
        self.tgt_image_transform = get_default_transformation(
            convert_to_lab=True, normalize_inputs=False
        )

        wikiart_colors_filepath = os.path.join(
            constants.WIKIART_DIRECTORY,
            'colors%s.pkl' % HUE_SHIFT_NUM_COLORS_IN_PALETTE
        )
        with open(wikiart_colors_filepath, 'rb') as f:
            self.src_rgb_palettes = pickle.load(f)

        with open(constants.FEELING_BLUE_SPLITS_FILE, 'r') as f:
            self.filenames = list(sorted(json.load(f)[subset]))

    def __len__(self):
        return (len(self.augmentations) + 1) * self.num_hue_shifts * len(self.filenames)

    def __getitem__(self, idx):
        aug_id = idx // (self.num_hue_shifts * len(self.filenames))
        hue_shift = list(range(1, 360, 20))[
            (idx // len(self.filenames)) % self.num_hue_shifts]
        image_filepath = self.filenames[idx % len(self.filenames)]

        src_image_palette = get_palette(
            True, image_filepath, self.src_rgb_palettes, normalize_inputs=self.normalize_inputs
        )
        if src_image_palette.shape[0] < HUE_SHIFT_NUM_COLORS_IN_PALETTE:
            src_image_palette = torch.cat(
                (
                    src_image_palette,
                    torch.zeros(
                        HUE_SHIFT_NUM_COLORS_IN_PALETTE - src_image_palette.shape[0], 3, dtype=src_image_palette.dtype
                    )
                ), dim=0
            )
        src_image_palette = src_image_palette.reshape(HUE_SHIFT_NUM_COLORS_IN_PALETTE * 3).float()
        src_image_tensor = self.src_image_transform(Image.open(
            os.path.join(constants.WIKIART_DIRECTORY, image_filepath)
        ).convert('RGB')).float()

        hue_shift_filepath = os.path.basename(image_filepath).split('.jpg')[0] + '-shift%s' % hue_shift
        with open(os.path.join(HUE_SHIFT_PALETTE_DIRECTORY, hue_shift_filepath + '.pkl'), 'rb') as f:
            hue_shift_palette = pickle.load(f)

        if self.subset == 'train' and self.add_noise:
            noise = torch.randn(hue_shift_palette.shape)
            hue_shift_palette += noise

        # we extend the colorgram palette to the desired length
        if hue_shift_palette.shape[0] < HUE_SHIFT_NUM_COLORS_IN_PALETTE:
            hue_shift_palette = torch.cat(
                (
                    hue_shift_palette,
                    torch.zeros(
                        HUE_SHIFT_NUM_COLORS_IN_PALETTE - hue_shift_palette.shape[0],
                        3, dtype=hue_shift_palette.dtype
                    )
                ), dim=0
            )

        if self.normalize_inputs:
            hue_shift_palette = normalize_lab_colors(hue_shift_palette, 1)

        hue_shift_palette = hue_shift_palette.reshape(HUE_SHIFT_NUM_COLORS_IN_PALETTE * 3).float()
        hue_shift_tensor = self.tgt_image_transform(Image.open(
            os.path.join(HUE_SHIFT_DIRECTORY, hue_shift_filepath + '.jpg')
        ).convert('RGB')).float()

        if aug_id == 0:
            augmenter = lambda x: x
        else:
            aug_type = self.augmentations[aug_id - 1]
            if aug_type == 'horizontal':
                augmenter = TF.hflip
            elif aug_type == 'vertical':
                augmenter = TF.vflip
            elif aug_type == 'rotate90':
                augmenter = (lambda image: TF.rotate(image, 90))
            elif aug_type == 'rotate180':
                augmenter = (lambda image: TF.rotate(image, 180))
            elif aug_type == 'rotate270':
                augmenter = (lambda image: TF.rotate(image, 270))
            else:
                raise Exception("Unsupported aug_type: %s" % aug_type)

        src_image_tensor = augmenter(src_image_tensor)
        hue_shift_tensor = augmenter(hue_shift_tensor)

        return image_filepath, hue_shift_filepath, src_image_palette, \
               src_image_tensor, hue_shift_palette, hue_shift_tensor
