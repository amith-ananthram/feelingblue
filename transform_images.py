import os
import sys
import json
import math
import glob
from PIL import Image
from tqdm import tqdm
from operator import itemgetter

import torch
from torch.optim import Adam
from torch.nn import functional as F
from torchvision.utils import save_image

import utils
import constants
from datasets import get_default_transformation
from models import EmotionClassifier, PaletteApplier

LR = 0.01
REUSE = False
L_BIAS_RANGE = 10
COLORS_IN_PALETTE = 6
USE_SRC_PALETTE = True


def load_palette_applier(device, palette_applier_path):
    variant_dir = os.path.dirname(palette_applier_path)
    with open(os.path.join(variant_dir, 'config.json'), 'r') as f:
        config = json.load(f)
    palette_applier = PaletteApplier(
        use_pretrained_resnet=False,
        conditioning_length=3 * COLORS_IN_PALETTE,
        dropout=0,
        normalization=config['model_normalization'],
        decode_mode=config['decode_mode'],
        output_mode=config['output_mode']
    )
    palette_applier.load_state_dict(
        torch.load(
            palette_applier_path,
            map_location=device
        )
    )
    utils.no_grad(palette_applier)
    palette_applier = palette_applier.eval()
    return palette_applier.to(device)


def load_emotion_classifier(device, emotion_classifier_path):
    variant_dir = os.path.dirname(emotion_classifier_path)
    with open(os.path.join(variant_dir, 'config.json'), 'r') as f:
        config = json.load(f)

    emotion_classifier = EmotionClassifier(
        device,
        use_pretrained_resnet=False,
        resnet_size=50,
        normalization=config['model_normalization'],
        dropout=0
    )
    emotion_classifier.load_state_dict(
        torch.load(
            emotion_classifier_path,
            map_location=device
        )
    )
    utils.no_grad(emotion_classifier)
    emotion_classifier = emotion_classifier.eval()
    return emotion_classifier.to(device)


def recolor(palette_applier, src_images, padding_mask, lab_palettes, l_biases):
    ab = palette_applier(src_images, lab_palettes).squeeze()
    l = src_images[:, 0].unsqueeze(dim=1)

    recolored = torch.cat((l, ab), dim=1)

    assert recolored.shape[1] == 3
    assert recolored.shape[0] == l_biases.shape[0]

    for i in range(recolored.shape[0]):
        recolored[i, 0, :, :] = torch.clamp(recolored[i, 0, :, :] + l_biases[i], min=-1, max=1)

    # we don't want backpropagation for either the EC or L2 component
    # to try to encourage differences in the masked areas of the image
    recolored = recolored * padding_mask

    return recolored


def unpad(image, padding):
    assert image.shape[0] == 3
    left_padding, top_padding, right_padding, bottom_padding = padding
    return image[
           :,
           top_padding:(image.shape[2] - bottom_padding),
           left_padding:(image.shape[1] - right_padding)
           ]


def get_padding_mask(src_image_shape, padding):
    channels, width, height = src_image_shape
    left_padding, top_padding, right_padding, bottom_padding = padding

    assert channels == 3

    mask = torch.zeros(width, height)
    mask[top_padding:(height - bottom_padding), left_padding:(width - right_padding)] = 1

    return mask


class RandomPalettes(torch.nn.Module):
    def __init__(self, l_bias_range, num_palettes):
        super(RandomPalettes, self).__init__()
        self.l_bias_range = l_bias_range
        self.num_palettes = num_palettes

        self.l_biases = torch.nn.Parameter(
            torch.zeros(num_palettes).detach()
        )
        self.palettes = torch.nn.Parameter(
            torch.atanh(torch.FloatTensor(num_palettes, 6 * 3).uniform_(-1, 1)).detach()
        )

    def forward(self):
        return self.l_bias_range / 50 * torch.tanh(self.l_biases), torch.tanh(self.palettes)


NUM_PALETTES = 100


def generate_images(images, palette_applier_path, emotion_classifier_path):
    if torch.cuda.is_available():
        print("using GPU...")
        device = torch.device("cuda")
    else:
        print("using CPU...")
        device = torch.device("cpu")

    palette_applier = load_palette_applier(device, emotion_classifier_path)
    emotion_classifier = load_emotion_classifier(device, palette_applier_path)

    zeros = torch.zeros(
        NUM_PALETTES, dtype=torch.long
    ).to(device)
    ones = torch.ones(
        NUM_PALETTES, dtype=torch.long
    ).to(device)

    for src_image_id, src_path in enumerate(
            tqdm(
                list(sorted(images.keys())),
                desc='transforming images (random palettes)'
            )
    ):
        src_image = Image.open(src_path).convert('RGB')
        _, _, padding = utils.get_padding_details(src_image)

        src_images = get_default_transformation(
            convert_to_lab=True, normalize_inputs=True
        )(src_image).float().unsqueeze(dim=0).expand(NUM_PALETTES, -1, -1, -1).to(device)
        padding_mask = get_padding_mask(src_images[0].shape, padding).to(device)

        for emotion, transformed_path in images[src_path]:
            emotions = F.one_hot(
                torch.tensor([constants.EMOTIONS.index(emotion) for _ in range(NUM_PALETTES)]),
                num_classes=len(constants.EMOTIONS)
            ).to(device)
            palettes = RandomPalettes(L_BIAS_RANGE, NUM_PALETTES).to(device)

            optimizer = Adam(palettes.parameters(), lr=LR)

            best_palettes_and_l_biases = [
                (math.inf,) for _ in range(NUM_PALETTES)
            ]

            step = 0
            while True:
                step += 1
                optimizer.zero_grad()

                l_biases, lab_palettes = palettes()

                # num palettes x 3 x 224 x 224
                recolored = recolor(
                    palette_applier, src_images, padding_mask, lab_palettes, l_biases
                )

                assert recolored.shape[0] == NUM_PALETTES

                l_preds = emotion_classifier(recolored, src_images, emotions)
                l_losses = F.cross_entropy(l_preds, zeros, reduction='none')

                r_preds = emotion_classifier(src_images, recolored, emotions)
                r_losses = F.cross_entropy(r_preds, ones, reduction='none')

                losses = l_losses + r_losses

                assert losses.shape[0] == NUM_PALETTES

                tot_loss = torch.sum(losses)

                tot_loss.backward()
                optimizer.step()

                for palette_idx, loss in enumerate(losses):
                    loss = loss.item()
                    if loss < best_palettes_and_l_biases[palette_idx][0]:
                        best_palettes_and_l_biases[palette_idx] = (
                            loss, lab_palettes[palette_idx].detach().cpu(), l_biases[palette_idx].detach().cpu()
                        )

                if step > 100:
                    break

            best_palettes_and_l_biases.sort(key=itemgetter(0))

            best_lab_palettes = torch.stack(
                [lab_palette for (_, lab_palette, __) in best_palettes_and_l_biases]
            ).to(device)
            best_l_biases = torch.stack(
                [l_bias for (_, __, l_bias) in best_palettes_and_l_biases]
            ).to(device)

            with torch.no_grad():
                best_images = recolor(
                    palette_applier, src_images, padding_mask,
                    best_lab_palettes.to(device), best_l_biases.to(device)
                )

            assert best_images.shape[0] == NUM_PALETTES

            for palette_idx in range(NUM_PALETTES):
                if palette_idx > 0:
                    filename, ext = os.path.splitext(transformed_path)
                    transformed_path_ = '%s-%s%s' % (filename, palette_idx, ext)
                else:
                    transformed_path_ = transformed_path
                save_image(
                    torch.tensor(
                        utils.get_image(
                            unpad(best_images[palette_idx], padding),
                            is_lab=True, unnormalize=True
                        )
                    ).permute(2, 0, 1).unsqueeze(dim=0),
                    transformed_path_
                )


if __name__ == '__main__':
    image_dir = sys.argv[1]
    palette_applier_path = sys.argv[2]
    emotion_classifier_path = sys.argv[3]
    output_dir = sys.argv[4]

    images = {
        image_path: [
            (
                emotion,
                os.path.join(output_dir, '%s_%s.%s' % (
                    os.path.splitext(os.path.basename(image_path))[0],
                    emotion,
                    'jpg'
                ))
            ) for emotion in constants.EMOTIONS
        ]
        for image_path in glob.glob(os.path.join(image_dir, '*jpg'))
    }
    generate_images(images, palette_applier_path, emotion_classifier_path)