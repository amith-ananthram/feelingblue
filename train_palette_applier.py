import os
import math
import json
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from optparse import OptionParser
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.optim import Adam, AdamW
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from torchvision import models

import constants
import utils
from utils.nn import clip_grad_norm_
from utils.colors import unnormalize_lab_colors
from models import PaletteApplier
from datasets import PaletteApplicationDataset

COLORS_IN_PALETTE = 6

DEFAULT_LR = 1e-3
DEFAULT_BETA1 = 0.9
DEFAULT_BETA2 = 0.999
DEFAULT_DROPOUT = 0.0
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_EPOCHS = 10
DEFAULT_NUM_HUE_SHIFTS = 18
DEFAULT_USE_ADVERSARY = False

MODEL_DIRECTORY = os.path.join(constants.MODEL_DIRECTORY, 'paletteapplier')


class ApplicationAdversary(nn.Module):
    def __init__(self, use_pretrained_resnet):
        super(ApplicationAdversary, self).__init__()

        base_resnet = models.resnet50(pretrained=use_pretrained_resnet)
        self.resnet = nn.Sequential(
            *(list(base_resnet.children())[:-1])
        )
        self.fc1 = nn.Linear(
            base_resnet.fc.in_features + 3 * COLORS_IN_PALETTE,
            int(base_resnet.fc.in_features / 2))
        self.fc2 = nn.Linear(
            int(base_resnet.fc.in_features / 2) + 3 * COLORS_IN_PALETTE,
            int(base_resnet.fc.in_features / 4))
        self.fc3 = nn.Linear(
            int(base_resnet.fc.in_features / 4) + 3 * COLORS_IN_PALETTE,
            int(base_resnet.fc.in_features / 8))
        self.fc4 = nn.Linear(int(base_resnet.fc.in_features / 8) + 3 * COLORS_IN_PALETTE, 1)

    def forward(self, images, palettes):
        output = self.resnet(images).squeeze()
        output = torch.relu(self.fc1(torch.cat((output, palettes), dim=1)))
        output = torch.relu(self.fc2(torch.cat((output, palettes), dim=1)))
        output = torch.relu(self.fc3(torch.cat((output, palettes), dim=1)))
        return self.fc4(torch.cat((output, palettes), dim=1)).squeeze()


def recolor_images(options, palette_applier, src_images, palettes, input_l_only):
    if input_l_only:
        l = src_images[:, 0, :, :].unsqueeze(dim=1)
        src_images = torch.stack([l, l, l], dim=1)

    ab = palette_applier(src_images, palettes)

    if options.normalize_inputs.lower() == 'yes' and options.output_mode == 'linear':
        l = unnormalize_lab_colors(src_images, 1)[:, 0, :, :].unsqueeze(dim=1)
    else:
        l = src_images[:, 0, :, :].unsqueeze(dim=1)

    trans_images = torch.cat((l, ab), dim=1)

    if options.output_mode == 'activation':
        trans_images = unnormalize_lab_colors(trans_images, 1)

    return trans_images


def calculate_l2_loss(loss_type, recolored_images, tgt_images):
    assert recolored_images.shape[1] == 3
    assert tgt_images.shape[1] == 3

    if loss_type.endswith('_ab'):
        recolored_channels = recolored_images[:, [1, 2], :, :]
        tgt_channels = tgt_images[:, [1, 2], :, :]
    else:
        recolored_channels = recolored_images
        tgt_channels = tgt_images

    loss_type = loss_type.split('_')[0]
    if loss_type == 'l2':
        return F.mse_loss(tgt_channels, recolored_channels)
    elif loss_type == 'l2-elementwise':
        # batch size x 3 x r x c (squared diff at each pixel)
        elementwise = F.mse_loss(tgt_channels, recolored_channels, reduction='none')
        return torch.sum(torch.sqrt(torch.sum(elementwise, dim=1)))
    elif loss_type == 'l2-elementwise-sq-avg':
        # batch size x 3 x r x c (squared diff at each pixel)
        elementwise = F.mse_loss(tgt_channels, recolored_channels, reduction='none')
        return torch.sum(torch.mean(torch.sqrt(torch.sum(elementwise, dim=1)), dim=[1, 2])**2)
    else:
        raise Exception("Unsupported loss_type: %s" % loss_type)


def persist_recolored_images(
        options, prefix, epoch, batch_id, src_images, palettes, tgt_images, recolored_images, model_dir
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for id, (src_image, palette, tgt_image, recolored_image) in enumerate(
                zip(src_images, palettes, tgt_images, recolored_images)):
            if id >= 25:
                break
            try:
                image_prefix = "%s-%s-%s-%s" % (
                    prefix, str(epoch).zfill(4), str(batch_id).zfill(4), str(id).zfill(4))

                fig, (a1, a2, a3, a4) = plt.subplots(
                    4, 1, gridspec_kw={'height_ratios': [8, 1, 8, 8]}, figsize=(12, 12))
                utils.plot_image(a1, utils.get_image(
                    src_image, True,
                    unnormalize=options.normalize_inputs.lower() == 'yes'
                ), "Source")
                utils.plot_image(a2, utils.get_palette(
                    palette.reshape(int(palette.shape[0] / 3), 3), True,
                    unnormalize=options.normalize_inputs.lower() == 'yes'
                )[1], "Palette")
                utils.plot_image(a3, utils.get_image(
                    tgt_image, True, unnormalize=False
                ), "Target")
                utils.plot_image(a4, utils.get_image(
                    recolored_image, True, unnormalize=False), "Recolored")
                plt.savefig(os.path.join(model_dir, '%s.jpg' % image_prefix))
                plt.close()
            except:
                plt.close()
                continue


def calculate_validation_loss(
    options, device, epoch, t_pa_l2_loss, t_pa_adv_loss, t_ad_real_loss, t_ad_fake_loss, palette_applier,
    adversary, loader, save_samples, model_dir, first_batch_only=False
):
    v_pa_l2_loss = 0.0
    v_pa_adv_loss = 0.0
    palette_applier.eval()
    for batch_id, batch in enumerate(loader):
        if first_batch_only and batch_id > 0:
            return

        with torch.no_grad():
            _, _, src_palettes, src_images, tgt_palettes, tgt_images = utils.move_to_device(batch, device)
            recolored_images = recolor_images(options, palette_applier, src_images, tgt_palettes, options.input_l_only)
            v_pa_l2_loss += calculate_l2_loss(options.loss_type, recolored_images, tgt_images).item()

            if adversary is not None:
                v_pa_adv_loss += F.binary_cross_entropy_with_logits(
                    adversary(recolored_images, tgt_palettes),
                    torch.ones(recolored_images.shape[0]).float().to(device)
                )

            if save_samples:
                persist_recolored_images(
                    options, "val", epoch, batch_id, src_images, tgt_palettes, tgt_images, recolored_images, model_dir
                )

    print("%s: EPOCH %s COMPLETE, t_loss=(%.3f, %.3f, %.3f, %.3f, %.3f), v_loss=(%.3f, %.3f, %.3f))" % (
        datetime.now(), epoch, t_pa_l2_loss, t_pa_adv_loss, t_ad_real_loss, t_ad_fake_loss,
        sum([t_pa_l2_loss, t_pa_adv_loss, t_ad_real_loss, t_ad_fake_loss]),
        v_pa_l2_loss, v_pa_adv_loss, v_pa_l2_loss + v_pa_adv_loss
    ))

    return v_pa_l2_loss, v_pa_adv_loss


if __name__ == '__main__':
    usage = "usage: %prog [options]"
    parser = OptionParser(usage=usage)
    parser.add_option(
        '-v',
        '--variant',
        dest='variant',
        default=datetime.now().strftime("%Y%m%d%H%m")
    )
    parser.add_option(
        '--lr',
        '--learning-rate',
        dest='lr',
        default=DEFAULT_LR
    )
    parser.add_option(
        '--beta1',
        dest='beta1',
        default=DEFAULT_BETA1
    )
    parser.add_option(
        '--adamw',
        dest='use_adamw',
        action='store_true',
        default=False
    )
    parser.add_option(
        '--grad-clip',
        dest='grad_clip',
        default=None
    )
    parser.add_option(
        '--save-samples',
        dest='save_samples',
        action='store_true',
        default=False
    )
    parser.add_option(
        '--num-hue-shifts',
        dest='num_hue_shifts',
        default=DEFAULT_NUM_HUE_SHIFTS
    )
    parser.add_option(
        '--use-pretrained-resnet',
        dest='use_pretrained_resnet',
        type='choice',
        choices=['yes', 'no'],
        default='yes'
    )
    parser.add_option(
        '--input-l-only',
        dest='input_l_only',
        action='store_true',
        default=False
    )
    parser.add_option(
        '--normalize-inputs',
        dest='normalize_inputs',
        type='choice',
        choices=['yes', 'no']
    )
    parser.add_option(
        '--augmentations',
        dest='augmentations',
        default=''
    )
    parser.add_option(
        '--model-normalization',
        dest='model_normalization',
        type='choice',
        choices=['batch', 'instance'],
        default='batch'
    )
    parser.add_option(
        '--decode-mode',
        dest='decode_mode',
        type='choice',
        choices=['upsamp', 'deconv'],
        default='upsamp'
    )
    parser.add_option(
        '--output-mode',
        dest='output_mode',
        type='choice',
        choices=['linear', 'activation'],
        default='linear'
    )
    parser.add_option(
        '--loss-type',
        dest='loss_type',
        type='choice',
        choices=[
            'l2', 'l2_ab',
            'l2-elementwise', 'l2-elementwise_ab',
            'l2-elementwise-sq-avg', 'l2-elementwise-sq-avg_ab'
        ],
        default='l2-elementwise'
    )
    parser.add_option(
        '--checkpoint',
        dest='checkpoint'
    )
    parser.add_option(
        '--add-noise',
        dest='add_noise',
        type='choice',
        choices=['yes', 'no'],
        default='no'
    )
    parser.add_option(
        '--use-adversary',
        dest='use_adversary',
        action='store_true',
        default=DEFAULT_USE_ADVERSARY
    )
    parser.add_option(
        '--lambda-l2',
        dest='lambda_l2',
        default=10
    )
    parser.add_option(
        '--dropout',
        dest='dropout',
        default=DEFAULT_DROPOUT
    )
    parser.add_option(
        '-b',
        '--batch-size',
        dest='batch_size',
        default=DEFAULT_BATCH_SIZE
    )
    parser.add_option(
        '-e',
        '--num-epochs',
        dest='num_epochs',
        default=DEFAULT_NUM_EPOCHS
    )
    parser.add_option(
        '--detect-anomalies',
        dest='detect_anomalies',
        action='store_true',
        default=False
    )
    parser.add_option(
        '-s',
        '--seed',
        dest='seed',
        default=100
    )
    (options, args) = parser.parse_args()

    model_dir = os.path.join(MODEL_DIRECTORY, options.variant)
    Path(model_dir).mkdir(parents=True, exist_ok=False)

    config = {
        'variant': options.variant,
        'lr': float(options.lr),
        'beta1': float(options.beta1),
        'use_adamw': options.use_adamw,
        'grad_clip': float(options.grad_clip) if options.grad_clip else None,
        'num_hue_shifts': int(options.num_hue_shifts),
        'use_pretrained_resnet': options.use_pretrained_resnet,
        'input_l_only': options.input_l_only,
        'normalize_inputs': options.normalize_inputs,
        'augmentations': options.augmentations,
        'model_normalization': options.model_normalization,
        'decode_mode': options.decode_mode,
        'output_mode': options.output_mode,
        'loss_type': options.loss_type,
        'add_noise': options.add_noise,
        'use_adversary': options.use_adversary,
        'lambda_l2': float(options.lambda_l2),
        'dropout': float(options.dropout),
        'batch_size': int(options.batch_size),
        'num_epochs': int(options.num_epochs),
        'seed': int(options.seed),
        'detect_anomalies': options.detect_anomalies
    }
    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump(config, f)

    if options.seed:
        np.random.seed(int(options.seed))
        torch.manual_seed(int(options.seed))

    if options.detect_anomalies:
        torch.autograd.set_detect_anomaly(True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        print("GPU detected, will use for training...")

    use_pretrained_resnet = options.use_pretrained_resnet.lower() == 'yes'
    if use_pretrained_resnet:
        print("Using pretrained ResNet weights...")
    else:
        print("Using randomly initialized ResNet weights...")

    ##
    # MODEL INITIALIZATION
    ##
    palette_applier = PaletteApplier(
        use_pretrained_resnet=use_pretrained_resnet,
        conditioning_length=3 * COLORS_IN_PALETTE,
        dropout=float(options.dropout),
        decode_mode=options.decode_mode,
        normalization=options.model_normalization,
        output_mode=options.output_mode
    ).to(device)

    if options.checkpoint:
        print("Loading weights from checkpoint %s" % options.checkpoint)
        palette_applier.load_state_dict(torch.load(options.checkpoint))

    opt_args = {
        'lr': float(options.lr),
        'betas': (float(options.beta1), DEFAULT_BETA2)
    }
    if options.use_adamw:
        pa_optimizer = AdamW(palette_applier.parameters(), **opt_args)
    else:
        pa_optimizer = Adam(palette_applier.parameters(), **opt_args)

    if options.use_adversary:
        adversary = ApplicationAdversary(
            use_pretrained_resnet=use_pretrained_resnet,
        ).to(device)

        if options.use_adamw:
            ad_optimizer = AdamW(adversary.parameters(), lr=float(options.lr))
        else:
            ad_optimizer = Adam(adversary.parameters(), lr=float(options.lr))
    else:
        adversary = None
        ad_optimizer = None

    augmentations = options.augmentations.split(',')
    dataset_args = {
        'num_hue_shifts': int(options.num_hue_shifts),
        'add_noise': options.add_noise.lower() == 'yes',
        'normalize_inputs': options.normalize_inputs.lower() == 'yes'
    }
    print(dataset_args)
    valid, train = (
        PaletteApplicationDataset("valid", **dataset_args),
        PaletteApplicationDataset("train", augmentations=augmentations, **dataset_args)
    )

    valid_indices = np.array(list(range(len(valid))))
    np.random.shuffle(valid_indices)
    valid = Subset(valid, valid_indices)

    print("%s: Loaded datasets..." % datetime.now())
    print("%s: valid=%s" % (datetime.now(), len(valid)))
    print("%s: train=%s" % (datetime.now(), len(train)))

    valid_loader, train_loader = \
        (
            DataLoader(valid, batch_size=2 * int(options.batch_size), num_workers=8),
            DataLoader(train, batch_size=int(options.batch_size), num_workers=8, shuffle=True)
        )

    calculate_validation_loss(
        options, device, -1, -1, -1, -1, -1, palette_applier,
        adversary, valid_loader, options.save_samples, model_dir
    )

    non_finite_count = 0
    best_l2_loss = math.inf
    save_next_t_batch = False
    for epoch in range(int(options.num_epochs)):
        t_pa_l2_loss = 0
        t_pa_adv_loss = 0
        t_ad_real_loss = 0
        t_ad_fake_loss = 0

        palette_applier.train()
        for batch_id, batch in enumerate(train_loader):
            pa_optimizer.zero_grad()

            _, _, src_palettes, src_images, tgt_palettes, tgt_images = utils.move_to_device(batch, device)
            recolored_images = recolor_images(options, palette_applier, src_images, tgt_palettes, options.input_l_only)
            l2_loss = calculate_l2_loss(options.loss_type, recolored_images, tgt_images)

            if options.use_adversary:
                pa_adv_loss = F.binary_cross_entropy_with_logits(
                    adversary(recolored_images, tgt_palettes),
                    torch.ones(recolored_images.shape[0]).float().to(device)
                )
                train_loss = float(options.lambda_l2) * l2_loss + pa_adv_loss
            else:
                pa_adv_loss = torch.tensor(0.0)
                train_loss = l2_loss

            train_loss.backward()
            if options.grad_clip:
                is_non_finite, _ = clip_grad_norm_(
                    palette_applier.parameters(), float(options.grad_clip))
                if is_non_finite:
                    non_finite_count += 1
                    continue
            pa_optimizer.step()

            t_pa_l2_loss += l2_loss.item()
            t_pa_adv_loss += pa_adv_loss.item()

            if options.use_adversary:
                ad_optimizer.zero_grad()
                real_preds = adversary(src_images, src_palettes)
                real_loss = F.binary_cross_entropy_with_logits(
                    real_preds, torch.ones(recolored_images.shape[0]).float().to(device)
                )
                fake_preds = adversary(
                    torch.cat(
                        (src_images, recolored_images.detach(), recolored_images.detach()), dim=0
                    ),
                    torch.cat(
                        (tgt_palettes, src_palettes, tgt_palettes), dim=0
                    )
                )
                fake_loss = F.binary_cross_entropy_with_logits(
                    fake_preds,
                    torch.zeros(3 * src_images.shape[0]).float().to(device)
                )
                adv_loss = real_loss + fake_loss

                adv_loss.backward()
                if options.grad_clip:
                    nn.utils.clip_grad_norm_(
                        adversary.parameters(), float(options.grad_clip))
                ad_optimizer.step()

                t_ad_real_loss += real_loss.item()
                t_ad_fake_loss += fake_loss.item()
            else:
                real_loss = torch.tensor(0.0)
                fake_loss = torch.tensor(0.0)
                adv_loss = torch.tensor(0.0)

            if save_next_t_batch:
                save_next_t_batch = False
                persist_recolored_images(
                    options, "train", epoch, batch_id, src_images, tgt_palettes, tgt_images, recolored_images, model_dir
                )

            if batch_id % 10 == 0:
                print(
                    "%s: EPOCH %s/%s, STEP %s/%s, pa_loss=(%.1f, %.1f, %.1f), adv_loss=(%.1f, %.1f, %.1f), nfc=%s" % (
                        datetime.now(), epoch, int(options.num_epochs), batch_id, len(train_loader), l2_loss.item(), pa_adv_loss.item(), train_loss.item(), real_loss.item(), fake_loss.item(), adv_loss.item(), non_finite_count
                    )
                )

        del src_palettes
        del src_images
        del tgt_palettes
        del tgt_images
        del train_loss
        torch.cuda.empty_cache()

        v_l2_loss, v_pa_loss = calculate_validation_loss(
            options, device, epoch, t_pa_l2_loss, t_pa_adv_loss, t_ad_real_loss,
            t_pa_adv_loss, palette_applier, adversary, valid_loader, False, model_dir
        )

        if v_l2_loss < best_l2_loss:
            save_next_t_batch = True
            best_l2_loss = v_l2_loss
            calculate_validation_loss(
                options, device, epoch, t_pa_l2_loss, t_pa_adv_loss, t_ad_real_loss, t_pa_adv_loss,
                palette_applier, adversary, valid_loader, True, model_dir, first_batch_only=True
            )
            torch.save(
                palette_applier.state_dict(), os.path.join(model_dir, 'best_checkpoint.pt')
            )
