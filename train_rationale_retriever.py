import os
import json
import math
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from datetime import datetime
from optparse import OptionParser

import clip
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

import constants
from models import RationaleRetriever
from datasets import RationaleRetrievalDataset

DEFAULT_NUM_LAYERS = 1
DEFAULT_USE_LAYER_NORM = False
DEFAULT_EMOTION_REPS = "concatenate"
DEFAULT_DROPOUT = 0
DEFAULT_LR = 0.001
DEFAULT_BALANCE_WEIGHTS = False
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_EPOCHS = 30
DEFAULT_SEED = 100

MODEL_DIRECTORY = os.path.join(constants.MODEL_DIRECTORY, 'rationaleretriever')


def get_weights(dataset, num_rationale_ids):
    y = [dataset.__getitem__(idx)[-2] for idx in range(len(dataset))]

    return torch.tensor(
        compute_class_weight(
            class_weight='balanced', classes=list(range(num_rationale_ids)), y=y
        )
    ).float()


def normalize_reps(reps):
    return reps / reps.norm(dim=-1, keepdim=True)


def handle_batch(batch, logit_scale, rationale_reps, criterion, model):
    image1_reps, image2_reps, emotions, rationale_ids, min_maxes = batch
    rationale_ids = rationale_ids.to(device)
    min_reps, max_reps = model(
        image1_reps.to(device),
        image2_reps.to(device),
        emotions.to(device)
    )
    min_reps = normalize_reps(min_reps)
    max_reps = normalize_reps(max_reps)

    min_ids = []
    max_ids = []
    for idx, min_max in enumerate(min_maxes):
        if min_max == 'MIN':
            min_ids.append(idx)
        elif min_max == 'MAX':
            max_ids.append(idx)
        else:
            raise Exception("Unsupported min/max=%s" % min_max)

    min_logits = logit_scale.exp() * min_reps[min_ids] @ rationale_reps.t()
    max_logits = logit_scale.exp() * max_reps[max_ids] @ rationale_reps.t()

    min_loss = criterion(min_logits, rationale_ids[min_ids])
    max_loss = criterion(max_logits, rationale_ids[max_ids])

    return (min_logits, max_logits), \
           (rationale_ids[min_ids], rationale_ids[max_ids]), \
           (min_loss, max_loss), \
           min_loss + max_loss


def calculate_irs(all_image_logits, all_rationale_ids):
    # @1, 3, 5, 10, 20, 30, 100
    ir_correct = [0, 0, 0, 0, 0, 0, 0]
    for image_logits, rationale_ids in zip(all_image_logits, all_rationale_ids):
        ranked_rationales = torch.argsort(image_logits, descending=True)
        if len(rationale_ids.shape) == 0:
            rationale_ids = rationale_ids.unsqueeze(dim=0)
        else:
            rationale_ids = rationale_ids.nonzero().flatten()
        found = False
        for rationale_id in rationale_ids:
            if rationale_id == ranked_rationales[0]:
                ir_correct[0] += 1
                found = True

            if rationale_id in ranked_rationales[:3]:
                ir_correct[1] += 1
                found = True

            if rationale_id in ranked_rationales[:5]:
                ir_correct[2] += 1
                found = True

            if rationale_id in ranked_rationales[:10]:
                ir_correct[3] += 1
                found = True

            if rationale_id in ranked_rationales[:20]:
                ir_correct[4] += 1
                found = True

            if rationale_id in ranked_rationales[:30]:
                ir_correct[5] += 1
                found = True

            if rationale_id in ranked_rationales[:100]:
                ir_correct[6] += 1
                found = True

            if found:
                break
    return np.array(ir_correct)


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
        '--num-layers',
        dest='num_layers',
        default=DEFAULT_NUM_LAYERS
    )
    parser.add_option(
        '--use-layer-norm',
        dest='use_layer_norm',
        action='store_true',
        default=DEFAULT_USE_LAYER_NORM
    )
    parser.add_option(
        '--emotion-reps',
        type='emotion_reps',
        choices=['concatenate', 'sum'],
        default=DEFAULT_EMOTION_REPS
    )
    parser.add_option(
        '--lr',
        dest='lr',
        default=DEFAULT_LR
    )
    parser.add_option(
        '--balance',
        dest='balance_weights',
        action='store_true',
        default=DEFAULT_BALANCE_WEIGHTS
    )
    parser.add_option(
        '--dropout',
        dest='dropout',
        default=DEFAULT_DROPOUT
    )
    parser.add_option(
        '--batch-size',
        dest='batch_size',
        default=DEFAULT_BATCH_SIZE
    )
    parser.add_option(
        '--num-epochs',
        dest='num_epochs',
        default=DEFAULT_NUM_EPOCHS
    )
    parser.add_option(
        '--seed',
        dest='seed',
        default=DEFAULT_SEED
    )
    (options, args) = parser.parse_args()

    model_dir = os.path.join(MODEL_DIRECTORY, options.variant)
    Path(model_dir).mkdir(parents=True, exist_ok=False)

    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump(vars(options), f, indent=4)

    if torch.cuda.is_available():
        print("Using GPU!")
        device = torch.device('cuda')
    else:
        print("Using CPU...")
        device = torch.device('cpu')

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model = clip_model.eval()

    # collect all unique images and rationales

    image_filepaths, rationale_subsets = set(), {'train': set(), 'valid': set()}
    for subset in ['train', 'valid']:
        dataset = RationaleRetrievalDataset(subset)
        for i in range(len(dataset)):
            min_filepath, max_filepath, _, rationale, _ = dataset.__getitem__(i)
            image_filepaths.update([min_filepath, max_filepath])
            rationale_subsets[subset].add(rationale.lower())
    image_filepaths = list(sorted(image_filepaths))
    rationale_subsets['train'] = list(sorted(rationale_subsets['train']))
    rationale_subsets['valid'] = list(sorted(rationale_subsets['valid']))

    # generating CLIP representations for images

    image_reps = []
    for image_filepath in tqdm(image_filepaths, desc='generating image representations'):
        image = preprocess(Image.open(image_filepath)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_rep = clip_model.encode_image(image)
            image_reps.append(image_rep.squeeze().cpu())
    image_reps = normalize_reps(torch.stack(image_reps))

    # generating CLIP representations for rationales

    rationale_reps_subsets = {}
    for subset, rationales in rationale_subsets.items():
        rationale_reps = []
        for rationale in tqdm(rationales, desc='generating %s rationale representations' % subset):
            with torch.no_grad():
                rationale_reps.append(
                    clip_model.encode_text(
                        clip.tokenize(
                            [rationale], truncate=True
                        ).to(device)
                    ).squeeze().cpu()
                )
        rationale_reps_subsets[subset] = normalize_reps(torch.stack(rationale_reps))

    # extract the CLIP model's logit_scale and then delete it

    logit_scale = clip_model.logit_scale.detach().cpu()
    del clip_model

    train = RationaleRetrievalDataset(
        'train', image_filepaths_and_reps=(image_filepaths, image_reps), rationales=rationale_subsets['train']
    )
    valid = RationaleRetrievalDataset(
        'valid', image_filepaths_and_reps=(image_filepaths, image_reps), rationales=rationale_subsets['valid']
    )

    train_weights = get_weights(train, len(rationale_subsets['train'])).to(device)
    valid_weights = get_weights(valid, len(rationale_subsets['valid'])).to(device)

    np.random.seed(options.seed)
    torch.manual_seed(options.seed)

    model = RationaleRetriever(
        dropout=options.dropout,
        num_layers=options.num_layers,
        use_layer_norm=options.use_layer_norm,
        emotion_reps=options.emotion_reps
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=options.lr)

    if not options.balance_weights:
        train_criterion = CrossEntropyLoss()
    else:
        train_criterion = CrossEntropyLoss(weight=train_weights)
    valid_criterion = CrossEntropyLoss(weight=valid_weights)

    train_loader = DataLoader(train, batch_size=options.batch_size, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=2 * options.batch_size)

    v_best_epoch = None
    v_best_irs = None
    v_best_loss = (None, None, math.inf)
    for epoch in tqdm(range(options.num_epochs), desc='training %s' % options.variant):
        t_tot_loss = 0.0
        model = model.train()
        train_rationale_reps = rationale_reps_subsets['train'].to(device)
        for t_batch_num, t_batch in enumerate(train_loader):
            optimizer.zero_grad()
            _, _, _, t_loss = handle_batch(
                t_batch, logit_scale,
                train_rationale_reps, train_criterion, model
            )
            t_loss.backward()
            optimizer.step()

            t_tot_loss += t_loss.detach().cpu().item()

        v_tot_min_loss = 0.0
        v_tot_max_loss = 0.0
        v_tot_loss = 0.0
        v_tot_min_ir_correct = np.array([0, 0, 0, 0, 0, 0, 0])
        v_tot_max_ir_correct = np.array([0, 0, 0, 0, 0, 0, 0])
        v_tot_ir_correct = np.array([0, 0, 0, 0, 0, 0, 0])

        model = model.eval()
        valid_rationale_reps = rationale_reps_subsets['valid'].to(device)
        with torch.no_grad():
            for v_batch_num, v_batch in enumerate(valid_loader):
                (v_min_logits, v_max_logits), \
                (v_min_rationale_ids, v_max_rationale_ids), \
                (v_min_loss, v_max_loss), v_loss = handle_batch(
                    v_batch, logit_scale,
                    valid_rationale_reps, valid_criterion, model
                )

                v_tot_min_loss += v_min_loss.detach().cpu().item()
                v_tot_max_loss += v_max_loss.detach().cpu().item()
                v_tot_loss += v_loss.detach().cpu().item()

                v_min_ir_correct = calculate_irs(v_min_logits, v_min_rationale_ids)
                v_max_ir_correct = calculate_irs(v_max_logits, v_max_rationale_ids)

                v_tot_ir_correct += v_min_ir_correct
                v_tot_ir_correct += v_max_ir_correct

        v_irs = ', '.join(map(lambda v_ir: '%0.2f' % v_ir, v_tot_ir_correct / len(valid)))

        if v_tot_loss < v_best_loss[-1]:
            v_best_epoch = epoch
            v_best_irs = v_irs
            v_best_loss = (v_tot_min_loss, v_tot_max_loss, v_tot_loss)
            torch.save(
                model.state_dict(), os.path.join(model_dir, 'best_checkpoint.pt')
            )

    print('END OF %s: best=%s, v_loss=(%0.4f, %0.4f, %0.4f), v_irs(1,3,5,10,20,30,100)=%s' % (
        options.variant, v_best_epoch, *v_best_loss, v_best_irs))
