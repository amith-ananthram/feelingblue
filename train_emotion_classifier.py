import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from optparse import OptionParser
from collections import defaultdict
from sklearn.metrics import accuracy_score

import torch
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader

import utils
import constants
from models import EmotionClassifier
from datasets import EmotionClassificationDataset

DEFAULT_LR = 1e-3
DEFAULT_DROPOUT = 0.1
DEFAULT_USE_PRETRAINED = False
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_EPOCHS = 3
DEFAULT_RESNET_SIZE = 50
DEFAULT_COLOR_SPACE = "lab"
DEFAULT_SEED = 100

RESNET_OUTPUT_SIZE = 2048

MODEL_DIRECTORY = os.path.join(constants.MODEL_DIRECTORY, 'emotionclassifier')


def get_preds_and_labels_by_emotion(labels, preds, emotions):
    lhs_preds, rhs_preds = defaultdict(list), defaultdict(list)
    lhs_labels, rhs_labels = defaultdict(list), defaultdict(list)
    for label, pred, emotion in zip(labels, torch.softmax(preds, 1), emotions):
        label = label.item()
        pred = torch.argmax(pred).item()
        emotion = utils.get_emotion(emotion)
        if label == 0:
            rhs_preds[emotion].append(pred)
            rhs_labels[emotion].append(label)
        else:
            lhs_preds[emotion].append(pred)
            lhs_labels[emotion].append(label)

    return (lhs_preds, rhs_preds), (lhs_labels, rhs_labels)


def get_formatted_accuracy_lines(lhs_rhs_preds, lhs_rhs_labels):
    lines = []
    both_preds, both_labels = defaultdict(list), defaultdict(list)
    for (preds, labels, side) in [(lhs_rhs_preds[0], lhs_rhs_labels[0], 'lhs'), (lhs_rhs_preds[1], lhs_rhs_labels[1], 'rhs')]:
        tot_preds = []
        tot_labels = []
        emotion_lines = []
        for emotion in constants.EMOTIONS:
            tot_preds.extend(preds[emotion])
            tot_labels.extend(labels[emotion])

            both_preds[emotion].extend(preds[emotion])
            both_labels[emotion].extend(labels[emotion])

            emotion_lines.append("%s: %.3f" % (emotion, accuracy_score(labels[emotion], preds[emotion])))
        lines.append("%s: %.3f (%s)" % (side, accuracy_score(tot_labels, tot_preds), ', '.join(emotion_lines)))

    tot_preds = []
    tot_labels = []
    emotion_lines = []
    for emotion in constants.EMOTIONS:
        tot_preds.extend(both_preds[emotion])
        tot_labels.extend(both_labels[emotion])

        emotion_lines.append("%s: %.3f" % (emotion, accuracy_score(both_labels[emotion], both_preds[emotion])))
    lines.append("both: %.3f (%s)" % (accuracy_score(tot_labels, tot_preds), ', '.join(emotion_lines)))

    return lines


def calculate_validation_accuracy(epoch, t_train_loss, t_train_preds, t_train_labels, emotion_classifier, loader):
    v_loss = 0.0
    v_lhs_preds, v_rhs_preds = defaultdict(list), defaultdict(list)
    v_lhs_labels, v_rhs_labels = defaultdict(list), defaultdict(list)
    emotion_classifier.eval()
    for batch in loader:
        with torch.no_grad():
            _, _, image1s, image2s, emotions, batch_labels = utils.move_to_device(batch, device)
            batch_preds = emotion_classifier(image1s, image2s, emotions)

            v_loss += F.cross_entropy(batch_preds, batch_labels.squeeze()).item()
            (b_lhs_preds, b_rhs_preds), (b_lhs_labels, b_rhs_labels) = \
                get_preds_and_labels_by_emotion(batch_labels, batch_preds, emotions)

            for emotion in constants.EMOTIONS:
                v_lhs_preds[emotion].extend(b_lhs_preds[emotion])
                v_rhs_preds[emotion].extend(b_rhs_preds[emotion])

                v_lhs_labels[emotion].extend(b_lhs_labels[emotion])
                v_rhs_labels[emotion].extend(b_rhs_labels[emotion])

    now = datetime.now()
    print("%s: EPOCH %s COMPLETE, t_loss=%.3f, v_loss=%s" % (now, epoch, t_train_loss, v_loss))
    for line in get_formatted_accuracy_lines(t_train_preds, t_train_labels):
        print("%s: EPOCH %s COMPLETE, train, %s" % (now, epoch, line))
    for line in get_formatted_accuracy_lines((v_lhs_preds, v_rhs_preds), (v_lhs_labels, v_rhs_labels)):
        print("%s: EPOCH %s COMPLETE, valid, %s" % (now, epoch, line))


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
        '--rs',
        '--resnet-size',
        dest='resnet_size',
        default=DEFAULT_RESNET_SIZE
    )
    parser.add_option(
        '-c',
        '--color-space',
        dest='color_space',
        default=DEFAULT_COLOR_SPACE
    )
    parser.add_option(
        '--lr',
        '--learning-rate',
        dest='lr',
        default=DEFAULT_LR
    )
    parser.add_option(
        '-d',
        '--dropout',
        dest='dropout',
        default=DEFAULT_DROPOUT
    )
    parser.add_option(
        '-p',
        '--use-pretrained',
        dest='use_pretrained',
        type='choice',
        choices=['yes', 'no'],
        default='yes'
    )
    parser.add_option(
        '--normalize-inputs',
        dest='normalize_inputs',
        type='choice',
        choices=['yes', 'no']
    )
    parser.add_option(
        '--model-normalization',
        dest='model_normalization',
        type='choice',
        choices=['none', 'batch', 'instance'],
        default='none'
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
        '-s',
        '--seed',
        dest='seed',
        default=DEFAULT_SEED
    )
    (options, args) = parser.parse_args()

    model_dir = os.path.join(MODEL_DIRECTORY, options.variant)
    Path(model_dir).mkdir(parents=True, exist_ok=False)

    config = {
        'variant': options.variant,
        'resnet_size': options.resnet_size,
        'color_space': options.color_space,
        'lr': options.lr,
        'dropout': options.dropout,
        'use_pretrained': options.use_pretrained,
        'normalize_inputs': options.normalize_inputs,
        'model_normalization': options.model_normalization,
        'batch_size': options.batch_size,
        'num_epochs': options.num_epochs,
        'seed': options.seed
    }
    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump(config, f)

    if options.seed:
        np.random.seed(int(options.seed))
        torch.manual_seed(int(options.seed))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        print("%s: GPU detected, will use for training..." % datetime.now())

    use_lab = options.color_space.lower() == 'lab'
    if use_lab:
        print("%s: Using LAB space..." % datetime.now())

    emotion_classifier = EmotionClassifier(
        device,
        use_pretrained_resnet=options.use_pretrained.lower() == 'yes',
        resnet_size=int(options.resnet_size),
        normalization=options.model_normalization,
        dropout=float(options.dropout)
    ).to(device)

    optimizer = Adam(emotion_classifier.parameters(), lr=float(options.lr))

    normalize_inputs = options.normalize_inputs.lower() == 'yes'
    dataset_args = {'convert_to_lab': use_lab, 'normalize_inputs': normalize_inputs}
    valid, train = (
        EmotionClassificationDataset("valid", **dataset_args),
        EmotionClassificationDataset("train", **dataset_args)
    )

    print("%s: Loaded datasets..." % datetime.now())
    print("%s: valid=%s" % (datetime.now(), len(valid)))
    print("%s: train=%s" % (datetime.now(), len(train)))

    valid_loader, train_loader = \
        DataLoader(valid, batch_size=2 * int(options.batch_size), num_workers=8), \
        DataLoader(train, batch_size=int(options.batch_size), shuffle=True, num_workers=8)

    calculate_validation_accuracy(
        -1, -1, (defaultdict(list), defaultdict(list)),
        (defaultdict(list), defaultdict(list)), emotion_classifier, valid_loader
    )

    for epoch in range(int(options.num_epochs)):
        t_tot_loss = 0
        t_lhs_preds, t_rhs_preds = defaultdict(list), defaultdict(list)
        t_lhs_labels, t_rhs_labels = defaultdict(list), defaultdict(list)

        emotion_classifier.train()
        for batch_id, batch in enumerate(train_loader):
            optimizer.zero_grad()

            _, _, image1s, image2s, emotions, labels = utils.move_to_device(batch, device)
            preds = emotion_classifier(image1s, image2s, emotions)

            train_loss = F.cross_entropy(preds, labels.squeeze())

            train_loss.backward()
            optimizer.step()

            t_tot_loss += train_loss.item()
            (b_lhs_preds, b_rhs_preds), (b_lhs_labels, b_rhs_labels) = \
                get_preds_and_labels_by_emotion(labels, preds, emotions)

            for emotion in constants.EMOTIONS:
                t_lhs_preds[emotion].extend(b_lhs_preds[emotion])
                t_rhs_preds[emotion].extend(b_rhs_preds[emotion])

                t_lhs_labels[emotion].extend(b_lhs_labels[emotion])
                t_rhs_labels[emotion].extend(b_rhs_labels[emotion])

            if batch_id % 10 == 0:
                print(
                    "%s: EPOCH %s/%s, STEP %s/%s, train_loss=%.3f" % (
                        datetime.now(), epoch, int(options.num_epochs), batch_id, len(train_loader), train_loss.item()
                    )
                )

        torch.save(
            emotion_classifier.state_dict(), os.path.join(model_dir, 'checkpoint%s.pt' % epoch)
        )

        del image1s
        del image2s
        del emotions
        del labels
        del preds
        del train_loss

        calculate_validation_accuracy(
            epoch, t_tot_loss, (t_lhs_preds, t_rhs_preds), (t_lhs_labels, t_rhs_labels), emotion_classifier, valid_loader
        )
