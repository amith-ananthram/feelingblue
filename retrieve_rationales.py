import os
import sys
import json
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import clip

import constants
from models import RationaleRetriever


def get_image_rep(device, clip_model, preprocess, image):
    with torch.no_grad():
        image_rep = clip_model.encode_image(
            preprocess(Image.open(image)).unsqueeze(0).to(device)
        ).squeeze().cpu()
        return image_rep / image_rep.norm()


def get_image_reps(device, clip_model, preprocess, rationale_task_csv):
    image_reps = {}
    for _, rationale_task in tqdm(pd.read_csv(rationale_task_csv).iterrows(), desc='generating image representations'):
        min_image, max_image = rationale_task['less_image_filepath'], rationale_task['more_image_filepath']
        if min_image not in image_reps:
            image_reps[min_image] = get_image_rep(device, clip_model, preprocess, min_image)

        if max_image not in image_reps:
            image_reps[max_image] = get_image_rep(device, clip_model, preprocess, max_image)
    return image_reps


def get_rationale_reps(device, clip_model):
    rationales = set()
    for _, row in pd.read_csv(constants.FEELING_BLUE_ANNOTATIONS_FILE).iterrows():
        rationales.add(row[constants.RATIONALE_COLUMN])
    rationales = list(sorted(rationales))

    rationale_reps = []
    for rationale in tqdm(rationales, desc='generating rationale representations'):
        with torch.no_grad():
            rationale_reps.append(
                clip_model.encode_text(
                    clip.tokenize([rationale], truncate=True).to(device)
                ).squeeze().cpu()
            )
    rationale_reps = torch.stack(rationale_reps).float().to(device)

    return rationales, rationale_reps / rationale_reps.norm(dim=-1, keepdim=True)


if __name__ == '__main__':
    rationale_task_csv = sys.argv[1]
    rationale_retriever_path = sys.argv[2]
    output_csv = sys.argv[3]

    if torch.cuda.is_available():
        print("Using GPU!")
        device = torch.device('cuda')
    else:
        print("Using CPU...")
        device = torch.device('cpu')

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model = clip_model.eval()

    logit_scale = clip_model.logit_scale.float()

    variant_dir = os.path.dirname(rationale_retriever_path)
    with open(os.path.join(variant_dir, 'config.json'), 'r') as f:
        config = json.load(f)

    rationale_retriever = RationaleRetriever(
        dropout=config['dropout'],
        num_layers=config['num_layers'],
        use_layer_norm=config['use_layer_norm'],
        emotion_reps=config['emotion_reps']
    )
    rationale_retriever.load_state_dict(torch.load(rationale_retriever_path))
    rationale_retriever = rationale_retriever.to(device)
    rationale_retriever.eval()

    image_reps = get_image_reps(device, clip_model, preprocess, rationale_task_csv)
    rationales, rationale_reps = get_rationale_reps(device, clip_model)

    rationale_results = []
    for _, rationale_task in tqdm(pd.read_csv(rationale_task_csv).iterrows(), desc='picking rationales'):
        min_image = rationale_task['less_image_filepath']
        max_image = rationale_task['more_image_filepath']
        emotion = rationale_task['emotion']

        with torch.no_grad():
            min_rep, max_rep = rationale_retriever(
                image_reps[min_image].unsqueeze(dim=0).to(device),
                image_reps[max_image].unsqueeze(dim=0).to(device),
                torch.tensor([constants.EMOTIONS.index(emotion)]).to(device)
            )
        min_rep = min_rep / min_rep.norm(dim=-1, keepdim=True)
        max_rep = max_rep / max_rep.norm(dim=-1, keepdim=True)

        min_logits = (logit_scale.exp() * min_rep @ rationale_reps.t()).squeeze()
        max_logits = (logit_scale.exp() * max_rep @ rationale_reps.t()).squeeze()

        top_min_rationale_ids = torch.argsort(min_logits, descending=True)[:5]
        top_max_rationale_ids = torch.argsort(max_logits, descending=True)[:5]

        rationale_results.append(
            (
                os.path.basename(min_image),
                os.path.basename(max_image),
                emotion,
                '||'.join(
                    [
                        rationales[rationale_id] for rationale_id in top_min_rationale_ids
                    ]
                ),
                '||'.join(
                    [
                        rationales[rationale_id] for rationale_id in top_max_rationale_ids
                    ]
                )
            )
        )

    pd.DataFrame.from_records(
        rationale_results, columns=[
            'less_image_filepath', 'more_image_filepath',
            'emotion', 'less_rationales', 'more_rationales'
        ]
    ).to_csv(output_csv, index=False)
