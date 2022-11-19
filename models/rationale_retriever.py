import torch
from torch import nn

import constants

CLIP_DIM = 512


class RationaleRetriever(nn.Module):
    def __init__(self, dropout, num_layers, use_layer_norm, emotion_reps):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.emotion_reps = emotion_reps

        if self.emotion_reps == 'concatenate':
            self.emotion_dim = CLIP_DIM
            self.combined_dim = 3 * CLIP_DIM
        elif self.emotion_reps == 'sum':
            self.emotion_dim = 2 * CLIP_DIM
            self.combined_dim = 2 * CLIP_DIM
        else:
            raise Exception("Unsupported emotion_reps=%s" % emotion_reps)

        self.emotion_embeddings = nn.Embedding(
            len(constants.EMOTIONS), self.emotion_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(self.combined_dim, 2 * CLIP_DIM)
        self.ln1 = nn.LayerNorm(2 * CLIP_DIM)

        self.num_layers = num_layers
        if self.num_layers == 2:
            self.fc2 = nn.Linear(self.combined_dim, 2 * CLIP_DIM)
            self.ln2 = nn.LayerNorm(2 * CLIP_DIM)
        else:
            assert self.num_layers == 1
        self.min_head = nn.Linear(self.combined_dim, CLIP_DIM)
        self.max_head = nn.Linear(self.combined_dim, CLIP_DIM)

    def combine(self, elements):
        if self.emotion_reps == 'concatenate':
            return torch.cat(elements, dim=1)
        elif self.emotion_reps == 'sum':
            if len(elements) == 2:
                return elements[0] + elements[1]
            elif len(elements) == 3:
                images = torch.cat((elements[0], elements[1]), dim=1)
                return images + elements[2]
            else:
                raise Exception("Unsupported length=%s" % len(elements))
        else:
            raise Exception("Unsupported emotion_reps=%s" % self.emotion_reps)

    def forward(self, image1, image2, emotion):
        emotion = self.emotion_embeddings(emotion)
        image_reps = self.dropout(torch.relu(self.fc1(self.combine((image1, image2, emotion)))))
        if self.use_layer_norm:
            image_reps = self.ln1(image_reps)
        if self.num_layers == 2:
            image_reps = self.dropout(torch.relu(self.fc2(self.combine((image_reps, emotion)))))
            if self.use_layer_norm:
                image_reps = self.ln2(image_reps)
        else:
            assert self.num_layers == 1

        min_image_reps = self.min_head(self.combine((image_reps, emotion)))
        max_image_reps = self.max_head(self.combine((image_reps, emotion)))

        return min_image_reps, max_image_reps
