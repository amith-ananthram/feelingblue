import os

CORPUS_DIRECTORY = 'corpora'
WIKIART_DIRECTORY = os.path.join(CORPUS_DIRECTORY, 'wikiart')

EMOTIONS = list(sorted(['anger', 'disgust', 'fear', 'happiness', 'sadness']))

FEELING_BLUE_ANNOTATIONS_FILE = os.path.join(CORPUS_DIRECTORY, 'feeling_blue.csv')
FEELING_BLUE_SPLITS_FILE = os.path.join(CORPUS_DIRECTORY, 'feelingblue_splits.json')

CHOICE_COLUMN = 'selected_image_id'
MIN_MAX_COLUMN = 'min/max'
EMOTION_COLUMN = 'emotion'
RATIONALE_COLUMN = 'rationale'

NUM_HUE_SHIFTS = 18
HUE_SHIFT_NUM_COLORS_IN_PALETTE = 6

MODEL_DIRECTORY = 'trained_models'


