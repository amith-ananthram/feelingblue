# FeelingBlue: A Corpus for Understanding the Emotional Connotation of Color in Context

![FeelingBlue](fixtures/corpus_sample.png)

Representative examples spanning **FeelingBlue**'s emotion subsets. Each image in an emotion subset has a score between 
-1 and 1 derived from its Best-Worst Scaling annotations.  Images selected as the _least_/_most_ emotional in a 4-tuple 
(not shown here) have rationales explaining why they are _less_/_more_ emotional than the rest. The names and artists 
behind these works can be found [here](fixtures/corpus_sample_artists.txt).

## Dataset

The subset of DeviantArt/WikiArt annotated in **FeelingBlue** can be downloaded [here](https://drive.google.com/drive/folders/1wIPGNa7AppDY5hI9nPFfe_XrVV8-XyWg?usp=share_link).

### Raw 4-Tuple Data

A CSV containing the annotations in **FeelingBlue** can be found [here](corpora/feelingblue.csv).
It is additionally hosted as a HuggingFace Dataset [here](https://huggingface.co/datasets/owinn/feelingblue_data).

Each row in the CSV (and each example in the HuggingFace Dataset) corresponds to the annotation of a single 
4-tuple of images for a single emotion and direction (less/more).  For example, the first example in the CSV
contains an annotation for the 4-images specified in the columns image(1|2|3|4)_id.  The image id in the selected_image_id
column corresponds to the image_id which the annotator believed was angriest of the four images.

Briefly, the columns in the CSV are as follows:

task_id - a unique identifier for the combination of emotion and 4-tuple

emotion - one of anger, disgust, fear, happiness or sadness

annotator - a unique identifier to the annotator who completed this annotation

min/max - the direction of the annotation, either MIN or MAX

selected_image_id - the image_id of the image which the annotator believed was the least/most emotional of the 4-tuple

rationale - the rationale provided by the annotator for their selection

image1_id - the image_id of the first image in the 4-tuple

image2_id - the image_id of the second image in the 4-tuple

image3_id - the image_id of the third image in the 4-tuple

image4_id - the image_id of the fourth image in the 4-tuple

image1_filename - the filename of the first image in the 4-tuple

image2_filename - the filename of the second image in the 4-tuple

image3_filename - the filename of the third image in the 4-tuple

image4_filename - the filename of the fourth image in the 4-tuple

### Derived Relative Emotion Scores

A CSV containing the derived relative emotion scores from the BWS annotations in **FeelingBlue** can be found [here](corpora/feelingblue_relative_scores.csv).
The identifiers and filenames match those in the raw 4-tuple data.

## Training 

### Preprocessing

### Training the EmotionClassifier

### Training the PaletteApplier

### Training the RationaleRetriever

## Evaluation

Our pretrained models can be found [here](https://drive.google.com/drive/folders/1NmxwxeVydREtIo8kprapzhu-0i_LJQxP?usp=share_link).

### Transforming Images

