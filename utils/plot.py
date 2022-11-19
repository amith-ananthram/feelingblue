import skimage
import numpy as np

import constants
from utils.colors import unnormalize_lab_colors


# returns the image as a numpy array in RGB space
def get_image(image, is_lab, unnormalize=False):
    if unnormalize:
        if is_lab:
            image = unnormalize_lab_colors(image, 0)
        else:
            image = 255 * image

    image = image.detach().cpu().permute(1, 2, 0)

    if is_lab:
        return skimage.color.lab2rgb(image)
    else:
        return image.numpy()


def get_palette(palette, is_lab, max_palette_length=6, height=10, width=10, allow_errs=False, unnormalize=False):
    if unnormalize:
        if is_lab:
            palette = unnormalize_lab_colors(palette, 1)
        else:
            palette = 255 * palette

    bad_ids = []
    full_palette = np.zeros((height, max_palette_length * width, 3))
    for color_id, color in enumerate(palette):
        if is_lab:
            try:
                color = skimage.color.lab2rgb(color.detach().cpu())
            except:
                if allow_errs:
                    bad_ids.append(color_id)
                    color = np.zeros(3)
                else:
                    raise
        for r in range(height):
            for c in range(color_id * width, (color_id + 1) * width):
                full_palette[r, c, :] = color
    return bad_ids, full_palette


def plot_image(image_plot, image, title):
    image_plot.imshow(image)
    image_plot.axis('off')
    image_plot.set_title(title)


def get_emotion(emotion):
    return constants.EMOTIONS[np.argmax(emotion.cpu().numpy())]
