import os
import math
import torch
import pickle
import skimage
import numpy as np
from PIL import Image
from skimage import color
from torchvision import transforms
from torchvision.utils import save_image

import webcolors
import colorgram
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000

from collections import defaultdict

CACHE_DIR = 'cache/color_regions/'

TARGET_IMAGE_SIZE = 224


class ColorRegion:
    def __init__(self, canonical_rgb, canonical_lab, points, orig_image):
        self.points = points
        self.canonical_rgb = canonical_rgb
        self.canonical_lab = canonical_lab
        self.region = np.zeros(orig_image.shape)
        for (x, y) in points:
            self.region[x, y, :] = orig_image[x, y, :]
        self.region = transforms.ToTensor()(self.region).float()

    def get_bounding_box(self, offset):
        top_left_x = math.inf
        top_left_y = math.inf
        bottom_right_x = -math.inf
        bottom_right_y = -math.inf
        for (x, y) in self.points:
            top_left_x = min(top_left_x, x)
            top_left_y = min(top_left_y, y)
            bottom_right_x = max(bottom_right_x, x)
            bottom_right_y = max(bottom_right_y, y)

        if offset:
            top_left_x = TARGET_IMAGE_SIZE + top_left_x
            bottom_right_x = TARGET_IMAGE_SIZE + bottom_right_x

        return (top_left_x, top_left_y), (bottom_right_x, bottom_right_y)


# not perfect as LAB space isn't perfectly rectangular but oh well
# values from https://github.com/scikit-image/scikit-image/issues/4506
def clamp_lab_channel(value, channel):
    if channel == 'l':
        return min(max(0.0, value), 100.0)
    elif channel == 'a':
        return min(max(-86.18302974, value), 98.23305386)
    elif channel == 'b':
        return min(max(-107.85730021, value), 94.47812228)
    else:
        raise Exception("Unsupported channel: %s" % channel)


def get_padding_details(image):
    width = image.width
    height = image.height

    if width < height:
        new_width = int(width * (TARGET_IMAGE_SIZE / height))
        new_height = TARGET_IMAGE_SIZE

        left_padding = int((TARGET_IMAGE_SIZE - new_width) / 2)
        right_padding = (TARGET_IMAGE_SIZE - new_width) - left_padding
        padding = (left_padding, 0, right_padding, 0)
    else:
        new_width = TARGET_IMAGE_SIZE
        new_height = int(height * (TARGET_IMAGE_SIZE / width))

        top_padding = int((TARGET_IMAGE_SIZE - new_height) / 2)
        bottom_padding = (TARGET_IMAGE_SIZE - new_height) - top_padding
        padding = (0, top_padding, 0, bottom_padding)
    return new_width, new_height, padding


# maintains aspect ratio and reduces image size to
# 224 x 224, padding the shorter length with zeros
def transform_and_pad(image):
    new_width, new_height, padding = get_padding_details(image)

    image = image.resize((new_width, new_height))

    return transforms.Pad(padding)(image)


CORRECTED_COLOR_NAMES = {
    'aliceblue': 'alice blue',
    'antiquewhite': 'antique white',
    'aquamarine': 'aquamarine',
    'azure': 'azure',
    'beige': 'beige',
    'bisque': 'bisque',
    'black': 'black',
    'blanchedalmond': 'blanched almond',
    'blue': 'blue',
    'blueviolet': 'blue violet',
    'brown': 'brown',
    'burlywood': 'burly wood',
    'cadetblue': 'cadet blue',
    'chartreuse': 'chartreuse',
    'chocolate': 'chocolate',
    'coral': 'coral',
    'cornflowerblue': 'cornflower blue',
    'cornsilk': 'corn silk',
    'crimson': 'crimson',
    'cyan': 'cyan',
    'darkblue': 'dark blue',
    'darkcyan': 'dark cyan',
    'darkgoldenrod': 'dark goldenrod',
    'darkgray': 'dark gray',
    'darkgreen': 'dark green',
    'darkkhaki': 'dark khaki',
    'darkmagenta': 'dark magenta',
    'darkolivegreen': 'dark olive green',
    'darkorange': 'dark orange',
    'darkorchid': 'dark orchid',
    'darkred': 'dark red',
    'darksalmon': 'dark salmon',
    'darkseagreen': 'dark sea green',
    'darkslateblue': 'dark slate blue',
    'darkslategray': 'dark slate gray',
    'darkturquoise': 'dark turquoise',
    'darkviolet': 'dark violet',
    'deeppink': 'deep pink',
    'deepskyblue': 'deep sky blue',
    'dimgray': 'dim gray',
    'dodgerblue': 'dodger blue',
    'firebrick': 'fire brick',
    'floralwhite': 'floral white',
    'forestgreen': 'forest green',
    'gainsboro': 'gainsboro',
    'ghostwhite': 'ghost white',
    'gold': 'gold',
    'goldenrod': 'goldenrod',
    'gray': 'gray',
    'green': 'green',
    'greenyellow': 'green yellow',
    'honeydew': 'honey dew',
    'hotpink': 'hot pink',
    'indianred': 'indian red',
    'indigo': 'indigo',
    'ivory': 'ivory',
    'khaki': 'khaki',
    'lavender': 'lavender',
    'lavenderblush': 'lavenderblush',
    'lawngreen': 'lawn green',
    'lemonchiffon': 'lemon chiffon',
    'lightblue': 'light blue',
    'lightcoral': 'light coral',
    'lightcyan': 'light cyan',
    'lightgoldenrodyellow': 'light goldenrod yellow',
    'lightgray': 'light gray',
    'lightgreen': 'light green',
    'lightpink': 'light pink',
    'lightsalmon': 'light salmon',
    'lightseagreen': 'light sea green',
    'lightskyblue': 'light sky blue',
    'lightslategray': 'light slate gray',
    'lightsteelblue': 'light steel blue',
    'lightyellow': 'light yellow',
    'lime': 'lime',
    'limegreen': 'lime green',
    'linen': 'linen',
    'magenta': 'magenta',
    'maroon': 'maroon',
    'mediumaquamarine': 'medium aquamarine',
    'mediumblue': 'medium blue',
    'mediumorchid': 'medium orchid',
    'mediumpurple': 'medium purple',
    'mediumseagreen': 'medium seagreen',
    'mediumslateblue': 'medium slate blue',
    'mediumspringgreen': 'medium spring green',
    'mediumturquoise': 'medium turquoise',
    'mediumvioletred': 'medium violet red',
    'midnightblue': 'midnight blue',
    'mintcream': 'mint cream',
    'mistyrose': 'misty rose',
    'moccasin': 'moccasin',
    'navajowhite': 'navajo white',
    'navy': 'navy',
    'oldlace': 'old lace',
    'olive': 'olive',
    'olivedrab': 'olive drab',
    'orange': 'orange',
    'orangered': 'orange red',
    'orchid': 'orchid',
    'palegoldenrod': 'pale goldenrod',
    'palegreen': 'pale green',
    'paleturquoise': 'pale turquoise',
    'palevioletred': 'pale violet red',
    'papayawhip': 'papaya whip',
    'peachpuff': 'peach puff',
    'peru': 'peru',
    'pink': 'pink',
    'plum': 'plum',
    'powderblue': 'powder blue',
    'purple': 'purple',
    'red': 'red',
    'rosybrown': 'rosy brown',
    'royalblue': 'royal blue',
    'saddlebrown': 'saddle brown',
    'salmon': 'salmon',
    'sandybrown': 'sandy brown',
    'seagreen': 'sea green',
    'seashell': 'seashell',
    'sienna': 'sienna',
    'silver': 'silver',
    'skyblue': 'sky blue',
    'slateblue': 'slate blue',
    'slategray': 'slate gray',
    'snow': 'snow',
    'springgreen': 'spring green',
    'steelblue': 'steel blue',
    'tan': 'tan',
    'teal': 'teal',
    'thistle': 'thistle',
    'tomato': 'tomato',
    'turquoise': 'turquoise',
    'violet': 'violet',
    'wheat': 'wheat',
    'white': 'white',
    'whitesmoke': 'whitesmoke',
    'yellow': 'yellow',
    'yellowgreen': 'yellow green'
}


def get_lab_color(color_tensor):
    l, a, b = color_tensor.cpu().numpy()
    return LabColor(lab_l=l, lab_a=a, lab_b=b)


# adapted from https://github.com/yongzx/PaletteNet-PyTorch/blob/master/palettenet.ipynb
def apply_hue_shift(lab_image, hue_shift, reuse_l):
    assert 0 <= hue_shift < 1

    if reuse_l:
        orig_l = lab_image[:, :, 0].copy()
    else:
        orig_l = None

    a_2d_index = np.array([
        [1, 0, 0] for _ in range(lab_image.shape[1])]).astype('bool')
    hsv_image = skimage.color.rgb2hsv(skimage.color.lab2rgb(lab_image))
    hsv_image[:, a_2d_index] = (hsv_image[:, a_2d_index] + hue_shift) % 1

    rgb_image = skimage.color.hsv2rgb(hsv_image)
    lab_image = skimage.color.rgb2lab(rgb_image)

    if reuse_l:
        lab_image[:, :, 0] = orig_l
        rgb_image = skimage.color.lab2rgb(lab_image)

    rgb_palette = colorgram.extract(
        Image.fromarray(np.uint8(rgb_image * 255)), 6)
    lab_palette = []
    for color in rgb_palette:
        rgb = [channel / 255.0 for channel in [color.rgb.r, color.rgb.g, color.rgb.b]]
        lab_palette.append(skimage.color.rgb2lab(rgb))
    lab_palette = torch.tensor(lab_palette)

    return lab_palette, lab_image, rgb_image


def closest_color(requested_rgb_color):
    r, g, b = requested_rgb_color
    requested_lab_color = LabColor(*color.rgb2lab([[r / 255, g / 255, b / 255]])[0])

    color_dists = []
    for hex, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(hex)
        lab_color = LabColor(*color.rgb2lab([[r_c / 255, g_c / 255, b_c / 255]])[0])
        delta_e_dist = delta_e_cie2000(requested_lab_color, lab_color)
        color_dists.append((name, delta_e_dist))
    return min(color_dists, key=lambda name_and_dist: name_and_dist[-1])[0]


def get_color_name(requested_color):
    return CORRECTED_COLOR_NAMES[closest_color(requested_color)]


def get_palette_color_regions(image_path, num_colors, pad=False, visualize=False):
    # cache_filename = '%s-color-regions.pkl' % (os.path.basename(image_path))
    # if os.path.exists(cache_filename):
    #     with open(cache_filename, 'rb') as f:
    #         regions = pickle.load(f)
    #     return regions

    image = Image.open(image_path).convert('RGB')

    rgb_palette = [
        [c.rgb.r, c.rgb.g, c.rgb.b] for c in colorgram.extract(image_path, num_colors)]
    lab_palette = transforms.ToTensor()(
        color.rgb2lab([[r / 255, g / 255, b / 255] for (r, g, b) in rgb_palette]))

    if pad:
        image = transform_and_pad(image)

    rgb_image = np.array(image)
    lab_image = transforms.ToTensor()(color.rgb2lab(image))

    palette_clusters_to_locations = defaultdict(list)
    for r in range(lab_image.shape[1]):
        for c in range(lab_image.shape[2]):
            image_color = get_lab_color(lab_image[:, r, c])

            min_palette_color_idx = 0
            min_palette_color_dist = math.inf
            for palette_color_idx, palette_color in enumerate(lab_palette[0]):
                palette_color = get_lab_color(palette_color)

                dist = delta_e_cie2000(image_color, palette_color)
                if dist < min_palette_color_dist:
                    min_palette_color_idx = palette_color_idx
                    min_palette_color_dist = dist

            palette_clusters_to_locations[min_palette_color_idx].append((r, c))

    if visualize:
        for palette_color_idx in range(len(lab_palette[0])):
            cluster_image = torch.zeros(lab_image.shape)
            for (row, col) in palette_clusters_to_locations[palette_color_idx]:
                r, g, b = rgb_image[row, col, :]
                cluster_image[0, row, col] = r / 255
                cluster_image[1, row, col] = g / 255
                cluster_image[2, row, col] = b / 255
            save_image(cluster_image, 'cluster-%s.png' % palette_color_idx)

    regions = [
        ColorRegion(rgb_palette[color_idx], lab_palette[0][color_idx], points, rgb_image)
        for (color_idx, points) in palette_clusters_to_locations.items()
    ]
    # with open(os.path.join(CACHE_DIR, cache_filename), 'wb') as f:
    #     pickle.dump(regions, f)

    return regions


def get_contiguous_color_regions(image_path, lab_image, collapse_into=None, delta_e_distance=1.0):
    cache_filename = '%s-%s-%s.pkl' % (os.path.basename(image_path), collapse_into, delta_e_distance)
    print(cache_filename)
    if os.path.exists(os.path.join(CACHE_DIR, cache_filename)):
        with open(os.path.join(CACHE_DIR, cache_filename), 'rb') as f:
            return pickle.load(f)

    current_region = -1
    locations_to_regions = {}
    regions_to_locations = defaultdict(list)
    for r in range(lab_image.shape[1]):
        for c in range(lab_image.shape[2]):
            if (r, c) in locations_to_regions:
                continue

            current_region += 1
            locations_processed = set()
            locations_to_process = [(None, (r, c))]
            locations_to_regions[(r, c)] = current_region
            regions_to_locations[current_region].append((r, c))
            while len(locations_to_process) > 0:
                neighbor, image_location = locations_to_process.pop()

                if image_location in locations_processed:
                    continue
                locations_processed.add(image_location)

                include_neighbors = False
                if neighbor:
                    neighbor_color = get_lab_color(lab_image[:, neighbor[0], neighbor[1]])
                    image_color = get_lab_color(lab_image[:, image_location[0], image_location[1]])

                    if delta_e_cie2000(neighbor_color, image_color) < delta_e_distance:
                        include_neighbors = True
                        regions_to_locations[current_region].append(image_location)
                        locations_to_regions[image_location] = current_region
                else:
                    include_neighbors = True

                if include_neighbors:
                    for r_step in [-1, 0, 1]:
                        for c_step in [-1, 0, 1]:
                            new_r = image_location[0] + r_step
                            new_c = image_location[1] + c_step

                            if not 0 <= new_r < lab_image.shape[1]:
                                continue

                            if not 0 <= new_c < lab_image.shape[2]:
                                continue

                            if (new_r, new_c) in locations_to_regions:
                                continue

                            if (new_r, new_c) in locations_processed:
                                continue

                            locations_to_process.append((image_location, (new_r, new_c)))

    regions = list(map(lambda points: ColorRegion(points), regions_to_locations.values()))

    if collapse_into and len(regions) > collapse_into:
        sorted_regions = list(sorted(regions, key=lambda region: len(region.points)))

        biggest_regions = sorted_regions[-collapse_into:]
        for other_region in sorted_regions[:-collapse_into]:
            for (r, c) in other_region.points:
                closest_region = None
                closest_distance = math.inf
                for biggest_region in biggest_regions:
                    for (br, bc) in biggest_region.points:
                        distance = math.sqrt((r - br) ** 2 + (c - bc) ** 2)
                        if distance < closest_distance:
                            closest_region = biggest_region
                            closest_distance = distance

                        if closest_distance == 1:
                            break
                    if closest_distance == 1:
                        break
                closest_region.points.append((r, c))

        regions = biggest_regions

    with open(os.path.join(CACHE_DIR, cache_filename), 'wb') as f:
        pickle.dump(regions, f)

    return regions


def visualize_color_regions(image_path, collapse_into=None, distances='1'):
    image = Image.open(image_path).convert('RGB')
    lab_image = transforms.ToTensor()(color.rgb2lab(image))

    rgb_image = np.array(image)
    for distance in map(float, distances.split(',')):
        mapped_image = torch.zeros(lab_image.shape)
        contiguous_regions = get_contiguous_color_regions(
            image_path, lab_image, collapse_into=collapse_into, delta_e_distance=distance)
        points_to_regions = {point: region \
                             for region in contiguous_regions for point in region.points}
        for r_idx in range(mapped_image.shape[1]):
            for c_idx in range(mapped_image.shape[2]):
                if (r_idx, c_idx) not in points_to_regions:
                    print("Missing: (%s, %s)" % (r_idx, c_idx))
                    continue
                region = points_to_regions[(r_idx, c_idx)]
                r, g, b = rgb_image[region.points[0][0], region.points[0][1], :]
                mapped_image[0, r_idx, c_idx] = r / 255
                mapped_image[1, r_idx, c_idx] = g / 255
                mapped_image[2, r_idx, c_idx] = b / 255

        print("Distance=%s, num regions=%s" % (distance, len(contiguous_regions)))
        save_image(mapped_image, 'regions-%s.png' % distance)


def normalize_l(l):
    return l / 50 - 1

def unnormalize_l(l):
    return 50 * (l + 1)

def normalize_a(a):
    return ((a + 86.18) / 92.2) - 1

def unnormalize_a(a):
    return 92.2 * (a + 1) - 86.18

def normalize_b(b):
    return ((b + 107.85) / 101.16) - 1

def unnormalize_b(b):
    return 101.16 * (b + 1) - 107.85


def normalize_lab_image(image):
    return normalize_lab_colors(image, 0)


# to between -1 and 1
def normalize_lab_colors(image, dim):
    assert image.shape[dim] == 3

    if dim == 0:
        l, a, b = image[0], image[1], image[2]
    elif dim == 1:
        l, a, b = image[:, 0], image[:, 1], image[:, 2]
    else:
        raise Exception("Unsupported dim: %s" % dim)

    return torch.stack([
        normalize_l(l).squeeze(),
        normalize_a(a).squeeze(),
        normalize_b(b).squeeze()
    ], dim=dim)


# to between -1 and 1
def unnormalize_lab_colors(image, dim):
    assert image.shape[dim] == 3

    if dim == 0:
        # 3 x 224 x 224
        l, a, b = image[0], image[1], image[2]
    elif dim == 1:
        # batch_size x 3 x 224 x 224
        # or num_colors x 3
        l, a, b = image[:, 0], image[:, 1], image[:, 2]
    else:
        raise Exception("Unsupported dim: %s" % dim)

    return torch.stack([
        unnormalize_l(l).squeeze(),
        unnormalize_a(a).squeeze(),
        unnormalize_b(b).squeeze()
    ], dim=dim)
