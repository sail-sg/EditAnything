import numpy as np
import cv2
from PIL import Image
from skimage.color import rgb2lab
from skimage.color import lab2rgb
from sklearn.cluster import KMeans


def count_high_freq_colors(image):
    im = image.getcolors(maxcolors=1024 * 1024)
    sorted_colors = sorted(im, key=lambda x: x[0], reverse=True)

    freqs = [c[0] for c in sorted_colors]
    mean_freq = sum(freqs) / len(freqs)

    high_freq_colors = [c for c in sorted_colors if c[0] > max(2, mean_freq * 1.25)]
    return high_freq_colors


def get_high_freq_colors(image, similarity_threshold=30):
    image_copy = image.copy()
    high_freq_colors = count_high_freq_colors(image)
    # Check for similar colors and replace the lower frequency color with the higher frequency color in the image
    for i, (freq1, color1) in enumerate(high_freq_colors):
        for j, (freq2, color2) in enumerate(high_freq_colors):
            if (color_distance(color1, color2) < similarity_threshold) or (
                    color_distance(color1, opaque_color_on_white(color2, 0.5)) < 5):
                if (freq2 > freq1):
                    replace_color(image_copy, color1, color2)

    high_freq_colors = count_high_freq_colors(image_copy)
    print(high_freq_colors)
    return [high_freq_colors, image_copy]


def color_quantization(image, color_frequency_list):
    # Convert the color frequency list to a set of unique colors
    unique_colors = set([color for _, color in color_frequency_list])

    # Create a mask for the image with True where the color is in the unique colors set
    mask = np.any(np.all(image.reshape(-1, 1, 3) == np.array(list(unique_colors)), axis=2), axis=1).reshape(
        image.shape[:2])

    # Create a new image with all pixels set to white
    new_image = np.full_like(image, 255)

    # Copy the pixels from the original image that have a color in the color frequency list
    new_image[mask] = image[mask]
    return new_image


def create_binary_matrix(img_arr, target_color):
    # Create mask of pixels with target color
    mask = np.all(img_arr == target_color, axis=-1)

    # Convert mask to binary matrix
    binary_matrix = mask.astype(int)
    from datetime import datetime
    binary_file_name = f'mask-{datetime.now().timestamp()}.png'
    cv2.imwrite(binary_file_name, binary_matrix * 255)

    # binary_matrix = torch.from_numpy(binary_matrix).unsqueeze(0).unsqueeze(0)
    return binary_file_name


def color_distance(color1, color2):
    return sum((a - b) ** 2 for a, b in zip(color1, color2)) ** 0.5


def replace_color(image, old_color, new_color):
    pixels = image.load()
    width, height = image.size
    for x in range(width):
        for y in range(height):
            if pixels[x, y] == old_color:
                pixels[x, y] = new_color


def opaque_color_on_white(color, a):
    r, g, b = color
    opaque_red = int((1 - a) * 255 + a * r)
    opaque_green = int((1 - a) * 255 + a * g)
    opaque_blue = int((1 - a) * 255 + a * b)
    return (opaque_red, opaque_green, opaque_blue)
