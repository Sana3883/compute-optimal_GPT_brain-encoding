import numpy as np
from PIL import Image
from torch import Tensor


def images_from_array(array):
    if isinstance(array, Tensor):
        array = array.numpy()
    mode = "P" if (array.shape[1] == 1 or len(array.shape) == 3) else "RGB"
    if array.shape[1] == 1:
        array = np.squeeze(array, axis=1)
    if mode == "RGB":
        array = np.moveaxis(array, 1, 3)
    if array.min() < 0 or array.max() < 1: # if pixel values in [-0.5, 0.5]
        array = 255 * (array + 0.5)

    images = [Image.fromarray(np.uint8(arr), mode) for arr in array]
    return images


def save_GIF(array, path, duration=200, optimize=False):
    """Save a GIF from an array of shape (n_frames, channels, width, height), also accepts
    (n_frames, width, height) for grey levels.
    """
    assert path[-4:] == ".gif"
    images = images_from_array(array)
    images[0].save(path, save_all=True, append_images=images[1:], optimize=optimize,
                   loop=0, duration=duration)
