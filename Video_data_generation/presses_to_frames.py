import retro
import numpy as np
from PIL import Image


def presses_to_frames(path, emulator, size=None):
    """Replay a bk2 file and return the images as a numpy array
    of shape (n_frames, channels=3, width, height)
    """
    movie = retro.Movie(path)
    emulator.initial_state = movie.get_state()
    emulator.reset()
    images = []

    while movie.step():
        keys = []
        for p in range(movie.players):
            for i in range(emulator.num_buttons):
                keys.append(movie.get_key(i, p))
        obs, _rew, _done, _info = emulator.step(keys)
        if size is not None:
            obs = resample(obs, size)
        images.append(obs)
    return np.moveaxis(np.array(images),-1, 1)


def resample(image, size=(64,64)):
    pilimage = Image.fromarray(image, mode="RGB").resize(size)
    return np.array(pilimage)

