# Adapted from https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/utils.py

# reshapes tensor start from dim i (inclusive)
# to dim j (exclusive) to the desired shape
# e.g. if x.shape = (b, thw, c) then
# view_range(x, 1, 2, (t, h, w)) returns
# x of shape (b, t, h, w, c)
def view_range(x, i, j, shape):
    shape = tuple(shape)

    n_dims = len(x.shape)
    if i < 0:
        i = n_dims + i

    if j is None:
        j = n_dims
    elif j < 0:
        j = n_dims + j

    assert 0 <= i < j <= n_dims

    x_shape = x.shape
    target_shape = x_shape[:i] + shape + x_shape[j:]
    return x.view(target_shape)


def tensor_slice(x, begin, size):
    assert all([b >= 0 for b in begin])
    size = [l - b if s == -1 else s
            for s, b, l in zip(size, begin, x.shape)]
    assert all([s >= 0 for s in size])

    slices = [slice(b, b + s) for b, s in zip(begin, size)]
    return x[slices]


# import math
# import numpy as np
# import skvideo.io
# def save_video_grid(video, fname, nrow=None):
#     b, c, t, h, w = video.shape
#     video = video.permute(0, 2, 3, 4, 1)
#     video = (video.cpu().numpy() * 255).astype('uint8')
#
#     if nrow is None:
#         nrow = math.ceil(math.sqrt(b))
#     ncol = math.ceil(b / nrow)
#     padding = 1
#     video_grid = np.zeros((t, (padding + h) * nrow + padding,
#                            (padding + w) * ncol + padding, c), dtype='uint8')
#     for i in range(b):
#         r = i // ncol
#         c = i % ncol
#
#         start_r = (padding + h) * r
#         start_c = (padding + w) * c
#         video_grid[:, start_r:start_r + h, start_c:start_c + w] = video[i]
#
#     skvideo.io.vwrite(fname, video_grid, inputdict={'-r': '5'})
#     print('saved videos to', fname)
