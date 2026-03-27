import torch
from torch.utils.data import DataLoader

def get_stats(dataset, batch_size=64):
    """
    Calculates the per-channel mean and std for a PyTorch dataset.
    Note: Dataset must return (image_tensor, label).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in loader:

        b, c, h, w = images.shape
        nb_pixels = b * h * w


        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])


        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    mean = fst_moment

    std = torch.sqrt(snd_moment - fst_moment ** 2)

    return mean.tolist(), std.tolist()
