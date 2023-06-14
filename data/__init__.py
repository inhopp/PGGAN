from .dataset import Dataset
from math import log2
import torch
import torchvision.transforms as transforms


def generate_loader(opt, image_size):
    dataset = Dataset

    batch_size = opt.batch_size[int(log2(image_size / 4))]

    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(mean=mean, std=std),
    ])

    dataset = dataset(opt, transform=transform)

    kwargs = {
        "batch_size": batch_size,
        "num_workers": opt.num_workers,
        "shuffle": True,
        "drop_last": True,
    }

    return torch.utils.data.DataLoader(dataset, **kwargs), dataset
