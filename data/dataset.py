import os
from PIL import Image
import torch.utils.data as data


class Dataset(data.Dataset):
    def __init__(self, opt, transform):
        self.data_dir = opt.data_dir
        self.data_name = opt.data_name
        self.transform = transform

        self.list_files = os.listdir(self.data_dir + self.data_name)

    def __getitem__(self, index):
        img_file = self.list_files[index]

        image = Image.open(os.path.join(
            self.data_dir, self.data_name, img_file))
        image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.list_files)
