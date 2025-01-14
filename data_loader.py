from glob import glob

import cv2
from torch.utils.data import Dataset
from torchvision import transforms


class Data(Dataset):
    def __init__(self, path="facades/train/"):
        self.filenames = glob(path + "*.jpg")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        image = cv2.imread(filename)
        image_width = image.shape[1]
        image_width = image_width // 2
        real = image[:, :image_width, :]
        condition = image[:, image_width:, :]

        real = transforms.functional.to_tensor(real)
        condition = transforms.functional.to_tensor(condition)

        return real, condition
