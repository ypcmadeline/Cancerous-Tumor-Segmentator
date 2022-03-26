from termios import XCASE
import torch
import h5py
import os
import matplotlib.pyplot as plt
import random


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for trans in self.transforms:
            image, label = trans(image, label)
        return image, label

class RandomFlip_LeftRight:
    def __init__(self):
        self.p = 0.5

    def _flip(self, image, p):
        if p <= self.p:
            image = image.flip(2)
        return image

    def __call__(self, image, label):
        p = random.uniform(0, 1)
        flip_img = self._flip(image, p)
        flip_mask = self._flip(label, p)
        return flip_img, flip_mask

class RandomFlip_UpDown:
    def __init__(self):
        self.p = 0.5

    def _flip(self, image, p):
        if p <= self.p:
            image = image.flip(3)
        return image

    def __call__(self, image, label):
        p = random.uniform(0, 1)
        flip_img = self._flip(image, p)
        flip_mask = self._flip(label, p)
        return flip_img, flip_mask


def read_h5(path):
    data = h5py.File(path, 'r')
    image = data['image'][:]
    label = data['label'][:]
    return image, label

def plot_image(image, path):
    n_i, n_j, n_k = image.shape
    slices = [image[int((n_i - 1) / 2), :, :], image[:, int((n_j - 1) / 2), :], image[:, :, int((n_k - 1) / 2)]]
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
        plt.savefig(path)



class LAHeart(torch.utils.data.Dataset):
    def __init__(self, split='train', transform=None):
        self.transforms = Compose([RandomFlip_LeftRight(),RandomFlip_UpDown()])
        self.image = []
        self.label = []
        self.split = split
        for idx, i in enumerate(os.listdir(f"datas/{split}")):
            image, label = read_h5(f"datas/{split}/{i}")
            # crop (112,112,80)
            n_i, n_j, n_k = image.shape
            i = random.randint(0, n_i-112)
            j = random.randint(0, n_j-112)
            k = random.randint(0, n_k-80)
            image = image[i:i+112, j:j+112, k:k+80] 
            label = label[i:i+112, j:j+112, k:k+80]

            self.image.append(image)
            self.label.append(label)
        

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        X = self.image[index]
        y = self.label[index]
        X = torch.FloatTensor(X).unsqueeze(0)
        y = torch.FloatTensor(y).unsqueeze(0)
        if self.transforms and self.split == "train":
            X,y = self.transforms(X, y)  
        return X, y.squeeze(0)


if __name__ == '__main__':
    train_dst = LAHeart(split='train', transform=None)
