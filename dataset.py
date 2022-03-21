import torch
import h5py
import os
import matplotlib.pyplot as plt
import random


def read_h5(path):
    data = h5py.File(path, 'r')
    image = data['image'][:]
    label = data['label'][:]
    return image, label

def plot_image(image):
    n_i, n_j, n_k = image.shape
    # sagittal (left image)
    center_i1 = int((n_i - 1) / 2)
    # coronal (center image)
    center_j1 = int((n_j - 1) / 2)
    # axial slice (right image)
    center_k1 = int((n_k - 1) / 2)
    slices = [image[center_i1, :, :], image[:, center_j1, :], image[:, :, center_k1]]
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
        plt.savefig('foo.jpg')



class LAHeart(torch.utils.data.Dataset):
    def __init__(self, split='train', transform=None):
        self.image = []
        self.label = []
        for i in os.listdir(f"datas/{split}"):
            image, label = read_h5(f"datas/{split}/{i}")
            plot_image(label)
            n_i, n_j, n_k = image.shape
            i = random.randint(0, n_i-112)
            j = random.randint(0, n_j-112)
            k = random.randint(0, n_k-80)
            image = image[None, i:i+112, j:j+112, k:k+80] 
            label = label[None, i:i+112, j:j+112, k:k+80] 
            self.image.append(image)
            # print(image.shape)
            self.label.append(label)
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        X = self.image[index]
        y = self.label[index]
        return X, y
