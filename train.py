from dataset import LAHeart
from torch.utils.data import DataLoader
from model import UNet
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
import torch

import glob
from dataset import read_h5, plot_image
import math
from medpy import metric

import numpy as np

use_cuda = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def plot_graph(loss):
    print(loss)
    plt.plot(loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig('loss.jpg')
    plt.show()


if __name__ == '__main__':
    max_epoch = 1000
    batch_size = 1

    model = UNet()
    model = model.to(device)

    train_dst = LAHeart(split='train', transform=None)
    # test_dst = LAHeart(split='test', transform=None)

    train_loader = DataLoader(
        train_dst, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4, 
        pin_memory=True
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.CrossEntropyLoss()
    loss_list = []
    
    for epoch in range(max_epoch):
        running_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            images, labels = batch
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()

            outputs = model(images)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        loss = running_loss / len(train_loader)
        print(f"Epoch: {epoch}, Loss: {loss}")
        loss_list.append(loss)


    plot_graph(loss_list)
    torch.save(model.state_dict(), "model.pt")

