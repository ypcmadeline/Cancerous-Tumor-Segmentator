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
from torchvision import transforms

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


class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, out, labels):
        num_label = out.size(1)
        dice = 0.
        for i in range(num_label):
            true_n_pred = (out[:,i] * labels[:,i]).sum(dim=1).sum(dim=1).sum(dim=1)
            out_sq = out[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1)
            labels_sq = labels[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1)
            dice += 2 * true_n_pred/(out_sq + labels_sq + self.smooth)
        dice = torch.clamp((1 - (dice / num_label)).mean(), 0, 1)
        return dice


if __name__ == '__main__':
    max_epoch = 1000
    batch_size = 1

    model = UNet()
    model = model.to(device)


    train_dst = LAHeart(split='train', transform=None)

    train_loader = DataLoader(
        train_dst, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4, 
        pin_memory=True
    )
    
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.00001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=600, gamma=0.1)
    criterion_bd = DiceLoss()
    criterion_ce = nn.CrossEntropyLoss()
    loss_list = []
    
    for epoch in range(max_epoch):
        running_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            images, labels = batch
            images = images.float()
            labels = labels.long()
            n, s, h, w = labels.size()
            labels = torch.zeros(n, 2, s, h, w).scatter_(1, labels.view(n, 1, s, h, w), 1)
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()
            outputs = model(images)

            loss_0 = criterion_bd(outputs[0], labels)
            loss_1 = criterion_bd(outputs[1], labels)
            loss_2 = criterion_bd(outputs[2], labels)
            loss_3 = criterion_bd(outputs[3], labels)
            loss_bd = loss_3 + 0.4 * (loss_0 + loss_1 + loss_2)

            loss_0 = criterion_ce(outputs[0], labels)
            loss_1 = criterion_ce(outputs[1], labels)
            loss_2 = criterion_ce(outputs[2], labels)
            loss_3 = criterion_ce(outputs[3], labels)
            loss_ce = loss_3 + 0.4 * (loss_0 + loss_1 + loss_2)

            loss = 0.5 * loss_bd + 0.5 * loss_ce
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        loss = running_loss / len(train_loader)
        print(f"Epoch: {epoch}, Loss: {loss}")
        loss_list.append(loss)
        scheduler.step()


    plot_graph(loss_list)
    torch.save(model.state_dict(), "model.pth")
    torch.save(model, "model_1.pth")

