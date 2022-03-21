from dataset import LAHeart
from torch.utils.data import DataLoader
from model import UNet
import torch.optim as optim
from torch import nn
from medpy import metric
from dataset import plot_image
import matplotlib.pyplot as plt
import torch

import numpy as np

def plot_graph(loss):
    plt.plot(loss)
    plt.show()
    plt.savefig('loss.jpg')


if __name__ == '__main__':
    max_epoch = 1000
    batch_size = 1

    model = UNet()
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
    loss_list = []
    
    for epoch in range(max_epoch):
        running_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            images, labels = batch
            outputs = model(images)

            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # running_dc += metric.binary.dc(outputs.detach().numpy(), labels.detach().numpy())
            # running_jac += metric.binary.jc(outputs.detach().numpy(), labels.detach().numpy())
            # running_asd += metric.binary.asd(outputs.detach().numpy(), labels.detach().numpy())
            # running_hd += metric.binary.hd(outputs.detach().numpy(), labels.detach().numpy())

        loss = running_loss / len(train_loader)
        # dc = running_dc / len(train_loader)
        # jac = running_jac / len(train_loader)
        # asd = running_asd / len(train_loader)
        # hd = running_hd / len(train_loader)
        print(f"Epoch: {epoch}, Loss: {loss}")
        loss_list.append(loss)
        # dice_list.append(dc)
        # jaccard_list.append(jac)
        # sad_list.append(asd)
        # hd_list.append(hd)

    plot_graph(loss_list)
    torch.save(model.state_dict(), "model.pt")
