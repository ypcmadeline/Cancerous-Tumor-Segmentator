import torch
import numpy as np
import math
from glob import glob

from dataset import read_h5, plot_image
from model import UNet
from medpy import metric


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = torch.load("model_1.pth")
    dice = []
    jac = []
    asd = []
    hd = []
    model = model.to(device)
    patch_size = (112, 112, 80)
    stride_xy = 18
    stride_z = 4

    counter = 0

    path_list = glob('./datas/test/*.h5')
    model.eval()
    for path in path_list:
        image, label = read_h5(path)

        w, h, d = image.shape
        sx = math.ceil((w - patch_size[0]) / stride_xy) + 1
        sy = math.ceil((h - patch_size[1]) / stride_xy) + 1
        sz = math.ceil((d - patch_size[2]) / stride_z) + 1

        scores = np.zeros((2, ) + image.shape).astype(np.float32)
        counts = np.zeros(image.shape).astype(np.float32)
        
        # inference all windows (patches)
        for x in range(0, sx):
            xs = min(stride_xy * x, w - patch_size[0])
            for y in range(0, sy):
                ys = min(stride_xy * y, h - patch_size[1])
                for z in range(0, sz):
                    zs = min(stride_z * z, d - patch_size[2])

                    # extract one patch for model inference
                    test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                    with torch.no_grad():
                        test_patch = torch.from_numpy(test_patch).cuda() # if use cuda
                        test_patch = test_patch.unsqueeze(0).unsqueeze(0) # [1, 1, w, h, d]
                        out = model(test_patch)
                        out = torch.softmax(out, dim=1)
                        out = out.cpu().data.numpy() # [1, 2, w, h, d]
                    
                    # record the predicted scores
                    scores[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += out[0, ...]
                    counts[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += 1

        scores = scores / np.expand_dims(counts, axis=0)
        predictions = np.argmax(scores, axis = 0) # final prediction: [w, h, d]
        plot_image(image,f"data/image/{counter}.jpg")
        plot_image(label,f"data/label/{counter}.jpg")
        plot_image(predictions,f"data/output/{counter}.jpg")
        counter += 1
        metrics_dc = metric.binary.dc(predictions, label)
        metrics_jac = metric.binary.jc(predictions, label)
        metrics_asd = metric.binary.asd(predictions, label)
        metrics_hd = metric.binary.hd(predictions, label)
        print(f"Dice: {metrics_dc}, JAC: {metrics_jac},ASD: {metrics_asd}, HD: {metrics_hd}")
        dice.append(metrics_dc)
        jac.append(metrics_jac)
        asd.append(metrics_asd)
        hd.append(metrics_hd)

    print(f"average dice: {sum(dice)/len(dice)}")
    print(f"average jac: {sum(jac)/len(jac)}")
    print(f"average asd: {sum(asd)/len(asd)}")
    print(f"average hd: {sum(hd)/len(hd)}")

