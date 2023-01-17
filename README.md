# CT Image Segmentation by 3D UNET

This repository is inspired by https://github.com/lee-zq/3DUNet-Pytorch.

## Check dependencies:
- pytorch=1.11.0=py3.9_cuda11.3_cudnn8.2.0_0
- pytorch-metric-learning=1.2.1
- torchvision=0.12.0
- tqdm=4.63.0
- h5py=3.6.0
- matplotlib=3.5
- medpy=0.4.0
- numpy=1.21.2
- pandas=1.4.1

## Model
3D UNet is implemented as the training model. Poor network performance is observed with 4 layers UNet model, with depth being 16, 32, 64, 128 respectively. Therefore, residual connection is added to avoid degradation and results a significant improvement in performance. <br />
![image](https://github.com/ypcmadeline/Image-segmentation-by-UNET/blob/master/models/unet.png)

## Data
Since the train data size is small, data augmentation is implemented to increase variation to the dataset, including random cropping of size (112,112,80), random flipping horizontally and vertically.

## Training
To accurately reflect the similarity between the ground-truth label and the predictions, dice loss is implemented. A final weighted loss (0.5*dice loss + 0.5*cross entropy loss) is backpropagated to update the weight. SGD optimizer is used with momentum to avoid being trapped at local minima. Since the training loss converges at an early stage, learning rate decay is applied to the optimizer, smaller update helps it to get closer to the global minima. <br />
Run `python train.py` to launch the training.<br />
The trained model will be saved as `model_1.pth`.<br />
Training loss curve will be saved as `loss.jpg`.

## Testing
Run `python test.py` to launch the testing.<br />
The testing module will load the trained model.<br />
Test result images will be saved to ``data/output``

## Result
The loss plot during training.<br />
![image](https://github.com/ypcmadeline/Image-segmentation-by-UNET/blob/master/models/loss.jpg)<br />

Segmentation result on test data (2D slices are listed in sagittal, coronal and axial
order):<br />
label                      |  Prediction
:-------------------------:|:-------------------------:
![image](https://github.com/ypcmadeline/Image-segmentation-by-UNET/blob/master/data/label/0.jpg)  |  ![image](https://github.com/ypcmadeline/Image-segmentation-by-UNET/blob/master/models/outputs/0.jpg)

Evaluation Metrics:
Dice: 0.8856  Jaccard: 0.7947   Average Surface Distance: 2.4237  95% Hausdroff Distance: 40.4351
| Dice        | Jaccard     | Average Surface Distance | 95% Hausdroff Distance |
| ----------- | ----------- | ------------------------ | ---------------------- |
| 0.8856      | 0.7947      | 2.4237                   | 40.4351                |
