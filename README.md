# Canerous Tumour Segmentator

## Background

### CT Images
CT images are particularly useful for diagnosing a wide range of medical conditions, including injuries, tumors, infections, and internal bleeding. They provide detailed information about the internal structures of the body, helping doctors make accurate diagnoses and plan appropriate treatments. CT scans are commonly used in various medical specialties, including radiology, oncology, neurology, and emergency medicine.

CT imagees can be analyzed in three views—axial, sagittal, and coronal. They refer to different planes or orientations in which the cross-sectional images of the body are viewed. These orientations are crucial for interpreting and understanding the anatomical structures and abnormalities captured by CT scans.

This is an example of how a CT image looks like in axial, sagittal, and coronal views.

![image](/data/image/0.jpg)

### Cancerous tumor segmentation   
Cancerous tumor segmentation is incredibly important in medical imaging and diagnostics, particularly in fields like radiology and oncology. Tumor segmentation involves identifying and delineating the boundaries of cancerous lesions or tumors. 
1. Accurate Diagnosis: Tumor segmentation enables medical professionals to accurately identify and diagnose the presence of cancerous tumors. Precise delineation of tumor boundaries helps differentiate between healthy tissue and abnormal growths, aiding in early detection and appropriate treatment planning.
2. Treatment Planning: Once a tumor is identified and segmented, it becomes essential for treatment planning. The size, shape, and location of the tumor are crucial factors in determining the most effective treatment approach, whether it's surgery, radiation therapy, chemotherapy, targeted therapy, or a combination of these.

The image above is an CT image that shows a tumor. This is the segmented tumor image.
![image](/data/label/0.jpg)

In this project, deep learning techniques, such as convolutional neural networks (CNNs), is used to perform tumor segmentation by analyzing patterns and features within medical images to precisely outline and differentiate cancerous growths from healthy tissues.

## Model
3D UNet is implemented as the training model. The 3D U-Net is a deep learning architecture designed for semantic segmentation tasks, particularly in the context of medical image analysis. It builds upon the original U-Net architecture by extending it to handle three-dimensional data. The 3D U-Net architecture consists of an encoding path, a bottleneck, and a decoding path. The encoding path captures hierarchical features from the input volume, and the decoding path gradually recovers the spatial resolution while refining the segmentation output. 
During the training, poor network performance is observed with 4 layers UNet model, with depth being 16, 32, 64, 128 respectively. Therefore, residual connection is added to avoid degradation and results a significant improvement in performance. <br />
![image](https://github.com/ypcmadeline/Image-segmentation-by-UNET/blob/master/models/unet.png)

## Data
Since the train data size is small, data augmentation is implemented to increase variation to the dataset. 3D random cropping of size (112,112,80), random flipping horizontally and vertically are implemented.

## Training
To accurately reflect the similarity between the ground-truth label and the predictions, dice loss is implemented. Dice loss is a commonly used loss function in image segmentation tasks, especially in medical image analysis. It measures the similarity or overlap between the predicted segmentation mask and the ground truth mask of a target object or region. 
A final weighted loss (0.5*dice loss + 0.5*cross entropy loss) is backpropagated to update the weight. SGD optimizer is used with momentum to avoid being trapped at local minima. Since the training loss converges at an early stage, learning rate decay is applied to the optimizer, smaller update helps it to get closer to the global minima. <br />
Run `python train.py` to launch the training.<br />
The trained model will be saved as `model_1.pth`.<br />
Training loss curve will be saved as `loss.jpg`.

## Testing
Run `python test.py` to launch the testing.<br />
The testing module will load the trained model.<br />
Test result images will be saved to ``data/output``

## Result
Segmentation result on test data (2D slices are listed in sagittal, coronal and axial
order):<br />
label                      |  Prediction
:-------------------------:|:-------------------------:
![image](https://github.com/ypcmadeline/Image-segmentation-by-UNET/blob/master/data/label/0.jpg)  |  ![image](https://github.com/ypcmadeline/Image-segmentation-by-UNET/blob/master/models/outputs/0.jpg)

Evaluation Metrics:
| Dice        | Jaccard     | Average Surface Distance | 95% Hausdroff Distance |
| ----------- | ----------- | ------------------------ | ---------------------- |
| 0.8856      | 0.7947      | 2.4237                   | 40.4351                |

The loss plot during training.<br />
![image](https://github.com/ypcmadeline/Image-segmentation-by-UNET/blob/master/models/loss.jpg)<br />


