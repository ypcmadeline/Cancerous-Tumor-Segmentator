# [ELEC4010N] Assignment-02, Problem 1

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

## Set up training
    Run ``python train.py`` to launch the training
    The trained model will be saved as ``model_1.pth``

## Set up testing
    Run ``python test.py`` to launch the testing
    The testing module will load the trained model and save the outputs to ``data/output``
