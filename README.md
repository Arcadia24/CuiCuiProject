# CuiCuiProject

The project aims to create a bird sound classifier to pursue the birdCLEF 2023 competitions.

### Preprocessing data

It uses the birdCLEF 2021 dataset with more than 300 hundred species. The ``` main.py ``` file transforms audio files in Mel-spectrogram using librosa features and saves them in images of shape (128,313). It can apply or not augmentations which are GaussianNoise, LowPassFilter, or tanhdistorsion. 

### utils tool

The project implements a dataset to use the preprocess data. Different models are implemented for research purposes like a Vision Transformer and two architectures that use Efficient Net for extraction features and LSTM or attention Block for classification.

### Mixup
To train the model, it implements mixup method to get better regularization of the predictions.
If you want to use mixup, change the variable ``` mixup_bool``` to use it or not.

### report weight and biases to explore the project data and result
https://wandb.ai/nathan-vidal/cuitcuit/reports/Cui-Cui-Project--VmlldzozMjk1MzUy
