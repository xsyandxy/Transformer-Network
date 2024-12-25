## Classification：Implementation of Classification Model in Pytorch
---

## Contents
1. [Top News](#Repository Updates Top News)
2. [Environment](#Required Environment)
3. [Download](#File Download)
4. [How2train](#Training Steps)
5. [How2predict](#Prediction Steps)
6. [How2eval](#Evaluation Steps)
7. [Reference](#Reference)

## Top News
**`2022-03`**:**Major updates, including support for step and cosine learning rate decay, optimizer selection (Adam, SGD), and adaptive learning rate adjustment based on batch size.Major updates, including support for step and cosine learning rate decay, optimizer selection (Adam, SGD), and adaptive learning rate adjustment based on batch size.**  
The original repository in the BiliBili video:https://github.com/bubbliiiing/classification-pytorch/tree/bilibili

**`2021-01`**:**Repository created, supports model training, includes extensive comments, and many adjustable parameters. Supports top-1 to top-5 accuracy evaluation.**   

## Environment
pytorch == 1.2.0

## Download
The pretrained weights required for training can be downloaded from Baidu Cloud.
Link: https://pan.baidu.com/s/18Ze7YMvM5GpbTlekYO8bcA
Extraction code: 5wym  

The example cat-dog dataset used for training can also be downloaded from Baidu Cloud.
Link: https://pan.baidu.com/s/1hYBNG0TnGIeWw1-SwkzqpA
Extraction code: ass8   

## How2train
1. The images in the datasets folder are divided into two parts: training images in the train folder and testing images in the test folder.  
2. Before training, you need to prepare the dataset. Create different folders in the train or test folder. The name of each folder should correspond to the class name, and the images inside each folder should belong to that class. The folder structure should look like this:
```
|-datasets
    |-train
        |-cat
            |-123.jpg
            |-234.jpg
        |-dog
            |-345.jpg
            |-456.jpg
        |-...
    |-test
        |-cat
            |-567.jpg
            |-678.jpg
        |-dog
            |-789.jpg
            |-890.jpg
        |-...
```
3. After preparing the dataset, run txt_annotation.py in the root directory to generate cls_train.txt for training. Before running, modify the classes section to reflect the classes you want to categorize.   
4. Then, modify cls_classes.txt in the model_data folder to match the classes you want to categorize.  
5. After adjusting the network and weights in train.py, you can start training!  

## How2predict
### a、Using Pretrained Weights
1. After downloading and extracting the repository, a pretrained cat-dog model mobilenet025_catvsdog.h5 is already available in model_data. Run predict.py and input:  
```python
img/cat.jpg
```
### b、Using Your Own Trained Weights
1. Follow the training steps to train the model.  
2. In classification.py, modify model_path, classes_path, backbone, and alpha to point to your trained files. model_path corresponds to the weight file in the logs folder, classes_path corresponds to the class list, backbone refers to the backbone feature extraction network, and alpha is the alpha value when using MobileNet**.  
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   Make sure to modify `model_path` and `classes_path` when using your own trained model for prediction!
    #   `model_path` points to the weight file in the `logs` folder, `classes_path` points to the `.txt` file in `model_data`.
    #   If there is a shape mismatch, adjust `model_path` and `classes_path` accordingly during training.
    #--------------------------------------------------------------------------#
    "model_path"    : 'model_data/mobilenet_catvsdog.pth',
    "classes_path"  : 'model_data/cls_classes.txt',
    #--------------------------------------------------------------------#
    #   Input image size
    #--------------------------------------------------------------------#
    "input_shape"   : [224, 224],
    #--------------------------------------------------------------------#
    #   所用模型种类：
    #   Model types:
    #   mobilenet, resnet50, and vgg16 are commonly used classification networks.
    #   cspdarknet53 is used to demonstrate how to train your own pretrained weights using mini_imagenet.
    #--------------------------------------------------------------------#
    "backbone"      : 'mobilenet',
    #-------------------------------#
    #   Whether to use Cuda
    #   Set to False if there is no GPU
    #-------------------------------#
    "cuda"          : True
}
```
3. Run predict.py and input:
```python
img/cat.jpg
```  

## How2eval
1. The images in the datasets folder are divided into two parts: train (training images) and test (testing images). During evaluation, we use the images in the test folder.
2. Before evaluating, prepare the dataset. Create different folders in the train or test directories, each named according to the class. The images in each folder should belong to that class. The folder structure should look like this:
```
|-datasets
    |-train
        |-cat
            |-123.jpg
            |-234.jpg
        |-dog
            |-345.jpg
            |-456.jpg
        |-...
    |-test
        |-cat
            |-567.jpg
            |-678.jpg
        |-dog
            |-789.jpg
            |-890.jpg
        |-...
```
3. After preparing the dataset, run txt_annotation.py in the root directory to generate cls_test.txt for evaluation. Before running, modify the classes section to reflect the classes you want to categorize.
4. Then, modify model_path, classes_path, backbone, and alpha in classification.py to point to your trained files. model_path corresponds to the weight file in the logs folder, classes_path corresponds to the class list, backbone refers to the backbone feature extraction network, and alpha is the alpha value when using MobileNet**.  
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   Make sure to modify `model_path` and `classes_path` when using your own trained model for prediction!
    #   `model_path` points to the weight file in the `logs` folder, `classes_path` points to the `.txt` file in `model_data`.
    #   If there is a shape mismatch, adjust `model_path` and `classes_path` accordingly during training.
    #--------------------------------------------------------------------------#
    "model_path"    : 'model_data/mobilenet_catvsdog.pth',
    "classes_path"  : 'model_data/cls_classes.txt',
    #--------------------------------------------------------------------#
    #   Input image size
    #--------------------------------------------------------------------#
    "input_shape"   : [224, 224],
    #--------------------------------------------------------------------#
    #   Model types:
    #   mobilenet, resnet50, and vgg16 are commonly used classification networks.
    #   cspdarknet53 is used to demonstrate how to train your own pretrained weights using mini_imagenet.
    #--------------------------------------------------------------------#
    "backbone"      : 'mobilenet',
    #-------------------------------#
    #   Whether to use Cuda
    #   Set to False if there is no GPU
    #-------------------------------#
    "cuda"          : True
}
```
5. Run eval_top1.py and eval_top5.py to evaluate model accuracy.

## Reference
https://github.com/keras-team/keras-applications   

