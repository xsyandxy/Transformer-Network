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
**`2022-03`**:**Major update, supports step and cosine learning rate decay, supports adam and sgd optimizers, and supports adaptive learning rate adjustment based on batch_size.**  
The original repository link from the BiliBili video:https://github.com/bubbliiiing/classification-pytorch/tree/bilibili

**`2021-01`**:**Repository created, supports model training with extensive comments and adjustable parameters. Supports top-1 and top-5 accuracy evaluation.**   

## Environment
pytorch == 1.2.0

## Download
The pre-trained weights required for training can be downloaded from Baidu Cloud.
Link: https://pan.baidu.com/s/18Ze7YMvM5GpbTlekYO8bcA
Extraction code: 5wym

The example cat-dog dataset used for training can also be downloaded from Baidu Cloud.
Link: https://pan.baidu.com/s/1hYBNG0TnGIeWw1-SwkzqpA
Extraction code: ass8   

## 训练步骤
1. The images in the datasets folder are divided into two parts: the "train" folder contains training images, and the "test" folder contains testing images.
2. Before training, you need to prepare the dataset. In the "train" or "test" folder, create different subfolders, with the subfolder name being the class name, and the images inside corresponding to that class. The folder structure should look like this:
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
3. After preparing the dataset, run the txt_annotation.py script in the root directory to generate the cls_train.txt file needed for training. Modify the classes field in the script to reflect the classes you want to use.
4. Then, modify the cls_classes.txt file in the model_data folder to match the classes you want to use.
5. After adjusting the network and weights in train.py to your requirements, you can start training!！  

## How2train
### a、Using Pre-trained Weights
1. After downloading and extracting the library, the model data already includes a pre-trained cat-dog model, mobilenet025_catvsdog.h5. Run predict.py and input:  
```python
img/cat.jpg
```
### b、Using Your Own Trained Weights
1. Train the model following the training steps above.  
2. In the classification.py file, modify the following fields—model_path, classes_path, backbone, and alpha—to match your trained files. model_path corresponds to the weights file in the logs folder, classes_path corresponds to the classes file, backbone corresponds to the backbone network used for feature extraction, and alpha is used when employing the MobileNet alpha value**.  
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   Make sure to modify `model_path` and `classes_path` when using your own trained model for prediction!
    #   `model_path` points to the weights file in the `logs` folder, and `classes_path` points to the txt file in `model_data`
    #   If there is a shape mismatch, ensure that `model_path` and `classes_path` are consistent with the training setup.
    #--------------------------------------------------------------------------#
    "model_path"    : 'model_data/mobilenet_catvsdog.pth',
    "classes_path"  : 'model_data/cls_classes.txt',
    #--------------------------------------------------------------------#
    #   Input image size
    #--------------------------------------------------------------------#
    "input_shape"   : [224, 224],
    #--------------------------------------------------------------------#
    #   Types of models used:
    #   mobilenet, resnet50, vgg16 are commonly used classification networks
    #   cspdarknet53 is used for showing how to train custom pre-trained weights using mini_imagenet
    #--------------------------------------------------------------------#
    "backbone"      : 'mobilenet',
    #-------------------------------#
    #   Whether to use Cuda
    #   Set to False if you don't have a GPU
    #-------------------------------#
    "cuda"          : True
}
```
3. Run predict.py and input:  
```python
img/cat.jpg
```  

## How2eval
1. The images in the datasets folder are divided into two parts: the "train" folder contains training images, and the "test" folder contains testing images. During evaluation, we use the images in the "test" folder.
2. Before evaluation, ensure the dataset is prepared in the same structure as described in the training steps.
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
3. After preparing the dataset, run the txt_annotation.py script in the root directory to generate the cls_test.txt file needed for evaluation. Modify the classes field in the script to match the classes you want to use.
4. Then, modify the following fields in the classification.py file—model_path, classes_path, backbone, and alpha—to match your trained files. model_path corresponds to the weights file in the logs folder, classes_path corresponds to the classes file, backbone corresponds to the backbone network used, and alpha is used when employing MobileNet's alpha value**.  
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   Make sure to modify `model_path` and `classes_path` when using your own trained model for evaluation!
    #   `model_path` points to the weights file in the `logs` folder, and `classes_path` points to the txt file in `model_data`
    #   If there is a shape mismatch, ensure that `model_path` and `classes_path` are consistent with the training setup.
    #--------------------------------------------------------------------------#
    "model_path"    : 'model_data/mobilenet_catvsdog.pth',
    "classes_path"  : 'model_data/cls_classes.txt',
    #--------------------------------------------------------------------#
    #   Input image size
    #--------------------------------------------------------------------#
    "input_shape"   : [224, 224],
    #--------------------------------------------------------------------#
    #   Types of models used:
    #   mobilenet, resnet50, vgg16 are commonly used classification networks
    #   cspdarknet53 is used for showing how to train custom pre-trained weights using mini_imagenet
    #--------------------------------------------------------------------#
    "backbone"      : 'mobilenet',
    #-------------------------------#
    #   Whether to use Cuda
    #   Set to False if you don't have a GPU
    #-------------------------------#
    "cuda"          : True
}
```
5. Run eval_top1.py and eval_top5.py to evaluate the model's accuracy.

## Reference
https://github.com/keras-team/keras-applications   

