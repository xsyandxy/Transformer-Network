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
**`2022-03`**:**Major update, supports step and cosine learning rate decay, supports choice of Adam and SGD optimizers, supports adaptive learning rate adjustment based on batch_size.**  
The original repository link in the BiliBili video is:https://github.com/bubbliiiing/classification-pytorch/tree/bilibili

**`2021-01`**:**Repository created, supports model training, extensive comments, and multiple adjustable parameters. Supports top-1 to top-5 accuracy evaluation.**   

## Environment
pytorch == 1.2.0

## File Download
Pretrained weights required for training can be downloaded from Baidu Cloud.     
Link: https://pan.baidu.com/s/18Ze7YMvM5GpbTlekYO8bcA     
Extraction code: 5wym   

The example cat-dog dataset used for training can also be downloaded from Baidu Cloud.   
Link: https://pan.baidu.com/s/1hYBNG0TnGIeWw1-SwkzqpA     
Extraction code: ass8    

## Training Steps
1. The images stored in the datasets folder are divided into two parts: the train folder contains training images, and the test folder contains testing images.  
2. Before training, the dataset must be prepared. Create different folders inside the train or test folders, and name each folder after the corresponding class name. The images inside the folder will belong to that class. The folder structure should follow this format:
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
3. After preparing the dataset, run txt_annotation.py in the root directory to generate the cls_train.txt needed for training. Before running, modify the classes section to match the classes you want to classify.   
4. Then modify cls_classes.txt under the model_data folder to correspond to the classes you want to classify.  
5. After adjusting the network and weights you want to use in train.py, you can begin training!  

## Prediction Steps
### a、Using Pretrained Weights
1. After downloading and extracting the repository, model_data already contains a pretrained cat-dog model, mobilenet025_catvsdog.h5. Run predict.py and input:  
```python
img/cat.jpg
```
### b、Using Your Own Trained Weights
1. Follow the training steps to train the model.  
2. In the classification.py file, modify the model_path, classes_path, backbone, and alpha to match your trained files; model_path corresponds to the weight file in the logs folder, classes_path corresponds to the classes for the model, backbone is the feature extraction network used, and alpha is the alpha value when using mobilenet.  
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   Make sure to modify model_path and classes_path when using your own trained model for prediction!
    #   model_path points to the weight file in the logs folder, and classes_path points to the corresponding .txt file in the model_data folder.
    #   If shape mismatch occurs, also make sure to adjust the model_path and classes_path parameters in the training process.
    #--------------------------------------------------------------------------#
    "model_path"    : 'model_data/mobilenet_catvsdog.pth',
    "classes_path"  : 'model_data/cls_classes.txt',
    #--------------------------------------------------------------------#
    #    Input image size
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
3. Run predict.py and input the image path:  
```python
img/cat.jpg
```  

## Evaluation Steps
1. The images stored in the datasets folder are divided into two parts: train folder for training images and test folder for testing images. For evaluation, we use images from the test folder.  
2. Before evaluating, the dataset must be prepared. Create different folders inside the train or test directories. Name each folder according to the corresponding class name, and place the class images inside that folder. The folder structure should be as follows:
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
3. After preparing the dataset, run txt_annotation.py in the root directory to generate the cls_test.txt needed for evaluation. Before running, modify the classes section to match the classes you want to classify.   
4. Then, modify the model_path, classes_path, backbone, and alpha sections in classification.py to match your trained files. model_path corresponds to the weight file in the logs folder, classes_path corresponds to the classes for the model, backbone corresponds to the feature extraction network used, and alpha is the alpha value for mobilenet.**  
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   Make sure to modify `model_path` and `classes_path` when using your own trained model for prediction!
    #   `model_path` points to the weight file in the `logs` folder, `classes_path` points to the `.txt` file in `model_data`.
    #   If shape mismatch occurs, adjust the `model_path` and `classes_path` parameters accordingly.
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
5. Run eval_top1.py and eval_top5.py to evaluate the model accuracy.

## Reference
https://github.com/keras-team/keras-applications   

