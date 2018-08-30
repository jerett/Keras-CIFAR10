# Keras for CIFAR10
This project demonstrates some personal examples with Keras on CIFAR10.

 ---
 
 
## Introduction

---
The CIFAR10 dataset is 32x32 size, 50000 train images and 10000 test images.
The dataset is divided into 40000 train images, 10000 validation images, and 10000 images.

 
 
## Result
All result is tested on 10000 test images.You can view and run in the jupyter
environment.

 Model | Notebook | Accuracy
 :---: | :---: | :---: 
 SVM | [svm](svm.ipynb) | 33.36% | 
 Softmax | [softmax](svm.ipynb)  | 33.11% |
 simple_cnn | [simple_cnn](simple_cnn.ipynb) | 66.75%
 vgg | [vgg](vgg.ipynb)  | 92.32% 
 inceptionV1 | [GoogLeNet](GoogLeNet.ipynb)  | 93.08% 
 ResNet18 | [resnet18](resnet18.ipynb)  | 93.47%
 small-ResNet20 | [small_resnet20](small_resnet20.ipynb) | 91.25%
 small-ResNet32 | [small_resnet32](small_resnet32.ipynb) | 92.34%
 small-ResNet56 | [small_resnet56](small_resnet56.ipynb) | 92.37%