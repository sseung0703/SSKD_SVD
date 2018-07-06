# Self-supervised Knowledge Distillation using Singular Value Decomposition
![Alt text](dist.png)
## Feature
- Define knowledge by Singular value decomposition
- Fast and efficient learning by multi-task learning
 
## Requirments
- Tensorflow
- Scipy

Unfortunatly SVD is very slow on GPU. so if i recommend below installation method.
- Install Tensorflow from source which is removed SVD GPU op.(recommended)
- Install ordinary Tensorflow and make SVD using CPU.
- Install Tensorflow version former than 1.2.

## How to Use
The code is based on Tensorflow-slim example codes. so if you used that it is easy to understand. 
1. Recording Cifar100 dataset to tfrecording file 
2. Train teacher network
3. Train student network using teacher knowledge

## Results
![Alt text](results.PNG)

## Paper
This research accepted to ECCV2018 poster session, and i will link my papaer soon





