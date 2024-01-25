# Predicting single-cell drug sensitivity by adaptive weighted feature for adversarial multi-source domain aaptation

## Introduction

**scAdaDrug** is a multi-source adaptive weighting model to predict single-cell drug sensitivity.

## Model architecture

![](framework.jpg)

## Requirements

The deep learning models were trained on 2*NVIDIA GeForce RTX 4090 on linux.

+ Python 3.11
+ PyTorch 2.0
+ Pandas 2.1.1
+ Numpy 1.25.2
+ Scikit-learn 1.3.1

## Usage

To setup the environment, install conda and run (Must run on servers with multiple GPUs):

```bash
conda create --name <your_env_name> --file requirements.txt
```


## Directory structure

+ `scAdaDrug_2sources/scAdaDrug_3sources`: contains the code for the model, the dataset, the evaluation, and the training loop.

+ `train_2sources/train_3sources`: Contains the training loop, the hyperparameters, and the evaluation.

+ `loader`: Scripts for processing the data.

+ `loss_define`: Define loss.

+ `datasets`: Directory where data is stored.

    + `plot`: Training loss and evaluation curves.

