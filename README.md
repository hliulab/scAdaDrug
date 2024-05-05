# Predicting single-cell drug sensitivity by adaptive weighted feature for adversarial multi-source domain aaptation

## Introduction

**scAdaDrug** is a multi-source domain adaptation with adaptively generated feature weights to predict single-cell drug sensitivity.

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

To train the scAdaDrug model and obtain predicted results in target domain, you need to download the datasets (Example: GDSC), place it in the datasets folder, and then run:

```bash
python train_2.py
```
if you want to train baseline model, please run
```bash
python train_baseline.py
```

if you want to train the model using other datasets, you need to modify the ```data_loader.py``` file. 

## Directory structure

+ `baseline/scAdaDrug_2/scAdaDrug_3`: contains the code for the model, the dataset, the evaluation, and the training loop.

+ `train_baseline/train_2/train_3`: Contains the training loop, the hyperparameters, and the evaluation.

+ `data_loader`: Scripts for processing the data.

+ `loss_and_metrics`: Define loss.

+ `datasets`: Directory where data is stored.

    + `plot`: Training loss and evaluation curves.

