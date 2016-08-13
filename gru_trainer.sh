#!/bin/bash

# specify train and validation files here
traindata="./data/train_sf.txt"
#valdata="./data/validate_sf.txt"
valdata="./data/validate_sf.txt"

# specify model name here
exp="sf"

# model save path
modelpath="model/$exp/"
mkdir -p $modelpath

# train
echo "Training..."
python src/char.py $traindata $valdata $modelpath

