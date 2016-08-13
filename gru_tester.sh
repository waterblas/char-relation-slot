#!/bin/bash

# specify test file here
fulltestdata="./data/test_sf.txt"

# specify model path here
modelpath="model/sf/"

# specify result path here
resultpath="result/"

mkdir -p $resultpath

# test
python src/test_char.py $fulltestdata $modelpath $resultpath
