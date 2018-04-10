#!/bin/bash


pip uninstall -y protobuf
pip uninstall -y tensorflow
pip uninstall -y tensorflow-gpu

pip install requirements.txt

python train.py -i ../../dataset/wiki.mat