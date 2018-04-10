#!/bin/sh


pip uninstall -y protobuf
pip uninstall -y tensorflow
pip uninstall -y tensorflow-gpu

pip install -r requirements.txt

python train.py -i ../../dataset/wiki.mat