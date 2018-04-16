#!/bin/sh


pip uninstall -y protobuf
#pip uninstall -y tensorflow
pip uninstall -y tensorflow-gpu

pip install -r requirements.txt

#python create_db.py -o ../../dataset/wiki.mat

python train.py --input ../../dataset/wiki64.mat

mv models ../../dataset/models
