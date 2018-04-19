#!/bin/sh


#pip uninstall -y protobuf
#pip uninstall -y tensorflow
#pip uninstall -y tensorflow-gpu

#pip install -r requirements.txt

#python create_db.py --img_size 48 -o ../../dataset/wiki48.mat --db wiki

python train.py --input ../../dataset/wiki48.mat

mv models ../../dataset/models
