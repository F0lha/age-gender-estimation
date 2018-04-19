import pandas as pd
import logging
import argparse
import os
import numpy as np
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from wide_resnet import WideResNet
from utils import mk_dir, load_data
from keras.preprocessing.image import ImageDataGenerator
from mixup_generator import MixupGenerator
from random_eraser import get_random_eraser
from MyModel import MyModel
from tensorflow.python import debug as tf_debug
import keras.backend as K
import Augmentor

logging.basicConfig(level=logging.DEBUG)

class Schedule:
    def __init__(self, nb_epochs):
        self.epochs = nb_epochs

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return 0.1
        elif epoch_idx < self.epochs * 0.5:
            return 0.02
        elif epoch_idx < self.epochs * 0.75:
            return 0.004
        return 0.0008


def get_args():
    parser = argparse.ArgumentParser(description="This script trains the CNN model for age and gender estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="path to input database mat file")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=100,
                        help="number of epochs")
    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network (should be 10, 16, 22, 28, ...)")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    parser.add_argument("--validation_split", type=float, default=0.1,
                        help="validation split ratio")
    parser.add_argument("--aug", action="store_true",
                        help="use data augmentation if set true")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    input_path = args.input
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    depth = args.depth
    k = args.width
    validation_split = args.validation_split
    use_augmentation = args.aug
    
    image_size = 224
    
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

    logging.debug("Loading data and Augmentor...")
    

    
    #sess = K.get_session()
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #K.set_session(sess)
    
    #model = WideResNet(image_size, depth=22, k=k)()
    model = MyModel(image_size)()
    adam = Adam(lr=0.1)
    sgd = SGD(lr=0.0001, momentum=0.9, nesterov=True, decay=0.00001)
    model.compile(optimizer=adam, loss="categorical_crossentropy",
                  metrics=['accuracy'])

    logging.debug("Model summary...")
    model.count_params()
    model.summary()
    
    #model.load_weights(os.path.join("checkpoints", "weights.03-4.78.hdf5"))

    logging.debug("Saving model...")
    mk_dir("models")
    with open(os.path.join("models", "WRN_{}_{}.json".format(depth, k)), "w") as f:
        f.write(model.to_json())

    mk_dir("checkpoints")
    callbacks = [LearningRateScheduler(schedule=Schedule(38138  // batch_size)),
                 ModelCheckpoint("checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                 monitor="val_loss",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="auto")
                 ]

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
        shuffle = True)
    
    train_generator = train_datagen.flow_from_directory(
        '../../dataset/wiki_crop/new_database/',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')
    
    val_generator = train_datagen.flow_from_directory(
        '../../dataset/wiki_crop/new_database/',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation')
    
    h = model.fit_generator(
        train_generator,
        epochs=20,
        validation_data=val_generator,
        workers=12)

    logging.debug("Saving weights...")
    model.save_weights(os.path.join("models", "WRN_{}_{}.h5".format(depth, k)), overwrite=True)
    #pd.DataFrame(hist.history).to_hdf(os.path.join("models", "history_{}_{}.h5".format(depth, k)), "history")


if __name__ == '__main__':
    main()