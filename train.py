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
from collections import Counter

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

def get_class_weights(y, smooth_factor=0):
    """
    Returns the weights for each class based on the frequencies of the samples
    :param smooth_factor: factor that smooths extremely uneven weights
    :param y: list of true labels (the labels must be hashable)
    :return: dictionary with the weight for each class
    """
    counter = Counter(y)

    if smooth_factor > 0:
        p = max(counter.values()) * smooth_factor
        for k in counter.keys():
            counter[k] += p

    majority = max(counter.values())

    return {cls: float(majority) / count for cls, count in counter.items()}

def generate_data_generator(generator):
    for x_batch,y_batch in generator:
            ages = np.arange(0, 101).reshape(101, 1)
            round_age = [int(pred.dot(ages).flatten()) for pred in y_batch]
            yield x_batch, [y_batch, np.array(round_age)]

def hiearcical_softmax_loss(y_true, y_pred):
    pos = K.sum(y_true * y_pred, axis=-1)
    neg = K.max((1.0 - y_true) * y_pred, axis=-1)
    return K.mean(K.maximum(0.0, neg - pos + 1), axis=-1)
            
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
    model = MyModel(image_size,trainable=False)()
    adam = Adam(lr=0.01, decay=0.001)
    sgd = SGD(lr=0.00001, momentum=0.9, nesterov=True, decay=0.0001)
    model.compile(optimizer=sgd, loss=["categorical_crossentropy","MSE"],loss_weights=[0.5,1.0],
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

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        horizontal_flip=True,
        validation_split=0.2)
    
    train_generator = train_datagen.flow_from_directory(
        '../../dataset/imdb_crop/new_database/',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True)
    print(next(train_generator)[1].shape)
    val_generator = train_datagen.flow_from_directory(
        '../../dataset/imdb_crop/new_database/',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=True)
        
    callbacks = [LearningRateScheduler(schedule=Schedule(38138  // batch_size)),
                 ModelCheckpoint("checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                 monitor="val_loss",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="auto")
                 ]
    

    
    #class_weight = get_class_weights(train_generator.classes)
    
    #print(class_weight)
    
    h = model.fit_generator(
        generate_data_generator(train_generator),
        use_multiprocessing=True,
        epochs=10,
        validation_data=generate_data_generator(val_generator),
        workers=12,
        steps_per_epoch = len(train_generator.classes)/batch_size,
        validation_steps = len(val_generator.classes)/batch_size,
        #class_weight=class_weight
    )

    logging.debug("Saving weights...")
    model.save_weights(os.path.join("models", "WRN_{}_{}.h5".format(depth, k)), overwrite=True)
    #pd.DataFrame(hist.history).to_hdf(os.path.join("models", "history_{}_{}.h5".format(depth, k)), "history")


if __name__ == '__main__':
    main()
