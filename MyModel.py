from keras_vggface.vggface import VGGFace

import logging
import sys
import numpy as np
import os
from keras.models import Model
from keras.layers import Input, Activation, add, Dense, Flatten, Dropout, GlobalAveragePooling2D,MaxPooling2D, LeakyReLU
from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K

from keras.models import Sequential

from keras.layers import Lambda, concatenate
import tensorflow as tf

#from stackoverflow
def multi_gpu_model(model, gpus):
  if isinstance(gpus, (list, tuple)):
    num_gpus = len(gpus)
    target_gpu_ids = gpus
  else:
    num_gpus = gpus
    target_gpu_ids = range(num_gpus)

  def get_slice(data, i, parts):
    shape = tf.shape(data)
    batch_size = shape[:1]
    input_shape = shape[1:]
    step = batch_size // parts
    if i == num_gpus - 1:
      size = batch_size - step * i
    else:
      size = step
    size = tf.concat([size, input_shape], axis=0)
    stride = tf.concat([step, input_shape * 0], axis=0)
    start = stride * i
    return tf.slice(data, start, size)

  all_outputs = []
  for i in range(len(model.outputs)):
    all_outputs.append([])

  # Place a copy of the model on each GPU,
  # each getting a slice of the inputs.
  for i, gpu_id in enumerate(target_gpu_ids):
    with tf.device('/gpu:%d' % gpu_id):
      with tf.name_scope('replica_%d' % gpu_id):
        inputs = []
        # Retrieve a slice of the input.
        for x in model.inputs:
          input_shape = tuple(x.get_shape().as_list())[1:]
          slice_i = Lambda(get_slice,
                           output_shape=input_shape,
                           arguments={'i': i,
                                      'parts': num_gpus})(x)
          inputs.append(slice_i)

        # Apply model on slice
        # (creating a model replica on the target device).
        outputs = model(inputs)
        if not isinstance(outputs, list):
          outputs = [outputs]

        # Save the outputs for merging back together later.
        for o in range(len(outputs)):
          all_outputs[o].append(outputs[o])

  # Merge outputs on CPU.
  with tf.device('/cpu:0'):
    merged = []
    for name, outputs in zip(model.output_names, all_outputs):
      merged.append(concatenate(outputs,
                                axis=0, name=name))
    return Model(model.inputs, merged)


class MyModel:
    def __init__(self, image_size, nb_class=101, trainable= False):
        self.nb_class = nb_class
        self.image_size = image_size
        if nb_class > 1:
            self.activation = "softmax"
        else:
            self.activation = "none"
        self.trainable = trainable
        self.model = None
    
    def get_model():
        return self.model
    
    def load_model(path):
        model.load_weights(path);
        
    def get_cnn():
        if self.vgg_model_size is not None and self.model is not None:
            return self.model.layers[:self.vgg_model_size]
        
    def get_vgg_face(model_type = 'vgg16'):
        return VGGFace(include_top=False, model=model_type, input_shape=(self.image_size, self.image_size, 3), pooling='avg')
    
    def __call__(self):
        logging.debug("Creating model...")
        
        os.environ["CUDA_VISIBLE_DEVICES"]="1"
        
        nb_class = 101
        vgg_model = VGGFace(include_top=False, input_shape=(self.image_size, self.image_size, 3), pooling='avg')

        out = vgg_model.layers[-1].output
        
        self.vgg_model_size = len(vgg_model.layers) 
        
        for layer in vgg_model.layers:
            layer.trainable = self.trainable
        
        x = Dense(units=4096,kernel_regularizer=l2(0.0005),kernel_initializer="he_normal", activation="relu")(out)
        x = Dropout(0.5)(x)
        x = Dense(units=4096,kernel_regularizer=l2(0.0005),kernel_initializer="he_normal", activation="relu")(x)
        x = Dropout(0.5)(x)
        class_pred = Dense(units=101, kernel_initializer="he_normal", use_bias=False,
                              kernel_regularizer=l2(0.0005), activation="softmax",
                              name="pred_age")(x)
        
        
        regression = Dense(units=1,kernel_initializer='normal', name="regress_age")(out)

        model = Model(vgg_model.input, outputs=[class_pred,regression])
        

        #model = multi_gpu_model(model,gpus=2)
        #if self.trainable:
        #    model.load_weights("models/WRN_16_8.h5");
        
        return model
        