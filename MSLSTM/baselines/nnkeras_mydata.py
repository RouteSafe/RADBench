import time
import evaluation
from sklearn.feature_selection import RFE
from collections import defaultdict
import numpy as np
np.random.seed(1337)  # for reproducibility
import keras
import sklearn
from numpy import *
import loaddata_mydata
from sklearn import tree
from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers import Dense
from keras.models import Model
from keras.layers.recurrent import LSTM,SimpleRNN
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
import tensorflow as tf
import loaddata
import printlog
import sys
import os
flags = tf.app.flags
import matplotlib.pyplot as plt
from keras.utils import to_categorical
#import ucr_load_data
import logging

# 配置日志 - 只输出到控制台
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

FLAGS = flags.FLAGS

def pprint(msg,method=''):
    if not 'Warning' in msg:
        logging.info(msg)
        try:
            sys.stderr.write(msg+'\n')
        except:
            pass
def Basemodel(_model,filename,trigger_flag,evalua_flag,is_binary_class):
    x_train, y_train = loaddata_mydata.get_data_withoutS(FLAGS.pooling_type, FLAGS.is_add_noise, FLAGS.noise_ratio, FLAGS.data_dir,
                                            filename, FLAGS.sequence_window, trigger_flag,is_binary_class,
                                            multiScale=False, waveScale=FLAGS.scale_levels,
                                            waveType=FLAGS.wave_type)
    FLAGS.input_dim = x_train.shape[-1]
    y_train = y_train.astype(int)
    num_classes = len(np.unique(y_train))
    FLAGS.number_class = num_classes
    
    y_train = to_categorical(y_train, num_classes=FLAGS.number_class)

    # using MLP to train
    if _model == "MLP":
        logging.info(_model + " is running..............................................")
        start = time.clock()
        model = Sequential()
        model.add(Dense(FLAGS.num_neurons1, activation="sigmoid", input_dim=FLAGS.input_dim))
        #model.add(Dense(output_dim=FLAGS.num_neurons1))
        #model.add(Dense(output_dim=FLAGS.num_neurons1))
        model.add(Dense(output_dim=FLAGS.number_class))
        model.add(Activation("sigmoid"))
        sgd = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
        #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, batch_size=FLAGS.batch_size, nb_epoch=FLAGS.max_epochs)
        end = time.clock()
        logging.info("The Time For MLP is " + str(end - start))

    elif _model == "RNN":
        logging.info(_model + " is running..............................................")
        start = time.clock()
        x_train, y_train = loaddata_mydata.get_data(FLAGS.pooling_type, FLAGS.is_add_noise,
                                                                           FLAGS.noise_ratio, FLAGS.data_dir,
                                                                           filename, FLAGS.sequence_window,
                                                                           trigger_flag,is_binary_class,
                                                                           multiScale=False,
                                                                           waveScale=FLAGS.scale_levels,
                                                                           waveType=FLAGS.wave_type)
        logging.info(f'nnkeras.Basemodel{_model} get_data成功')
        logging.info(f'x_train: {x_train.shape}, y_train: {y_train.shape}')
        rnn_object1 = SimpleRNN(FLAGS.num_neurons1, input_length=len(x_train[0]), input_dim=FLAGS.input_dim)
        model = Sequential()
        model.add(rnn_object1)  # X.shape is (samples, timesteps, dimension)
        #model.add(Dense(30, activation="sigmoid"))
        #rnn_object2 = SimpleRNN(FLAGS.num_neurons2, input_length=len(x_train[0]), input_dim=FLAGS.input_dim)
        #model.add(rnn_object2)  # X.shape is (samples, timesteps, dimension)
        model.add(Dense(output_dim=FLAGS.number_class))
        model.add(Activation("sigmoid"))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, batch_size=FLAGS.batch_size, nb_epoch=FLAGS.max_epochs)

        end = time.clock()
        logging.info("The Time For RNN is " + str(end - start))

        # print(result)
    elif _model == "LSTM":
        logging.info(_model + " is running..............................................")
        start = time.clock()
        x_train, y_train = loaddata.get_data(FLAGS.pooling_type, FLAGS.is_add_noise,
                                                                           FLAGS.noise_ratio, FLAGS.data_dir,
                                                                           filename, FLAGS.sequence_window,
                                                                           trigger_flag,is_binary_class,
                                                                           multiScale=False,
                                                                           waveScale=FLAGS.scale_levels,
                                                                           waveType=FLAGS.wave_type)
        logging.info(f'nnkeras.Basemodel{_model} get_data成功')
        initi_weight = keras.initializers.RandomNormal(mean=0.0, stddev= 1, seed=None)
        initi_bias = keras.initializers.Constant(value=0.1)
        lstm_object = LSTM(FLAGS.num_neurons1, input_length=x_train.shape[1], input_dim=FLAGS.input_dim)
        model = Sequential()
        model.add(lstm_object,kernel_initializer=initi_weight,bias_initializer=initi_bias)  # X.shape is (samples, timesteps, dimension)
        #model.add(Dense(30, activation="relu"))
        model.add(Dense(output_dim=FLAGS.number_class))
        model.add(Activation("sigmoid"))
        sgd = keras.optimizers.SGD(lr=0.02, momentum=0.0, decay=0.0, nesterov=False)
        model.compile(optimizer=sgd,loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, batch_size=FLAGS.batch_size, nb_epoch=FLAGS.max_epochs)
        end = time.clock()
        logging.info("The Time For LSTM is " + str(end - start))

    return model

def get_trained_model(_model, filename, trigger_flag, evalua_flag, is_binary_class):
    """获取训练好的模型"""
    try:
        x_train, y_train = loaddata_mydata.get_data_withoutS(
            FLAGS.pooling_type,
            FLAGS.is_add_noise,
            FLAGS.noise_ratio,
            FLAGS.data_dir,
            filename,
            FLAGS.sequence_window,
            trigger_flag,
            is_binary_class,
            multiScale=False,
            waveScale=FLAGS.scale_levels,
            waveType=FLAGS.wave_type
        )
        
        if x_train is None or y_train is None:
            return None
            
        y_train = to_categorical(y_train.astype(int), num_classes=FLAGS.number_class)
        
        # 创建并训练模型
        model = Sequential()
        model.add(Dense(FLAGS.num_neurons1, activation="sigmoid", input_dim=FLAGS.input_dim))
        model.add(Dense(FLAGS.number_class))
        model.add(Activation("sigmoid"))
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, batch_size=FLAGS.batch_size, epochs=FLAGS.max_epochs, verbose=0)
        
        return model
        
    except Exception as e:
        logging.error(f"训练模型时出错: {str(e)}")
        return None

