#_*_coding:utf8_*_
from keras.models import Sequential
from keras.layers import MaxPool2D,Flatten,Dense
from keras.layers.convolutional import Conv3D, Conv2D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from keras.optimizers import Adam
from numpy import concatenate
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras import backend as K
from keras.layers.core import Lambda
import h5py
import hdf5storage
import os
from numpy.random import seed
import random as python_random

# lon lat dim time

row = 32
col = 32
dim = 6
count_train_steps = 72
count_val_steps = 36
count_test_steps = 10
epochs = 1000

filters_ensemble = [20,30,40,50]
#model.compile(optimizer='sgd', loss=tf.keras.losses.CosineSimilarity(axis=1))
# seq.load_weights('//mnt//md0//XL//temPredict//weights')

# (samples, time, row, col, dim)
count = 1
def generate_arrays_from_file(path):


    global count

    while 1:
        train_data = hdf5storage.loadmat(path + 'train_data_' + str(count) + '.mat')
        train_data_label = hdf5storage.loadmat(path + 'train_data_label_' + str(count) + '.mat')
        train_data = train_data['train_data'] #29*18*var*time*sample
        train_data_label = train_data_label['train_data_label']# lon*lat*sample

        # train_data = np.transpose(train_data, (4, 3, 0, 1, 2))
        # train_data_label = np.transpose(train_data_label, (  2, 0, 1))
        # train_data_label = train_data_label.reshape(train_data_label.shape[0], train_data_label.shape[1],
        #                                             train_data_label.shape[2],  1)

        # print(train_data.shape)

        count = count + 1
        if count > count_train_steps:
            count = 1

        yield (train_data, train_data_label)


count_val = 1
def generate_arrays_from_file_val(path):
    # x_y 是我们的训练集包括标签，每一行的第一个是我们的图片路径，后面的是我们的独热化后的标签

    global count_val

    while 1:
        validate_data = hdf5storage.loadmat(path + 'validate_data_'+ str(count_val) + '.mat')
        validate_data_label = hdf5storage.loadmat(path + 'validate_data_label_' + str(count_val) + '.mat')
        validate_data = validate_data['validate_data']
        validate_data_label = validate_data_label['validate_data_label']

        # validate_data = np.transpose(validate_data, (4, 3, 0, 1, 2))
        # validate_data_label = np.transpose(validate_data_label, ( 2, 0, 1))
        # validate_data_label = validate_data_label.reshape(validate_data_label.shape[0], validate_data_label.shape[1],
        #                                             validate_data_label.shape[2], 1)

        count_val = count_val + 1
        if count_val > count_val_steps:
            count_val = 1

        yield (validate_data, validate_data_label)

for index_filter in range(len(filters_ensemble)):
    for index_point in range(0, 1):
        for index_lead in range(0, 12):
            # if index_point==1 and (index_lead<2 or index_lead>=5):
            #     continue
            # if index_point==2 and (index_lead<2 or index_lead>=5):
            #     continue
            # if index_point==3 and (index_lead<3 or index_lead>=5):
            #     continue
            # if index_point==4 and (index_lead<3 or index_lead>=5):
            #     continue
            # if index_point==5 and (index_lead<3 or index_lead>=5):
            #     continue
            # if index_point==6 and (index_lead<3 or index_lead>=5):
            #     continue
            # if index_point==7 and (index_lead<2 or index_lead>=5):
            #     continue
            # if index_point==8 and (index_lead<2 or index_lead>=5):
            #     continue

            np.random.seed(1)
            python_random.seed(1)
            tf.set_random_seed(1)

            # 指定第一块GPU可用
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            config = tf.ConfigProto()
            # config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
            sess = tf.Session(config=config)
            KTF.set_session(sess)

            seq = Sequential()
            seq.add(Conv2D(filters=filters_ensemble[index_filter], kernel_size=(2, 2),
                           input_shape=(row, col, dim),
                           activation='relu', padding='same'))  # 32*32*30
            # seq.add(BatchNormalization())
            seq.add(MaxPool2D(pool_size=(2, 2)))  # 16*16*30

            seq.add(Conv2D(filters=filters_ensemble[index_filter], kernel_size=(2, 2),
                           activation='relu', padding='same'))
            seq.add(MaxPool2D(pool_size=(2, 2)))  # 8*8*30

            seq.add(Conv2D(filters=filters_ensemble[index_filter], kernel_size=(2, 2),
                           activation='relu', padding='same'))
            seq.add(MaxPool2D(pool_size=(2, 2)))  # 4*4*30

            seq.add(Conv2D(filters=filters_ensemble[index_filter], kernel_size=(2, 2),
                           activation='relu', padding='same'))
            seq.add(MaxPool2D(pool_size=(2, 2)))  # 2*2*30

            # seq.add(Conv2D(filters=30, kernel_size=(2, 2),
            #                activation='relu', padding='same'))
            # seq.add(MaxPool2D(pool_size=(2, 2)))  # 2*2*30
            # seq.add(BatchNormalization())

            seq.add(Conv2D(filters=filters_ensemble[index_filter], kernel_size=(2, 2),
                           activation='relu', padding='valid'))

            # seq.add(BatchNormalization())
            seq.add(Flatten())
            # seq.add(tf.keras.layers.Dropout(0.5))
            seq.add(Lambda(lambda x: K.dropout(x, level=0.5)))
            seq.add(Dense(1))

            # seq.add(ConvLSTM2D(filters=1, kernel_size=(3, 3),
            #               padding='same',return_sequences=True,  data_format='channels_last'))

            opt = Adam(lr=0.001)  # learning_rate

            seq.compile(loss="mean_squared_error", optimizer=opt)  # mean_squared_error
            print(seq.summary())
            # /scratch/xulei/temPredict/    I://data//temPredict6//  G://甘肃马拉松极端天气预测//code//test4//      I://data//temPredict//  //mnt//md0//XL//temPredict//    //scratch//xulei//temPredict//
            checkpointer = ModelCheckpoint(
                '//mnt//md0/XL//era5_predict//t2m//' + 'point_' + str(index_point + 1) + '//lead_' + str(
                    index_lead + 1) + '//' + 'weights_'+str(index_filter), monitor='val_loss', verbose=2, save_weights_only=False,
                save_best_only=True, mode='min')
            earlystop = EarlyStopping(monitor='val_loss', min_delta=0.000001, patience=100, verbose=1, mode='min')
            reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=30, verbose=1, mode='min',
                                         min_delta=1e-8, cooldown=1, min_lr=0.0000001)

            # seq.load_weights('//mnt//md0/XL//era5_predict//t2m//' + 'point_' + str(index_point + 1) + '//lead_' + str(
            #     index_lead + 1) + '//' + 'weights')
            #fit model
            history = seq.fit_generator(generate_arrays_from_file('//mnt//md0/XL//era5_predict//t2m//' + 'point_' + str(index_point + 1) +  '//lead_' + str(
                index_lead + 1) + '//'),steps_per_epoch = count_train_steps, validation_data=generate_arrays_from_file_val('//mnt//md0/XL//era5_predict//t2m//' +
                'point_' + str(index_point + 1) +  '//lead_' + str(index_lead + 1) + '//'),
                              validation_steps=count_val_steps, epochs=epochs, verbose=1, callbacks=[checkpointer, earlystop, reducelr])
            # print(h.history)
            error = np.empty(shape=[epochs, 2])  # train, validate
            t = history.history['loss']
            error[0:len(t), 0] = t
            t = history.history['val_loss']
            error[0:len(t), 1] = t
            hdf5storage.savemat('//mnt//md0/XL//era5_predict//t2m//' + 'point_' + str(index_point + 1) +  '//lead_' + str(
                index_lead + 1) + '//' + 'error_' +str(index_filter)+'.mat',
                                {'error': error})

            # # read test data
            for i in range(count_test_steps):
                test_data = hdf5storage.loadmat(
                    '//mnt//md0/XL//era5_predict//t2m//' + 'point_' + str(index_point + 1) + '//lead_' + str(
                        index_lead + 1) + '//' + 'test_data_' + str(i + 1) + '.mat')
                # test_data_label = hdf5storage.loadmat('E://xulei//test_data_label_' + str(i+1) + '.mat')
                test_data = test_data['test_data']
                # test_data_label = test_data_label['test_data_label']
                # test_data = np.transpose(test_data, (4, 3, 0, 1, 2))
                y_pre_all = np.empty([test_data.shape[0], 1], dtype=float)
                # for j in range(10):
                #     y_pre_all[:, :, j] = seq.predict(test_data)
                y_pre_all[:, :] = seq.predict(test_data)
                # y_pre = np.mean(y_pre_all, axis=2)  # (samples, time, row, col,)
                # y_pre_std = np.std(y_pre_all, axis=2)
                y_pre = y_pre_all
                hdf5storage.savemat(
                    '//mnt//md0/XL//era5_predict//t2m//' + 'point_' + str(index_point + 1) + '//lead_' + str(
                        index_lead + 1) + '//' + 'test_data_predict_' + str(i + 1) +'filter_' + str(index_filter) + '.mat', {'y_pre': y_pre})
                # hdf5storage.savemat(
                #     '//mnt//md0/XL//era5_predict//t2m//' + 'point_' + str(index_point + 1) + '//lead_' + str(
                #         index_lead + 1) + '//' + 'test_data_predict_std_' + str(i + 1) + '.mat',
                #     {'y_pre_std': y_pre_std})
