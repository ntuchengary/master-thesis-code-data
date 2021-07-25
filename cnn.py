#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 08:59:23 2021

@author: sora
"""

#from fcmeans import FCM
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tkinter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
#from loaddat import load_dat_file
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Activation,Flatten, Dense, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D, BatchNormalization, InputLayer, Dropout

from tensorflow.keras.utils import get_source_inputs, plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.optimizers import RMSprop,Adam, SGD, Nadam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.callbacks import TensorBoard, Callback
import tensorflow as tf
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn import preprocessing
from numpy.random import seed
import codecs
import copy
import sys
import time
from sklearn.preprocessing import MinMaxScaler,QuantileTransformer
from tensorflow.keras import initializers
from statistics import mean 
import tensorflow_addons as tfa
from tensorflow_addons.activations import mish
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import random
from collections import Counter
import time
import statistics
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from PIL import Image

from sklearn.cluster import KMeans, AgglomerativeClustering
import gc

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

#設定 Keras 使用的 Session
tf.compat.v1.keras.backend.set_session(sess)


#now = round(time.time() * 1000)
#tf.random.set_seed(1)

class SensitivitySpecificityCallback(Callback):

    def on_epoch_end(self, epoch, logs=None):
        def rank_rank(predictions, metadata, byte_number, real_key, min_trace_idx, max_trace_idx, last_key_bytes_proba):
        # Compute the rank
            if len(last_key_bytes_proba) == 0:
                # If this is the first rank we compute, initialize all the estimates to zero
                key_bytes_proba = np.zeros(256)
            else:
                # This is not the first rank we compute: we optimize things by using the
                # previous computations to save time!
                key_bytes_proba = last_key_bytes_proba
        
            for p in range(0, max_trace_idx-min_trace_idx):
        
                plaintext = metadata[min_trace_idx + p][byte_number]
                for i in range(0, 256):
                    # Our candidate key byte probability is the sum of the predictions logs
                    proba = predictions[p][SBOX_NP[int(plaintext) ^ i]]
        
                    if proba != 0:
                        key_bytes_proba[i] += np.log(proba)
                    else:
                        # We do not want an -inf here, put a very small epsilon
                        # that correspondis to a power of our min non zero proba
                        min_proba_predictions = predictions[p][np.array(predictions[p]) != 0]
                        if len(min_proba_predictions) == 0:
                            print("Error: got a prediction with only zeroes ... this should not happen!")
                            sys.exit(-1)
                        min_proba = min(min_proba_predictions)
                        key_bytes_proba[i] += np.log(min_proba**2)
            # Now we find where our real key candidate lies in the estimation.
            # We do this by sorting our estimates and find the rank in the sorted array.
            sorted_proba = np.array(list(map(lambda a : key_bytes_proba[a], key_bytes_proba.argsort()[::-1])))
            real_key_rank = np.where(sorted_proba == key_bytes_proba[real_key])[0][0]
            return (real_key_rank, key_bytes_proba)
    
        def full_ranks(model, dataset, metadata, byte_number, min_trace_idx, max_trace_idx, rank_step):
        	# Real key byte value that we will use. '2' is the index of the byte (third byte) of interest.
        	
            #real_key=['00','11','22','33','44','55','66','77','88','99','AA','BB','CC','DD','EE','FF']
            key=[0,17,34,51,68,85,102,119,136,153,170,187,204,221,238,255]
            
            real_key=key[byte_number]
        
            input_data = dataset[min_trace_idx:max_trace_idx, :]
        
        
        	# Predict our probabilities
            predictions = self.model.predict(input_data)
        
            index = np.arange(min_trace_idx + rank_step, max_trace_idx, rank_step)
            
            f_ranks = np.zeros((len(index), 2), dtype=np.uint32)
            key_bytes_proba = []
            for t, i in zip(index, range(0, len(index))):
                real_key_rank, key_bytes_proba = rank(predictions[t-rank_step:t], metadata, byte_number, real_key, t-rank_step, t, key_bytes_proba)
                f_ranks[i] = [t - min_trace_idx, real_key_rank]

            return f_ranks
        
        ranks = full_ranks(self, test_data, pt_test, byte_number, 0, reserved_num_trace+1 , 1)
        y = [ranks[i][1] for i in range(0, ranks.shape[0])]
        counter=256
        for converge in range(len(y),0,-1):
            if y[converge-1]==0 :
                counter=converge
            elif (y[converge-1]!=0):
            
                break   
        records_for_all_epochs.append(converge)  
        print('\n')
        if converge >= reserved_num_trace:
            print("Cant crack within " + str(reserved_num_trace) + " traces")
        else:
            print("Converge at trace number: " +str(converge))
            
            
def load_input(input_file):
    return np.loadtxt(input_file, dtype = np.int32,
                      converters = {_: lambda s: int(s, 16) for _ in range(16)})

def mark_sbox(ax,poi,grid_size,color,flag):
    loc = poi
    loc = np.array(loc).T
    for ii in range(len(loc[0])):
        #ax.text(loc[1][ii],loc[0][ii],str(ii),horizontalalignment='center',verticalalignment='center',color='r')
        patch = plt.Rectangle((loc[1][ii] - grid_size//2 - 0.5, loc[0][ii] -grid_size//2 - 0.5), grid_size, grid_size, facecolor='none', edgecolor='black')
        if flag==True:
            ax.add_patch(patch)
        ax.text(loc[1][ii],loc[0][ii],str(ii),horizontalalignment='center',verticalalignment='center',color=color, size=7)

def rank(predictions, metadata, byte_number, real_key, min_trace_idx, max_trace_idx, last_key_bytes_proba):
	# Compute the rank
	if len(last_key_bytes_proba) == 0:
		# If this is the first rank we compute, initialize all the estimates to zero
		key_bytes_proba = np.zeros(256)
	else:
		# This is not the first rank we compute: we optimize things by using the
		# previous computations to save time!
		key_bytes_proba = last_key_bytes_proba

	for p in range(0, max_trace_idx-min_trace_idx):

		plaintext = metadata[min_trace_idx + p][byte_number]
		for i in range(0, 256):
			# Our candidate key byte probability is the sum of the predictions logs
			proba = predictions[p][SBOX_NP[int(plaintext) ^ i]]

			if proba != 0:
				key_bytes_proba[i] += np.log(proba)
			else:
				# We do not want an -inf here, put a very small epsilon
				# that correspondis to a power of our min non zero proba
				min_proba_predictions = predictions[p][np.array(predictions[p]) != 0]
				if len(min_proba_predictions) == 0:
					print("Error: got a prediction with only zeroes ... this should not happen!")
					sys.exit(-1)
				min_proba = min(min_proba_predictions)
				key_bytes_proba[i] += np.log(min_proba**2)
	# Now we find where our real key candidate lies in the estimation.
	# We do this by sorting our estimates and find the rank in the sorted array.
	sorted_proba = np.array(list(map(lambda a : key_bytes_proba[a], key_bytes_proba.argsort()[::-1])))
	real_key_rank = np.where(sorted_proba == key_bytes_proba[real_key])[0][0]
	return (real_key_rank, key_bytes_proba)

def full_ranks(model, dataset, metadata, byte_number, min_trace_idx, max_trace_idx, rank_step):
	# Real key byte value that we will use. '2' is the index of the byte (third byte) of interest.
	
    #real_key=['00','11','22','33','44','55','66','77','88','99','AA','BB','CC','DD','EE','FF']
    key=[0,17,34,51,68,85,102,119,136,153,170,187,204,221,238,255]
    
    real_key=key[byte_number]

    input_data = dataset[min_trace_idx:max_trace_idx, :]


	# Predict our probabilities
    predictions = model.predict(input_data)


    index = np.arange(min_trace_idx + rank_step, max_trace_idx, rank_step)
    
    f_ranks = np.zeros((len(index), 2), dtype=np.uint32)
    key_bytes_proba = []
    for t, i in zip(index, range(0, len(index))):
        real_key_rank, key_bytes_proba = rank(predictions[t-rank_step:t], metadata, byte_number, real_key, t-rank_step, t, key_bytes_proba)
        f_ranks[i] = [t - min_trace_idx, real_key_rank]

    return f_ranks

#for trials in range (0,1,1):
reserved_num_trace=3000
multiply=17000
time_start = time.time() 

sbox_poi_index = [(94, 161), (65, 171), (70, 179), (93, 166), (92, 173), (81, 175), (65, 166), (65, 174), (70, 175), (88, 171), (81, 170), (59, 166), (66, 160), (64, 182), (88, 178), (89, 177)] 

data_2D_full=np.load('/usr/python/full_chip_temperature/thermal_20000_inter.npy')

thermal_2000_inter_filtered= np.zeros((multiply,201,201))


for trials in range (0,1,1):
    SBOX_NP = np.array([
        99, 124, 119, 123, 242, 107, 111, 197,  48,   1, 103,  43, 254, 215, 171, 118,
        202, 130, 201, 125, 250,  89,  71, 240, 173, 212, 162, 175, 156, 164, 114, 192,
        183, 253, 147,  38,  54,  63, 247, 204,  52, 165, 229, 241, 113, 216,  49,  21,
        4, 199,  35, 195,  24, 150,   5, 154,   7,  18, 128, 226, 235,  39, 178, 117,
        9, 131,  44,  26,  27, 110,  90, 160,  82,  59, 214, 179,  41, 227,  47, 132,
        83, 209, 0, 237, 32, 252, 177, 91, 106, 203, 190, 57, 74, 76, 88, 207,
        208, 239, 170, 251, 67, 77, 51, 133, 69, 249, 2, 127, 80, 60, 159, 168,
        81, 163, 64, 143, 146, 157, 56, 245, 188, 182, 218, 33, 16, 255, 243, 210,
        205, 12, 19, 236, 95, 151, 68, 23, 196, 167, 126, 61, 100, 93, 25, 115,
        96, 129, 79, 220, 34, 42, 144, 136, 70, 238, 184, 20, 222, 94, 11, 219,
        224, 50, 58, 10, 73, 6, 36, 92, 194, 211, 172, 98, 145, 149, 228, 121,
        231, 200, 55, 109, 141, 213, 78, 169, 108, 86, 244, 234, 101, 122, 174, 8,
        186, 120, 37, 46, 28, 166, 180, 198, 232, 221, 116, 31, 75, 189, 139, 138,
        112, 62, 181, 102, 72, 3, 246, 14, 97, 53, 87, 185, 134, 193, 29, 158,
        225, 248, 152, 17, 105, 217, 142, 148, 155, 30, 135, 233, 206, 85, 40, 223,
        140, 161, 137, 13, 191, 230, 66, 104, 65, 153, 45, 15, 176, 84, 187, 22])

    key = np.array([0,17,34,51,68,85,102,119,136,153,170,187,204,221,238,255],dtype=np.int32)
    
    np_records_for_all_trials=[]
    


    plaintext_ori_no_shu = load_input('multi_tile_ft_plaintext.csv')
    plaintext=plaintext_ori_no_shu[0:20000,0:16]

        
    train_data, test_data = np.split(data_2D_full, [20000-reserved_num_trace ])
    scaler =MinMaxScaler(feature_range=(0, 1))
    
    train_data=train_data.reshape((17000,40401))
    train_data = scaler.fit_transform (train_data)
    train_data=train_data.reshape((17000,201,201))
    
    test_data=test_data.reshape((3000,40401))
    test_data =scaler.transform(test_data)
    test_data=test_data.reshape((3000,201,201))


    pt_train, pt_test =  np.split(plaintext, [20000-reserved_num_trace ])

    label = SBOX_NP[key^plaintext]
    label=label[0:20000,0:16];
    label_train, label_test =  np.split(label, [20000-reserved_num_trace ])
  

    #test_byte_list= [9]
    test_byte_list= [0,1,2,3,4,5,6,7,8,10,11,12,13,14,15]
        

    del_flag=True
    for n in test_byte_list :
        

        
        byte_number = n
        records_for_all_epochs=[]
        np_records_for_all_epochs=[]
        epochs=5000
        batch_size=128
        
        y_train=to_categorical(label_train[:,byte_number] ,num_classes=256)
        y_test=to_categorical(label_test[:,byte_number] ,num_classes=256)


        input_shape = (201,201,1)
        img_input = Input(shape=input_shape)
        x = Conv2D(64, 3,strides=2, kernel_initializer='VarianceScaling', activation= mish, padding='same', name='block1_conv1',kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01))(img_input)
        x = BatchNormalization()(x)
        x = AveragePooling2D(2, strides=2, name='block1_pool')(x)
        
        # 2nd convolutional block
        x = Conv2D(32, 3, strides=2, kernel_initializer='VarianceScaling', activation= mish, padding='same', name='block2_conv1',kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        x = AveragePooling2D(2, strides=2, name='block2_pool')(x)
        
        # 3rd convolutional block
        x = Conv2D(16, 3, strides=2, kernel_initializer='VarianceScaling', activation= mish, padding='same', name='block3_conv1',kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01))(x)      
        x = BatchNormalization()(x)
        x = AveragePooling2D(2, strides=2, name='block3_pool')(x)

        x = Flatten(name='flatten')(x)

        # Classification part
        x = Dense(64, kernel_initializer='VarianceScaling', activation= mish, name='fc1')(x)
        x = BatchNormalization()(x)
        x = Dense(64, kernel_initializer='VarianceScaling', activation= mish, name='fc2')(x)
        x = BatchNormalization()(x)
        # x = Dense(128, kernel_initializer='VarianceScaling', activation= mish, name='fc3')(x)
        # x = BatchNormalization()(x)

        # Logits layer
        x = Dense(256, activation='softmax', name='predictions',kernel_regularizer=regularizers.l2(0.1),activity_regularizer=regularizers.l2(0.1))(x)



        inputs = img_input 
        model= Model(inputs=inputs, outputs=x, name='FC_output')
        adam = tf.keras.optimizers.Adam(learning_rate=1e-2)
        #radam=tfa.optimizers.RectifiedAdam(learning_rate=0.001,beta_1=0.95,epsilon=1e-05)
        radam=tfa.optimizers.RectifiedAdam(learning_rate=0.001,beta_1=0.95,epsilon=1e-05,total_steps=300000,warmup_proportion=0.1,min_lr=1e-5)
        ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
        
        model.compile(optimizer= ranger, loss='categorical_crossentropy', metrics=[tf.keras.metrics.TopKCategoricalAccuracy(k=5)])
        model.summary()

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=50,min_delta=0.00001, restore_best_weights=True)
        now=time.time()

        Sbox16= model.fit(train_data, y_train, epochs=epochs, validation_split=0.2, batch_size= batch_size,verbose=1,callbacks=[callback])
        
        #callbacks=[callback]
        now1=time.time()
        print('byte' + str(byte_number))
        print((now1-now))
        ranks = full_ranks(model, test_data, pt_test, byte_number, 0, reserved_num_trace +1 , 1)
        
        x = [ranks[i][0] for i in range(0, ranks.shape[0])]
        y = [ranks[i][1] for i in range(0, ranks.shape[0])]
        
        
        
        evaluate= model.evaluate(test_data, y_test, batch_size=20, verbose=1)
        test_acc  = evaluate[1]
        print('test_top 5 acc:' +str(test_acc))
        
        
        #np.save(str(foldername) +'/epoch'+ str(epochs)+ ' byte' + str(byte_number) + ' trial' + str(trials)+' :trace vs rank ' ,ranks)
        epoch_list=list(range(1, epochs+1))
        
        counter= reserved_num_trace
        for converge in range(len(y),0,-1):
        
            if y[converge-1]==0 :
                counter=converge
            elif (y[converge-1]!=0):
            
                break
        
        print("\n")
        if counter>(reserved_num_trace -1) :
            print("Can not converge within " + str(reserved_num_trace) +" traces")
        else:
            print("Converge at trace number: " +str(converge))
        
        print("Last rank: " + str(y[2999]))
        print("\n")
        train_loss = Sbox16.history['loss']
        val_loss   = Sbox16.history['val_loss']
        train_acc  = Sbox16.history['top_k_categorical_accuracy']
        val_acc    = Sbox16.history['val_top_k_categorical_accuracy']
        epochs= len(train_loss)
        xc         = range(epochs)
        
        records_for_all_epochs.insert(0,byte_number)

        
        model.save( os.path.abspath("crack_model"+' b' + str(byte_number) +'e'+ str(epochs)+ 'trial' + str(trials) +'vgg') )
        np.save("result_np" +'\\ byte' + str(byte_number) + 'epoch'+ str(epochs)+ ' trial ' + str(trials)+'trace vs rank '  , ranks)
        np.save("result_np" +'\\ byte' + str(byte_number) +'epoch'+ str(epochs)+  ' trial ' + str(trials)+ 'epoch vs result'  , records_for_all_epochs)
        np.save("result_np" +'\\ byte'+ str(byte_number) + 'loss', train_loss)
        np.save("result_np" +'\\ byte'+ str(byte_number) + 'val_loss', val_loss)
        np.save("result_np" +'\\ byte'+ str(byte_number) + 'train_top5_acc', train_acc)
        np.save("result_np" +'\\ byte'+ str(byte_number) + 'val_top5_acc', val_acc)
        

        
        
        