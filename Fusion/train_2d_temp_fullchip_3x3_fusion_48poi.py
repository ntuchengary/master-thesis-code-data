#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 16:37:13 2021

@author: sora
"""
import tkinter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
#from loaddat import load_dat_file
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Activation,Flatten, Dense, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D, BatchNormalization, InputLayer, Dropout
from tensorflow.keras.layers import concatenate, add, average, maximum, minimum, subtract, multiply, dot,subtract
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
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import initializers
from statistics import mean 
import tensorflow_addons as tfa
from tensorflow_addons.activations import mish
from tensorflow.keras import regularizers
#read data
#seed_value= 1
#tf.compat.v1.set_random_seed(seed_value)

#data_10cyc_power = np.fromfile('dpv_2610_20480_rhsc_prop.dat').reshape((2610,-1)).T
#data_10cyc_power_poi = np.fromfile('poi_dpv_power_rhsc_prop.dat').reshape((-1,16))

def load_input(input_file):
    return np.loadtxt(input_file, dtype = np.int32,
                      converters = {_: lambda s: int(s, 16) for _ in range(16)})

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

# poi_index = np.array([[46,13],[17,22],[22,30],[45,17],[45,25],[33,26],[17,18],[17,26],
#                      [23,26],[40,22],[33,22],[11,17],[18,11],[16,33],[40,30],[41,28]])

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
            predictions = self.model.predict([b0_test, b1_test, b2_test, b3_test, b4_test, 
                                       b5_test, b6_test, b7_test, b8_test, b9_test, 
                                       b10_test, b11_test, b12_test, b13_test, b14_test,
                                       b15_test, b16_test, b17_test, b18_test, b19_test,
                                       b20_test, b21_test, b22_test, b23_test, b24_test,
                                       b25_test, b26_test, b27_test, b28_test, b29_test,
                                       b30_test, b31_test, b32_test, b33_test, b34_test,
                                       b35_test, b36_test, b37_test, b38_test, b39_test,
                                       b40_test, b41_test, b42_test, b43_test, b44_test,
                                       b45_test, b46_test, b47_test])
        
            index = np.arange(min_trace_idx + rank_step, max_trace_idx, rank_step)
            
            f_ranks = np.zeros((len(index), 2), dtype=np.uint32)
            key_bytes_proba = []
            for t, i in zip(index, range(0, len(index))):
                real_key_rank, key_bytes_proba = rank(predictions[t-rank_step:t], metadata, byte_number, real_key, t-rank_step, t, key_bytes_proba)
                f_ranks[i] = [t - min_trace_idx, real_key_rank]

            return f_ranks
        
        ranks = full_ranks(self, b0_test , pt_test, byte_number, 0, 3001, 1)
        y = [ranks[i][1] for i in range(0, ranks.shape[0])]
        counter=256
        for converge in range(len(y),0,-1):
            if y[converge-1]==0 :
                counter=converge
            elif (y[converge-1]!=0):
            
                break   
        records_for_all_epochs.append(converge)  
        print('\n')
        print("Converge at trace number: " +str(converge))

        

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

    #input_data = dataset[min_trace_idx:max_trace_idx, :]


	# Predict our probabilities
    predictions = model.predict([b0_test, b1_test, b2_test, b3_test, b4_test, 
                                       b5_test, b6_test, b7_test, b8_test, b9_test, 
                                       b10_test, b11_test, b12_test, b13_test, b14_test,
                                       b15_test, b16_test, b17_test, b18_test, b19_test,
                                       b20_test, b21_test, b22_test, b23_test, b24_test,
                                       b25_test, b26_test, b27_test, b28_test, b29_test,
                                       b30_test, b31_test, b32_test, b33_test, b34_test,
                                       b35_test, b36_test, b37_test, b38_test, b39_test,
                                       b40_test, b41_test, b42_test, b43_test, b44_test,
                                       b45_test, b46_test, b47_test])

    index = np.arange(min_trace_idx + rank_step, max_trace_idx, rank_step)
    
    f_ranks = np.zeros((len(index), 2), dtype=np.uint32)
    key_bytes_proba = []
    for t, i in zip(index, range(0, len(index))):
        real_key_rank, key_bytes_proba = rank(predictions[t-rank_step:t], metadata, byte_number, real_key, t-rank_step, t, key_bytes_proba)
        f_ranks[i] = [t - min_trace_idx, real_key_rank]
    aaaa=real_key_rank
    bbbb=key_bytes_proba 
    return f_ranks


np_records_for_all_trials=[]
#data= np.load('2d_data.npy')
#data = data.reshape(-1, 58, 45, 1)
#data_temp_fullchip=np.load('2d_data_reset_minmax.npy')

"""
1D  3x3 windows for 16 poi shape=16*20000*9
"""
data_3x3_poi= np.load('1D_48x20000x9_temp_fullchip_inter_minmax.npy')

#data_3x3_poi= np.load('1D_48x20000x9_temp_fullchip_inter_minmax.npy')

train_data, test_data = np.split(data_3x3_poi, [17000], axis=1)
# scaler =MinMaxScaler(feature_range=(0, 1))
# train_data = scaler.fit_transform (train_data_ori)
# test_data =scaler.transform(test_data_ori)

plaintext = load_input('multi_tile_ft_plaintext.csv')
plaintext=plaintext[0:20000,0:16]
pt_train, pt_test =  np.split(plaintext, [17000])

label = SBOX_NP[key^plaintext]
label=label[0:20000,0:16];
label_train, label_test =  np.split(label, [17000])
  

for m in range(0,1):
    
        
    for n in range(15,16):

        byte_number = n
        records_for_all_epochs=[]
        np_records_for_all_epochs=[]
        epochs=200
        batch_size=100
        
        y_train=to_categorical(label_train[:,byte_number] ,num_classes=256)
        
        b0_train, b0_test= train_data[0], test_data[0]
        b1_train, b1_test= train_data[1], test_data[1]
        b2_train, b2_test= train_data[2], test_data[2]
        b3_train, b3_test= train_data[3], test_data[3]
        b4_train, b4_test= train_data[4], test_data[4]
        b5_train, b5_test= train_data[5], test_data[5]
        b6_train, b6_test= train_data[6], test_data[6]
        b7_train, b7_test= train_data[7], test_data[7]
        b8_train, b8_test= train_data[8], test_data[8]
        b9_train, b9_test= train_data[9], test_data[9]
        b10_train, b10_test= train_data[10], test_data[10]
        b11_train, b11_test= train_data[11], test_data[11]
        b12_train, b12_test= train_data[12], test_data[12]
        b13_train, b13_test= train_data[13], test_data[13]
        b14_train, b14_test= train_data[14], test_data[14]
        b15_train, b15_test= train_data[15], test_data[15]
        b16_train, b16_test= train_data[16], test_data[16]
        b17_train, b17_test= train_data[17], test_data[17]
        b18_train, b18_test= train_data[18], test_data[18]
        b19_train, b19_test= train_data[19], test_data[19]
        b20_train, b20_test= train_data[20], test_data[20]
        b21_train, b21_test= train_data[21], test_data[21]
        b22_train, b22_test= train_data[22], test_data[22]
        b23_train, b23_test= train_data[23], test_data[23]
        b24_train, b24_test= train_data[24], test_data[24]
        b25_train, b25_test= train_data[25], test_data[25]
        b26_train, b26_test= train_data[26], test_data[26]
        b27_train, b27_test= train_data[27], test_data[27]
        b28_train, b28_test= train_data[28], test_data[28]
        b29_train, b29_test= train_data[29], test_data[29]
        b30_train, b30_test= train_data[30], test_data[30]
        b31_train, b31_test= train_data[31], test_data[31]        
        b32_train, b32_test= train_data[32], test_data[32]
        b33_train, b33_test= train_data[33], test_data[33]
        b34_train, b34_test= train_data[34], test_data[34]
        b35_train, b35_test= train_data[35], test_data[35]
        b36_train, b36_test= train_data[36], test_data[36]
        b37_train, b37_test= train_data[37], test_data[37]
        b38_train, b38_test= train_data[38], test_data[38]
        b39_train, b39_test= train_data[39], test_data[39]
        b40_train, b40_test= train_data[40], test_data[40]
        b41_train, b41_test= train_data[41], test_data[41]
        b42_train, b42_test= train_data[42], test_data[42]
        b43_train, b43_test= train_data[43], test_data[43]
        b44_train, b44_test= train_data[44], test_data[44]
        b45_train, b45_test= train_data[45], test_data[45]
        b46_train, b46_test= train_data[46], test_data[46]
        b47_train, b47_test= train_data[47], test_data[47]
        
        byte_0_input = Input(shape=(9,), name='0_input')
        byte_0_output = Dense(32, activation= mish)(byte_0_input)
        byte_1_input = Input(shape=(9,), name='1_input')
        byte_1_output = Dense(32, activation= mish)(byte_1_input)
        byte_2_input = Input(shape=(9,), name='2_input')
        byte_2_output = Dense(32, activation= mish)(byte_2_input)
        byte_3_input = Input(shape=(9,), name='3_input')
        byte_3_output = Dense(32, activation= mish)(byte_3_input)
        byte_4_input = Input(shape=(9,), name='4_input')
        byte_4_output = Dense(32, activation= mish)(byte_4_input)
        byte_5_input = Input(shape=(9,), name='5_input')
        byte_5_output = Dense(32, activation= mish)(byte_5_input)
        byte_6_input = Input(shape=(9,), name='6_input')
        byte_6_output = Dense(32, activation= mish)(byte_6_input)
        byte_7_input = Input(shape=(9,), name='7_input')
        byte_7_output = Dense(32, activation= mish)(byte_7_input)
        byte_8_input = Input(shape=(9,), name='8_input')
        byte_8_output = Dense(32, activation= mish)(byte_8_input)
        byte_9_input = Input(shape=(9,), name='9_input')
        byte_9_output = Dense(32, activation= mish)(byte_9_input)
        byte_10_input = Input(shape=(9,), name='10_input')
        byte_10_output = Dense(32, activation= mish)(byte_10_input)
        byte_11_input = Input(shape=(9,), name='11_input')
        byte_11_output = Dense(32, activation= mish)(byte_11_input)
        byte_12_input = Input(shape=(9,), name='12_input')
        byte_12_output = Dense(32, activation= mish)(byte_12_input)
        byte_13_input = Input(shape=(9,), name='13_input')
        byte_13_output = Dense(32, activation= mish)(byte_13_input)
        byte_14_input = Input(shape=(9,), name='14_input')
        byte_14_output = Dense(32, activation= mish)(byte_14_input)
        byte_15_input = Input(shape=(9,), name='15_input')
        byte_15_output = Dense(32, activation= mish)(byte_15_input)
        byte_16_input = Input(shape=(9,), name='16_input')
        byte_16_output = Dense(32, activation= mish)(byte_16_input)
        byte_17_input = Input(shape=(9,), name='17_input')
        byte_17_output = Dense(32, activation= mish)(byte_17_input)
        byte_18_input = Input(shape=(9,), name='18_input')
        byte_18_output = Dense(32, activation= mish)(byte_18_input)
        byte_19_input = Input(shape=(9,), name='19_input')
        byte_19_output = Dense(32, activation= mish)(byte_19_input)
        byte_20_input = Input(shape=(9,), name='20_input')
        byte_20_output = Dense(32, activation= mish)(byte_20_input)
        byte_21_input = Input(shape=(9,), name='21_input')
        byte_21_output = Dense(32, activation= mish)(byte_21_input)
        byte_22_input = Input(shape=(9,), name='22_input')
        byte_22_output = Dense(32, activation= mish)(byte_22_input)
        byte_23_input = Input(shape=(9,), name='23_input')
        byte_23_output = Dense(32, activation= mish)(byte_23_input)
        byte_24_input = Input(shape=(9,), name='24_input')
        byte_24_output = Dense(32, activation= mish)(byte_24_input)
        byte_25_input = Input(shape=(9,), name='25_input')
        byte_25_output = Dense(32, activation= mish)(byte_25_input)
        byte_26_input = Input(shape=(9,), name='26_input')
        byte_26_output = Dense(32, activation= mish)(byte_26_input)
        byte_27_input = Input(shape=(9,), name='27_input')
        byte_27_output = Dense(32, activation= mish)(byte_27_input)
        byte_28_input = Input(shape=(9,), name='28_input')
        byte_28_output = Dense(32, activation= mish)(byte_28_input)
        byte_29_input = Input(shape=(9,), name='29_input')
        byte_29_output = Dense(32, activation= mish)(byte_29_input)
        byte_30_input = Input(shape=(9,), name='30_input')
        byte_30_output = Dense(32, activation= mish)(byte_30_input)
        byte_31_input = Input(shape=(9,), name='31_input')
        byte_31_output = Dense(32, activation= mish)(byte_31_input)
        byte_32_input = Input(shape=(9,), name='32_input')
        byte_32_output = Dense(32, activation= mish)(byte_32_input)
        byte_33_input = Input(shape=(9,), name='33_input')
        byte_33_output = Dense(32, activation= mish)(byte_33_input)
        byte_34_input = Input(shape=(9,), name='34_input')
        byte_34_output = Dense(32, activation= mish)(byte_34_input)
        byte_35_input = Input(shape=(9,), name='35_input')
        byte_35_output = Dense(32, activation= mish)(byte_35_input)
        byte_36_input = Input(shape=(9,), name='36_input')
        byte_36_output = Dense(32, activation= mish)(byte_36_input)
        byte_37_input = Input(shape=(9,), name='37_input')
        byte_37_output = Dense(32, activation= mish)(byte_37_input)
        byte_38_input = Input(shape=(9,), name='38_input')
        byte_38_output = Dense(32, activation= mish)(byte_38_input)
        byte_39_input = Input(shape=(9,), name='39_input')
        byte_39_output = Dense(32, activation= mish)(byte_39_input)
        byte_40_input = Input(shape=(9,), name='40_input')
        byte_40_output = Dense(32, activation= mish)(byte_40_input)
        byte_41_input = Input(shape=(9,), name='41_input')
        byte_41_output = Dense(32, activation= mish)(byte_41_input)
        byte_42_input = Input(shape=(9,), name='42_input')
        byte_42_output = Dense(32, activation= mish)(byte_42_input)
        byte_43_input = Input(shape=(9,), name='43_input')
        byte_43_output = Dense(32, activation= mish)(byte_43_input)
        byte_44_input = Input(shape=(9,), name='44_input')
        byte_44_output = Dense(32, activation= mish)(byte_44_input)
        byte_45_input = Input(shape=(9,), name='45_input')
        byte_45_output = Dense(32, activation= mish)(byte_45_input)
        byte_46_input = Input(shape=(9,), name='46_input')
        byte_46_output = Dense(32, activation= mish)(byte_46_input)
        byte_47_input = Input(shape=(9,), name='47_input')
        byte_47_output = Dense(32, activation= mish)(byte_47_input)
        
        
        
        
        
  
        fusion = multiply([byte_0_output, byte_1_output, byte_2_output, byte_3_output, byte_4_output, byte_5_output,
                           byte_6_output, byte_7_output, byte_8_output, byte_9_output, byte_10_output, byte_11_output, 
                           byte_12_output, byte_13_output, byte_14_output, byte_15_output, byte_16_output, byte_17_output, 
                           byte_18_output, byte_19_output, byte_20_output, byte_21_output, byte_22_output, byte_23_output,
                           byte_24_output, byte_25_output, byte_26_output, byte_27_output, byte_28_output, byte_29_output,
                           byte_30_output, byte_31_output, byte_32_output, byte_33_output, byte_34_output, byte_35_output,
                           byte_36_output, byte_37_output, byte_38_output, byte_39_output, byte_40_output, byte_41_output, 
                           byte_42_output, byte_43_output, byte_44_output, byte_45_output, byte_46_output, byte_47_output], name='Fusion_multiply_layer')
        
        dense1 = Dense(64, activation= mish )(fusion)
        batchnormal1= BatchNormalization()(dense1)
        dropout1=Dropout(0.5)(batchnormal1)
        
        dense2 = Dense(64, activation= mish )(dropout1)
        batchnormal2= BatchNormalization()(dense2)
        #dense5 = Dense(32, activation= mish )(dense4)
        #dense6 = Dense(32, activation= mish )(dense5)
        output = Dense(256, activation='softmax',kernel_regularizer= regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01))( batchnormal2)
        
        model_3x3 = Model(inputs=[byte_0_input, byte_1_input, byte_2_input, byte_3_input, byte_4_input, byte_5_input, 
                                  byte_6_input, byte_7_input, byte_8_input, byte_9_input, byte_10_input, byte_11_input, 
                                  byte_12_input, byte_13_input, byte_14_input, byte_15_input, byte_16_input, byte_17_input,
                                  byte_18_input, byte_19_input, byte_20_input, byte_21_input, byte_22_input, byte_23_input,
                                  byte_24_input, byte_25_input, byte_26_input, byte_27_input, byte_28_input, byte_29_input,
                                  byte_30_input, byte_31_input, byte_32_input, byte_33_input, byte_34_input, byte_35_input, 
                                  byte_36_input, byte_37_input, byte_38_input, byte_39_input, byte_40_input, byte_41_input, 
                                  byte_42_input, byte_43_input, byte_44_input, byte_45_input, byte_46_input, byte_47_input], outputs=output, name='Final_output')
        
        #radam=tfa.optimizers.RectifiedAdam(learning_rate=0.001,beta_1=0.95,epsilon=1e-05)
        radam=tfa.optimizers.RectifiedAdam(learning_rate=0.001,beta_1=0.95,epsilon=1e-05,total_steps=10000,warmup_proportion=0.1,min_lr=1e-4)
        ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
        
        model_3x3.compile(optimizer= ranger, loss='categorical_crossentropy', metrics=['accuracy'])
        model_3x3.summary()
        plot_model(model_3x3, to_file='48poi_3x3_inputs.png')
        
        POI_3x3= model_3x3.fit([b0_train, b1_train, b2_train, b3_train, b4_train, 
                                b5_train, b6_train, b7_train, b8_train, b9_train, 
                                b10_train, b11_train, b12_train, b13_train, b14_train, 
                                b15_train, b16_train, b17_train, b18_train, b19_train,
                                b20_train, b21_train, b22_train, b23_train, b24_train, 
                                b25_train, b26_train, b27_train, b28_train, b29_train,
                                b30_train, b31_train, b32_train, b33_train, b34_train, 
                                b35_train, b36_train, b37_train, b38_train, b39_train,
                                b40_train, b41_train, b42_train, b43_train, b44_train, 
                                b45_train, b46_train, b47_train], 
                               y_train, epochs=epochs, validation_split=0.085, batch_size= batch_size, callbacks=[SensitivitySpecificityCallback()])
        #POI_3x3= model_3x3.fit([b0_train, b1_train, b2_train, b3_train, b4_train, b5_train, b6_train, b7_train, b8_train, b9_train, b10_train, b11_train, b12_train, b13_train, b14_train, b15_train], y_train, epochs=epochs, validation_split=0.085, batch_size= batch_size)
        
        ranks = full_ranks(model_3x3, [b0_test, b1_test, b2_test, b3_test, b4_test, 
                                       b5_test, b6_test, b7_test, b8_test, b9_test, 
                                       b10_test, b11_test, b12_test, b13_test, b14_test,
                                       b15_test, b16_test, b17_test, b18_test, b19_test,
                                       b20_test, b21_test, b22_test, b23_test, b24_test,
                                       b25_test, b26_test, b27_test, b28_test, b29_test,
                                       b30_test, b31_test, b32_test, b33_test, b34_test,
                                       b35_test, b36_test, b37_test, b38_test, b39_test,
                                       b40_test, b41_test, b42_test, b43_test, b44_test,
                                       b45_test, b46_test, b47_test] , pt_test, byte_number, 0, 3001, 1)
        
        x = [ranks[i][0] for i in range(0, ranks.shape[0])]
        y = [ranks[i][1] for i in range(0, ranks.shape[0])]
        
        epoch_list=list(range(1, epochs+1))
        
        counter=3000
        for converge in range(len(y),0,-1):
        
            if y[converge-1]==0 :
                counter=converge
            elif (y[converge-1]!=0):
            
                break
        
        print("\n")
        if counter>2999:
            print("Can not converge within 3000traces")
        else:
            print("Converge at trace number: " +str(converge))
        fig=plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(1, 1, 1)
        major_ticks = np.arange(0, 1600, 100)
        minor_ticks = np.arange(0, 1600, 50)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
        ax.grid(which='minor', alpha=0.5)
        ax.grid(which='major', alpha=0.7)
        
        subtitle_str='Trial' + str(m)+ "Byte:" + str(byte_number) + " , Epochs:" + str(epochs) 
        plt.title(subtitle_str, fontsize=16)
        plt.suptitle("  Converge Trace: " +str(converge),fontsize=20,y=0.95)
        plt.xlabel('number of traces')
        plt.ylabel('rank')
        plt.grid(True)
        plt.plot(x, y)
        plt.savefig( 'multiply/'+'epoch'+ str(epochs)+ ' byte' + str(byte_number) + ' trial' + str(m)+' :trace vs rank '  )
        

        fig1=plt.figure(figsize=(15, 8))
        ax1 = fig.add_subplot(1, 1, 1)
        major_ticks1 = np.arange(0, epochs+10, 5)
        minor_ticks1 = np.arange(0, epochs+10, 1)
        ax1.set_xticks(major_ticks1)
        ax1.set_xticks(minor_ticks1, minor=True)
        ax1.set_yticks(major_ticks1)
        ax1.set_yticks(minor_ticks1, minor=True)
        ax1.grid(which='minor', alpha=0.5)
        ax1.grid(which='major', alpha=0.7)
        
        subtitle_str='Trial' + str(m)+ " Byte:" + str(byte_number) + " Required traces vs. Epochs:" + str(epochs) 
        plt.title(subtitle_str, fontsize=16)
        plt.suptitle("  Converge Trace: " +str(converge),fontsize=20,y=0.95)
        plt.xlabel('Epochs')
        plt.ylabel('Required traces')
        plt.grid(True)
        plt.plot(epoch_list, records_for_all_epochs)
        plt.savefig('multiply/'+'epoch'+ str(epochs)+ ' byte' + str(byte_number) + ' trial' + str(m)+ ' :epoch vs result'  )
        
        
        records_for_all_epochs.insert(0,byte_number)
        #np_records_for_all_trials.append(np_records_for_all_epochs.append (np.array(records_for_all_epochs)))
        np_records_for_all_trials.append(np.array(records_for_all_epochs))
        
        train_loss = POI_3x3.history['loss']
        val_loss   = POI_3x3.history['val_loss']
        train_acc  = POI_3x3.history['accuracy']
        val_acc    = POI_3x3.history['val_accuracy']
        xc         = range(epochs)


        fig2=plt.figure(figsize=(15, 8))
        plt.plot(xc, train_loss, 'r-',label='Training Loss')
        plt.plot(xc, val_loss,'b-',label='Validation Loss')
        plt.legend(loc="upper left")
        #plt.ylim(0, 2.0)
        #plt.plot(xc, records_for_all_epochs)
        plt.autoscale(tight=True) 
        plt.savefig('multiply/'+'epoch'+ str(epochs)+ ' byte' + str(byte_number) + ' trial' + str(m)+ ' :epoch vs loss '  )
        

        fig3=plt.figure(figsize=(15, 8))
        plt.plot(xc, train_acc,'r-',label='Training Accuracy')
        plt.plot(xc, val_acc,'b-',label='Validation Accuracy')
        plt.legend(loc="upper left")
        #plt.ylim(0, 2.0)
        #plt.plot(xc, records_for_all_epochs)
        plt.autoscale(tight=True) 
        plt.savefig('multiply/'+'epoch'+ str(epochs)+ ' byte' + str(byte_number) + ' trial' + str(m)+ ' :epoch vs acc '  )
np_records_for_all_trials=np.array(np_records_for_all_trials)

    