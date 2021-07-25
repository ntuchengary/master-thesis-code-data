#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 08:59:23 2021

@author: sora
"""
#from fcmeans import FCM
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tkinter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
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
from tensorflow.keras.models import load_model  # or another method - but this one is simpliest
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier  
from IPython.core.display import HTML
from IPython.display import display
from sklearn.cluster import KMeans, AgglomerativeClustering
import gc
import eli5
from eli5.sklearn import PermutationImportance
tf.random.set_seed(5)

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

data_2D_full=np.load('C:\\Users\\Sora\\Documents\\Python\\thermal_20000_inter.npy')

thermal_2000_inter_filtered= np.zeros((multiply,201,201))


"""
coordinates you selected from stamdard deviation
"""
poi_all_rounds_all=[(73, 172),(73, 171),(72, 169),(73, 169),(70, 172),
                (70, 169),(69, 158),(68, 158),(70, 157),(68, 160),
                (71, 170),(85, 166),(84, 163),(71, 173),(70, 158),
                (70, 171),(73, 170),(82, 161),(89, 159),(72, 173),
                (72, 171),(90, 163),(82, 163),(88, 161),(71, 169),
                (72, 172),(84, 162),(65, 171),(68, 161),(87, 160),
                (88, 178),(89, 163),(88, 159),(93, 166),(63, 182),
                (89, 161),(70, 160),(59, 166),(89, 176),(68, 157),
                (68, 156),(88, 171),(88, 170),(71, 171),(64, 182),
                (90, 176),(87, 162),(65, 166),(94, 161),(68, 112),
                (64, 181),(72, 170),(66, 170),(81, 150),(92, 173),
                (93, 162),(58, 166),(67, 157),(69, 156),(94, 162),
                (85, 151),(87, 159),(81, 175),(85, 153),(88, 179),
                (69, 159),(90, 161),(81, 174),(81, 163),(85, 154),
                (65, 174),(66, 171),(71, 172),(69, 157),(81, 170),
                (70, 159),(68, 159),(70, 178),(67, 158),(82, 167),
                (66, 166),(70, 173),(91, 162),(83, 163),(88, 152),
                (83, 139),(94, 166),(69, 162),(69, 160),(83, 164),
                (70, 170),(93, 174),(89, 160),(88, 154),(73, 173),
                (82, 166),(84, 114),(69, 161),(81, 162),(70, 179),
                (71, 137),(85, 152),(71, 175),(82, 131),(83, 166),
                (88, 160),(74, 172),(81, 171),(71, 174),(80, 175),
                (82, 162),(90, 177),(80, 170),(70, 161),(89, 158),
                (67, 160),(81, 164),(83, 169),(88, 162),(58, 167),
                (81, 114),(66, 159),(89, 177),(74, 171),(93, 161),
                (71, 178),(81, 167),(70, 174),(90, 142),(73, 168),
                (64, 174),(72, 168),(89, 162),(83, 167),(74, 169),
                (64, 175),(71, 142),(83, 162),(69, 172),(88, 177),
                (84, 166),(70, 156),(58, 154),(88, 150),(80, 146),
                (67, 159),(93, 173),(70, 168),(89, 179),(88, 163),
                (92, 174),(69, 169),(80, 171),(81, 132),(66, 160),
                (87, 161),(71, 163),(70, 175),(68, 119),(88, 164),
                (57, 155),(69, 135),(85, 114),(60, 150),(80, 176),
                (87, 158),(71, 157),(82, 113),(71, 158),(67, 161),
                (81, 166),(65, 160),(73, 156),(88, 122),(90, 160),
                (92, 172),(80, 113),(84, 167),(71, 179),(88, 158),
                (69, 171),(71, 140),(84, 164),(69, 114),(87, 171),
                (89, 178),(81, 161),(82, 169),(64, 171),(74, 170),
                (71, 168),(91, 163),(65, 167),(83, 161),(85, 142),
                (64, 132),(91, 161),(66, 121),(90, 162),(82, 124)]


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



#sequential forward selecting
order_list=[]
poi_all_rounds= []
order_loss=[]
while len(poi_all_rounds_all) >0 :

    totla=len(poi_all_rounds_all)
    crack_results=np.zeros((int(totla),9)) 

    for trials in range (0, len(poi_all_rounds_all) ,1):

        test_byte_list= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    
        poi_all_rounds_temp= poi_all_rounds + [poi_all_rounds_all[trials]]

        aim_poi=len(poi_all_rounds_temp)
        
    
        
        np_records_for_all_trials=[]
        
        byte_trace=np.zeros((20000,aim_poi))
        
        """
        form the new data from selected subset
        """
        for poi_num in range (len(poi_all_rounds_temp)):
            all_trace=[]
            for trace_num in range (20000):
               
                a=poi_all_rounds_temp[poi_num ][0]
                b=poi_all_rounds_temp[poi_num ][1]
                temp=data_2D_full[trace_num,a,b]
                all_trace.append(temp)
        
            byte_trace[:,poi_num]=all_trace


        
        plaintext_ori_no_shu = load_input('multi_tile_ft_plaintext.csv')
        plaintext=plaintext_ori_no_shu[0:20000,0:16]
    
        train_data_ori, test_data_ori = np.split(byte_trace, [20000-reserved_num_trace ])
        scaler =MinMaxScaler(feature_range=(0, 1))
        train_data = scaler.fit_transform (train_data_ori)
        test_data =scaler.transform(test_data_ori)
    
    
        pt_train, pt_test =  np.split(plaintext, [20000-reserved_num_trace ])
    
        label = SBOX_NP[key^plaintext]
        label=label[0:20000,0:16];
        label_train, label_test =  np.split(label, [20000-reserved_num_trace ])
      
    
        del_flag=True
        for n in test_byte_list :
            
    
            
            byte_number = n
            records_for_all_epochs=[]
            np_records_for_all_epochs=[]
            epochs=1000
            batch_size=256
            
            y_train=to_categorical(label_train[:,byte_number] ,num_classes=256)
            y_test=to_categorical(label_test[:,byte_number] ,num_classes=256)
            
            """
            NN training
            """
            hidden_neuron=20
            input_shape = (aim_poi,)
            trace_input = Input(shape=input_shape)
            
            dense1 = Dense(hidden_neuron, activation= mish)(trace_input)
            batchnormal1= BatchNormalization()(dense1)
            dense2 = Dense(hidden_neuron, activation= mish)(batchnormal1)
            batchnormal2= BatchNormalization()(dense2)
            
            output = Dense(256, activation='softmax' ,kernel_regularizer=regularizers.l2(0.1),activity_regularizer=regularizers.l2(0.1))(batchnormal2)
    
            inputs = trace_input
            model= Model(inputs=inputs, outputs=output, name='FC_output')

            radam=tfa.optimizers.RectifiedAdam(learning_rate=0.001,beta_1=0.95,epsilon=1e-05,total_steps=50000,warmup_proportion=0.1,min_lr=1e-5)
            ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
            

            model.compile(optimizer= ranger, loss='categorical_crossentropy', metrics=["top_k_categorical_accuracy"])
    
            callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=100, restore_best_weights=True)
            #,min_delta=0.0000
            now=time.time()
            Sbox16= model.fit(train_data, y_train, epochs=epochs, validation_split=0.2, batch_size= batch_size, callbacks=[callback],verbose=0)
            

            now1=time.time()
            usedtime=now1-now
            print('byte' + str(byte_number))
            print((usedtime))
            
            """
            attacking
            """
            ranks = full_ranks(model, test_data, pt_test, byte_number, 0, reserved_num_trace +1 , 1)
            
            x = [ranks[i][0] for i in range(0, ranks.shape[0])]
            y = [ranks[i][1] for i in range(0, ranks.shape[0])]
            
            
            
            evaluate= model.evaluate(test_data, y_test, batch_size=3000, verbose=1)
            
            test_acc  = evaluate[1]
            
            test_loss= evaluate[0]
            print('test_val:' +str(test_loss))
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
            
            crack_results[trials][0]=trials
            crack_results[trials][1]=val_loss[-101]
            crack_results[trials][2]=val_acc[-101]
            crack_results[trials][3]=test_loss
            crack_results[trials][4]=test_acc
    
            crack_results[trials][5]=converge
            crack_results[trials][6]=y[2999]
            crack_results[trials][7]=usedtime
            crack_results[trials][8]=len(poi_all_rounds_temp)
            
    """
    Order list and order loss are the final feature selected from SFS method and its coorsponding loss
    choose the best subset with lowest loss 
    """
    min_loss_index= crack_results.argmin(axis=0) [1]
    order_loss= order_loss + [crack_results[min_loss_index][1]]
    poi_all_rounds=  poi_all_rounds + [poi_all_rounds_all[min_loss_index]]
    print( poi_all_rounds )
    print()
    order_list=order_list + [poi_all_rounds_all[min_loss_index]]
    del poi_all_rounds_all[min_loss_index]
        
    