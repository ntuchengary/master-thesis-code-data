"""
Multi-gate Mixture-of-Experts demo with census income data.

Copyright (c) 2018 Drawbridge, Inc
Licensed under the MIT License (see LICENSE for details)
Written by Alvin Deng
"""

import random
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import VarianceScaling, HeUniform
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler,RobustScaler,QuantileTransformer
from mmoe import MMoE
from tensorflow_addons.activations import mish
import tensorflow_addons as tfa
import sys
from tensorflow.keras import regularizers
import time

#2 good

# Fix numpy seed for reproducibility
#np.random.seed(SEED)

# Fix random seed for reproducibility
#random.seed(SEED)

# Fix TensorFlow graph-level seed for reproducibility
tf.random.set_seed(2)
tf_session = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
tf.compat.v1.keras.backend.set_session(tf_session)

class SensitivitySpecificityCallback(Callback):
    def __init__(self, patience=1000000):
        super(SensitivitySpecificityCallback, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
        self.converge_flag= True
    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf
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
    
        def full_ranks(model, dataset, metadata, byte_number_all, min_trace_idx, max_trace_idx, rank_step):
        	# Real key byte value that we will use. '2' is the index of the byte (third byte) of interest.
        	
            #real_key=['00','11','22','33','44','55','66','77','88','99','AA','BB','CC','DD','EE','FF']
            key=[0,17,34,51,68,85,102,119,136,153,170,187,204,221,238,255]
            
            predictions_all = self.model.predict(dataset)
            
            f_total_rank=[]
            for x in range (16):
                
                byte_number=byte_number_all[x]
                real_key=key[byte_number]
                predictions=predictions_all[x]
                index = np.arange(min_trace_idx + rank_step, max_trace_idx, rank_step)
                
                f_ranks = np.zeros((len(index), 2), dtype=np.uint32)
                key_bytes_proba = []
                for t, i in zip(index, range(0, len(index))):
                    real_key_rank, key_bytes_proba = rank(predictions[t-rank_step:t], metadata, byte_number, real_key, t-rank_step, t, key_bytes_proba)
                    f_ranks[i] = [t - min_trace_idx, real_key_rank]
                f_total_rank.append(f_ranks)
            return f_total_rank
                
        if epoch%5==1:
            now = round(time.time() * 1000)
            self.model.save(str(now))
            byte_number=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
            ranks = full_ranks(self, test_data, pt_test, byte_number, 0, reserved_num_trace+1 , 1)
            for z in range(16):
                x = [ranks[z][i][0] for i in range(0, reserved_num_trace)]
                y = [ranks[z][i][1] for i in range(0, reserved_num_trace)]
                
                epoch_list=list(range(1, epochs+1))
                
                counter= reserved_num_trace
                for converge in range(len(y),0,-1):
                
                    if y[converge-1]==0 :
                        counter=converge
                    elif (y[converge-1]!=0):
                    
                        break
                
                print("\n")
                if counter>(reserved_num_trace -1) :
                    print("Byte:" + str(z) +" Can not converge within " + str(reserved_num_trace) +" traces " + "Last Rank: " +str(y[4999]))
                else:
                    print("Byte:" + str(z) +"Converge at trace number: " +str(converge))
                #self.converge_flag=True
            now1 = round(time.time() * 1000)
            print(now1-now)
            print(now)
            #if converge<100:
               # self.converge_flag=False
                
        # if self.converge_flag==True:
        #     current = logs.get("val_loss")
        #     if np.less(current, self.best):
        #         self.best = current
        #         self.wait = 0
        #         # Record the best weights if current results is better (less).
        #         self.best_weights = self.model.get_weights()
        #     else:
        #         self.wait += 1
        #         if self.wait >= self.patience:
        #             self.stopped_epoch = epoch
        #             self.model.stop_training = True
        #             print("Restoring model weights from the end of the best epoch.")
        #             self.model.set_weights(self.best_weights)
        
            
   
    
def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)
# Simple callback to print out ROC-AUC
class ROCCallback(Callback):
    def __init__(self, training_data, validation_data, test_data):
        self.train_X = training_data[0]
        self.train_Y = training_data[1]
        self.validation_X = validation_data[0]
        self.validation_Y = validation_data[1]
        self.test_X = test_data[0]
        self.test_Y = test_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        train_prediction = self.model.predict(self.train_X)
        validation_prediction = self.model.predict(self.validation_X)
        test_prediction = self.model.predict(self.test_X)

        # Iterate through each task and output their ROC-AUC across different datasets
        for index, output_name in enumerate(self.model.output_names):
            train_roc_auc = roc_auc_score(self.train_Y[index], train_prediction[index])
            validation_roc_auc = roc_auc_score(self.validation_Y[index], validation_prediction[index])
            test_roc_auc = roc_auc_score(self.test_Y[index], test_prediction[index])
            print(
                'ROC-AUC-{}-Train: {} ROC-AUC-{}-Validation: {} ROC-AUC-{}-Test: {}'.format(
                    output_name, round(train_roc_auc, 4),
                    output_name, round(validation_roc_auc, 4),
                    output_name, round(test_roc_auc, 4)
                )
            )

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def load_input(input_file):
    return np.loadtxt(input_file, dtype = np.int32,
                      converters = {_: lambda s: int(s, 16) for _ in range(16)})

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

def full_ranks(model, dataset, metadata, byte_number_all, min_trace_idx, max_trace_idx, rank_step):
	# Real key byte value that we will use. '2' is the index of the byte (third byte) of interest.
	
    #real_key=['00','11','22','33','44','55','66','77','88','99','AA','BB','CC','DD','EE','FF']
    key=[0,17,34,51,68,85,102,119,136,153,170,187,204,221,238,255]
    
    

    #input_data = dataset[min_trace_idx:max_trace_idx, :]


	# Predict our probabilities
    predictions_all = model.predict(dataset)
    
    f_total_rank=[]
    for x in range (16):
        
        byte_number=byte_number_all[x]
        real_key=key[byte_number]
        predictions=predictions_all[x]
        index = np.arange(min_trace_idx + rank_step, max_trace_idx, rank_step)
        
        f_ranks = np.zeros((len(index), 2), dtype=np.uint32)
        key_bytes_proba = []
        for t, i in zip(index, range(0, len(index))):
            real_key_rank, key_bytes_proba = rank(predictions[t-rank_step:t], metadata, byte_number, real_key, t-rank_step, t, key_bytes_proba)
            f_ranks[i] = [t - min_trace_idx, real_key_rank]
        f_total_rank.append(f_ranks)
    return f_total_rank

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


data_2D_full=np.load('C:\\Users\\Sora\\Documents\\Python\\thermal_20000_inter.npy')


poi_all_rounds=[(73, 172),(73, 171),(72, 169),(73, 169),(70, 172),
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
                (57, 155),(69, 135),(85, 114),(60, 150),(80, 176),(87, 158),(71, 157),(82, 113),(71, 158),(67, 161),(81, 166),(65, 160),(73, 156),(88, 122),(90, 160),(92, 172),(80, 113),(84, 167),(71, 179),(88, 158),(69, 171),(71, 140),(84, 164),(69, 114),(87, 171),(89, 178),(81, 161),(82, 169),(64, 171),(74, 170),(71, 168),(91, 163),(65, 167),(83, 161),(85, 142),(64, 132),(91, 161),(66, 121),(90, 162),(82, 124)]



reserved_num_trace=3000
train_valid_split=13600
used_trace=17000
aim_poi=200
multiply=17000

np_records_for_all_trials=[]

byte_trace=np.zeros((20000,aim_poi))

for poi_num in range (len(poi_all_rounds)):
    all_trace=[]
    for trace_num in range (20000):
       
        a=poi_all_rounds[poi_num ][0]
        b=poi_all_rounds[poi_num ][1]
        temp=data_2D_full[trace_num,a,b]
        all_trace.append(temp)

    byte_trace[:,poi_num]=all_trace
#np.save('train_np\\' + str(multiply),byte_trace)

plaintext_ori_no_shu = load_input('multi_tile_ft_plaintext.csv')
plaintext=plaintext_ori_no_shu[0:20000,0:16]

train_data_ori, test_data_ori = np.split(byte_trace, [20000-reserved_num_trace ])
scaler =MinMaxScaler(feature_range=(0, 1))
#scaler= QuantileTransformer()
train_data_valid = scaler.fit_transform (train_data_ori)

train_data, val_data= np.split(train_data_valid , [train_valid_split ])

test_data =scaler.transform(test_data_ori)

pt_train, pt_test =  np.split(plaintext, [20000-reserved_num_trace ])

label = SBOX_NP[key^plaintext]
label=label[0:20000,0:16];
label_train, label_test =  np.split(label, [20000-reserved_num_trace ])
test_byte_list= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

# train_valid=train_data_orii
# test_data= test_data_ori
# train_data, val_data= np.split(train_valid, [train_valid_split ])

# train_valid = scaler.transform(train_data_orii)
# test_data =scaler.transform(test_data_ori)
# train_data, val_data = np.split(train_valid, [train_valid_split ])

b_0_y=to_categorical(label_train[:,0] ,num_classes=256)
b_0_y_train, b_0_y_valid =  np.split(b_0_y, [train_valid_split])

b_1_y=to_categorical(label_train[:,1] ,num_classes=256)
b_1_y_train, b_1_y_valid =  np.split(b_1_y, [train_valid_split])

b_2_y=to_categorical(label_train[:,2] ,num_classes=256)
b_2_y_train, b_2_y_valid =  np.split(b_2_y, [train_valid_split])


b_3_y=to_categorical(label_train[:,3] ,num_classes=256)
b_3_y_train, b_3_y_valid =  np.split(b_3_y, [train_valid_split])

b_4_y=to_categorical(label_train[:,4] ,num_classes=256)
b_4_y_train, b_4_y_valid =  np.split(b_4_y, [train_valid_split])


b_5_y=to_categorical(label_train[:,5] ,num_classes=256)
b_5_y_train, b_5_y_valid =  np.split(b_5_y, [train_valid_split])

b_6_y=to_categorical(label_train[:,6] ,num_classes=256)
b_6_y_train, b_6_y_valid =  np.split(b_6_y, [train_valid_split])

b_7_y=to_categorical(label_train[:,7] ,num_classes=256)
b_7_y_train, b_7_y_valid =  np.split(b_7_y, [train_valid_split])

b_8_y=to_categorical(label_train[:,8] ,num_classes=256)
b_8_y_train, b_8_y_valid =  np.split(b_8_y, [train_valid_split])

b_9_y=to_categorical(label_train[:,9] ,num_classes=256)
b_9_y_train, b_9_y_valid =  np.split(b_9_y, [train_valid_split])


b_10_y=to_categorical(label_train[:,10] ,num_classes=256)
b_10_y_train, b_10_y_valid =  np.split(b_10_y, [train_valid_split])

b_11_y=to_categorical(label_train[:,11] ,num_classes=256)
b_11_y_train, b_11_y_valid =  np.split(b_11_y, [train_valid_split])

b_12_y=to_categorical(label_train[:,12] ,num_classes=256)
b_12_y_train, b_12_y_valid =  np.split(b_12_y, [train_valid_split])

b_13_y=to_categorical(label_train[:,13] ,num_classes=256)
b_13_y_train, b_13_y_valid =  np.split(b_13_y, [train_valid_split])

b_14_y=to_categorical(label_train[:,14] ,num_classes=256)
b_14_y_train, b_14_y_valid =  np.split(b_14_y, [train_valid_split])

b_15_y=to_categorical(label_train[:,15] ,num_classes=256)
b_15_y_train, b_15_y_valid =  np.split(b_15_y, [train_valid_split])


b_0_yt=to_categorical(label_test[:,0] ,num_classes=256)
b_1_yt=to_categorical(label_test[:,1] ,num_classes=256)
b_2_yt=to_categorical(label_test[:,2] ,num_classes=256)
b_3_yt=to_categorical(label_test[:,3] ,num_classes=256)
b_4_yt=to_categorical(label_test[:,4] ,num_classes=256)
b_5_yt=to_categorical(label_test[:,5] ,num_classes=256)
b_6_yt=to_categorical(label_test[:,6] ,num_classes=256)
b_7_yt=to_categorical(label_test[:,7] ,num_classes=256)
b_8_yt=to_categorical(label_test[:,8] ,num_classes=256)
b_9_yt=to_categorical(label_test[:,9] ,num_classes=256)
b_10_yt=to_categorical(label_test[:,10] ,num_classes=256)
b_11_yt=to_categorical(label_test[:,11] ,num_classes=256)
b_12_yt=to_categorical(label_test[:,12] ,num_classes=256)
b_13_yt=to_categorical(label_test[:,13] ,num_classes=256)
b_14_yt=to_categorical(label_test[:,14] ,num_classes=256)
b_15_yt=to_categorical(label_test[:,15] ,num_classes=256)



y_train=[b_0_y_train,b_1_y_train,b_2_y_train,b_3_y_train,
         b_4_y_train,b_5_y_train,b_6_y_train,b_7_y_train,
         b_8_y_train,b_9_y_train,b_10_y_train,b_11_y_train,
         b_12_y_train,b_13_y_train,b_14_y_train,b_15_y_train]

y_valid=[b_0_y_valid,b_1_y_valid,b_2_y_valid,b_3_y_valid,
         b_4_y_valid,b_5_y_valid,b_6_y_valid,b_7_y_valid,
         b_8_y_valid,b_9_y_valid,b_10_y_valid,b_11_y_valid,
         b_12_y_valid,b_13_y_valid,b_14_y_valid,b_15_y_valid]

y_test=[b_0_yt,b_1_yt,b_2_yt,b_3_yt,
         b_4_yt,b_5_yt,b_6_yt,b_7_yt,
         b_8_yt,b_9_yt,b_10_yt,b_11_yt,
         b_12_yt,b_13_yt,b_14_yt,b_15_yt]

output_info=['byte0','byte1','byte2','byte3','byte4','byte5','byte6','byte7','byte8','byte9','byte10','byte11','byte12','byte13','byte14','byte15',]



num_features = train_data.shape[1]


print('Training data shape = {}'.format(train_data.shape))
print('Validation data shape = {}'.format(val_data.shape))
print('Test data shape = {}'.format(test_data.shape))

# Set up the input layer
input_layer = Input(shape=(num_features,))

epochs=3000
batchsize=256
number_experts=8
mmoe_units=20
tower_units=20
tower1_units=64

# Set up MMoE layer
mmoe_layers = MMoE(
    units= mmoe_units,
    num_experts=number_experts,
    num_tasks=16
)(input_layer)

output_layers = []

#Build tower layer from MMoE layer
for index, task_layer in enumerate(mmoe_layers):
    
    tower_layer = Dense(
        units=tower_units,
        activation=mish,
        kernel_initializer=HeUniform())(task_layer)
    
    batchnormal1= BatchNormalization()(tower_layer)
    
    # tower_layer1 = Dense(
    #     units=tower1_units,
    #     activation=mish,
    #     kernel_initializer=VarianceScaling())(batchnormal1)
    
    # batchnormal2= BatchNormalization()(tower_layer1)
    
    # tower_layer2 = Dense(
    #     units=tower1_units,
    #     activation=mish,
    #     kernel_initializer=VarianceScaling())(batchnormal2)
    
    output_layer = Dense(
        units=256,
        name=output_info[index],
        activation='softmax',kernel_initializer=VarianceScaling(),kernel_regularizer=regularizers.l2(0.1),
        activity_regularizer=regularizers.l2(0.1))(batchnormal1)
    #        
    #        kernel_initializer=VarianceScaling(),
    output_layers.append(output_layer)

# Compile model
model = Model(inputs=[input_layer], outputs=output_layers)
adam= tf.keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,name='Adam')
rmsdrop= tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.7, momentum=0.0, epsilon=1e-07, centered=False,name='RMSprop')
radam=tfa.optimizers.RectifiedAdam(learning_rate=0.001,beta_1=0.95,epsilon=1e-05,total_steps=10000,warmup_proportion=0.1,min_lr=1e-4)
ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=50,min_delta=0.005, restore_best_weights=True)

model.compile(
    loss={'byte0': 'categorical_crossentropy', 'byte1': 'categorical_crossentropy','byte2': 'categorical_crossentropy','byte3': 'categorical_crossentropy',
          'byte4': 'categorical_crossentropy', 'byte5': 'categorical_crossentropy','byte6': 'categorical_crossentropy','byte7': 'categorical_crossentropy',
          'byte8': 'categorical_crossentropy', 'byte9': 'categorical_crossentropy','byte10': 'categorical_crossentropy','byte11': 'categorical_crossentropy',
          'byte12': 'categorical_crossentropy', 'byte13': 'categorical_crossentropy','byte14': 'categorical_crossentropy','byte15': 'categorical_crossentropy'},
    optimizer=ranger,
    metrics=[tf.keras.metrics.TopKCategoricalAccuracy(k=5)]
)

# Print out model architecture summary
model.summary()

now=time.time()
global_mode=model.fit(
    x=train_data,
    y=y_train,
    validation_data=(val_data, y_valid),
    batch_size=batchsize,
    epochs=epochs,callbacks=[callback],verbose=1
)
now1=time.time()
print(now1-now)
byte_number=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
ranks = full_ranks(model, test_data, pt_test, byte_number, 0, reserved_num_trace +1 , 1)


for z in range(16):
    x = [ranks[z][i][0] for i in range(0, reserved_num_trace)]
    y = [ranks[z][i][1] for i in range(0, reserved_num_trace)]
    
    epoch_list=list(range(1, epochs+1))
    
    counter= reserved_num_trace
    for converge in range(len(y),0,-1):
    
        if y[converge-1]==0 :
            counter=converge
        elif (y[converge-1]!=0):
        
            break
    
    print("\n")
    if counter>(reserved_num_trace -1) :
        print("Byte:" + str(z) +" Can not converge within " + str(reserved_num_trace) +" traces " + "Last Rank: " +str(y[2999]))
    else:
        print("Byte:" + str(z) +"Converge at trace number: " +str(converge))
    
model.save('newnew')
train_loss = global_mode.history['loss']
val_loss   = global_mode.history['val_loss']
train_acc  = global_mode.history['byte0_top_k_categorical_accuracy']
val_acc    = global_mode.history['val_byte0_top_k_categorical_accuracy']
#byte0_loss = global_mode.history['']
epochs= len(train_loss)
xc         = range(epochs)

evaluate= model.evaluate(test_data, y_test, batch_size=3000, verbose=1)


for byte in range (16):
    val_acc    = global_mode.history['val_byte'+ str(byte)+'_top_k_categorical_accuracy']
    np.save('byte' + str(byte) + 'val_acc',val_acc   )
    
for byte in range (16):
    val_loss    = global_mode.history['val_byte' + str(byte) + '_loss']
    np.save('byte' + str(byte) + 'val_loss',val_loss   )
    
for byte in range (16):
    train_acc    = global_mode.history['byte'+ str(byte)+'_top_k_categorical_accuracy']
    np.save('byte' + str(byte) + 'train_acc',train_acc   )
    
for byte in range (16):
    train_loss    = global_mode.history['byte' + str(byte) + '_loss']
    np.save('byte' + str(byte) + 'train_loss',train_loss   )

whole_loss= global_mode.history['loss']
np.save('whole_train_loss',whole_loss  )

whole_val_loss= global_mode.history['val_loss']
np.save('whole_val_loss',whole_val_loss  )