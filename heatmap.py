#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 14:09:13 2021

@author: sora

"""

import tensorflow as tf
import tensorflow_addons as tfa
import cv2
from keras import backend as K
from keras.preprocessing import image
from keras.applications import imagenet_utils
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler,QuantileTransformer
from tensorflow.keras.utils import to_categorical
tf.compat.v1.disable_eager_execution()
import sys
data_2D_full=np.load('/usr/python/full_chip_temperature/thermal_20000_inter.npy')

epochs=5000
byte_number=15
test_trace=391

tfa.register_all()
reserved_num_trace=3000
#path='/usr/python/full_chip_temperature/2D_fullchip_CNN_crack/new_cnn_new_new_new/123/crack_modelb' + str(test_byte) + 'trial0'
path='/usr/python/full_chip_temperature/2D_fullchip_CNN_crack/cnn_new/crack_model b15e230trial0vgg'
model=tf.keras.models.load_model(path)
train_data, test_data = np.split(data_2D_full, [20000-reserved_num_trace ])
scaler =MinMaxScaler(feature_range=(0, 1))

train_data=train_data.reshape((17000,40401))
train_data = scaler.fit_transform (train_data)
train_data=train_data.reshape((17000,201,201))

test_data=test_data.reshape((3000,40401))
test_data =scaler.transform(test_data)
test_data=test_data.reshape((3000,201,201))

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
    input_data=dataset

	# Predict our probabilities
    predictions = model.predict(input_data)
    # predictions1= model.predict(input_data)
    # top=5
    # y_test1d=np.reshape(metadata[0:3000,byte_number],(3000,1))
    # max_k_preds = predictions1.argsort(axis=1)[:, -top:][:, ::-1]
    
    # match_array = np.logical_or.reduce(max_k_preds==y_test1d, axis=1)
    
    # topk_acc_score = match_array.sum() / match_array.shape[0]
    # print('Test_Top 5 Acc:' + str(topk_acc_score))
    

    index = np.arange(min_trace_idx + rank_step, max_trace_idx, rank_step)
    
    f_ranks = np.zeros((len(index), 2), dtype=np.uint32)
    key_bytes_proba = []
    for t, i in zip(index, range(0, len(index))):
        real_key_rank, key_bytes_proba = rank(predictions[t-rank_step:t], metadata, byte_number, real_key, t-rank_step, t, key_bytes_proba)
        f_ranks[i] = [t - min_trace_idx, real_key_rank]

    return f_ranks


def plot_heatmap(heatmap, img_path,byte_number):
    # ReLU
    heatmap = np.maximum(heatmap, 0)
    
    # 正規化
    heatmap /= np.max(heatmap)
    
    # 讀取影像
    #img = cv2.imread(img_path)
    img=img_path
    im=img_path
    
    fig, ax = plt.subplots()
    
    #im = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), (img.shape[1], img.shape[0]))

    # 拉伸 heatmap
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap = np.uint8(255 * heatmap)
    
    # 以 0.6 透明度繪製原始影像
    ax.imshow(im, alpha=0.5, cmap='hot')
    
    # 以 0.4 透明度繪製熱力圖
    ax.imshow(heatmap, cmap='jet', alpha=0.6)
    
    plt.title(byte_number)
    
    plt.show()
    
    return heatmap

def load_input(input_file):
    return np.loadtxt(input_file, dtype = np.int32,
                      converters = {_: lambda s: int(s, 16) for _ in range(16)})

plaintext_ori_no_shu = load_input('multi_tile_ft_plaintext.csv')
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

plaintext=plaintext_ori_no_shu[0:20000,0:16]
pt_train, pt_test =  np.split(plaintext, [20000-reserved_num_trace ])

label = SBOX_NP[key^plaintext]
label=label[0:20000,0:16];
label_train, label_test =  np.split(label, [20000-reserved_num_trace ])
y_test=to_categorical(label_test[:,byte_number] ,num_classes=256)
#evaluate= model.evaluate(test_data, y_test, batch_size=20, verbose=1)

trace_number=test_trace

final_heatmap=np.zeros((201,201))
for x in range (100):
    trace_number=x
    img_ori = test_data[trace_number,:,:]
    img = test_data[trace_number,:,:]
    
    processed_img=img.reshape((1,201,201,1))
    
    test_data_back=test_data
    tast_data_back=test_data_back.reshape((3000,201,201,1))
    
    preds_all_train = model.predict(tast_data_back)
    y_classes = preds_all_train.argmax(axis=-1)
    
    label_for_math=label_test[:,byte_number]
    evaluate= model.evaluate(tast_data_back, y_test, batch_size=20, verbose=1)
    
    for x in range (3000):
        if (label_for_math[x]==y_classes[x]):
            print(x)
            
    print(evaluate)
    
    
    ranks = full_ranks(model, tast_data_back, pt_test, byte_number, 0, reserved_num_trace +1 , 1)
            
    x = [ranks[i][0] for i in range(0, ranks.shape[0])]
    y = [ranks[i][1] for i in range(0, ranks.shape[0])]
    
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
    preds = model.predict(processed_img)
    max_label=np.argmax(preds[0])
    
    print(max_label)
    byte0_output=model.output[:,max_label]
    #byte0_output=model.output[]
    
    last_conv_layer = model.get_layer('block3_conv1')
    
    grads = K.gradients(byte0_output, last_conv_layer.output)[0] 
    
    pooled_grads = K.mean(grads, axis=(0, 1, 2)) 
    
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    
    pooled_grads_value, conv_layer_output_value = iterate([processed_img])
    
    
    for i in range(16):
    
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]  # 將特徵圖數組的每個通道乘以這個通道對大象類別重要程度
        
    
    heatmap = np.mean(conv_layer_output_value, axis=-1) 
    
    # heatmap = np.maximum(heatmap, 0)  # heatmap與0比較，取其大者
    
    # #heatmap /= np.max(heatmap)
    
    # plt.matshow(heatmap)
    
    # plt.show()



    byte_number_str='CNN heatmap for byte '+str(byte_number)
    final_hitmap=plot_heatmap(heatmap, img_ori, byte_number_str)
    
    final_heatmap=final_heatmap+final_hitmap