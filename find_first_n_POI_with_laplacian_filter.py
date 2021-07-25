#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 08:59:23 2021

@author: sora
"""
#from fcmeans import FCM

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
from numpy.random import seed
import codecs
import copy
import sys
import time
from statistics import mean 
import os
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
from skimage import filters
import gc

now = round(time.time() * 1000)

def mark_sbox(ax,poi,grid_size,color,flag):
    loc = poi
    loc = np.array(loc).T
    for ii in range(len(loc[0])):
        #ax.text(loc[1][ii],loc[0][ii],str(ii),horizontalalignment='center',verticalalignment='center',color='r')
        patch = plt.Rectangle((loc[1][ii] - grid_size//2 - 0.5, loc[0][ii] -grid_size//2 - 0.5), grid_size, grid_size, facecolor='none', edgecolor='black')
        if flag==True:
            ax.add_patch(patch)
        ax.text(loc[1][ii],loc[0][ii],str(ii),horizontalalignment='center',verticalalignment='center',color=color, size=7)

"""
first 1700 images as training data
last 3000 images for attacking (test data)
"""
attack_trace_number=3000
train_trace_number=17000
time_start = time.time() 

sbox_poi_index = [(94, 161), (65, 171), (70, 179), (93, 166), (92, 173), (81, 175), (65, 166), (65, 174), (70, 175), (88, 171), (81, 170), (59, 166), (66, 160), (64, 182), (88, 178), (89, 177)] 


"""
thermal data path
"""
data_2D_full=np.load('C:\\Users\\Sora\\Documents\\Python\\thermal_20000_inter.npy')


thermal_2000_inter_filtered= np.zeros((train_trace_number,201,201))



data_2D=data_2D_full[0]
rows,cols = data_2D.shape
sd_record=np.zeros((rows,cols))

"""
apply Laplacian filter to all training images, any other filters can be applied here
"""
for x in range (train_trace_number):
    trace= data_2D_full[x]
    filtered= filters.laplace(trace)
    thermal_2000_inter_filtered[x]= filtered

"""
calcualte each tile's standard deviation value and form a sstd map (sd_record)
"""
for x in range(rows):
    for y in range(cols):
        
        selected_trace_temp=[]
        
        for z in range(train_trace_number):
            trace=thermal_2000_inter_filtered[z]
            temperature=trace[x][y]
            selected_trace_temp.append(temperature)
        
        sd=statistics.pstdev(selected_trace_temp)
        sd_record[x][y]=sd

#save the standard deviation map
np.save(str(train_trace_number) + "_stdmap", sd_record)



std_list=[]
index_list=[]

for x in range (rows):
    for y in range(cols):
        std=sd_record[x][y]
        std_list.append(std)
        index_list.append((x,y))
        
temp=list(enumerate(std_list))
temp=sorted(temp,key= (lambda x: x[1]), reverse=True)
maximum=max(enumerate(std_list), key=(lambda x: x[1]))
        
first,snd= zip(*temp)

#start searching
overlap=[]

"""
aim_poi is how many POI you want to cover (starting from high std to low std)

grid_size is the searching window size, for example size 3 is 3x3 search size
and here we allow 1 pixel size overlapping for all the windows
"""
total_samples=17000
aim_poi=100
for x in range (total_samples):
    overlap.append(index_list[int(first[x])])
    
grid_size=[3]

for size in  grid_size:       
    poi_all_rounds=[]
    no_overlap=[]
    counter=0
    for x in range (total_samples):

        flag=True
        if x==0:
            element= index_list[int(first[x])]
            no_overlap.append(element)
            counter=counter+1
        
        if x>0:
            x_index, y_index=zip(*no_overlap)
            if counter<aim_poi:
                
                accept_overlap_size=size-1
                
                for y in range(len(no_overlap)):

                    element= index_list[int(first[x])]

                    if abs(x_index[y]-element[0])<accept_overlap_size  and abs(y_index[y]-element[1])<accept_overlap_size :

                        flag=False

                if flag==True:

                    no_overlap.append(element)
                    counter=counter+1
                    flag=True
                    
    poi_all_rounds=poi_all_rounds+ no_overlap                
    
    #Plotting
    fig_sbox,ax_sbox = plt.subplots(figsize=(40,40))
    im1=ax_sbox.imshow(sd_record , cmap=plt.cm.hot_r)
    plt.colorbar(im1)
    mark_sbox(ax_sbox,poi_all_rounds,size,color='dodgerblue',flag=True )
    mark_sbox(ax_sbox,sbox_poi_index,1,color='lime', flag=False)
    
    plt.savefig(str(train_trace_number)+'_graph') 
    plt.close('all')
    
    """
    save the found coordinates in a txt file
    """
    with open("gridsize_" +str(size)+ 'sample no'+ str(train_trace_number) + '.txt', 'w') as f:
        counter=0
        for item in poi_all_rounds:
            
            f.write(str(counter)+str(item))
            counter=counter+1
        f.write('\n')
        for item in poi_all_rounds:
            f.write(str(item)+",")

        
        
    