#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 21:23:37 2021

@author: sora
"""
import numpy as np
#data_minmax_1D=np.load('1d_data_reset_minmax.npy')
#data_2D=np.load('2d_data_3x3.npy')


data_temp_fullchip =np.load('/usr/python/full_chip_temperature/thermal_20000_inter.npy')

poi_9=data_temp_fullchip.squeeze()



poi_index = [(94, 161), (65, 171), (70, 179), (93, 166), (92, 173), (81, 175), (65, 166), (65, 174), (70, 175), (88, 171), (81, 170), (59, 166), (66, 160), (64, 182), (88, 178), (89, 177)] 


# poi_index= [(73, 172),(73, 169),(69, 158),(70, 172),(70, 169),(84, 163),(85, 166),(89, 159),(90, 163),
#             (68, 161),(65, 171),(88, 178),(59, 166),(93, 166),(63, 182),(88, 170),(87, 162),(65, 166),
#             (94, 161),(68, 112),(92, 173),(81, 150),(81, 175),(85, 151),(81, 163),(65, 174),(82, 167),
#             (88, 152),(81, 170),(85, 154),(70, 178),(83, 139),(84, 114),(71, 137),(71, 175),(82, 131),
#             (81, 114),(90, 142),(71, 142),(58, 154),(80, 146),(71, 163),(65, 160),(60, 150),(68, 119),
#             (73, 156),(88, 122),(85, 142)]

byte_trace=np.zeros([len(poi_index),20000,9])



for poi_num in range (len(poi_index)):
    all_trace=[]
    for trace_num in range (20000):
       
        a=poi_index[poi_num ][0]
        b=poi_index[poi_num ][1]
        temp=[poi_9[trace_num,a-1, b-1], poi_9[trace_num,a-1, b], poi_9[trace_num,a-1, b+1], 
                        poi_9[trace_num,a, b-1]  ,poi_9[trace_num,a, b]   , poi_9[trace_num,a, b+1], 
                        poi_9[trace_num,a+1, b-1], poi_9[trace_num,a+1, b], poi_9[trace_num,a+1, b+1]]
        all_trace.append(temp)
        #all_trace=np.array(all_trace)
        #byte_trace=np.expand_dims(byte_trace,axis=poi_num)
    byte_trace[poi_num,:,:]=all_trace
#byte0_np= np.array(byte0_trace)
#np.save('1D_data_reset_byte0_3x3poi', byte0_np)

#byte0=byte_trace[0]

result = np.where(byte_trace == np.amin(byte_trace))
print('Returned tuple of arrays :', result)
print('List of Indices of minimum element :', result[0])

np.save('1D_16x20000x9_temp_fullchip_inter_sbox',byte_trace)