# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 17:29:47 2022

@author: pkb
"""
from os import listdir, mkdir
import os
#clr, lndm, msk, orig
input_dir = r"../dataset/exp24_mp/msk/"
#output_dir = r"../dataset/celeba/"
folder_list = [f for f in listdir(input_dir)]
index = 0
for fld in folder_list[:]:
    os.rename(input_dir+fld, input_dir+fld.zfill(5))
    
folder_list = [f for f in listdir(input_dir)]
for fld in folder_list[:]: 
    file = str(int(fld))
    if(file != str(index)):
        os.rename(input_dir+fld+'/'+file+'.jpg', input_dir+fld+'/'+str(index)+'.jpg')
    os.rename(input_dir+fld+'/'+file+'.jpg', input_dir+fld+'/'+str(index).zfill(6)+'.jpg')    
    os.rename(input_dir+fld, input_dir+str(index))
    index+=1
    
"""
from os import listdir, mkdir
import os

input_dir = r"../dataset/exp24_mp/clr/"
#output_dir = r"../dataset/celeba/"
folder_list = [f for f in listdir(input_dir)]
index = 0
for fld in folder_list[:]:
    os.rename(input_dir+fld, input_dir+fld.zfill(5))
    
folder_list = [f for f in listdir(input_dir)]
for fld in folder_list[:]:  
    os.rename(input_dir+fld, input_dir+str(index))
    index+=1
    
"""