# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:20:04 2022

@author: pkb
"""

from os import listdir, mkdir
import os
import numpy as np
import cv2
import shutil

input_dir = r"../dataset/celeba_id/"
output_dir = r"../dataset/celeba/"
right_folder = np.load('../dataset/legit_indices.npy')
folder_list = [f for f in listdir('../dataset/celeba_id/')]
for fld in folder_list[:]:
    if int(fld) in right_folder:
        save_dir = os.path.join(output_dir, fld)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for file in os.listdir(input_dir+fld):
            source = os.path.join(input_dir, fld, file)
            destination = os.path.join(output_dir, fld, file)
            shutil.copy2(source, destination)