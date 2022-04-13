# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 11:28:30 2022

@author: pkb
"""
import numpy as np
import os
import cv2
th = 20

def crop(image):
    y_nonzero, x_nonzero, _ = np.nonzero(image>th)
    img = image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]
    return img

folder_dir = "dataset/exp04_r/"

for image in os.listdir(folder_dir):
    img_path = os.path.join(folder_dir,image)
    img = cv2.imread(img_path)    
    img = crop(img)
    resized = cv2.resize(img, (100,100))
    cv2.imwrite(img_path, resized)  
