# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 17:43:38 2022

@author: pkb
"""

import os
import shutil

dir = "exp01/"

count = 0
for file in os.listdir(dir):

    filename = str(count)+".jpg"
    os.rename(dir+file, f"exp01/"+filename)
    dir_name = str(count)
    count +=1
    print(f'dir_name: {dir_name}')

    dir_path = dir + dir_name
    print(f'dir_path: {dir_path}')
    
    # check if directory exists or not yet
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    if os.path.exists(dir_name):
        file_path = dir + str(filename)
        print(f'file_path: {file_path}')
        
        # move files into created directory
        shutil.move(file_path, dir_name)
    