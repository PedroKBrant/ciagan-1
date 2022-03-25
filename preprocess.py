# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 17:43:38 2022

@author: pkb
"""

import os
import shutil

dir = "exp24_/"

for file in os.listdir(dir):
    # get all but the last 8 characters to remove
    # the index number and extension
    dir_name = file.replace('.jpg', '')
    print(f'dir_name: {dir_name}')

    dir_path = dir + dir_name
    print(f'dir_path: {dir_path}')
    
    # check if directory exists or not yet
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    if os.path.exists(dir_name):
        file_path = dir + file
        print(f'file_path: {file_path}')
        
        # move files into created directory
        shutil.move(file_path, dir_name)