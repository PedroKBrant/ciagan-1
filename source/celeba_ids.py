# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 10:07:36 2022

@author: pkb
"""
import os
import numpy as np
import cv2
import shutil

def read_identity(identities_filename):
    identities = []
    with open(identities_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            identities.append(pair)
            #print(identities)
    return np.array(identities)

def create_identity(celebA_dir,identity_path,output_dir):

    img_identity = read_identity(os.path.expanduser(identity_path))
    print(img_identity[0][1])
    i=0
    for filename in os.listdir(celebA_dir):

        if filename == img_identity[i][0]:
            print("Good ", filename)
            # # ----create the sub folder in the output folder
            save_dir = os.path.join(output_dir,img_identity[i][1])
            # i += 1
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # # ----copy image source to its identity destination
            destination = os.path.join(output_dir, img_identity[i][1])
            source = os.path.join(celebA_dir, img_identity[i][0])
            i += 1
            shutil.copy2(source, destination)

if __name__ == '__main__':

    celebA_dir = r"../dataset/img_align_celeba/" #Add your directory where dataset exist
    output_dir = r"../dataset/celeba/" #Add directory where you want to save datasets with identities
    identity_path = r"../dataset/identity_CelebA.txt"

create_identity(celebA_dir,identity_path,output_dir)