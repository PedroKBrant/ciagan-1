# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 17:43:38 2022

@author: pkb
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

dir = "dataset/exp04_r/"

count = 0
for file in os.listdir(dir):

    #passo 1 dar zfill em todo o dataset para ordenar direitinho
    print("train_"+os.path.splitext(file)[0].split("_")[1].zfill(5)+"_aligned"+".jpg")
    os.rename(dir+file, f"dataset/exp04_r/"+"train_"+os.path.splitext(file)[0].split("_")[1].zfill(5)+"_aligned"+".jpg")
    #os.rename(dir+file, f"dataset/exp04_r/"+str(int(os.path.splitext(file)[0])+1).zfill(6)+".jpg")
    #passo 2, passar a parte de train para outra pasta e renome-la
    #os.rename(dir+file, f"dataset/exp04_r1/"+"test_"+os.path.splitext(file)[0][2:]+"_aligned"+".jpg")
    #os.rename(dir+file, f"dataset/exp04_r/"+"train_"+str(int(os.path.splitext(file)[0])-3067).zfill(5)+"_aligned"+".jpg")
    #("test_"+os.path.splitext(file)[0][2:]+"_aligned"+".jpg")