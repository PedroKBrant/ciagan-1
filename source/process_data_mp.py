# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 14:32:20 2022

@author: pkb
"""
import os
import numpy as np
import mediapipe as mp
import cv2
from os import listdir, mkdir
from os.path import isfile, join, isdir
from PIL import Image
import argparse
import itertools
import matplotlib.pyplot as plt

def get_lndm(path_img, path_out, start_id = 0, dlib_path=""):
    dir_proc = {'msk':'msk', 'org':'orig', 'clr':'clr', 'lnd':'lndm'}

    for dir_it in dir_proc:
        if os.path.isdir(path_out + dir_proc[dir_it]) == False:
            os.mkdir(path_out + dir_proc[dir_it])

    folder_list = [f for f in listdir(path_img)]
    folder_list.sort()

    line_px = 1
    res_w = 178
    res_h = 218
    NOSE = [1, 4, 5, 195, 197, 6]
    CONTOUR = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
               397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
               172, 58, 132, 137, 127, 162, 21, 54, 103, 67, 109]

    for fld in folder_list[:]:
        imglist_all = [f[:-4] for f in listdir(join(path_img, fld)) 
                                   if isfile(join(path_img, fld, f)) 
                                       and f[-4:] == ".jpg"]
        imglist_all.sort(key=int)
        imglist_all = imglist_all[start_id:]

        for dir_it in dir_proc:
            if os.path.isdir(join(path_out, dir_proc[dir_it], fld)) == False:
                os.mkdir(join(path_out, dir_proc[dir_it], fld))

        land_mask = True
        crop_coord = []
        for it in range(len(imglist_all)):
            clr = cv2.imread(join(path_img, fld, imglist_all[it]+".jpg"), cv2.IMREAD_ANYCOLOR)
            img_mp = clr.copy()
            
            # Face Mesh
            mp_face_mesh = mp.solutions.face_mesh
            face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, 
                                                     max_num_faces=1,
                                                     min_detection_confidence=0.5)
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            
            face_mesh_results = face_mesh_images.process(img_mp[:,:,::-1])#BGR to RGB
          
            img_copy = img_mp.copy()
            

            left_eye =  face_mesh_results.multi_face_landmarks[0].landmark[133]
            right_eye = face_mesh_results.multi_face_landmarks[0].landmark[362]
         
            # centering
            c_x = int(((right_eye.x + left_eye.x)*100) / 2)
            c_y = int(((right_eye.y + left_eye.y)*100) / 2)
            w_r = int(((right_eye.x - left_eye.x)*100)*4)
            h_r = int(((right_eye.x - left_eye.x)*100)*5)
            w_r = int(h_r/res_h*res_w)

            w, h = int(w_r * 2), int(h_r * 2)
            pd = int(w) # padding size
            
            img_p = np.zeros((img_copy.shape[0]+pd*2, img_copy.shape[1]+pd*2, 3), np.uint8) * 255
            img_p[:, :, 0] = np.pad(img_copy[:, :, 0], pd, 'constant')
            img_p[:, :, 1] = np.pad(img_copy[:, :, 1], pd, 'constant')
            img_p[:, :, 2] = np.pad(img_copy[:, :, 2], pd, 'constant')
            
            visual = img_p[c_y - h_r+pd:c_y + h_r+pd, c_x - w_r+pd:c_x + w_r+pd]

            crop_coord.append([c_y - h_r, c_y + h_r, c_x - w_r, c_x + w_r, pd, imglist_all[it]+".jpg"])
            t_x, t_y = int(c_x - w_r), int(c_y - h_r)

            ratio_w, ratio_h = res_w/w, res_h/h

            visual = cv2.resize(visual, dsize=(res_w, res_h), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(join(path_out, dir_proc['clr'], fld, imglist_all[it]+".jpg"), visual) #saving crop
            cv2.imwrite(join(path_out, dir_proc['org'], fld, imglist_all[it]+".jpg"), clr) # saving original
            
            face_mesh_results = face_mesh_images.process(visual[:,:,::-1])#BGR to RGB
            
            if face_mesh_results.multi_face_landmarks:
                img_lndm = np.ones((res_h, res_w, 3), np.uint8) * 255 #white image
                for face_landmarks in face_mesh_results.multi_face_landmarks:
                    
                    mp_drawing.draw_landmarks(image=img_lndm, landmark_list=face_landmarks,connections=mp_face_mesh.FACEMESH_CONTOURS,
                                              landmark_drawing_spec=None, 
                                              connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

            plt.title("Resultant Image");plt.axis('off');plt.imshow(img_lndm);plt.show() 
            
            result = Image.fromarray((img_lndm).astype(np.uint8))
            result.save(join(path_out, dir_proc['lnd'], fld, imglist_all[it]+".jpg"))
            
            
            img_msk = np.ones((res_h, res_w, 3), np.uint8) * 255         
            contours = np.array(CONTOUR)
            contours_lndm = np.zeros((0, 2),dtype=np.int32)
            
            for lndm in contours:
                arr = np.array([[((round(face_mesh_results.multi_face_landmarks[0].landmark[lndm].x*100) - t_x) * ratio_w), 
                                 ((round(face_mesh_results.multi_face_landmarks[0].landmark[lndm].y*100) - t_y) * ratio_h)]],
                                 dtype=np.int32)
                contours_lndm = np.concatenate((contours_lndm, arr))
                         
            cv2.fillPoly(img_msk, [contours_lndm], color=(0, 0, 0))
            img_msk = cv2.resize(img_msk, dsize=(res_w, res_h), interpolation=cv2.INTER_CUBIC)
            plt.title("Resultant Image");plt.axis('off');plt.imshow(img_msk);plt.show() 
            result = Image.fromarray((img_msk).astype(np.uint8))
            result.save(join(path_out, dir_proc['msk'], fld, imglist_all[it]+".jpg"))
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, help='directory with input data', default='../dataset/celeba/clr/')
    parser.add_argument('--output', type=str, help='directory for output', default='../output/')
    parser.add_argument('--dlib', type=str, help='directory with dlib predictor', default='')
    args = parser.parse_args()
    
    get_lndm(args.input, args.output, dlib_path=args.dlib)
    