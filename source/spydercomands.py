# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 11:52:47 2022

@author: pkb
"""

plt.figure(0)
plt.imshow(img_dlib)
for pkb in landmarks.parts():
    pkb_x, pkb_y = pkb.x, pkb.y
    plt.plot(pkb_x, pkb_y, 'rx')
plt.imshow(img_msk, alpha=0.6)

plt.figure(0)
plt.imshow(img_dlib)
for pkb in range(17):
    pkb_x, pkb_y = landmarks.part(pkb).x, landmarks.part(pkb).y
    plt.plot(pkb_x, pkb_y, 'rx')

plt.imshow(img_msk, alpha=0.6)

plt.figure(0)
plt.imshow(img_dlib) 
pkb_x, pkb_y = landmarks.part(0).x, landmarks.part(0).y
plt.plot(pkb_x, pkb_y, 'rx')
pkb_x, pkb_y = landmarks.part(16).x, landmarks.part(16).y
plt.plot(pkb_x, pkb_y, 'rx')
plt.imshow(img_msk, alpha=0.6)