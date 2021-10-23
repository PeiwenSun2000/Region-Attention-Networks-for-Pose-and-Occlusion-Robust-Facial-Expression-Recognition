# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 10:30:30 2018

@author: shen1994
"""
import cv2
import os
import dlib
import numpy as np
from PIL import Image

# Dlib 预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/home/DataBase4/sunpeiwen/ODS_group/model/shape_predictor_68_face_landmarks.dat')

def crop_from_image(src_file, des_file, crop_size):
    img = cv2.imread(src_file)  
    # Dlib 检测
    faces = detector(img, 1)

    print("人脸数：", len(faces), '\n')
    max_height = 0
    for k, d in enumerate(faces):
        # 计算矩形大小
        # (x,y), (宽度width, 高度height)
        pos_start = tuple([d.left(), d.top()])
        pos_end = tuple([d.right(), d.bottom()])
        y1 = d.top() if d.top() > 0 else 0
        y2 = d.bottom() if d.bottom() > 0 else 0
        x1 = d.left() if d.left() > 0 else 0
        x2 = d.right() if d.right() > 0 else 0
        
        height = y2-y1
        width = x2-x1
        if (len(faces) >= 2):
            if (height >= max_height):
                max_height = height;
                img_crop = img[y1:y2,x1:x2]
                img_crop = cv2.resize(img_crop,crop_size,)
                cv2.imwrite(des_file, img_crop)
        else:
            if(len(faces) == 1):
                img_crop = img[y1:y2,x1:x2]
                img_crop = cv2.resize(img_crop,crop_size,)
                cv2.imwrite(des_file, img_crop)
            else:
                print('No face is detected.')
        if(k == len(faces)-1):
            max_height = 0
        

def folder_for_crop(db_folder, result_folder, crop_size):
    # if "FERplus" in db_folder:
    #     if not os.path.exists(result_folder):
    #         os.mkdir(result_folder)
    #     counter = 0
    #     for pers_file in os.listdir(db_folder):
    #         pers_folder = db_folder  + os.sep + pers_file
    #         dest_folder = result_folder + os.sep + pers_file
    #         if not os.path.exists(dest_folder):
    #             os.mkdir(dest_folder)
    #         for img_file in os.listdir(pers_folder):
    #             counter += 1
    #             src_img_path = pers_folder  + os.sep + img_file
    #             des_img_path = dest_folder  + os.sep + img_file
    #             print(des_img_path)
    #             if "png" not in src_img_path:
    #                 continue
    #             img = cv2.imread(src_img_path) 
    #             img = cv2.resize(img,crop_size,)
    #             cv2.imwrite(des_img_path, img)
    # else:
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
    counter = 0
    for pers_file in os.listdir(db_folder):
        pers_folder = db_folder  + os.sep + pers_file
        dest_folder = result_folder + os.sep + pers_file
        if not os.path.exists(dest_folder):
            os.mkdir(dest_folder)
        for img_file in os.listdir(pers_folder):
            counter += 1

            src_img_path = pers_folder  + os.sep + img_file
            des_img_path = dest_folder  + os.sep + img_file
            print(des_img_path)
            if "png" not in src_img_path:
                continue
            crop_from_image(src_img_path, des_img_path, crop_size)
            #print(counter)

def run():
    list_file=["FER2013Test","FER2013Train","FER2013Valid"]
    db_folder = "/home/DataBase3/sunpeiwen/RAN/FERplus_dir/dataset/FERPlus/data"
    result_folder = "/home/DataBase3/sunpeiwen/RAN/FERplus_dir/dataset/FERPlus_crop"
    folder_for_crop(db_folder, result_folder, (224, 224))
    
if __name__ == "__main__":
    run()


 
 
