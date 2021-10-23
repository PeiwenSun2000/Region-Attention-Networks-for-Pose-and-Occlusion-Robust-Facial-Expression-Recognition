import os 

text_file_path1 = "./FERPlus_list/dlib_ferplus_train_center_crop_range_list.txt"
text_file_path2 = "./FERPlus_list/dlib_ferplus_val_center_crop_range_list.txt"
dict_folder={}
for item in [text_file_path1,text_file_path2]:
    with open(item, 'r') as imf:
        for line in imf:
            space_index = line.find(' ')
            video_name = line[0:space_index]
    #         print(video_name)
            list_path_split = video_name.split("/")
    #         print(list_path_split)
            dict_folder[list_path_split[1]]=list_path_split[0]




dict_folder

import cv2
from  matplotlib import pyplot as plt
def crop5region(pic_path,crop_save_path):
#     print(crop_save_path)
    img = cv2.imread(pic_path)
    top_y,top_x=img.shape[0],img.shape[1]
    
    whole_cropped = img
    whole_cropped = cv2.resize(whole_cropped,(224,224))
    top_left_cropped = img[0:round(top_y*0.75), 0:round(top_x*0.75)] # 裁剪坐标为[y0:y1, x0:x1]
    top_left_cropped = cv2.resize(top_left_cropped,(224,224))
    top_right_cropped = img[0:round(top_y*0.75), round(0.25*top_x):top_x] # 裁剪坐标为[y0:y1, x0:x1]
    top_right_cropped = cv2.resize(top_right_cropped,(224,224))
    center_down_cropped = img[round(0.25*top_y):top_y,round(0.125*top_x):round(0.875*top_x)] # 裁剪坐标为[y0:y1, x0:x1]
    center_down_cropped = cv2.resize(center_down_cropped,(224,224))
    center_big_cropped = img[round(0.05*top_y):round(0.95*top_y), round(0.05*top_x):round(0.95*top_x)] # 裁剪坐标为[y0:y1, x0:x1]
    center_big_cropped = cv2.resize(center_big_cropped,(224,224))
    center_small_cropped = img[round(0.075*top_y):round(0.925*top_y), round(0.075*top_x):round(0.925*top_x)] # 裁剪坐标为[y0:y1, x0:x1]
    center_small_cropped = cv2.resize(center_small_cropped,(224,224))
    
#     plt.imshow(top_left_cropped)
#     plt.show()
#     plt.imshow(top_right_cropped)
#     plt.show()
#     plt.imshow(center_down_cropped)
#     plt.show()
#     plt.imshow(center_big_cropped)
#     plt.show()
#     plt.imshow(center_small_cropped)
#     plt.show()
    cv2.imwrite(crop_save_path+os.sep+"0.jpg", whole_cropped)
    cv2.imwrite(crop_save_path+os.sep+"1.jpg", top_left_cropped)
    cv2.imwrite(crop_save_path+os.sep+"2.jpg", top_right_cropped)
    cv2.imwrite(crop_save_path+os.sep+"3.jpg", center_down_cropped)
    cv2.imwrite(crop_save_path+os.sep+"4.jpg", center_big_cropped)
    cv2.imwrite(crop_save_path+os.sep+"5.jpg", center_small_cropped)

import shutil
def file2folder_copy(pic_file_path,pic_output_path):
    for file in os.listdir(pic_file_path):
        print(file)
        try:
            file_name=dict_folder[file[:-4]]
            if not os.path.exists(pic_output_path+os.sep+file_name):
                os.makedirs(pic_output_path+os.sep+file_name)
            crop_save_path=pic_output_path+os.sep+file_name+os.sep+file[:-4]
            if not os.path.exists(crop_save_path):
                os.makedirs(crop_save_path)
#             shutil.copy(pic_file_path+os.sep+file,
#                         pic_output_path+os.sep+file_name+os.sep+file)
        except:
#             print(file,"  not exist in txt")
            continue
        crop5region(pic_file_path+os.sep+file,crop_save_path)


import threading
overall_pic_path="./FERPlus/data"
overall_output_path="./FERPlus_reshape_crop_folder"
idx=0
t=[]
for folder in os.listdir(overall_pic_path):
    print(folder,folder[7:])
    if folder[7:] == "Valid":
        t.append(threading.Thread(target=file2folder_copy, args=(overall_pic_path+os.sep+folder,overall_output_path+os.sep+"Train",)))
    else:
        t.append(threading.Thread(target=file2folder_copy, args=(overall_pic_path+os.sep+folder,overall_output_path+os.sep+folder[7:],)))
    t[idx].start()
    idx+=1






