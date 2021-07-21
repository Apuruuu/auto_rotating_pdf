from genericpath import isdir
import os
import numpy as np
import cv2
from PIL import Image

x_train_new = []
y_train_new = []
x_test_new = []
y_test_new = []


use_label = ['0@0','1@0','2@0','3@0','4@0','5@0','6@0','7@0','8@0','9@0',
                        '0@90','1@90','2@90','3@90','4@90','5@90','6@90','7@90','8@90','9@90',
                        '2@180','3@180','4@180','5@180','7@180','2@270','3@270','4@270','5@270','7@270','others']

n=0

for _class in use_label:
    if _class == 'others':continue
    input_path = os.path.join("dataset", "pos",_class)
    if not os.path.isdir(input_path):
        continue

    file_names = os.listdir(input_path)
    file_list = [os.path.join(input_path,file) for file in file_names]

    for img_file in file_list:
        input_img = cv2.imread(img_file)
        b, _, _ = cv2.split(input_img)
        input_img = np.asarray(b)
        white_count = len(input_img[input_img>=128])
        if white_count <= 60 or white_count >= 190: continue
        else : 
            num, deg = _class.split("@")
            
            save_img = Image.fromarray(b)
            save_img = save_img.rotate(360-int(deg))


            # _save_class = num + "@" + deg
            # save_path = os.path.join('dataset','dataset',_save_class)
            # if not isdir(save_path):
            #     os.mkdir(save_path)
            # save_img.save(os.path.join(save_path,'MTR'+str(n)+'.jpg'))
            # n +=1

            for rotate_deg in [0,90,180,270]:
                _save_img = save_img.rotate(rotate_deg)
                _save_class = num + "@" + str(rotate_deg)
                if _save_class == "6@180":
                    _save_class = "9@0"
                elif _save_class == "6@270":
                    _save_class = "9@90"
                elif _save_class == "9@180":
                    _save_class = "6@0"
                elif _save_class == "9@270":
                    _save_class = "6@90"

                elif _save_class == "0@180":
                    _save_class = "0@0"
                elif _save_class == "0@270":
                    _save_class = "0@90"

                elif _save_class == "1@180":
                    _save_class = "1@0"
                elif _save_class == "1@270":
                    _save_class = "1@90"

                elif _save_class == "8@180":
                    _save_class = "8@0"
                elif _save_class == "8@270":
                    _save_class = "8@90"

                save_path = os.path.join('dataset','dataset',_save_class)
                if not isdir(save_path):
                    os.mkdir(save_path)
                _save_img.save(os.path.join(save_path,'MTR'+str(n)+'.jpg'))
                n +=1

                save_img_array = np.asarray(_save_img).reshape((28,28))
                x_train_new.append(save_img_array)
                y_train_new.append(use_label.index(_save_class))

for _class in ['others']:
    input_path = os.path.join("dataset", "pos",_class)
    if not os.path.isdir(input_path):
        continue

    file_names = os.listdir(input_path)
    file_list = [os.path.join(input_path,file) for file in file_names]

    for img_file in file_list:
        input_img = cv2.imread(img_file)
        b, _, _ = cv2.split(input_img)
        input_img = np.asarray(b)
        white_count = len(input_img[input_img>=128])
        if white_count <= 60 or white_count >= 190: continue
        else :           
            save_img = Image.fromarray(b)
            save_img = save_img.rotate(360-int(deg))

            for rotate_deg in [0,90,180,270]:
                _save_img = save_img.rotate(rotate_deg)
                _save_class = 'others'
                save_img_array = np.asarray(_save_img).reshape((28,28))
                x_train_new.append(save_img_array)
                y_train_new.append(use_label.index(_save_class))
                save_path = os.path.join('dataset','dataset',_save_class)
                if not isdir(save_path):
                    os.mkdir(save_path)
                _save_img.save(os.path.join(save_path,'MTR'+str(n)+'.jpg'))
                n +=1

np.savez('add_2.npz',x_train = x_train_new,
                        y_train = y_train_new,
                        x_test = [],
                        y_test = [])