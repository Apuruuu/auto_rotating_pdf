from genericpath import isdir
import numpy as np
from PIL import Image
import os

r2_label = ['0@0','1@0','6@0','8@0','9@0']

use_label = ['0@0','1@0','2@0','3@0','4@0','5@0','6@0','7@0','8@0','9@0',
                        '0@90','1@90','2@90','3@90','4@90','5@90','6@90','7@90','8@90','9@90',
                        '2@180','3@180','4@180','5@180','7@180','2@270','3@270','4@270','5@270','7@270','others']

x_train_new = []
y_train_new = []
x_test_new = []
y_test_new = []

count_train = 0

for _class in use_label:
    input_path = os.path.join("dataset",_class)
    if not os.path.isdir(input_path):
        continue

    if _class in r2_label:
        directions = [0, 90]
    elif _class in use_label:
        directions = [0, 90, 180, 270]
    else: continue

    

    file_names = os.listdir(input_path)
    file_list = [os.path.join(input_path,file) for file in file_names]
    # print(file_list)

    for img_file in file_list:
        input_img = Image.open(img_file)
        input_img = np.asarray(input_img)
        input_img  = input_img.reshape((28, 28))
        img = Image.fromarray(input_img)
        # print(input_img)
        # print(input_img.shape)

        for dir_15deg in [-15,0,15]:
            img = img.rotate(dir_15deg)

            for direction in directions:
                img = img.rotate(direction)
                _x_train = np.asarray(img)
                x_train_new.append(_x_train)
                if _class == 'others':
                    y_train_new.append(use_label.index(_class))
                else:
                    num, deg = _class.split('@')
                    y_train_new.append(use_label.index('%s@%d'%(num,int(deg)+direction)))
                count_train += 1

    print('added %d data to train_data'%count_train)

np.savez('add_1.npz',x_train = x_train_new,
                        y_train = y_train_new)