import os
import numpy as np
import cv2


use_label = ['0@0','1@0','2@0','3@0','4@0','5@0','6@0','7@0','8@0','9@0',
                        '0@90','1@90','2@90','3@90','4@90','5@90','6@90','7@90','8@90','9@90',
                        '2@180','3@180','4@180','5@180','7@180','2@270','3@270','4@270','5@270','7@270']
white_count = [0 for i in range(800)]

for _class in use_label:
    input_path = os.path.join("dataset", "pos",_class)
    if not os.path.isdir(input_path):
        continue

    file_names = os.listdir(input_path)
    file_list = [os.path.join(input_path,file) for file in file_names]

    for img_file in file_list:
        input_img = cv2.imread(img_file)
        b, _, _ = cv2.split(input_img)
        input_img = np.asarray(b)
        white_pixel = len(input_img[input_img>=128])
        white_count[white_pixel] +=1

print(white_count)