from genericpath import isdir
import numpy as np
from PIL import Image
import os


path='mnist.npz'
f = np.load(path)
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']

use_label = ['0@0','1@0','2@0','3@0','4@0','5@0','6@0','7@0','8@0','9@0',
                        '0@90','1@90','2@90','3@90','4@90','5@90','6@90','7@90','8@90','9@90',
                        '2@180','3@180','4@180','5@180','7@180','2@270','3@270','4@270','5@270','7@270','others']

for num in range(len(y_train)):
    img = Image.fromarray(x_train[num])
    save_path = os.path.join('dataset','dataset',use_label[y_train[num]])
    if not isdir(save_path):
        os.mkdir(save_path)
    img.save(os.path.join(save_path,'M1TR'+str(num)+'.jpg'))

for num in range(len(y_test)):
    img = Image.fromarray(x_test[num])
    save_path = os.path.join('dataset','dataset',use_label[y_test[num]])
    if not isdir(save_path):
        os.mkdir(save_path)
    img.save(os.path.join(save_path,'1MTE'+str(num)+'.jpg'))
