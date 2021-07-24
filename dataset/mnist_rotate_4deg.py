import numpy as np
from PIL import Image

path='mnist_new.npz'
f = np.load(path)
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']

use_arabic = [0,1,2,3,4,5,6,7,8,9]
use_label = ['0@0','1@0','2@0','3@0','4@0','5@0','6@0','7@0','8@0','9@0',
                        '0@90','1@90','2@90','3@90','4@90','5@90','6@90','7@90','8@90','9@90',
                        '2@180','3@180','4@180','5@180','7@180','2@270','3@270','4@270','5@270','7@270','others']

x_train_new = []
y_train_new = []
x_test_new = []
y_test_new = []

count_train = 0

for direction in [0, 90, 180, 270]: # rotating train dataset
    for num in range(len(y_train)):
        if y_train[num] not in use_arabic:
            continue
        img = Image.fromarray(x_train[num])
        img = img.rotate(direction)
        # img.show()
        _x_train = np.asarray(img)
        x_train_new.append(_x_train)
        y_train_new.append(use_label.index(y_train[num] + direction)) # will like 5 + 90deg = 95
        count_train += 1

print('added %d data to train_data'%count_train)
count_test = 0

for direction in [0, 90, 180, 270]: # rotating test dataset
    for num in range(len(y_test)):
        if y_test[num] not in use_arabic:
            continue
        img = Image.fromarray(x_test[num])
        img = img.rotate(direction)
        # img.show()
        _x_test = np.asarray(img)
        x_test_new.append(_x_test)
        y_test_new.append(use_label.index(y_test[num] + direction)) # will like 5 + 90deg = 95
        count_test+= 1

print('added %d data to test_data'%count_test)

np.savez('mnist_new.npz',x_train = x_train_new,
                        y_train = y_train_new,
                        x_test = x_test_new,
                        y_test = y_test_new)