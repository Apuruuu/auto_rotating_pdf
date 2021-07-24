import numpy as np
from PIL import Image
# path_1='mnist_new.npz'
# f1 = np.load(path_1)
# x_train_1, y_train_1 = f1['x_train'], f1['y_train']
# x_test_1, y_test_1 = f1['x_test'], f1['y_test']


x_train_1 =[]
y_train_1 =[]
x_test_1 =[]
y_test_1 =[]

test_pre_all=0.1


path_2='add_2.npz'
f2 = np.load(path_2)
x_train_2, y_train_2 = f2['x_train'], f2['y_train']
x_test_2, y_test_2 = f2['x_test'], f2['y_test']

train_cache = []
for n in range(len(y_train_1)):
    train_cache.append([x_train_1[n], y_train_1[n]])
for n in range(len(y_train_2)):
    train_cache.append([x_train_2[n], y_train_2[n]])
for n in range(len(y_test_1)):
    train_cache.append([x_test_1[n], y_test_1[n]])
for n in range(len(y_test_2)):
    train_cache.append([x_test_2[n], y_test_2[n]])

print('1')
np.random.shuffle(train_cache)
print('2')
x_train_new = []
y_train_new = []
x_test_new = []
y_test_new = []
count = len(train_cache)
for n in range(count):
    if n < int(count * (1-test_pre_all)):
        _x_train,  y_train = train_cache[n]
        x_train_new.append(_x_train)
        y_train_new.append(y_train)
    else :
        _x_test,  y_test = train_cache[n]
        x_test_new.append(_x_test)
        y_test_new.append(y_test)
print('5')

np.savez('mnist_new_add_1.npz',x_train = x_train_new,
                        y_train = y_train_new,
                        x_test = x_test_new,
                        y_test = y_test_new)