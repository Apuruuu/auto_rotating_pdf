import numpy as np
from PIL import Image
import random

use_label = ['0@0','1@0','2@0','3@0','4@0','5@0','6@0','7@0','8@0','9@0',
                        '0@90','1@90','2@90','3@90','4@90','5@90','6@90','7@90','8@90','9@90',
                        '2@180','3@180','4@180','5@180','7@180','2@270','3@270','4@270','5@270','7@270','others']
# path='mnist_new.npz'
path='add_2.npz'
f = np.load(path)
x_train, y_train = f['x_train'], f['y_train']
num = random.randrange(0,len(y_train)-1,1)

img = Image.fromarray(x_train[num])
print(use_label[y_train[num]])
img.show()


# x_test, y_test = f['x_test'], f['y_test']
# num = random.randrange(0,len(y_test)-1,1)

# img = Image.fromarray(x_test[num])
# print(use_label[y_test[num]])
# img.show()