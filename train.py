import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np

use_label = ['0@0','1@0','2@0','3@0','4@0','5@0','6@0','7@0','8@0','9@0',
            '0@90','1@90','2@90','3@90','4@90','5@90','6@90','7@90','8@90','9@90',
            '2@180','3@180','4@180','5@180','7@180','2@270','3@270','4@270','5@270','7@270','others']

def load_self_data(path = 'mnist_new_add_1.npz'):
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)

# (train_images, train_labels), (test_images,test_labels) = datasets.mnist.load_data
(train_images, train_labels), (test_images,test_labels) = load_self_data()

train_images = train_images.reshape((len(train_labels), 28, 28, 1))
test_images = test_images.reshape((len(test_labels), 28, 28, 1))
# Regularization
train_images = train_images / 255.0
test_images = test_images / 255.0

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(len(use_label), activation='softmax')) # 31 Classes

model.summary()
exit()
model.compile(optimizer='sgd',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=50)

model.save_weights('weights_3.h5', overwrite=True)
model.save('model_3.h5', overwrite=True, include_optimizer=True)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 2)