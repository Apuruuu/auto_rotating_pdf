import numpy as np
import cnn

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

_cnn = cnn.cnn(mode = 'train')

_cnn.model.summary()

_cnn.model.compile(optimizer='sgd',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

_cnn.model.fit(train_images, train_labels, epochs=50)

_cnn.model.save_weights('weights_3.h5', overwrite=True)
_cnn.model.save('model_3.h5', overwrite=True, include_optimizer=True)

test_loss, test_acc = _cnn.model.evaluate(test_images, test_labels, verbose = 2)