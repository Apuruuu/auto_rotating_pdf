import numpy as np
from keras.models import load_model
from tensorflow.keras import datasets, layers, models



class cnn():
    def __init__(self , mode, model_path = 'model_3.h5', labels_file = 'labels.txt'):
        # load labels
        self.use_label = self.Load_labels(labels_file)
        if mode == 'train':
            self.create_model()
        elif mode == 'predict':
            self.load_model(model_path)

    def Load_labels(self, label_file = 'labels.txt'):
        with open(label_file) as f: return f.read().splitlines()

    def create_model(self):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)))
        self.model.add(layers.MaxPooling2D((2,2)))
        self.model.add(layers.Conv2D(64,(3,3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2,2)))
        self.model.add(layers.Conv2D(64,(3,3), activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(len(self.use_label), activation='softmax')) # 31 Classes

    def load_model(self, model_path):
        self.model = load_model(model_path)
        pass

    def train():
        pass
        
    def Predict(self, input_roi):
        input_roi = np.asarray(input_roi).reshape((1, 28, 28, 1))
        input_roi = input_roi / 255.0

        # input_roi.shape = (28, 28)
        predict = self.model.predict(input_roi)
        predict_num = np.argmax(predict)  # 取最大值的位置
        accuracy = predict[0][predict_num]
        result = self.use_label[predict_num]

        return result, accuracy

if __name__ == '__main__':
    CNN = cnn(mode = 'predict')
    use_label = open('labels.txt').read().splitlines()
    
    import random

    # path='mnist_new.npz'
    # f = np.load(path)
    # x_test, y_test = f['x_test'], f['y_test']
    # use_label = ['2@0','3@0','4@0','5@0','7@0','2@90','3@90','4@90','5@90','7@90','2@180','3@180','4@180','5@180','7@180','2@270','3@270','4@270','5@270','7@270']
    # num = random.randrange(0,len(y_test)-1,1)
    # input_img = x_test[num]
    # input_img  = input_img.reshape((1, 28, 28, 1))
    # # print(input_img)
    # print(input_img.shape)
    # results = CNN.Predict(input_img)
    # print(results, use_label[y_test[num]])
    

    from PIL import Image
    input_img = Image.open('roi\\82.jpg')
    input_img = np.asarray(input_img)
    # input_img  = input_img.reshape((1, 28, 28, 1))
    # print(input_img)
    print(input_img.shape)
    results, predict = CNN.Predict(input_img)
    print(results, predict)