import numpy as np
from keras.models import load_model

def Load_labels(label_file = 'labels.txt'):
    with open(label_file) as f: return f.read().splitlines()

class cnn():
    def __init__(self , model_path = 'model_3.h5', labels_file = 'labels.txt'):
        # load model
        self.model = load_model(model_path)
        # load labels
        self.use_label = Load_labels(labels_file)

    def create_model():
        pass

    def load_model():
        pass

    def train():
        pass
        
    def Predict(self, input_rois):
        input_rois = np.asarray(input_rois)
        input_rois = input_rois / 255.0
        results = []
        accuracys = []

        for roi in input_rois:
            roi.shape = (1, 28, 28, 1)
            predict = self.model.predict(roi)
            predict_num = np.argmax(predict)  # 取最大值的位置

            accuracy = predict[0][predict_num]
            accuracys.append(accuracy)

            results.append(self.use_label[predict_num])

        return results, accuracys

if __name__ == '__main__':
    CNN = cnn()
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
    input_img = Image.open('roi\\test\\332.jpg')
    use_label = ['0@0','1@0','2@0','3@0','4@0','5@0','6@0','7@0','8@0','9@0',
                        '0@90','1@90','2@90','3@90','4@90','5@90','6@90','7@90','8@90','9@90',
                        '2@180','3@180','4@180','5@180','7@180','2@270','3@270','4@270','5@270','7@270','others']
    input_img = np.asarray(input_img)
    input_img  = input_img.reshape((1, 28, 28, 1))
    # print(input_img)
    print(input_img.shape)
    results, predict = CNN.Predict(input_img)
    print(results, predict)