import numpy as np
import os
import cnn
from PIL import Image

class main():
    def __init__(self, file_path = os.path.join('roi','test')):
        self.n = 0
        # load cnn
        self.CNN = cnn.cnn()

        self.file_name = os.listdir(file_path)
        self.pickup(file_path)

    def pickup(self,file_path,savepath = os.path.join('dataset')):
        if not os.path.isdir(savepath):
            os.mkdir(savepath)
        if not os.path.isdir(os.path.join(savepath,'pos')):
            os.mkdir(os.path.join(savepath,'pos'))
        if not os.path.isdir(os.path.join(savepath,'neg')):
            os.mkdir(os.path.join(savepath,'neg'))
        n = 0
        for img_file in self.file_name:
            img_file_path = os.path.join(file_path,img_file)

            img = Image.open(img_file_path)
            input_img = np.asarray(img)
            input_img  = input_img.reshape((1, 28, 28, 1))

            results, accuracy = self.CNN.Predict(input_img)

            results = results[0]
            accuracy = accuracy[0]

            if accuracy <= 0.96:
                _save_path = os.path.join(savepath, 'neg', results)
            else:
                _save_path = os.path.join(savepath, 'pos', results)

            if not os.path.isdir(_save_path):
                os.mkdir(_save_path)

            img.save(os.path.join(_save_path, str(n) + '.jpg'))
            print(n)
            n+=1

if __name__ == '__main__':
    main()