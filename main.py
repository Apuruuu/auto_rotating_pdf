# import pathlib
import time
# from numpy.core.numeric import isclose
# from numpy.lib.twodim_base import tri
# import pdfplumber  # install
# from PIL import Image, ImageFilter
import PyPDF4
# from io import BytesIO
import numpy as np
import cv2
import os
# import tensorflow as tf
import cnn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class main():
    def __init__(self, file_list, laod_path = 'load', save_path = 'save'):
        self.n = 0
        self.start_time=time.time()
        # load cnn
        self.CNN = cnn.cnn()
        print('Load Model used %.2fs'%(time.time()-self.start_time))
        for self.file_name in file_list:
            with open(os.path.join(save_path, self.file_name), "wb") as outfile:
                self.start_time=time.time()
                self.open_pdf(os.path.join(laod_path, self.file_name))
                self.output = PyPDF4.pdf.PdfFileWriter()
                self.load_page()
                self.output.write(outfile)

    def open_pdf(self, file_path):
        self.pdf = PyPDF4.pdf.PdfFileReader(file_path)
        # check password
        if self.pdf.isEncrypted:
            password = input()
            self.pdf.decrypt(password)
        print('Opend File: %.2fs'%(time.time()-self.start_time))

    def load_page(self):
        
        for page_num in range(self.pdf.getNumPages()):
            page = self.pdf.getPage(page_num)
            # page = self.pdf.getPage(1)
            for key in page['/Resources']['/XObject']:
                if key[0:3] == '/Im':
                    self.start_time=time.time()

                    img_raw = page['/Resources']['/XObject'][key].getData() # 获取图片二进制数据
                    img_raw = cv2.imdecode(np.array(bytearray(img_raw), dtype='uint8'), cv2.IMREAD_UNCHANGED)  # 从二进制图片数据中读取

                    print('P.%d: get image %.2fs'%(page_num, time.time()-self.start_time), end='')
                    self.start_time=time.time()

                    img_width = img_raw.shape[1]
                    img_height = img_raw.shape[0]
                    if img_raw.shape[-1] == 3:
                        img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY) # 转换灰度
                    else:
                        img_gray = img_raw

                    img_gray = hist_normalization(img_gray) # 均衡直方图
                    _,img_binner =cv2.threshold(img_gray,125,255,cv2.THRESH_BINARY) # 二值化
                    img_BINNER2RBG = cv2.merge([img_binner, img_binner, img_binner])
                    img_binner = cv2.bitwise_not(img_binner, img_binner) # 反色
                    contours, _ = cv2.findContours(img_binner,cv2.RETR_LIST,cv2.CHAIN_APPROX_TC89_L1) # 寻找连续块

                    ROIs_xy_cache = []
                    for c in contours[1:]:
                        x, y, w, h = cv2.boundingRect(c)  # 外接矩形
                        if w>img_width*0.1 or h>img_height*0.1:
                            continue
                        if w<10 or h<10:
                            continue
                        ROIs_xy_cache.append([x,y,w,h])

                    ROIs = [] # roi（图片）
                    ROIs_xy = [] # 保存需要的roi位置
                    
                    for x, y, w, h in ROIs_xy_cache:
                        ROI = img_binner[y:y+h, x:x+w]
                        ROI = cv2.resize(ROI, (20,20))

                        white_count = len(ROI[ROI>=128])
                        if white_count <= 60 or white_count >= 190: continue

                        ROI = cv2.copyMakeBorder(ROI, 4, 4, 4, 4, cv2.BORDER_CONSTANT,value=[0,0,0])
                        cv2.imwrite(os.path.join('roi','test', str(self.n) + '.jpg'), ROI)
                        ROIs.append(ROI)
                        ROIs_xy.append([x,y,w,h])
                        # cv2.rectangle(img_BINNER2RBG,(x,y),(x+w,y+h),(0,0,255),2)
                        self.n+=1

                    print('  get_roi %.2fs'%(time.time()-self.start_time), end='')
                    self.start_time=time.time()

                    # print(n,len(contours))
                    results, accuracys = self.CNN.Predict(ROIs)
                    # print(results)

                    print('  get_results %.2fs'%(time.time()-self.start_time), end='')
                    self.start_time=time.time()

                    def draw(ROI_xy,color,deg = None):
                        x, y, w, h = ROI_xy
                        cv2.rectangle(img_BINNER2RBG,(x,y),(x+w, y+h), color, thickness=17)
                        if not deg == None:
                            if deg == '0':
                                triangle = np.array([[x, y],[x+w, y],[x+w/2, y-20]], dtype=np.int32)
                            if deg == '90':
                                triangle = np.array([[x, y],[x, y+h],[x-20, y+h/2]], dtype=np.int32)
                            if deg == '180':
                                triangle = np.array([[x, y+h],[x+w, y+h],[x+w/2, y+h+20]], dtype=np.int32)
                            if deg == '270':
                                triangle = np.array([[x+w, y],[x+w, y+h],[x+w+20, y+h/2]], dtype=np.int32)
                            # print(triangle)
                            cv2.fillPoly(img_BINNER2RBG, [triangle], (255,0,0))
                            cv2.putText(img_BINNER2RBG, deg, [x,y], cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2)

                    deg_count = {'0':0, '90':0, '180':0, '270':0}
                    for num in range(len(results)):
                        ROI_xy = ROIs_xy[num]
                        if accuracys[num] <= 0.97:
                            draw(ROI_xy, (0,0,200))
                            continue
                        if results[num] == 'others':
                            draw(ROI_xy, (0,0,200))
                            continue
                        result, deg = results[num].split('@')
                        if result in ['2','3','4','5','7']:
                            deg_count[deg] += 1
                        elif result in ['6','8','9']:
                            if deg in ['0', '180']:
                                deg_count['0'] += 1
                                deg_count['180'] += 1
                            else:
                                deg_count['90'] += 1
                                deg_count['270'] += 1
                        draw(ROI_xy, (0,255,0), deg)

                    print('  draw %.2fs'%(time.time()-self.start_time), end='')
                    self.start_time=time.time()

                    for key in deg_count.keys():
                        if deg_count[key] == max(deg_count.values()):
                            if int(key) == 0: 
                                rotate_img = 0
                                rotate_pdf = 0
                            else: 
                                rotate_img = 360 - int(key)
                                rotate_pdf = int(key)

                    print('  rotate paf %.2fs'%(time.time()-self.start_time), end='')
                    self.start_time=time.time()

                    page_new = page.rotateClockwise(0)
                    self.output.addPage(page_new)

                    center = (w // 2, h // 2)
                    h, w = img_BINNER2RBG.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, rotate_img, 1)
                    img_rotated = cv2.warpAffine(img_BINNER2RBG, M, (w, h))
                    
                    cv2.putText(img_rotated, str(rotate_img), [int(img_width/2), int(img_height/2)], cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), thickness=4)

                    cv2.imwrite(os.path.join('report', self.file_name + str(page_num) + '.jpg'), img_rotated)

                    print('  save_img %.2fs'%(time.time()-self.start_time))
                    self.start_time=time.time()

                    # cv2.namedWindow('Result of filtered', 0)
                    # cv2.imshow('Result of filtered', img_rotated)
                    # cv2.waitKey(5000)

def hist_normalization(img, a=0, b=255):
    c = img.min()
    d = img.max()
 
    out = img.copy()
    # normalization
    out = (b - a) / (d - c) * (out - c) + a
    out[out < a] = a
    out[out > b] = b
 
    out = out.astype(np.uint8)
    return out

if __name__ == '__main__':
    # main(file_list = ["test1.pdf", "test2.pdf", "test3.pdf", "test4.pdf", "test5.pdf"])
    main(file_list = ["20210720160358956.pdf"])