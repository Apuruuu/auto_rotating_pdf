import numpy as np
import cv2
import os

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

n=0
file_name = 'input\\eng_r4.jpg'
img_raw = cv2.imread(file_name)

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
    # if w>img_width*0.07 or h>img_height*0.07:
    #     continue
    # if w<10 or h<10:
    #     continue
    ROIs_xy_cache.append([x,y,w,h])

ROIs = [] # roi（图片）
ROIs_xy = [] # 保存需要的roi位置

# os.mkdir(os.path.join('dataset'))

for x, y, w, h in ROIs_xy_cache:
    ROI = img_binner[y:y+h, x:x+w]
    ROI = cv2.resize(ROI, (20,20))

    ROI = cv2.copyMakeBorder(ROI, 4, 4, 4, 4, cv2.BORDER_CONSTANT,value=[0,0,0])
    cv2.imwrite(os.path.join('dataset', str(n) + '.jpg'), ROI)
    ROIs.append(ROI)
    ROIs_xy.append([x,y,w,h])
    # cv2.rectangle(img_BINNER2RBG,(x,y),(x+w,y+h),(0,0,255),2)
    n+=1
