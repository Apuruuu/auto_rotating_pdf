import os
import time
from PyPDF4 import PdfFileReader, PdfFileWriter
import numpy as np
import cv2
import shutil
import cnn

MAX_ROI_SIZE = [1*0.1, 1*0.1]
MIN_ROI_SIZE = (10,10)
Trust_Threshold = 0.97
Nunber_of_white_pxile_range = (60, 190)

# Colors setting (BGR) (0 to 255)
COLOR_RELIABLE = (0,255,0)
COLOR_UNRELIABLE = (0,0,200)
COLOR_DO_NOT_USE = (0,200,200)
COLOR_ARROW = (255,0,0)
COLOR_TEXT = (255,0,0)

# Debug Mode save path
ROI_SAVE_PATH = os.path.join('roi')
IMAGE_SAVE_PATH = os.path.join('report')
CLEAR = True


class Rotate_PDF():
    def __init__(self, file_path_lists, save_path = 'save', debug = False):

        self.debug = debug
        if self.debug:
            if CLEAR:
                if os.path.isdir(ROI_SAVE_PATH):shutil.rmtree(ROI_SAVE_PATH)
                if os.path.isdir(IMAGE_SAVE_PATH):shutil.rmtree(IMAGE_SAVE_PATH)
            self.n = 0

        # Load cnn
        self.CNN = cnn.cnn(mode = 'predict')
        
        for file_path in file_path_lists:
            self.open_pdf(file_path)

            for file_path in file_path_lists:
                _save_path = save_path
                path = os.path.normpath(file_path).split(os.sep)
                for _path in path[1:-1]:
                    _save_path = os.path.join(_save_path, _path)
                if not os.path.isdir(_save_path):
                    os.makedirs(_save_path)
                self._file_name = path[-1]

            print('{:s}'.format('-'*64))
            print('|{:^62s}|'.format(file_path[-57:]))
            print('{:s}'.format('-'*64))
            print('|{:^6s}|{:^7s}|{:^31s}|{:^15s}|'.format('', '', 'CNN Return', ''))
            print('|{:^6s}|{:^7s}|{:^7s}|{:^5s}|{:^5s}|{:^5s}|{:^5s}|{:^15s}|'.format(\
                    'Page', 'ROI', 'ALL','0', '90', '180', '270','STATUS'))
            print('{:s}'.format('-'*64))

            self.main()

            # Save rotated pdf
            with open(os.path.join(_save_path, self._file_name), "wb") as outfile:
                self.output.write(outfile)
            print('{:s}\nCompleted 完成\n'.format('-'*64))

    def open_pdf(self, file_path):
        self.pdf = PdfFileReader(file_path)
        self.output = PdfFileWriter()
        # Check password
        if self.pdf.isEncrypted:
            password = input()
            self.pdf.decrypt(password)

    def image_pretreatment(self, input_img):
        if input_img.shape[-1] == 3:
            img_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY) # Grayscale
        else:
            img_gray = input_img

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

        img_gray = hist_normalization(img_gray) # Balanced histogram
        _,img_binary =cv2.threshold(img_gray,125,255,cv2.THRESH_BINARY) # Binarization
        img_binary = cv2.bitwise_not(img_binary, img_binary) # Reverse bit

        return img_binary

    def Get_ROI(self, img_binary):
        contours, _ = cv2.findContours(img_binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_TC89_L1) # Find Contours
        width = img_binary.shape[1]
        height = img_binary.shape[0]
        # Get ROI
        ROIs = [] # roi（图片）
        ROIs_xy = [] # 保存需要的roi位置
        _max_roi_size = np.array([width, height]) * np.array(MAX_ROI_SIZE)

        for c in contours[1:]:
            x, y, w, h = cv2.boundingRect(c)  # 外接矩形
            # Filter by ROI size

            if w > _max_roi_size[0] or h > _max_roi_size[1]:
                continue
            if w < MIN_ROI_SIZE[0] or h < MIN_ROI_SIZE[1]:
                continue

            # Crop ROI
            ROI = img_binary[y:y+h, x:x+w]
            ROI = cv2.resize(ROI, (20,20))
            white_count = len(ROI[ROI>=128])

            # Filter by number of white pixels
            if white_count <= Nunber_of_white_pxile_range[0] or \
               white_count >= Nunber_of_white_pxile_range[1]: continue

            ROI = cv2.copyMakeBorder(ROI, 4, 4, 4, 4, cv2.BORDER_CONSTANT,value=[0,0,0])
            
            ROIs.append(ROI)
            ROIs_xy.append([x,y,w,h])

            # [Debug mode] save ROI to Jpg file
            if self.debug:
                if not os.path.isdir(ROI_SAVE_PATH):
                    os.mkdir(ROI_SAVE_PATH)
                cv2.imwrite(os.path.join(ROI_SAVE_PATH, str(self.n) + '.jpg'), ROI)
                self.n += 1
            self.num_ROI = len(ROIs_xy)
        return ROIs, ROIs_xy

    def main(self):
        for self.page_num in range(self.pdf.getNumPages()):
            page = self.pdf.getPage(self.page_num)
            self.start_time = time.time()
            print(end=f"\r{'|{:<6s}|{:^7d}|{:^7d}|{:^5d}|{:^5d}|{:^5d}|{:^5d}|{:^15s}|'.format('P.%d'%(self.page_num+1), 0, 0, 0, 0, 0, 0,'')}")
            # get image
            for key in page['/Resources']['/XObject']:
                self.rotated_angle = int(page['/Rotate'])
                if key[0:3] == '/Im':
                    img_raw = page['/Resources']['/XObject'][key].getData() # Get image binary data
                    img_raw = cv2.imdecode(np.array(bytearray(img_raw), dtype='uint8'), cv2.IMREAD_UNCHANGED)  # Decode to opencv image data

                    # Image pretreatment
                    img_binary = self.image_pretreatment(img_raw)
                    # Get ROI
                    ROIs, ROIs_xy = self.Get_ROI(img_binary)
                else : continue

            self.Num_ROIs = len(ROIs)
            print(end=f"\r{'|{:<6s}|{:^7d}|{:^7d}|{:^5d}|{:^5d}|{:^5d}|{:^5d}|{:^15s}|'.format('P.%d'%(self.page_num+1), self.Num_ROIs, 0, 0, 0, 0, 0,'WAITING CNN')}")

            # Predict direction
            # results like ['3@90','5@90',...]
            # accuracys like [0.9854,0.8712,...]
            self.Num_CNN_Report = 0
            results, accuracys = [],[]
            for _ROI in ROIs: 
                result, accuracy = self.CNN.Predict(_ROI)
                results.append(result)
                accuracys.append(accuracy)
                self.Num_CNN_Report += 1
                status = '%s %.2fs'%(u'\u25ae'*(int(self.Num_CNN_Report*10/self.Num_ROIs)), (self.Num_ROIs-self.Num_CNN_Report)*0.04)
                print(end=f"\r{'|{:<6s}|{:^7d}|{:^7d}|{:^5d}|{:^5d}|{:^5d}|{:^5d}|{:<15s}|'.format('P.%d'%(self.page_num+1), self.Num_ROIs, self.Num_CNN_Report, 0, 0, 0, 0, status)}")

            rotate_degree, draw_ROI = self.Classification_statistics(results, accuracys, ROIs_xy)
            print(end=f"\r{'|{:<6s}|{:^7d}|{:^7d}|{:^5d}|{:^5d}|{:^5d}|{:^5d}|{:^15s}|'.format('P.%d'%(self.page_num+1), self.Num_ROIs, self.Num_CNN_Report, self.deg_count['0'], self.deg_count['90'], self.deg_count['180'], self.deg_count['270'],'')}")

            page_new = page.rotateCounterClockwise(self.rotated_angle)
            page_new = page.rotateClockwise(rotate_degree)
            self.output.addPage(page_new)

            if self.debug:
                print(end=f"\r{'|{:<6s}|{:^7d}|{:^7d}|{:^5d}|{:^5d}|{:^5d}|{:^5d}|{:^15s}|'.format('P.%d'%(self.page_num+1), self.Num_ROIs, self.Num_CNN_Report, self.deg_count['0'], self.deg_count['90'], self.deg_count['180'], self.deg_count['270'],'OpenCV Painting')}")

                def Draw_ROI(ROI_xy, color, deg = None):
                    x, y, w, h = ROI_xy
                    cv2.rectangle(img_color,(x,y),(x+w, y+h), color, thickness=1)
                    if not deg == None:
                        if deg == '0':
                            triangle = np.array([[x, y],[x+w, y],[x+w/2, y-20]], dtype=np.int32)
                        if deg == '90':
                            triangle = np.array([[x, y],[x, y+h],[x-20, y+h/2]], dtype=np.int32)
                        if deg == '180':
                            triangle = np.array([[x, y+h],[x+w, y+h],[x+w/2, y+h+20]], dtype=np.int32)
                        if deg == '270':
                            triangle = np.array([[x+w, y],[x+w, y+h],[x+w+20, y+h/2]], dtype=np.int32)
                        cv2.fillPoly(img_color, [triangle], COLOR_ARROW)
                        cv2.putText(img_color, deg, [x,y], cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_TEXT, thickness=2)

                def Rotate_image(file_name,page_num):
                    width = img_color.shape[1]
                    height = img_color.shape[0]
                    center = (width // 2, height // 2)
                    height, width = img_color.shape[:2]
                    center = (width // 2, height // 2)
                    M = cv2.getRotationMatrix2D(center, Image_rotate_degree, 1)
                    img_rotated = cv2.warpAffine(img_color, M, (width, height))
                    cv2.putText(img_rotated, str(Image_rotate_degree), [int(width/2), int(height/2)], 
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), thickness=4)

                    if not os.path.isdir(IMAGE_SAVE_PATH):
                        os.mkdir(IMAGE_SAVE_PATH)
                    _save_path = os.path.join(IMAGE_SAVE_PATH, file_name)
                    if not os.path.isdir(_save_path):
                        os.mkdir(_save_path)

                    cv2.imwrite(os.path.join(_save_path, 'Page%d.jpg'%(page_num + 1)), img_rotated)

                img_binary = cv2.bitwise_not(img_binary, img_binary) # Reverse bit
                img_color = cv2.merge([img_binary, img_binary, img_binary]) # Create RGB image
                for ROI in draw_ROI:
                    ROI_xy, color, deg = ROI
                    Draw_ROI(ROI_xy, color, deg)
                Image_rotate_degree = 360 - rotate_degree
                Rotate_image(self._file_name,self.page_num)
            
            print(end=f"\r{'|{:<6s}|{:^7d}|{:^7d}|{:^5d}|{:^5d}|{:^5d}|{:^5d}|{:^15s}|'.format('P.%d'%(self.page_num+1), self.Num_ROIs, self.Num_CNN_Report, self.deg_count['0'], self.deg_count['90'], self.deg_count['180'], self.deg_count['270'],' %.3fs'%(time.time()-self.start_time))}")
            print('')

    def Classification_statistics(self, results, accuracys, ROIs_xy):
        deg_count = {'0':0, '90':0, '180':0, '270':0}
        draw_ROI = []
        for num in range(len(results)):
            ROI_xy = ROIs_xy[num]
            if accuracys[num] <= Trust_Threshold:
                draw_ROI.append([ROI_xy, COLOR_UNRELIABLE, None])
                continue
            if results[num] in ['0@0', '0@90', '1@0', '1@90', 'others']:
                draw_ROI.append([ROI_xy, COLOR_DO_NOT_USE, None])
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
            draw_ROI.append([ROI_xy, COLOR_RELIABLE, deg])
        
        for key in deg_count.keys():
            if deg_count[key] == max(deg_count.values()):
                rotate_degree = int(key)

        self.deg_count = deg_count

        return rotate_degree, draw_ROI



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(prog='Auto_Rotating_Scanned_PDF')
    parser.add_argument('--load','-l', metavar='L', default='load', type=str, help='Load PDF files from this directory')
    parser.add_argument('--save','-c', metavar='S', default='save', type=str, help='Save Rotated PDF to this directory')
    parser.add_argument('--debug','-m', metavar='D', type=bool, default=False, help='Run in debug mode')
    args = parser.parse_args()

    load_path = os.path.join(args.load)
    save_path = os.path.join(args.save)

    if not os.path.isdir(load_path):
        os.mkdir(load_path)
        print('Put the PDF files into ./load')
        print('将PDF文件放入load文件夹内')
        print('PDFファイルを/loadフォルダに入れてください')

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    file_paths = []
    for root, _, files in os.walk(load_path):
        for file in files:
            if os.path.splitext(file)[1] == '.pdf':
                file_paths.append(os.path.join(root,file))

    Rotate_PDF(file_paths, save_path, args.debug)