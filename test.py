from PyPDF4 import PdfFileReader, PdfFileWriter
import os

def rotate_pages(pdf_path):
    pdf_writer = PdfFileWriter()
    pdf_reader = PdfFileReader(pdf_path)
    # 顺时针旋转90度
    page_1 = pdf_reader.getPage(0).rotateClockwise(90)
    pdf_writer.addPage(page_1)
    # 逆时针旋转90度
    page_2 = pdf_reader.getPage(1).rotateCounterClockwise(90)
    pdf_writer.addPage(page_2)
    # 在正常方向上添加一页
    pdf_writer.addPage(pdf_reader.getPage(2))
    
    with open(os.path.join('save', 'test_out.pdf'), 'wb') as fh:
        pdf_writer.write(fh)
    
if __name__ == '__main__':
    file_path = os.path.join('load','test1.pdf')
    rotate_pages(file_path)
