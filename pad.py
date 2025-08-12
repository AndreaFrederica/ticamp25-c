from paddleocr import PaddleOCR

import paddle
paddle.utils.run_check()

ocr = PaddleOCR()  # need to run only once to download and load model into memory
img_path = './image/1.png'
result = ocr.predict(img_path)
for line in result:
    print(line)