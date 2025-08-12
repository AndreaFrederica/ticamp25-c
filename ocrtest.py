from time import time
import easyocr
import cv2
import numpy as np

reader = easyocr.Reader(['en'])        # 如果想识别中文：['ch_sim', 'en']

print(time())

results = reader.readtext(
    './image/1.png',
    allowlist='0123456789',            # 仅数字
    detail=1                           # 返回 3 元组：(bbox, text, conf)
)

print(time())

for bbox, text, conf in results:
    # bbox 是 4 个点的列表，顺时针：[左上, 右上, 右下, 左下]
    print(f'Text: {text}  (conf={conf:.2f})')
    print('  BBox:', bbox)



img = cv2.imread('./image/1.png')
for bbox, text, conf in results:
    pts = np.array(bbox, dtype=int)
    cv2.polylines(img, [pts], isClosed=True, color=(0,255,0), thickness=2)
    cv2.putText(img, text, pts[0], cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8, color=(0,0,255), thickness=2)
cv2.imwrite('annotated.png', img)
