import cv2
import numpy as np

# 1. 读入图像
img = cv2.imread('input.jpg')

# 2. 源点：你选出的梯形四个顶点，按顺时针序
#    比如左上、右上、右下、左下
src_pts = np.array([
    [x1, y1],
    [x2, y2],
    [x3, y3],
    [x4, y4]
], dtype=np.float32)

# 3. 物理尺寸（例如单位是毫米），以及像素/物理单位的转换比例
phys_width  = 200.0   # 物理宽度 mm
phys_height = 100.0   # 物理高度 mm
# 假设你量到的图中底边像素长度
measured_px_width = np.linalg.norm(src_pts[1] - src_pts[0])
scale = measured_px_width / phys_width  # px/mm

# 4. 输出矩形的像素尺寸
out_w = int(phys_width  * scale)
out_h = int(phys_height * scale)

# 5. 目标点：矩形的四个顶点
dst_pts = np.array([
    [0,     0],
    [out_w, 0],
    [out_w, out_h],
    [0,     out_h],
], dtype=np.float32)

# 6. 计算透视变换矩阵
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

# 7. 透视变换
rectified = cv2.warpPerspective(img, M, (out_w, out_h))

# 8. 显示或保存结果
cv2.imshow('Rectified', rectified)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('rectified.jpg', rectified)
