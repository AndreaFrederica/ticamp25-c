#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时检测带 2cm 黑框的 A4 纸，测量距离并显示结果

功能：
1. 打开默认摄像头（索引 0）
2. 对每一帧做 HSV → 二值化 → 形态学处理
3. 找出最大黑色轮廓并拟合成四边形
4. 在原图上绘制检测到的边框
"""

import cv2
import numpy as np

def detect_black_frame(frame):
    """
    输入 BGR 图像，返回：
      - pts: 检测到的最大黑色边框的四个顶点，shape=(4,2)，按任意顺序
      - mask: 二值掩码图像
    未检测到时返回 (None, mask)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 黑色阈值，可根据光照微调 V 上限
    lower_black = np.array([0,   0,   0])
    upper_black = np.array([180, 255, 50])
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # 形态学处理：闭运算填充 → 开运算去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    # 找外轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    # 取最大轮廓
    max_cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(max_cnt) < 10000:
        return None, mask

    # 拟合：优先多边形逼近，退而求其次最小外接矩形
    peri = cv2.arcLength(max_cnt, True)
    approx = cv2.approxPolyDP(max_cnt, 0.02 * peri, True)
    if len(approx) != 4:
        rect = cv2.minAreaRect(max_cnt)
        box = cv2.boxPoints(rect)
        approx = box.astype(int)

    pts = approx.reshape(-1, 2)
    return pts, mask

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头 (索引 0)")
        return

    # 可选：调低分辨率，加快处理速度
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("✅ 摄像头已打开，开始实时检测黑色边框…")
    while True:
        ret, frame = cap.read()
        print("cap.read() →", ret, frame.shape if ret else None)
        if not ret:
            break

        pts, mask = detect_black_frame(frame)

        # 在原图上绘制结果
        disp = frame.copy()
        if pts is not None:
            cv2.polylines(disp, [pts], isClosed=True, color=(0,255,0), thickness=3)
            for (x, y) in pts:
                cv2.circle(disp, (x, y), 5, (0,0,255), -1)

        # 显示窗口
        cv2.imshow('Original', disp)
        cv2.imshow('Mask',     mask)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
