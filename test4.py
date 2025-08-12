#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time

def detect_rectangles(frame):
    """
    输入 BGR 图像 frame，返回：
      - mask: 只保留黑色边框区域的二值掩码
      - rects: 一个列表，元素是所有检测到的四边形顶点，shape=(4,2)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0,   0,   0])
    upper_black = np.array([180, 255, 85])
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # 形态学填充，修补断裂
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 边缘检测
    edges = cv2.Canny(mask, 50, 150)

    # 找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for cnt in contours:
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 1:  # 面积阈值可调整
            pts = approx.reshape(4, 2)
            rects.append(pts)
    return mask, rects

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("✅ 摄像头已打开，按 ESC 退出")
    while True:
        time.sleep(0.01)
        ret, frame = cap.read()
        if not ret:
            break

        mask, rects = detect_rectangles(frame)
        vis = frame.copy()

        # 把所有检测到的矩形都画出来
        for pts in rects:
            pts = pts.astype(int)
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
            # 标记四个顶点
            for (x, y) in pts:
                cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)

        cv2.imshow("All Rectangles", vis)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
