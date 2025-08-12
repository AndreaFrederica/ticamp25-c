#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np

def detect_rectangle(frame):
    """
    输入 BGR 图像（frame），返回：
      - frame: 原图
      - result: 只保留黑色边框区域的图像
      - pts: 检测到的最大黑色矩形轮廓顶点，shape=(4,2)，未检测到时为 None
    """
    # 转 HSV，生成黑色掩码
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0,   0,   0])
    upper_black = np.array([180, 255, 85])
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # # 用掩码提取黑边区域
    # result = cv2.bitwise_and(frame, frame, mask=mask)

    # # 二值化后边缘检测
    # gray    = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # edges   = cv2.Canny(blurred, 50, 150)
    
    # 可选：先闭运算填充黑框，再开运算去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    # 直接对 mask 做 Canny
    edges = cv2.Canny(mask, 50, 150)

    # 查找外轮廓，筛选最大四边形
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    pts = None

    for cnt in contours:
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > max_area:
                max_area  = area
                pts        = approx.reshape(4, 2)

    return frame, mask, pts

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("✅ 摄像头已打开，按 ESC 退出。")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig, result, pts = detect_rectangle(frame)

        # 可视化原图和检测到的矩形
        vis = orig.copy()
        if pts is not None:
            cv2.polylines(vis, [pts.astype(int)], True, (0, 255, 0), 2)
            for (x, y) in pts:
                cv2.circle(vis, (int(x), int(y)), 5, (0, 0, 255), -1)

        # 展示三个窗口
        cv2.imshow('Original',     vis)
        cv2.imshow('Mask Result',  result)
        # 如果你还想看中间的 Canny 边缘：
        # cv2.imshow('Edges', edges)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC 键退出
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
