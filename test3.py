#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time

def detect_rectangle(frame):
    """
    输入 BGR 图像（frame），返回：
      - frame: 原图
      - result: 掩码与原图 bitwise_and 后的图，只保留黑色边框
      - pts: 最大矩形轮廓的 4 个顶点 (4,2)，未检测到时为 None
    """
    # 转 HSV，生成黑色掩码
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0,   0,   0])
    upper_black = np.array([180, 255, 85])  # 可根据实际光照微调 V 上限
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # 用掩码提取黑框区域
    # result = cv2.bitwise_and(frame, frame, mask=mask)

    # # 二值化图上做 Canny
    # gray    = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # edges   = cv2.Canny(blurred, 50, 150)
    # 可选：先闭运算填充黑框，再开运算去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    # 直接对 mask 做 Canny
    edges = cv2.Canny(mask, 50, 150)

    # 查找外轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    pts = None

    # 筛选面积最大的四边形
    for cnt in contours:
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > max_area:
                max_area = area
                pts      = approx.reshape(4, 2)

    return frame, mask, pts

def order_points(pts):
    """
    将无序 4 顶点排序为 [TL, TR, BR, BL]
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # 左上
    rect[2] = pts[np.argmax(s)]      # 右下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # 右上
    rect[3] = pts[np.argmax(diff)]   # 左下
    return rect

def sample_edge(p1, p2, n):
    """
    在 p1→p2 上等距采 n 个点
    """
    return [
        (
            int(p1[0] + (p2[0] - p1[0]) * i / (n - 1)),
            int(p1[1] + (p2[1] - p1[1]) * i / (n - 1))
        )
        for i in range(n)
    ]

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("✅ 摄像头已打开，按 ESC 退出")

    num_points_per_edge = 30

    while True:
        time.sleep(0.01)            # 等摄像头略微稳定
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 帧读取失败，退出")
            break

        orig, result, pts = detect_rectangle(frame)
        vis = orig.copy()

        if pts is not None:
            # 排序 + 画轮廓
            ordered = order_points(pts).astype(int)
            cv2.polylines(vis, [ordered], True, (0, 255, 0), 2)

            # 在每条边上采样并绘制红点
            for i in range(4):
                p1 = ordered[i]
                p2 = ordered[(i + 1) % 4]
                samples = sample_edge(p1, p2, num_points_per_edge)
                for x, y in samples:
                    cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)

        # 实时显示
        cv2.imshow("Original + Samples", vis)
        cv2.imshow("Mask & Result",    result)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
