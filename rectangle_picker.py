#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np

def detect_rectangle(frame):
    """
    输入 BGR 图像，返回：
      - mask: 只保留黑色边框区域的二值图
      - pts: 纸张四个顶点 (4,2)，如果未检测到返回 None
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0,   0,   0])
    upper_black = np.array([180, 255, 85])  # 根据实际环境调整 V 上限
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # 形态学填充，尽量修补断裂
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # 在“实心”mask 上直接找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask, None

    # 取最大轮廓
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 10000:
        return mask, None

    # 最小外接矩形 → 4 顶点
    rect = cv2.minAreaRect(cnt)
    box  = cv2.boxPoints(rect)         # float32 (4,2)
    pts  = box.astype(int).reshape(4,2)
    return mask, pts

def order_points(pts):
    """
    将无序的 4 个顶点排序为 [TL, TR, BR, BL]
    """
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def sample_edge(p1, p2, n):
    """
    在 p1→p2 之间等距采 n 个点
    """
    return [
        (
            int(p1[0] + (p2[0]-p1[0]) * i/(n-1)),
            int(p1[1] + (p2[1]-p1[1]) * i/(n-1))
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

    points_per_edge = 30  # 每条边采样点数

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask, pts = detect_rectangle(frame)
        vis = frame.copy()

        if pts is not None:
            # 排序顶点
            ordered = order_points(pts)

            # 画出矩形轮廓
            cv2.polylines(vis, [ordered.astype(int)], True, (0,255,0), 2)

            # 四条边上采样并绘制红点
            edge_names = ["上边","右边","下边","左边"]
            for i in range(4):
                p1 = ordered[i]
                p2 = ordered[(i+1)%4]
                samples = sample_edge(p1, p2, points_per_edge)
                for pt in samples:
                    cv2.circle(vis, pt, 3, (0,0,255), -1)

        # 显示结果
        cv2.imshow("原图 + 采样点", vis)
        cv2.imshow("黑框掩码", mask)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
