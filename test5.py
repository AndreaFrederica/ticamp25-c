#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import cv2
import numpy as np
import time


# A4 纸长宽比
A4_RATIO = 297.0 / 210.0  # 竖放时的长/短
TOL = 0.2  # ±20% 容差


def nothing(x):
    pass


def create_trackbars():
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    # HSV 滑条
    cv2.createTrackbar("H Min", "Controls", 0, 179, nothing)
    cv2.createTrackbar("H Max", "Controls", 180, 179, nothing)
    cv2.createTrackbar("S Min", "Controls", 0, 255, nothing)
    cv2.createTrackbar("S Max", "Controls", 255, 255, nothing)
    cv2.createTrackbar("V Min", "Controls", 0, 255, nothing)
    cv2.createTrackbar("V Max", "Controls", 85, 255, nothing)
    # Canny 滑条
    cv2.createTrackbar("Canny Min", "Controls", 50, 255, nothing)
    cv2.createTrackbar("Canny Max", "Controls", 150, 255, nothing)


def get_trackbar_values():
    h_min = cv2.getTrackbarPos("H Min", "Controls")
    h_max = cv2.getTrackbarPos("H Max", "Controls")
    s_min = cv2.getTrackbarPos("S Min", "Controls")
    s_max = cv2.getTrackbarPos("S Max", "Controls")
    v_min = cv2.getTrackbarPos("V Min", "Controls")
    v_max = cv2.getTrackbarPos("V Max", "Controls")
    c_min = cv2.getTrackbarPos("Canny Min", "Controls")
    c_max = cv2.getTrackbarPos("Canny Max", "Controls")
    return (h_min, h_max, s_min, s_max, v_min, v_max, c_min, c_max)


def detect_rectangles(frame, hsv_range, canny_range):
    """
    frame: BGR 图像
    hsv_range: (h_min, h_max, s_min, s_max, v_min, v_max)
    canny_range: (canny_min, canny_max)
    返回：mask, rects
    """
    h_min, h_max, s_min, s_max, v_min, v_max = hsv_range
    # 生成 HSV 掩码
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower, upper)

    # 形态学闭运算填平断裂
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Canny 边缘检测
    c_min, c_max = canny_range
    edges = cv2.Canny(mask, c_min, c_max)

    # 找轮廓并筛选四边形
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 500:
            pts = approx.reshape(4, 2)
            rects.append(pts)
    return mask, rects


# 计算四条边长度
def dist(p, q):
    return math.hypot(q[0] - p[0], q[1] - p[1])


def is_a4_quad(pts):
    """
    pts: (4,2) ndarray，无序顶点
    返回：是否近似 A4 纸的四边形（任意旋转角度）
    """
    # 排序：TL, TR, BR, BL
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    ordered = np.zeros((4, 2), dtype=float)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]

    # 计算四条边长度：top, right, bottom, left
    e0 = np.linalg.norm(ordered[1] - ordered[0])
    e1 = np.linalg.norm(ordered[2] - ordered[1])
    e2 = np.linalg.norm(ordered[2] - ordered[3])
    e3 = np.linalg.norm(ordered[3] - ordered[0])

    # 对边取平均
    long_edge = (e0 + e2) / 2.0
    short_edge = (e1 + e3) / 2.0
    if short_edge == 0:
        return False

    aspect = long_edge / short_edge
    # 考虑两种放置：竖放或横放
    return abs(aspect - A4_RATIO) < TOL or abs(aspect - (1.0 / A4_RATIO)) < TOL


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    create_trackbars()
    print("✅ 摄像头已打开，按 ESC 退出。")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv_vals = get_trackbar_values()[:6]
        canny_vals = get_trackbar_values()[6:]
        mask, rects = detect_rectangles(frame, hsv_vals, canny_vals)
        # 在画框之前，加入这段过滤代码：
        filtered = []
        for pts in rects:
            # pts 已是 (4,2) ndarray
            if is_a4_quad(pts):
                filtered.append(pts)
        rects = filtered

        vis = frame.copy()
        for pts in rects:
            pts = pts.astype(int)
            # 排序顶点 TL, TR, BR, BL
            s = pts.sum(axis=1)
            diff = np.diff(pts, axis=1).ravel()
            ordered = np.zeros((4, 2), dtype=int)
            ordered[0] = pts[np.argmin(s)]
            ordered[2] = pts[np.argmax(s)]
            ordered[1] = pts[np.argmin(diff)]
            ordered[3] = pts[np.argmax(diff)]

            # 画绿色轮廓
            cv2.polylines(vis, [ordered], True, (0, 255, 0), 2)

            # 计算四条边长度
            e_top = dist(ordered[0], ordered[1])
            e_right = dist(ordered[1], ordered[2])
            e_bottom = dist(ordered[2], ordered[3])
            e_left = dist(ordered[3], ordered[0])

            # 宽度取顶/底边平均，高度取左右边平均
            width_px = (e_top + e_bottom) / 2.0
            height_px = (e_right + e_left) / 2.0

            # 确定哪条是长/短
            if width_px < height_px:
                short_edge = width_px
                long_edge = height_px
                # 短边中点在顶底
                mid1 = (
                    (ordered[0][0] + ordered[1][0]) // 2,
                    (ordered[0][1] + ordered[1][1]) // 2,
                )
                mid2 = (
                    (ordered[2][0] + ordered[3][0]) // 2,
                    (ordered[2][1] + ordered[3][1]) // 2,
                )
            else:
                short_edge = height_px
                long_edge = width_px
                # 短边中点在左右
                mid1 = (
                    (ordered[1][0] + ordered[2][0]) // 2,
                    (ordered[1][1] + ordered[2][1]) // 2,
                )
                mid2 = (
                    (ordered[0][0] + ordered[3][0]) // 2,
                    (ordered[0][1] + ordered[3][1]) // 2,
                )

            # 画蓝色中点
            cv2.circle(vis, mid1, 6, (255, 0, 0), -1)
            cv2.circle(vis, mid2, 6, (255, 0, 0), -1)

            # 新长边：连接短边中点（红色）
            new_long = dist(mid1, mid2)
            cv2.line(vis, mid1, mid2, (0, 0, 255), 2)

            # 标注
            x0, y0 = ordered[0]
            cv2.putText(
                vis,
                f"L={long_edge:.1f}, S={short_edge:.1f}",
                (x0, y0 - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                vis,
                f"L_new={new_long:.1f}",
                (x0, y0 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                vis,
                f"Deep={(((524.38 / new_long) ** (1.0 / 1.003)) * 100):.1f} cm",
                (x0, y0 - 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )
            # TODO 强行塞了公式
        # for pts in rects:
        #     pts = pts.astype(int)
        #     cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
        #     for x, y in pts:
        #         cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)

        cv2.imshow("All Rectangles", vis)
        cv2.imshow("Mask", mask)
        cv2.imshow(
            "Controls", np.zeros((1, 400), dtype=np.uint8)
        )  # 只是为了保持滑条窗口不被最小化

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
