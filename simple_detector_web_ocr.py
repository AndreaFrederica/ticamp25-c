# main.py
import asyncio
from typing import Any, List
import cv2
import numpy as np
import threading
import time
import math
import json
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
import uvicorn
from scipy.spatial.distance import cdist
import easyocr

app = FastAPI()

vis: None | cv2.Mat = None
mask: None | cv2.Mat = None
crops: Any
cap_o = True

# 全局参数：支持双HSV范围和最小面积
params = {
    # range1
    "h1_min": 0,
    "h1_max": 179,
    "s1_min": 0,
    "s1_max": 255,
    "v1_min": 0,
    "v1_max": 85,
    # range2
    "h2_min": 0,
    "h2_max": 179,
    "s2_min": 0,
    "s2_max": 255,
    "v2_min": 0,
    "v2_max": 85,
    "use_range2": False,
    # "min_area": 200,
    # Canny
    "canny_min": 50,
    "canny_max": 150,
    "min_area": 40000,
    "max_area": 190000,
    "close_kernel_size": 10,  # Larger values close bigger gaps
    "max_line_gap": 50,  # Allow larger gaps in Hough lines
    "min_line_length": 100,
    "enable_ocr" : True
}

# 摄像头配置
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
if not cap.isOpened():
    raise RuntimeError("无法打开摄像头")

lock = threading.Lock()
frame = None
latest_stats = {}
clients: list[WebSocket] = []

# A4比例与容差
A4_RATIO = 297.0 / 210.0
TOL = 0.6

OCR_LANGS = ["la"]  # 若需中文改 ['ch_sim', 'en']
OCR_ALLOWLIST = "0123456789"  # 只识别数字；改 '' 为“不限制字符”
USE_GPU = False  # 树莓派/CPU 设 False

# 延迟初始化 EasyOCR.Reader（耗时操作）
_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(OCR_LANGS, gpu=USE_GPU)
    return _reader


def run_ocr_on_frame(frame: np.ndarray):
    """
    对单帧图像执行 OCR，返回列表，每项包含数字、置信度、四点坐标
    -------
    [{'text': '75', 'conf': 0.98,
      'bbox': [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}, ...]
    """
    if frame is None or frame.size == 0:
        return []

    reader = _get_reader()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = reader.readtext(
        rgb,
        allowlist=OCR_ALLOWLIST,
        detail=1,  # (bbox, text, conf)
    )

    parsed = []
    for bbox, text, conf in results:
        # 若只想要数字，可再次过滤
        if OCR_ALLOWLIST and any(c not in OCR_ALLOWLIST for c in text):
            continue
        parsed.append(
            {
                "text": text,
                "conf": float(conf),
                "bbox": [list(map(int, pt)) for pt in bbox],
            }
        )
    return parsed


# ------------------------------------------------------------------------------#
# 辅助函数 (完整替换后的版本)
# ------------------------------------------------------------------------------#


def dist(p, q):
    return math.hypot(q[0] - p[0], q[1] - p[1])


def midpoint(p, q):
    return ((p[0] + q[0]) // 2, (p[1] + q[1]) // 2)


def order_pts(pts):
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    tr, bl = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.int32)


def is_a4_quad(pts, a4_ratio=297 / 210, tol=0.2):
    o = order_pts(pts)
    e = [dist(o[i], o[(i + 1) % 4]) for i in range(4)]
    long_e, short_e = (e[0] + e[2]) / 2, (e[1] + e[3]) / 2
    if short_e == 0:
        return False
    asp = long_e / short_e
    return abs(asp - a4_ratio) < tol or abs(asp - 1 / a4_ratio) < tol


# 保持原有的identify_shape_by_area不变
# def identify_shape_by_area(contour):
#     ...
def identify_shape_by_area(contour, area_thresh=20):
    """
    识别轮廓的形状，并返回详细信息
    返回值：
        {
            "shape_type": "Circle"/"Triangle"/"Square"/"Unknown",
            "area": float,
            "width": int,
            "height": int,
            "contour": numpy.array,
            "info": str
        }
    """
    area = cv2.contourArea(contour)
    if area < area_thresh:
        return {
            "shape_type": "Unknown",
            "area": area,
            "width": 0,
            "height": 0,
            "contour": contour,
            "info": "面积太小",
        }

    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return {
            "shape_type": "Unknown",
            "area": area,
            "width": 0,
            "height": 0,
            "contour": contour,
            "info": "周长为0",
        }

    circularity = 4 * np.pi * area / (perimeter**2)
    x, y, w, h = cv2.boundingRect(contour)

    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = len(approx)

    shape_type = "Unknown"
    info = f"顶点数={vertices}, 圆度={circularity:.3f}, 宽高比={w / h:.3f}"

    if vertices == 3:
        shape_type = "Triangle"
        info += "; 三角形"
    elif vertices == 4:
        # 对于四边形，先计算四条边的长度
        side_lengths = [
            np.linalg.norm(approx[i][0] - approx[(i + 1) % 4][0]) for i in range(4)
        ]
        mean_len = np.mean(side_lengths)
        var_coeff = np.std(side_lengths) / mean_len if mean_len > 0 else 1
        aspect_ratio = float(w) / h if h > 0 else 0

        # 将变异系数加入 info
        info += f"; 变异系数={var_coeff:.3f}"

        # 判定为正方形：边长相似且长宽比接近 1
        if var_coeff < 0.4:
            shape_type = "Square"
            # 将宽高设置为平均边长
            w = h = int(mean_len)
            info += "; 正方形"
        else:
            info += "; 四边形但非正方形"
    elif circularity > 0.6:
        shape_type = "Circle"
        info += "; 圆形"
    elif vertices >= 6 and circularity > 0.4:
        shape_type = "Circle"
        info += "; 近似圆形"
    else:
        info += "; 未知形状"

    return {
        "shape_type": shape_type,
        "area": area,
        "width": w
        if shape_type != "Circle"
        else int(cv2.minEnclosingCircle(contour)[1] * 2),
        "height": h
        if shape_type != "Circle"
        else int(cv2.minEnclosingCircle(contour)[1] * 2),
        "contour": contour,
        "info": info,
    }


# ------------------------------------------------------------------------------#
# 阶段 1：生成二值掩码（双 HSV 范围 + 形态学）
# ------------------------------------------------------------------------------#
def build_mask(hsv, params):
    lower1 = np.array([params["h1_min"], params["s1_min"], params["v1_min"]])
    upper1 = np.array([params["h1_max"], params["s1_max"], params["v1_max"]])
    mask1 = cv2.inRange(hsv, lower1, upper1)

    if params["use_range2"]:
        lower2 = np.array([params["h2_min"], params["s2_min"], params["v2_min"]])
        upper2 = np.array([params["h2_max"], params["s2_max"], params["v2_max"]])
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        mask = mask1

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


# ------------------------------------------------------------------------------#
# 阶段 2：轮廓检测与过滤（只保留近似 A4 的四边形）
# ------------------------------------------------------------------------------#
# def find_a4_quads(mask, params):
#     edges = cv2.Canny(mask, params["canny_min"], params["canny_max"])
#     cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     quads = []
#     for cnt in cnts:
#         if cv2.contourArea(cnt) < MIN_AREA:
#             continue
#         peri = cv2.arcLength(cnt, True)
#         approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
#         if len(approx) == 4 and is_a4_quad(approx.reshape(4, 2)):
#             quads.append(approx.reshape(4, 2))
#     return quads
def find_a4_quads(mask: np.ndarray, params: dict) -> list[np.ndarray]:
    """
    参数
    ----
    mask   : 二值/灰度单通道图。白色(非零)区域为前景
    params : {
        "canny_min": int,
        "canny_max": int,
        # 可选：形态学核 & 迭代次数；不传则用默认
        "close_kernel": int (odd),   # e.g. 7
        "close_iter":   int          # e.g. 1~3
    }

    返回
    ----
    quads : list[np.ndarray]，每个元素形状为 (4, 2)，float32/int32 均可
    """
    # 1. 边缘检测
    edges = cv2.Canny(mask, params["canny_min"], params["canny_max"])

    # 2. 形态学闭运算：修补缝隙
    ksize = params.get("close_kernel", 3)
    iters = params.get("close_iter", 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=iters)

    # 3. 轮廓提取
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    quads: list[np.ndarray] = []
    for cnt in cnts:
        # 面积过滤
        if cv2.contourArea(cnt) < params["min_area"]:
            continue
        if cv2.contourArea(cnt) > params["max_area"]:
            continue

        # 3.1 取凸包，让断裂边补齐
        hull = cv2.convexHull(cnt)

        # 3.2 多边形近似
        peri = cv2.arcLength(hull, closed=True)
        approx = cv2.approxPolyDP(hull, 0.02 * peri, closed=True)

        # —— 情况 A：近似直接给出 4 点 —— #
        if len(approx) == 4:
            quad = approx.reshape(4, 2)
        else:
            # —— 情况 B：兜底用 minAreaRect —— #
            rect = cv2.minAreaRect(hull)
            quad = cv2.boxPoints(rect)  # shape = (4,2), float32
            quad = quad.astype(np.float32)

        # 3.3 自定义比例校验（纵横比 / 大小）
        if is_a4_quad(quad):  # 用户已有的判定函数
            quads.append(quad)

    return quads


# def annotate_outer(img, quads, stats, params):
#     crops = []
#     for idx, pts in enumerate(quads, start=1):
#         # —— 新增：过滤掉面积小于 min_area 的外框 —— #
#         area = cv2.contourArea(pts.reshape(-1,1,2))
#         if area < params['min_area']:
#             continue
#         # —— 下面为原有逻辑 —— #
#         o = order_pts(pts)
#         e = [dist(o[i], o[(i+1)%4]) for i in range(4)]
#         long_e = (e[0] + e[2]) / 2
#         short_e = (e[1] + e[3]) / 2
#         if e[0] < e[1]:
#             m1, m2 = midpoint(o[0], o[1]), midpoint(o[2], o[3])
#         else:
#             m1, m2 = midpoint(o[1], o[2]), midpoint(o[0], o[3])
#         new_e = dist(m1, m2)

#         x, y, w, h = cv2.boundingRect(o)
#         roi = img[y:y+h, x:x+w].copy()

#         stats['rects'].append({
#             'id': idx,
#             'outer_width': int(long_e),
#             'outer_height': int(short_e),
#             'new_long_px': float(new_e),
#             'area': int(area),
#             'outer_pts': o.tolist(),
#             'midpoints': [(int(m1[0]),int(m1[1])),(int(m2[0]),int(m2[1]))],
#             'position': (int(x),int(y))
#         })
#         crops.append(roi)
#     stats['count'] = len(stats['rects'])
#     return crops


# def annotate_inner(crops, params, stats):
#     hsv1_lower = np.array([params["h1_min"], params["s1_min"], params["v1_min"]])
#     hsv1_upper = np.array([params["h1_max"], params["s1_max"], params["v1_max"]])
#     hsv2_lower = (
#         np.array([params["h2_min"], params["s2_min"], params["v2_min"]])
#         if params["use_range2"]
#         else None
#     )
#     hsv2_upper = (
#         np.array([params["h2_max"], params["s2_max"], params["v2_max"]])
#         if params["use_range2"]
#         else None
#     )

#     for idx, roi in enumerate(crops):
#         gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#         _, white = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
#         w_cnts, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if not w_cnts:
#             continue

#         wx, wy, ww, wh = cv2.boundingRect(max(w_cnts, key=cv2.contourArea))
#         inner = roi[wy : wy + wh, wx : wx + ww]
#         hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
#         mask = cv2.inRange(hsv, hsv1_lower, hsv1_upper)
#         if params["use_range2"]:
#             mask2 = cv2.inRange(hsv, hsv2_lower, hsv2_upper)
#             mask = cv2.bitwise_or(mask, mask2)

#         k = np.ones((3, 3), np.uint8)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

#         s_cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if not s_cnts:
#             continue

#         shape = max(s_cnts, key=cv2.contourArea)
#         shape_result = identify_shape_by_area(shape)

#         stats["rects"][idx].update(
#             {
#                 "shape_type": shape_result["shape_type"],
#                 "inner_width": int(shape_result["width"]),
#                 "inner_height": int(shape_result["height"]),
#                 "inner_area": int(shape_result["area"]),
#                 "inner_info": shape_result["info"],
#                 "inner_contour": shape_result["contour"]
#                 .reshape(-1, 2)
#                 .tolist(),  # JSON serializable
#             }
#         )


def safe_crop(img: np.ndarray, rect) -> np.ndarray | None:
    """将 (x, y, w, h) 裁剪到合法范围；若宽高 ≤0 返回 None"""
    x, y, w, h = rect
    H, W = img.shape[:2]

    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)

    if x1 - x0 <= 0 or y1 - y0 <= 0:  # 空 ROI
        return None
    return img[y0:y1, x0:x1]


# ———————————————————— 外层矩形 ————————————————————
def annotate_outer(
    img: np.ndarray, quads: List[np.ndarray], stats: dict, params: dict
) -> List[np.ndarray]:
    """
    返回 crops: List[roi]；并向 stats['rects'] 追加条目
    """
    crops = []
    min_area = params.get("min_area", 8_000)
    min_side_px = params.get("min_side_px", 5)

    for idx, pts in enumerate(quads, 1):
        # 1. 面积过滤
        area = cv2.contourArea(pts.reshape(-1, 1, 2))
        if area < min_area:
            continue

        # 2. 原有逻辑
        o = order_pts(pts)
        e = [dist(o[i], o[(i + 1) % 4]) for i in range(4)]
        long_e, short_e = (e[0] + e[2]) / 2, (e[1] + e[3]) / 2
        if e[0] < e[1]:
            m1, m2 = midpoint(o[0], o[1]), midpoint(o[2], o[3])
        else:
            m1, m2 = midpoint(o[1], o[2]), midpoint(o[0], o[3])
        new_e = dist(m1, m2)

        # 3. ROI 裁剪 + 尺寸校验
        x, y, w, h = cv2.boundingRect(o)
        if w < min_side_px or h < min_side_px:
            continue
        roi = safe_crop(img, (x, y, w, h))
        if roi is None:
            continue

        # 4. 记录信息
        stats["rects"].append(
            {
                "id": idx,
                "outer_width": int(long_e),
                "outer_height": int(short_e),
                "new_long_px": float(new_e),
                "area": int(area),
                "outer_pts": o.tolist(),
                "midpoints": [(int(m1[0]), int(m1[1])), (int(m2[0]), int(m2[1]))],
                "position": (int(x), int(y)),
            }
        )
        crops.append(roi)

    stats["count"] = len(stats["rects"])
    return crops


# ———————————————————— 内层形状 ————————————————————
def annotate_inner(crops, params, stats):
    hsv1_lower = np.array([params["h1_min"], params["s1_min"], params["v1_min"]])
    hsv1_upper = np.array([params["h1_max"], params["s1_max"], params["v1_max"]])
    if params["use_range2"]:
        hsv2_lower = np.array([params["h2_min"], params["s2_min"], params["v2_min"]])
        hsv2_upper = np.array([params["h2_max"], params["s2_max"], params["v2_max"]])

    for idx, roi in enumerate(crops):
        # ---------- 双保险：空 ROI 直接跳过 ----------
        if roi is None or roi.size == 0:
            continue

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, white = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        w_cnts, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not w_cnts:
            continue

        wx, wy, ww, wh = cv2.boundingRect(max(w_cnts, key=cv2.contourArea))
        inner = safe_crop(roi, (wx, wy, ww, wh))
        if inner is None or inner.size == 0:
            continue

        hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv1_lower, hsv1_upper)
        if params["use_range2"]:
            mask2 = cv2.inRange(hsv, hsv2_lower, hsv2_upper)
            mask = cv2.bitwise_or(mask, mask2)

        k = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        s_cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not s_cnts:
            continue

        shape = max(s_cnts, key=cv2.contourArea)
        shape_result = identify_shape_by_area(shape)

        # 确保外层统计索引对应
        if idx < len(stats["rects"]):
            stats["rects"][idx].update(
                {
                    "shape_type": shape_result["shape_type"],
                    "inner_width": int(shape_result["width"]),
                    "inner_height": int(shape_result["height"]),
                    "inner_area": int(shape_result["area"]),
                    "inner_info": shape_result["info"],
                    "inner_contour": shape_result["contour"].reshape(-1, 2).tolist(),
                }
            )


def draw_annotations(img, stats):
    for rect in stats["rects"]:
        # 将列表转换回 numpy 数组
        o = np.array(rect["outer_pts"], dtype=np.int32)
        m1, m2 = rect["midpoints"]
        new_e = rect["new_long_px"]
        x, y = rect["position"]

        # 绘制外框
        cv2.polylines(img, [o], True, (0, 255, 0), 2)
        cv2.circle(img, tuple(m1), 6, (255, 0, 0), -1)
        cv2.circle(img, tuple(m2), 6, (255, 0, 0), -1)
        cv2.line(img, tuple(m1), tuple(m2), (0, 0, 255), 2)
        cv2.putText(
            img,
            f"L_new={new_e:.1f}",
            tuple(o[0] - [0, 5]),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

        # 绘制内部图形（坐标转换为原图位置）
        if "inner_contour" in rect:
            inner_cnt_global = np.array(
                rect["inner_contour"], dtype=np.int32
            ) + np.array([x, y])
            color = {
                "Circle": (255, 0, 255),
                "Triangle": (0, 255, 255),
                "Square": (255, 255, 0),
            }.get(rect["shape_type"], (0, 0, 0))
            cv2.polylines(img, [inner_cnt_global], True, color, 3)


def detect_and_annotate(img):
    global crops
    stats = {"count": 0, "rects": [], "area_sum": 0}
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = build_mask(hsv, params)
    # quads = find_broken_quads_combined(mask, params)
    quads = find_a4_quads(mask, params)

    # 先外部标注并裁剪图像
    crops = annotate_outer(img, quads, stats, params)

    # edges = cv2.Canny(mask, 50, 150)

    # # 2. 在边缘图上检测角点
    # corners = cv2.goodFeaturesToTrack(
    #     edges,        # 输入：边缘图
    #     maxCorners=100,
    #     qualityLevel=0.01,
    #     minDistance=10
    # )

    # # 3. 画出角点
    # for pt in corners:
    #     x, y = pt.ravel().astype(int)
    #     cv2.circle(img, (x, y), 5, (0, 0, 255), 2)

    # ocr_result = dict()
    # if crops and params["enable_ocr"]:
    #     for idx,image in enumerate(crops):
    #         ocr_result[idx] = run_ocr_on_frame(image)
    # stats["ocr_result"] = ocr_result
    # 内部标注，使用裁剪后的图像
    annotate_inner(crops, params, stats)
    draw_annotations(img, stats)
    # 明确转换为内置类型，确保可序列化
    for rect in stats["rects"]:
        for key, value in rect.items():
            if isinstance(value, (np.integer,)):
                rect[key] = int(value)
            elif isinstance(value, (np.floating,)):
                rect[key] = float(value)

    return img, mask, stats


# 捕获循环
prev_time = time.time()



def _capture_loop():
    global frame, latest_stats, prev_time, mask, vis, cap_o
    while True:
        if not cap_o:
            time.sleep(1)
            continue
        ret, img = cap.read()
        if not ret:
            continue
        with lock:
            frame = img.copy()
        vis, mask, stats = detect_and_annotate(img.copy())
        total_pixels = img.shape[0] * img.shape[1]
        black_pixels = int((mask == 255).sum())
        now = time.time()
        fps = 1 / (now - prev_time) if now != prev_time else 0
        prev_time = now
        frame_ratio = stats["area_sum"] / total_pixels * 100
        black_ratio = black_pixels / total_pixels * 100
        latest_stats = {
            "count": stats["count"],
            "total_pixels": int(total_pixels),
            "frame_ratio": int(frame_ratio),
            "black_ratio": int(black_ratio),
            "fps": int(fps),
            "rects": stats["rects"],
        }
        time.sleep(0.03)


threading.Thread(target=_capture_loop, daemon=True).start()


# MJPEG
async def broadcast_loop():
    while True:
        status = latest_stats.copy()
        # new = []
        if "rects" in status.keys():
            for item in status["rects"]:
                try:
                    item.pop("outer_pts")
                    item.pop("midpoints")
                    item.pop("inner_contour")
                # new.append(item)
                except:
                    pass

        text = json.dumps(status)
        bad = []
        for ws in clients:
            try:
                await ws.send_text(text)
            except WebSocketDisconnect:
                bad.append(ws)
        for ws in bad:
            clients.remove(ws)
        await asyncio.sleep(0.1)


def mjpeg_generator(proc):
    global mask, vis
    while True:
        with lock:
            img = frame.copy() if frame is not None else None
        if img is None:
            time.sleep(0.01)
            continue
        try:
            if proc == "vis":
                tgt = vis
            elif proc == "mask":
                tgt = mask

            _, buf = cv2.imencode(".jpg", tgt)
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )

        except:
            pass

        time.sleep(0.03)


# routes
@app.get("/video/processed")
def video_processed():
    return StreamingResponse(
        mjpeg_generator("vis"), media_type="multipart/x-mixed-replace;boundary=frame"
    )


@app.get("/video/mask")
def video_mask():
    return StreamingResponse(
        mjpeg_generator("mask"), media_type="multipart/x-mixed-replace;boundary=frame"
    )


@app.post("/control/hsv")
async def set_hsv(req: Request):
    d = await req.json()
    for k in [
        "h1_min",
        "h1_max",
        "s1_min",
        "s1_max",
        "v1_min",
        "v1_max",
        "h2_min",
        "h2_max",
        "s2_min",
        "s2_max",
        "v2_min",
        "v2_max",
    ]:
        if k not in d:
            raise HTTPException(400)
        params[k] = int(d[k])
    params["use_range2"] = bool(d.get("use_range2", 0))
    params["min_area"] = int(d.get("min_area", 200))
    return {"params": params}


@app.post("/control/canny")
async def set_canny(req: Request):
    d = await req.json()
    for k in ["canny_min", "canny_max"]:
        if k not in d:
            raise HTTPException(400)
        params[k] = int(d[k])
    return {"params": params}


@app.post("/control/area")
async def set_area(req: Request):
    d: dict = await req.json()
    print(d)
    if "max_area" in d.keys():
        params["max_area"] = d["max_area"]
    if "min_area" in d.keys():
        params["min_area"] = d["min_area"]
    return {"params": params}


@app.get("/get/params")
async def get_params(req: Request) -> dict:
    return params

@app.get("/ocr")
async def detect_numbers():
    global crops, cap_o
    cap_o = False
    ocr_result = dict()
    if crops and params["enable_ocr"]:
        for idx,image in enumerate(crops):
            ocr_result[idx] = run_ocr_on_frame(image)
    cap_o = True
    return ocr_result


@app.websocket("/ws")
async def ws_ep(ws: WebSocket):
    await ws.accept()
    clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        clients.remove(ws)


@app.on_event("startup")
async def startup():
    asyncio.create_task(broadcast_loop())


@app.get("/", response_class=HTMLResponse)
def index():
    with open("simple_detector.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
