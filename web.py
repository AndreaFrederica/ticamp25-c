# main.py
import asyncio
import cv2
import numpy as np
import threading
import time
import math
import json
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
import uvicorn

app = FastAPI()

# 全局参数
params = {
    "h_min": 0, "h_max": 179,
    "s_min": 0, "s_max": 255,
    "v_min": 0, "v_max": 85,
    "canny_min": 50, "canny_max": 150
}

# 摄像头读取配置
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
if not cap.isOpened():
    raise RuntimeError("无法打开摄像头")

lock = threading.Lock()
frame = None
latest_stats = {}
clients: list[WebSocket] = []

# A4 纸比例与容差
A4_RATIO = 297.0 / 210.0
TOL = 0.2

def dist(p, q):
    return math.hypot(q[0] - p[0], q[1] - p[1])

def midpoint(p, q):
    return ((p[0] + q[0]) // 2, (p[1] + q[1]) // 2)

def order_pts(pts):
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]; bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.int32)

def is_a4_quad(pts):
    o = order_pts(pts)
    e = [dist(o[i], o[(i+1)%4]) for i in range(4)]
    long_e = (e[0] + e[2]) / 2
    short_e = (e[1] + e[3]) / 2
    if short_e == 0:
        return False
    asp = long_e / short_e
    return abs(asp - A4_RATIO) < TOL or abs(asp - 1/A4_RATIO) < TOL

def detect_and_annotate(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([params['h_min'], params['s_min'], params['v_min']])
    upper = np.array([params['h_max'], params['s_max'], params['v_max']])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (1,1)), iterations=2)
    edges = cv2.Canny(mask, params['canny_min'], params['canny_max'])
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stats = {"count": 0, "rects": []}
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and cv2.contourArea(approx) > 500:
            pts = approx.reshape(4, 2)
            if is_a4_quad(pts):
                o = order_pts(pts)
                e = [dist(o[i], o[(i+1)%4]) for i in range(4)]
                long_e = (e[0] + e[2]) / 2
                short_e = (e[1] + e[3]) / 2
                # 计算新长边
                if e[0] < e[1]:
                    m1, m2 = midpoint(o[0], o[1]), midpoint(o[2], o[3])
                else:
                    m1, m2 = midpoint(o[1], o[2]), midpoint(o[0], o[3])
                new_e = dist(m1, m2)
                # 绘制标注
                cv2.polylines(img, [o], True, (0,255,0), 2)
                cv2.circle(img, m1, 6, (255,0,0), -1)
                cv2.circle(img, m2, 6, (255,0,0), -1)
                cv2.line(img, m1, m2, (0,0,255), 2)
                cv2.putText(img, f"L_new={new_e:.1f}", tuple(o[0] - [0,5]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                stats['rects'].append({
                    'width_px': long_e,
                    'height_px': short_e,
                    'new_long_px': new_e
                })
    stats['count'] = len(stats['rects'])
    return img, mask, stats

# 后台捕获循环

def capture_loop():
    global frame, latest_stats
    while True:
        ret, img = cap.read()
        if not ret:
            continue
        with lock:
            frame = img.copy()
        _, _, stats = detect_and_annotate(img.copy())
        latest_stats = stats
        time.sleep(0.03)

threading.Thread(target=capture_loop, daemon=True).start()

# MJPEG 流生成器

def mjpeg_generator(proc):
    while True:
        with lock:
            img = frame.copy() if frame is not None else None
        if img is None:
            time.sleep(0.01)
            continue
        vis, mask, _ = detect_and_annotate(img)
        tgt = vis if proc=='vis' else mask
        _, buf = cv2.imencode('.jpg', tgt)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.03)

@app.get('/video/processed')
def video_processed():
    return StreamingResponse(mjpeg_generator('vis'), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get('/video/mask')
def video_mask():
    return StreamingResponse(mjpeg_generator('mask'), media_type='multipart/x-mixed-replace; boundary=frame')

@app.post('/control/hsv')
async def set_hsv(req: Request):
    data = await req.json()
    for k in ['h_min','h_max','s_min','s_max','v_min','v_max']:
        if k not in data:
            raise HTTPException(status_code=400, detail=f"缺少 {k}")
        params[k] = int(data[k])
    return {"params": params}

@app.post('/control/canny')
async def set_canny(req: Request):
    data = await req.json()
    for k in ['canny_min','canny_max']:
        if k not in data:
            raise HTTPException(status_code=400, detail=f"缺少 {k}")
        params[k] = int(data[k])
    return {"params": params}

# WebSocket 广播
@app.websocket('/ws')
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        clients.remove(ws)

# 在启动时创建广播任务
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(broadcast_loop())

async def broadcast_loop():
    while True:
        text = json.dumps(latest_stats)
        bad = []
        for ws in clients:
            try:
                await ws.send_text(text)
            except WebSocketDisconnect:
                bad.append(ws)
        for ws in bad:
            clients.remove(ws)
        await asyncio.sleep(0.1)

# 静态页面路由
@app.get('/', response_class=HTMLResponse)
def index():
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        raise RuntimeError('index.html 文件未找到')

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)