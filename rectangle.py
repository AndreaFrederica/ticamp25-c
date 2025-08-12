import cv2
import numpy as np
import time


# 1. 图像预处理与矩形检测
def detect_rectangle(image_path):
    img = cv2.imread(image_path)
    # 黑色阈值，可根据光照微调 V 上限
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    mask = cv2.inRange(hsv, lower_black, upper_black)
    result = cv2.bitwise_and(img, img, mask=mask)
    #需要实时显示result
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    rect_contour = None

    # 筛选面积最大的矩形轮廓
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:  # 确保是四边形
            area = cv2.contourArea(approx)
            if area > max_area:
                max_area = area
                rect_contour = approx

    if rect_contour is None:
        raise ValueError("未检测到矩形")
    return img, rect_contour.reshape(4, 2)


# 2. 顶点排序（左上→右上→右下→左下）
def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下
    return rect


# 3. 边点采样（线性插值）
def sample_edge(pt1, pt2, num_points):
    return [
        (
            int(pt1[0] + (pt2[0] - pt1[0]) * i / (num_points - 1)),
            int(pt1[1] + (pt2[1] - pt1[1]) * i / (num_points - 1)),
        )
        for i in range(num_points)
    ]


# 4.红色点追踪
def find_green_point(image_path):
    ball_color = "red"
    color_dist = {
        "red": {"Lower": np.array([0, 60, 60]), "Upper": np.array([6, 255, 255])},
        "blue": {"Lower": np.array([100, 80, 46]), "Upper": np.array([124, 255, 255])},
        "green": {"Lower": np.array([35, 43, 35]), "Upper": np.array([90, 255, 255])},
    }
    frame = cv2.imread(image_path)
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    erode_hsv = cv2.erode(blurred, None, iterations=2)
    inRange_hsv = cv2.inRange(
        erode_hsv, color_dist[ball_color]["Lower"], color_dist[ball_color]["Upper"]
    )
    cnts = cv2.findContours(
        inRange_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[-2]
    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    cv2.drawContours(frame, [np.int0(box)], -1, (0, 255, 255), 2)


# 主流程
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # 创建摄像头对象，参数0表示默认摄像头
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 设置帧宽度
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 设置帧高度[1,4,7](@ref)
while True:  # 循环
    time.sleep(0.01)  # 等待摄像头稳定
    ret, frame = cap.read()
    if not ret:
        print("帧读取失败")
        break
    cv2.imwrite("picture.jpg", frame)
    image_path = "picture.jpg"  # 替换为你的图像路径
    try:
        img, vertices = detect_rectangle(image_path)
        ordered_vertices = order_points(vertices)

        # 每条边采样点（可调整密度）
        num_points_per_edge = 30
        edge_points = {}

        for i in range(4):
            start_pt = ordered_vertices[i]
            end_pt = ordered_vertices[(i + 1) % 4]  # 这两行能够把矩形连在一起
            edge_name = ["上边", "右边", "下边", "左边"][i]
            edge_points[edge_name] = sample_edge(start_pt, end_pt, num_points_per_edge)
            # 可视化：绘制采样点
            for pt in edge_points[edge_name]:
                cv2.circle(img, pt, 3, (0, 0, 255), -1)
        cv2.imshow("Image Matching System", img)
        # 输出坐标（按边分组）
        print("===== 矩形四条边的点坐标 =====")
        for edge_name, points in edge_points.items():
            print(f"\n**{edge_name}**（共{len(points)}点）:")
            for i, pt in enumerate(points):
                print(f"  点 {i + 1}: ({pt[0]}, {pt[1]})")
        # 保存结果
        cv2.imwrite("sampled_edges.jpg", img)
        print("\n可视化结果已保存至: sampled_edges.jpg")
        key = cv2.waitKey(1)  # 等待1ms并获取按键
        if key == 27:  # ESC键ASCII码
            break
    except Exception:
        pass
