#按D键切换调试模式
#按M键切换多颜色模式
#数字1-5切换颜色开关
import cv2
import numpy as np
import time
import serial
import threading
from queue import Queue, PriorityQueue
from usbttl import send_usb_ttl
HEADER = 0xAA, 0x55 # 2-byte header
FOOTER = 0x55       # 1-byte footer

# 全局优先级队列用于线程间通信
data_queue = PriorityQueue()
detection_running = True

# 数据类型标识
COLOR_BLOCK_TYPE = 'color_block'  # 改为通用的颜色块类型
RECTANGLE_TYPE = 'rectangle'

# 调试模式控制
debug_mode = True  # 设置为False可关闭所有调试窗口和输出

# 多颜色检测开关
multi_color_mode = False  # 设置为True启用多颜色同时检测，False为单颜色检测

# 是否检测矩形
detect_rectangles = False  # 设置为True启用矩形检测，False只检测颜色

# 多颜色模式下各颜色的开关状态
color_enabled = {
    0: True,   # 红色默认开启
    1: False,   # 绿色默认开启
    2: True,   # 蓝色默认开启
    3: False,  # 黄色默认关闭
    4: False,  # 橙色默认关闭
    5: False,  # 紫色默认关闭
}

# 目标颜色选择 (0=红色, 1=绿色, 2=蓝色, 3=黄色, 4=橙色, 5=紫色)
target_color_index = 5  # 默认设置为紫色

# 预定义颜色范围（包含颜色ID）
color_ranges = {
    0: {'name': '红色', 'id': 0x00, 'lower1': [161, 113, 18], 'upper1': [179, 255, 255], 
        'lower2': [161, 123, 217], 'upper2': [179, 255, 255], 'use_dual_range': True},
    1: {'name': '绿色', 'id': 0x01, 'lower1': [40, 50, 50], 'upper1': [80, 255, 255], 
        'lower2': [0, 0, 0], 'upper2': [0, 0, 0], 'use_dual_range': False},
    2: {'name': '蓝色', 'id': 0x02, 'lower1': [105, 92, 18], 'upper1': [118, 105, 255], 
        'lower2': [82, 72, 50], 'upper2': [161, 255, 255], 'use_dual_range': True},
    3: {'name': '黄色', 'id': 0x03, 'lower1': [20, 50, 50], 'upper1': [30, 255, 255], 
        'lower2': [0, 0, 0], 'upper2': [0, 0, 0], 'use_dual_range': False},
    4: {'name': '橙色', 'id': 0x04, 'lower1': [10, 50, 50], 'upper1': [20, 255, 255], 
        'lower2': [0, 0, 0], 'upper2': [0, 0, 0], 'use_dual_range': False},
    5: {'name': '紫色', 'id': 0x05, 'lower1': [130, 50, 50], 'upper1': [160, 255, 255], 
        'lower2': [0, 0, 0], 'upper2': [0, 0, 0], 'use_dual_range': False},
}

# 阈值调试参数（全局变量）- 详细说明
threshold_params = {
    # Canny边缘检测参数
    'canny_low': 50,         # Canny低阈值(0-200): 检测弱边缘，值越小检测到的边缘越多
    'canny_high': 150,       # Canny高阈值(50-300): 检测强边缘，值越大检测到的边缘越少
    'gaussian_blur': 5,      # 高斯模糊核大小(1-15): 降噪，值越大图像越模糊
    
    # HSV颜色检测参数
    'red_h_low1': 0,         # 色相下限1(0-179): HSV中的H值，决定颜色类型
    'red_h_high1': 10,       # 色相上限1(0-179): 配合H下限定义颜色范围
    'red_h_low2': 170,       # 色相下限2(0-179): 红色需要双范围(0-10和170-179)
    'red_h_high2': 180,      # 色相上限2(0-179): 红色在HSV色轮两端
    'red_s_low': 50,         # 饱和度下限(0-255): 值越高颜色越鲜艳，排除灰色
    'red_v_low': 50,         # 明度下限(0-255): 值越高排除暗色，避免阴影干扰
    
    # 形态学处理参数
    'morphology_kernel': 7   # 形态学核大小(3-15): 消除噪点和填补空洞，值越大效果越强
}

def update_color_params():
    """根据当前选择的颜色更新参数"""
    global threshold_params, target_color_index
    current_color = color_ranges[target_color_index]
    threshold_params['red_h_low1'] = current_color['lower1'][0]
    threshold_params['red_h_high1'] = current_color['upper1'][0]
    threshold_params['red_h_low2'] = current_color['lower2'][0]
    threshold_params['red_h_high2'] = current_color['upper2'][0]
    threshold_params['red_s_low'] = current_color['lower1'][1]
    threshold_params['red_v_low'] = current_color['lower1'][2]

def handle_keyboard_input(key):
    """处理键盘输入调节参数"""
    global threshold_params, target_color_index, multi_color_mode, detect_rectangles, color_enabled
    
    # 颜色切换 (1-6键)
    if ord('1') <= key <= ord('6'):
        target_color_index = key - ord('1')
        if target_color_index < len(color_ranges):
            update_color_params()
            color_name = color_ranges[target_color_index]['name']
            print(f"切换到检测颜色: {color_name}")
            return True
    
    # Canny参数调节
    elif key == ord('w'):  # W键增加Canny低阈值
        threshold_params['canny_low'] = min(200, threshold_params['canny_low'] + 5)
        print(f"Canny低阈值: {threshold_params['canny_low']}")
        return True
    elif key == ord('s'):  # S键减少Canny低阈值
        threshold_params['canny_low'] = max(0, threshold_params['canny_low'] - 5)
        print(f"Canny低阈值: {threshold_params['canny_low']}")
        return True
    elif key == ord('e'):  # E键增加Canny高阈值
        threshold_params['canny_high'] = min(300, threshold_params['canny_high'] + 10)
        print(f"Canny高阈值: {threshold_params['canny_high']}")
        return True
    elif key == ord('d'):  # D键减少Canny高阈值(冲突，改为x)
        return False  # 让D键保持调试模式切换功能
    elif key == ord('x'):  # X键减少Canny高阈值
        threshold_params['canny_high'] = max(50, threshold_params['canny_high'] - 10)
        print(f"Canny高阈值: {threshold_params['canny_high']}")
        return True
    
    # 颜色H值调节
    elif key == ord('r'):  # R键增加H低值1
        threshold_params['red_h_low1'] = min(179, threshold_params['red_h_low1'] + 5)
        print(f"颜色H低值1: {threshold_params['red_h_low1']}")
        return True
    elif key == ord('f'):  # F键减少H低值1
        threshold_params['red_h_low1'] = max(0, threshold_params['red_h_low1'] - 5)
        print(f"颜色H低值1: {threshold_params['red_h_low1']}")
        return True
    elif key == ord('t'):  # T键增加H高值1
        threshold_params['red_h_high1'] = min(179, threshold_params['red_h_high1'] + 5)
        print(f"颜色H高值1: {threshold_params['red_h_high1']}")
        return True
    elif key == ord('g'):  # G键减少H高值1
        threshold_params['red_h_high1'] = max(0, threshold_params['red_h_high1'] - 5)
        print(f"颜色H高值1: {threshold_params['red_h_high1']}")
        return True
    
    # 形态学核大小调节
    elif key == ord('z'):  # Z键增加核大小
        threshold_params['morphology_kernel'] = min(15, threshold_params['morphology_kernel'] + 2)
        print(f"形态学核大小: {threshold_params['morphology_kernel']}")
        return True
    elif key == ord('c'):  # C键减少核大小
        threshold_params['morphology_kernel'] = max(3, threshold_params['morphology_kernel'] - 2)
        print(f"形态学核大小: {threshold_params['morphology_kernel']}")
        return True
    
    # 多颜色模式切换
    elif key == ord('m') or key == ord('M'):  # M键切换多颜色模式
        multi_color_mode = not multi_color_mode
        mode_text = "多颜色同时检测" if multi_color_mode else "单一颜色检测"
        print(f"切换检测模式: {mode_text}")
        if multi_color_mode:
            enabled_colors = [color_ranges[i]['name'] for i, enabled in color_enabled.items() if enabled]
            print(f"当前启用的颜色: {', '.join(enabled_colors)}")
            print("按 Ctrl+1到6 切换对应颜色的开关状态")
        return True
    
    # 矩形检测开关
    elif key == ord('n') or key == ord('N'):  # N键切换矩形检测
        detect_rectangles = not detect_rectangles
        rect_text = "开启" if detect_rectangles else "关闭"
        print(f"矩形检测: {rect_text}")
        return True
    
    # 多颜色模式下的颜色开关控制 (Ctrl+数字键)
    elif 49 <= key <= 54:  # 数字键1-6 (ASCII码49-54)，在多颜色模式下用于切换颜色开关
        if multi_color_mode:
            color_idx = key - 49  # 转换为0-5的索引
            if color_idx < len(color_ranges):
                color_enabled[color_idx] = not color_enabled[color_idx]
                color_name = color_ranges[color_idx]['name']
                status = "开启" if color_enabled[color_idx] else "关闭"
                print(f"多颜色模式 - {color_name}检测: {status}")
                
                # 显示当前启用的颜色列表
                enabled_colors = [color_ranges[i]['name'] for i, enabled in color_enabled.items() if enabled]
                if enabled_colors:
                    print(f"当前启用的颜色: {', '.join(enabled_colors)}")
                else:
                    print("⚠️ 警告: 所有颜色都已关闭！")
                return True
        else:
            # 单颜色模式下保持原有的颜色切换功能
            target_color_index = key - 49
            if target_color_index < len(color_ranges):
                update_color_params()
                color_name = color_ranges[target_color_index]['name']
                print(f"切换到检测颜色: {color_name}")
                return True
    
    return False

def create_trackbars():
    """创建阈值调试滑动条"""
    if not debug_mode:
        return
        
    cv2.namedWindow('Threshold Controls', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Threshold Controls', 400, 600)
    
    # Canny边缘检测参数
    cv2.createTrackbar('Canny Low', 'Threshold Controls', threshold_params['canny_low'], 200, lambda x: None)
    cv2.createTrackbar('Canny High', 'Threshold Controls', threshold_params['canny_high'], 300, lambda x: None)
    cv2.createTrackbar('Gaussian Blur', 'Threshold Controls', threshold_params['gaussian_blur'], 15, lambda x: None)
    
    # 红色检测参数
    cv2.createTrackbar('Red H Low1', 'Threshold Controls', threshold_params['red_h_low1'], 179, lambda x: None)
    cv2.createTrackbar('Red H High1', 'Threshold Controls', threshold_params['red_h_high1'], 179, lambda x: None)
    cv2.createTrackbar('Red H Low2', 'Threshold Controls', threshold_params['red_h_low2'], 179, lambda x: None)
    cv2.createTrackbar('Red H High2', 'Threshold Controls', threshold_params['red_h_high2'], 179, lambda x: None)
    cv2.createTrackbar('Red S Low', 'Threshold Controls', threshold_params['red_s_low'], 255, lambda x: None)
    cv2.createTrackbar('Red V Low', 'Threshold Controls', threshold_params['red_v_low'], 255, lambda x: None)
    cv2.createTrackbar('Morphology Kernel', 'Threshold Controls', threshold_params['morphology_kernel'], 15, lambda x: None)

def update_threshold_params():
    """更新阈值参数"""
    global threshold_params
    if not debug_mode:
        return
        
    threshold_params['canny_low'] = cv2.getTrackbarPos('Canny Low', 'Threshold Controls')
    threshold_params['canny_high'] = cv2.getTrackbarPos('Canny High', 'Threshold Controls')
    threshold_params['gaussian_blur'] = max(1, cv2.getTrackbarPos('Gaussian Blur', 'Threshold Controls'))
    threshold_params['red_h_low1'] = cv2.getTrackbarPos('Red H Low1', 'Threshold Controls')
    threshold_params['red_h_high1'] = cv2.getTrackbarPos('Red H High1', 'Threshold Controls')
    threshold_params['red_h_low2'] = cv2.getTrackbarPos('Red H Low2', 'Threshold Controls')
    threshold_params['red_h_high2'] = cv2.getTrackbarPos('Red H High2', 'Threshold Controls')
    threshold_params['red_s_low'] = cv2.getTrackbarPos('Red S Low', 'Threshold Controls')
    threshold_params['red_v_low'] = cv2.getTrackbarPos('Red V Low', 'Threshold Controls')
    threshold_params['morphology_kernel'] = max(3, cv2.getTrackbarPos('Morphology Kernel', 'Threshold Controls'))

# 卡尔曼滤波器类
class KalmanFilter:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                 [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
        self.kalman.measurementNoiseCov = 0.1 * np.eye(2, dtype=np.float32)
        self.kalman.errorCovPost = 1.0 * np.eye(4, dtype=np.float32)
        self.initialized = False
        self.lost_frames = 0
        self.max_lost_frames = 10  # 最多允许丢失10帧
        
    def update(self, center):
        """更新卡尔曼滤波器"""
        if center is not None:
            if not self.initialized:
                # 初始化状态：[x, y, vx, vy]
                self.kalman.statePre = np.array([center[0], center[1], 0, 0], dtype=np.float32)
                self.kalman.statePost = np.array([center[0], center[1], 0, 0], dtype=np.float32)
                self.initialized = True
                self.lost_frames = 0
                return center
            else:
                # 预测
                prediction = self.kalman.predict()
                # 更新
                measurement = np.array([[center[0]], [center[1]]], dtype=np.float32)
                self.kalman.correct(measurement)
                self.lost_frames = 0
                return (int(prediction[0]), int(prediction[1]))
        else:
            if self.initialized:
                # 没有检测到目标，仅使用预测
                self.lost_frames += 1
                if self.lost_frames < self.max_lost_frames:
                    prediction = self.kalman.predict()
                    return (int(prediction[0]), int(prediction[1]))
                else:
                    # 丢失太久，重置滤波器
                    self.reset()
                    return None
            return None
    
    def reset(self):
        """重置滤波器"""
        self.initialized = False
        self.lost_frames = 0
        
    def get_prediction(self):
        """获取预测位置（不更新状态）"""
        if self.initialized:
            # 临时预测，不改变内部状态
            temp_kalman = cv2.KalmanFilter(4, 2)
            temp_kalman.measurementMatrix = self.kalman.measurementMatrix.copy()
            temp_kalman.transitionMatrix = self.kalman.transitionMatrix.copy()
            temp_kalman.processNoiseCov = self.kalman.processNoiseCov.copy()
            temp_kalman.statePre = self.kalman.statePost.copy()
            prediction = temp_kalman.predict()
            return (int(prediction[0]), int(prediction[1]))
        return None

# 创建多个卡尔曼滤波器实例（每种颜色一个）
color_trackers = {
    0: KalmanFilter(),  # 红色
    1: KalmanFilter(),  # 绿色
    2: KalmanFilter(),  # 蓝色
    3: KalmanFilter(),  # 黄色
    4: KalmanFilter(),  # 橙色
    5: KalmanFilter(),  # 紫色
}

# 1. 图像预处理与矩形检测（支持阈值调试）
def detect_rectangle(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 使用可调节的高斯模糊参数
    blur_size = threshold_params['gaussian_blur']
    if blur_size % 2 == 0:  # 确保是奇数
        blur_size += 1
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    
    # 使用可调节的Canny参数
    edges = cv2.Canny(blurred, threshold_params['canny_low'], threshold_params['canny_high'])
    
    # 仅在调试模式下显示灰度图和边缘检测结果
    if debug_mode:
        cv2.imshow("Gray Image", gray)
        cv2.imshow("Blurred Image", blurred)
        cv2.imshow("Canny Edges", edges)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    rect_contour = None
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000:  # 适当提高面积阈值
            continue
            
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and area > max_area:
            max_area = area
            rect_contour = approx
    
    if rect_contour is None:
        raise ValueError("未检测到矩形")
    
    return rect_contour.reshape(4, 2)

def clear_old_data_from_queue(data_type):
    """
    清理队列中指定类型的旧数据，只保留最新的数据
    """
    global data_queue
    temp_items = []
    
    # 取出所有数据
    while not data_queue.empty():
        try:
            item = data_queue.get_nowait()
            temp_items.append(item)
            data_queue.task_done()
        except:
            break
    
    # 筛选数据：只保留非指定类型的数据
    for item in temp_items:
        priority, timestamp, data = item
        if data['type'] != data_type:
            data_queue.put(item)

# 2. 顶点排序（仅处理数据，不发送）
def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下
    
    # 提取坐标并转换为整数
    top_left = (int(rect[0][0]), int(rect[0][1]))
    top_right = (int(rect[1][0]), int(rect[1][1]))
    bottom_right = (int(rect[2][0]), int(rect[2][1]))
    bottom_left = (int(rect[3][0]), int(rect[3][1]))
    
    # 将数据加入发送队列（矩形数据优先级较低）
    global data_queue
    
    # 清理队列中旧的矩形数据
    clear_old_data_from_queue(RECTANGLE_TYPE)
    
    rect_data = {
        'type': RECTANGLE_TYPE,
        'points': {
            'top_left': top_left,
            'top_right': top_right,
            'bottom_right': bottom_right,
            'bottom_left': bottom_left
        }
    }
    # 优先级2（数值越小优先级越高，矩形数据优先级较低）
    data_queue.put((2, time.time(), rect_data))
    
    return rect

# 3. 边点采样（保持不变）
def sample_edge(pt1, pt2, num_points=20):
    return [
        (
            int(pt1[0] + (pt2[0] - pt1[0]) * i / (num_points - 1)),
            int(pt1[1] + (pt2[1] - pt1[1]) * i / (num_points - 1))
        ) for i in range(num_points)
    ]

# 4. 单一颜色检测（支持多种颜色和阈值调试）
def detect_color_block(frame):
    global color_trackers, target_color_index, data_queue
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 使用当前选择的颜色范围
    current_color = color_ranges[target_color_index]
    
    # 使用可调节的颜色范围检测参数
    lower_color1 = np.array([threshold_params['red_h_low1'], threshold_params['red_s_low'], threshold_params['red_v_low']])
    upper_color1 = np.array([threshold_params['red_h_high1'], 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_color1, upper_color1)
    mask = mask1
    
    # 如果是红色或其他需要双范围的颜色
    if current_color['use_dual_range']:
        lower_color2 = np.array([threshold_params['red_h_low2'], threshold_params['red_s_low'], threshold_params['red_v_low']])
        upper_color2 = np.array([threshold_params['red_h_high2'], 255, 255])
        mask2 = cv2.inRange(hsv, lower_color2, upper_color2)
        mask = cv2.bitwise_or(mask1, mask2)
    
    # 使用可调节的形态学处理参数
    kernel_size = threshold_params['morphology_kernel']
    if kernel_size % 2 == 0:  # 确保是奇数
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_processed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_processed = cv2.morphologyEx(mask_processed, cv2.MORPH_OPEN, kernel)
    
    # 仅在调试模式下显示二值图和中间结果
    if debug_mode:
        cv2.imshow("HSV Image", hsv)
        cv2.imshow(f"{current_color['name']} Binary Mask", mask)  # 显示二值图
        cv2.imshow(f"{current_color['name']} Processed", mask_processed)
    
    # 初始化检测到的中心点
    detected_center = None
    
    # 查找轮廓并标注
    contours, _ = cv2.findContours(mask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        # 设置面积范围
        min_area = 100
        max_area = 20000
        
        valid_contours = [cnt for cnt in contours 
                         if min_area < cv2.contourArea(cnt) < max_area]
        
        if valid_contours:
            max_contour = max(valid_contours, key=cv2.contourArea)
            x,y,w,h = cv2.boundingRect(max_contour)
            detected_center = (x + w//2, y + h//2)
            
            # 长宽比检查
            aspect_ratio = w / h if h > 0 else 0
            if not (0.3 < aspect_ratio < 3.0):
                detected_center = None
    
    # 使用当前颜色对应的卡尔曼滤波器
    current_tracker = color_trackers[target_color_index]
    filtered_center = current_tracker.update(detected_center)
    
    # 绘制检测结果
    if detected_center:
        x, y, w, h = cv2.boundingRect(max_contour)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.circle(frame, detected_center, 3, (0,255,0), -1)
    
    if filtered_center:
        cv2.circle(frame, filtered_center, 8, (0,255,255), 2)
        cv2.putText(frame, f"({filtered_center[0]},{filtered_center[1]})", 
                   (filtered_center[0]-50, filtered_center[1]-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        
        status = "Tracking" if detected_center else f"Predicting({current_tracker.lost_frames})"
        cv2.putText(frame, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        
        # 将目标颜色数据加入发送队列（最高优先级）
        
        # 清理队列中旧的目标颜色数据
        clear_old_data_from_queue(COLOR_BLOCK_TYPE)
        
        color_data = {
            'type': COLOR_BLOCK_TYPE,
            'center': filtered_center,
            'color_id': color_ranges[target_color_index]['id'],
            'color_name': color_ranges[target_color_index]['name']
        }
        # 优先级1（数值越小优先级越高，目标颜色数据最高优先级）
        data_queue.put((1, time.time(), color_data))

    return frame, filtered_center

# 5. 多颜色同时检测
def detect_multi_color_blocks(frame):
    """同时检测所有颜色的色块（仅检测启用的颜色）"""
    global color_trackers, data_queue, color_enabled
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detected_blocks = []
    
    # 遍历所有颜色进行检测（仅检测启用的颜色）
    for color_idx, color_info in color_ranges.items():
        # 检查该颜色是否启用
        if not color_enabled.get(color_idx, False):
            continue  # 跳过未启用的颜色
        
        # 使用预设的颜色范围
        lower_color1 = np.array(color_info['lower1'])
        upper_color1 = np.array(color_info['upper1'])
        
        mask1 = cv2.inRange(hsv, lower_color1, upper_color1)
        mask = mask1
        
        # 如果需要双范围检测
        if color_info['use_dual_range']:
            lower_color2 = np.array(color_info['lower2'])
            upper_color2 = np.array(color_info['upper2'])
            mask2 = cv2.inRange(hsv, lower_color2, upper_color2)
            mask = cv2.bitwise_or(mask1, mask2)
        
        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_processed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_processed = cv2.morphologyEx(mask_processed, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # 设置面积范围
            min_area = 100
            max_area = 20000
            
            valid_contours = [cnt for cnt in contours 
                             if min_area < cv2.contourArea(cnt) < max_area]
            
            if valid_contours:
                max_contour = max(valid_contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(max_contour)
                detected_center = (x + w//2, y + h//2)
                
                # 长宽比检查
                aspect_ratio = w / h if h > 0 else 0
                if 0.3 < aspect_ratio < 3.0:
                    # 使用对应颜色的卡尔曼滤波器
                    tracker = color_trackers[color_idx]
                    filtered_center = tracker.update(detected_center)
                    
                    if filtered_center:
                        # 绘制检测结果（不同颜色用不同标记）
                        color_bgr = [(0,0,255), (0,255,0), (255,0,0), (0,255,255), (128,0,128), (255,0,255)][color_idx]
                        cv2.rectangle(frame, (x,y), (x+w,y+h), color_bgr, 2)
                        cv2.circle(frame, filtered_center, 8, color_bgr, 2)
                        cv2.putText(frame, f"{color_info['name']}({filtered_center[0]},{filtered_center[1]})", 
                                   (filtered_center[0]-40, filtered_center[1]-15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_bgr, 1)
                        
                        # 添加到检测结果
                        detected_blocks.append({
                            'color_idx': color_idx,
                            'color_name': color_info['name'],
                            'color_id': color_info['id'],
                            'center': filtered_center
                        })
                        
                        # 将颜色数据加入发送队列
                        clear_old_data_from_queue(f"{COLOR_BLOCK_TYPE}_{color_idx}")
                        
                        color_data = {
                            'type': COLOR_BLOCK_TYPE,
                            'center': filtered_center,
                            'color_id': color_info['id'],
                            'color_name': color_info['name']
                        }
                        # 优先级1（数值越小优先级越高）
                        data_queue.put((1, time.time(), color_data))
    
    return frame, detected_blocks

def data_sender_thread():
    """
    专门的数据发送线程，从优先级队列中获取数据并发送
    红色块数据（优先级1）将优先于矩形数据（优先级2）发送
    始终发送最新数据，自动丢弃旧数据
    """
    global data_queue, detection_running
    
    print("数据发送线程已启动（优先级队列模式 - 仅发送最新数据）")
    
    while detection_running:
        try:
            # 从优先级队列中获取数据，超时时间0.1秒
            # 格式: (priority, timestamp, data)
            priority, timestamp, data = data_queue.get(timeout=0.1)
            
            # 检查数据新鲜度（避免发送过时数据）
            current_time = time.time()
            data_age = current_time - timestamp
            
            # 如果数据太旧（超过0.2秒），跳过发送
            if data_age > 0.2:
                print(f"⚠️ 跳过过时数据: {data['type']}, 延迟 {data_age:.2f}s")
                data_queue.task_done()
                continue
            
            if data['type'] == COLOR_BLOCK_TYPE:
                # 发送目标颜色数据（实时优先）
                center = data['center']
                color_id = data['color_id']
                color_name = data['color_name']
                
                # 数据格式: 帧头(2) + 颜色ID(1) + X坐标(2) + Y坐标(2) + 帧尾(1)
                data_bytes = bytes([HEADER[0], HEADER[1], color_id, 
                                   center[0] // 256, center[0] % 256, 
                                   center[1] // 256, center[1] % 256, 
                                   FOOTER])
                send_usb_ttl(data_bytes)
                print(f"🎯 [最新] 发送{color_name}块(ID:0x{color_id:02X}): {center} (延迟 {data_age:.3f}s)")
                
            elif data['type'] == RECTANGLE_TYPE:
                # 发送矩形顶点数据
                points = data['points']
                
                # 发送左上顶点
                top_left = points['top_left']
                data_bytes = bytes([HEADER[0], HEADER[1], 0x02,
                                   top_left[0] // 256, top_left[0] % 256,
                                   top_left[1] // 256, top_left[1] % 256,
                                   FOOTER])
                send_usb_ttl(data_bytes)
                
                # 发送右上顶点
                top_right = points['top_right']
                data_bytes = bytes([HEADER[0], HEADER[1], 0x03,
                                   top_right[0] // 256, top_right[0] % 256,
                                   top_right[1] // 256, top_right[1] % 256,
                                   FOOTER])
                send_usb_ttl(data_bytes)
                
                # 发送右下顶点
                bottom_right = points['bottom_right']
                data_bytes = bytes([HEADER[0], HEADER[1], 0x04,
                                   bottom_right[0] // 256, bottom_right[0] % 256,
                                   bottom_right[1] // 256, bottom_right[1] % 256,
                                   FOOTER])
                send_usb_ttl(data_bytes)
                
                # 发送左下顶点
                bottom_left = points['bottom_left']
                data_bytes = bytes([HEADER[0], HEADER[1], 0x05,
                                   bottom_left[0] // 256, bottom_left[0] % 256,
                                   bottom_left[1] // 256, bottom_left[1] % 256,
                                   FOOTER])
                send_usb_ttl(data_bytes)
                
                print(f"🔵 [最新] 发送矩形: 左上{top_left}, 右上{top_right}, 右下{bottom_right}, 左下{bottom_left} (延迟 {data_age:.3f}s)")
            
            # 标记任务完成
            data_queue.task_done()
            
        except:
            # 队列为空或超时，继续循环
            continue
    
    print("数据发送线程已停止")

# 主流程（并行处理版本）
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    # 恢复原分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("启动多功能颜色检测程序...")
    print("分辨率: 640x480")
    print("并行处理: 检测线程 + 数据发送线程")
    print("优先级队列: 目标颜色数据(优先级1) > 矩形数据(优先级2)")
    print("数据策略: 仅发送最新数据，自动丢弃旧数据")
    print(f"调试模式: {'开启' if debug_mode else '关闭'} (运行时按D键切换)")
    print(f"检测模式: {'多颜色同时检测' if multi_color_mode else '单一颜色检测'} (按M键切换)")
    print(f"矩形检测: {'开启' if detect_rectangles else '关闭'} (按N键切换)")
    print(f"当前检测颜色: {color_ranges[target_color_index]['name']} (按1-6键切换，仅单色模式有效)")
    print("多颜色模式控制:")
    print("  默认启用: 红色、绿色、蓝色")
    print("  默认关闭: 黄色、橙色、紫色")
    print("  在多颜色模式下，按1-6键切换对应颜色的开关状态")
    print("数据格式: 帧头(0xAA55) + 颜色ID(0x00-0x05) + X坐标(2字节) + Y坐标(2字节) + 帧尾(0x55)")
    print("矩形格式: 帧头(0xAA55) + 顶点ID(0x10-0x13) + X坐标(2字节) + Y坐标(2字节) + 帧尾(0x55)")
    print("键盘控制:")
    print("  W/S=Canny低阈值, E/X=Canny高阈值, R/F=颜色H低值, T/G=颜色H高值, Z/C=形态学核大小")
    print("  M=切换多色模式, N=切换矩形检测, D=调试模式")
    print("  单色模式: 1-6=选择颜色")
    print("  多色模式: 1-6=切换对应颜色开关")
    
    # 初始化颜色参数
    update_color_params()
    
    # 创建阈值调试滑动条
    create_trackbars()
    
    # 启动数据发送线程
    detection_running = True
    sender_thread = threading.Thread(target=data_sender_thread, daemon=True)
    sender_thread.start()
    
    frame_count = 0
    rect_detect_interval = 5  # 每5帧检测一次矩形
    
    # 缓存上次检测到的矩形顶点
    last_vertices = None
    last_ordered_vertices = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("帧读取失败")
                break
                
            frame_count += 1
            should_detect_rect = (frame_count % rect_detect_interval == 0) and detect_rectangles
            
            # 更新阈值参数
            update_threshold_params()
                
            try:
                # 根据模式选择检测方式
                if multi_color_mode:
                    # 多颜色同时检测
                    processed_frame, color_blocks = detect_multi_color_blocks(frame.copy())
                    if color_blocks and frame_count % 30 == 0:
                        print(f"检测到 {len(color_blocks)} 个颜色块")
                else:
                    # 单一颜色检测（实时跟踪）
                    processed_frame, color_center = detect_color_block(frame.copy())
                
                # 定期检测矩形（如果启用）
                if should_detect_rect:
                    try:
                        vertices = detect_rectangle(frame)
                        ordered_vertices = order_points(vertices)
                        # 缓存检测结果
                        last_vertices = vertices
                        last_ordered_vertices = ordered_vertices
                    except Exception as e:
                        # 矩形检测失败，使用缓存的结果
                        if frame_count % 30 == 0:  # 每30帧打印一次错误
                            print(f"矩形检测失败: {e}")
                
                # 使用最后一次成功检测的矩形进行绘制
                if last_ordered_vertices is not None:
                    # 绘制矩形顶点连线（蓝色）
                    cv2.polylines(processed_frame, [last_ordered_vertices.astype(int)], True, (255,0,0), 2)
                    
                    # 绘制边缘点（每10帧绘制一次以提高性能）
                    if frame_count % 10 == 0:
                        num_points_per_edge = 15
                        
                        for i in range(4):
                            start_pt = last_ordered_vertices[i]
                            end_pt = last_ordered_vertices[(i + 1) % 4]
                            edge_points = sample_edge(start_pt, end_pt, num_points_per_edge)
                            
                            for pt in edge_points:
                                cv2.circle(processed_frame, tuple(map(int, pt)), 2, (0, 0, 255), -1)
                
                # 在画面上显示队列状态和参数信息
                queue_size = data_queue.qsize()
                cv2.putText(processed_frame, f"Queue: {queue_size}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(processed_frame, f"Frame: {frame_count}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # 显示检测模式和矩形检测状态
                mode_text = "多颜色模式" if multi_color_mode else "单色模式"
                cv2.putText(processed_frame, f"模式: {mode_text} (M键切换)", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                rect_text = "矩形:开" if detect_rectangles else "矩形:关"
                cv2.putText(processed_frame, f"{rect_text} (N键切换)", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 在多颜色模式下显示启用的颜色
                if multi_color_mode:
                    enabled_colors = [color_ranges[i]['name'] for i, enabled in color_enabled.items() if enabled]
                    if enabled_colors:
                        colors_text = "启用: " + ",".join(enabled_colors[:3])  # 最多显示3个颜色名
                        if len(enabled_colors) > 3:
                            colors_text += f"等{len(enabled_colors)}种"
                    else:
                        colors_text = "启用: 无 (1-6键切换)"
                    cv2.putText(processed_frame, colors_text, (10, 180), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    cv2.putText(processed_frame, "1-6键切换颜色开关", (10, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # 仅在调试模式下显示详细参数信息
                if debug_mode:
                    current_color_name = color_ranges[target_color_index]['name']
                    y_offset = 220 if multi_color_mode else 180
                    cv2.putText(processed_frame, f"Canny: {threshold_params['canny_low']}-{threshold_params['canny_high']}", (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    cv2.putText(processed_frame, f"{current_color_name} H: {threshold_params['red_h_low1']}-{threshold_params['red_h_high1']}", (10, y_offset+30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    cv2.putText(processed_frame, "DEBUG MODE ON (按D切换)", (10, y_offset+60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    if not multi_color_mode:
                        cv2.putText(processed_frame, f"颜色: {current_color_name} (1-6键切换)", (10, y_offset+90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                else:
                    y_offset = 220 if multi_color_mode else 180
                    cv2.putText(processed_frame, "DEBUG MODE OFF (按D切换)", (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    if not multi_color_mode:
                        current_color_name = color_ranges[target_color_index]['name']
                        cv2.putText(processed_frame, f"检测: {current_color_name} (1-6键切换)", (10, y_offset+30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # 显示结果
                cv2.imshow("Parallel Detection", processed_frame)
                
                # 仅在调试模式下打印当前参数（用于调试）
                if debug_mode and frame_count % 30 == 0:
                    current_color_name = color_ranges[target_color_index]['name']
                    print(f"当前参数 - Canny: {threshold_params['canny_low']}-{threshold_params['canny_high']}, "
                          f"{current_color_name}H: {threshold_params['red_h_low1']}-{threshold_params['red_h_high1']}, "
                          f"形态学核: {threshold_params['morphology_kernel']}")
                
            except Exception as e:
                # 显示错误信息
                cv2.putText(frame, f"Error: {str(e)[:50]}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.imshow("Parallel Detection", frame)
            
            # 退出和控制
            key = cv2.waitKey(1) & 0xFF
            
            # 处理键盘输入
            if handle_keyboard_input(key):
                continue  # 如果处理了键盘输入，继续循环
            
            if key == 27 or key == ord('q') or key == ord('Q'):  # ESC或q/Q键退出
                print("收到退出指令，正在关闭程序...")
                break
            elif key == ord('d') or key == ord('D'):  # D键切换调试模式
                debug_mode = not debug_mode
                print(f"调试模式: {'开启' if debug_mode else '关闭'}")
                
                if debug_mode:
                    # 开启调试模式，创建调试窗口
                    create_trackbars()
                else:
                    # 关闭调试模式，销毁调试窗口
                    try:
                        cv2.destroyWindow('Threshold Controls')
                        cv2.destroyWindow('Gray Image')
                        cv2.destroyWindow('Blurred Image')
                        cv2.destroyWindow('Canny Edges')
                        cv2.destroyWindow('HSV Image')
                        # 销毁所有颜色的掩码窗口
                        for color_info in color_ranges.values():
                            cv2.destroyWindow(f"{color_info['name']} Mask Raw")
                            cv2.destroyWindow(f"{color_info['name']} Mask Processed")
                    except:
                        pass
                
    except KeyboardInterrupt:
        print("\n程序被中断")
    
    finally:
        print("正在停止程序...")
        
        # 停止数据发送线程
        detection_running = False
        
        # 强制关闭所有OpenCV窗口
        try:
            cv2.destroyAllWindows()
            # 等待一小段时间确保窗口关闭
            cv2.waitKey(1)
        except:
            pass
        
        # 等待发送队列中剩余数据处理完成
        print("等待数据队列清空...")
        try:
            data_queue.join()
        except:
            pass
        
        # 等待发送线程结束
        try:
            sender_thread.join(timeout=2)
        except:
            pass
        
        # 清理资源
        try:
            cap.release()
        except:
            pass
        
        # 再次确保所有窗口关闭
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except:
            pass
        
        print("程序已退出")