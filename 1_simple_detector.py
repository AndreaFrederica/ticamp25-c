import cv2
import numpy as np
import time

def nothing(x):
    """滑条回调函数"""
    pass

def create_trackbars():
    """创建HSV调节滑条"""
    cv2.namedWindow("HSV Controls", cv2.WINDOW_NORMAL)
    
    # 第一个HSV范围 - 默认设置为检测黑色
    cv2.createTrackbar("H1 Min", "HSV Controls", 0, 179, nothing)
    cv2.createTrackbar("H1 Max", "HSV Controls", 179, 179, nothing)  # H范围改为0-179（OpenCV的H范围）
    cv2.createTrackbar("S1 Min", "HSV Controls", 0, 255, nothing)
    cv2.createTrackbar("S1 Max", "HSV Controls", 255, 255, nothing)  # S范围改为0-255
    cv2.createTrackbar("V1 Min", "HSV Controls", 0, 255, nothing)  # V最小值改为0
    cv2.createTrackbar("V1 Max", "HSV Controls", 80, 255, nothing)  # V最大值改为80检测黑色
    
    # 第二个HSV范围（用于双范围检测）
    cv2.createTrackbar("H2 Min", "HSV Controls", 0, 179, nothing)
    cv2.createTrackbar("H2 Max", "HSV Controls", 179, 179, nothing)
    cv2.createTrackbar("S2 Min", "HSV Controls", 0, 255, nothing)
    cv2.createTrackbar("S2 Max", "HSV Controls", 255, 255, nothing)
    cv2.createTrackbar("V2 Min", "HSV Controls", 0, 255, nothing)
    cv2.createTrackbar("V2 Max", "HSV Controls", 80, 255, nothing)
    
    # 控制选项
    cv2.createTrackbar("Use Range2", "HSV Controls", 0, 1, nothing)  # 是否启用第二范围
    cv2.createTrackbar("Min Area", "HSV Controls", 200, 5000, nothing)  # 降低最小面积

def get_hsv_values():
    """获取HSV滑条数值"""
    h1_min = cv2.getTrackbarPos("H1 Min", "HSV Controls")
    h1_max = cv2.getTrackbarPos("H1 Max", "HSV Controls")
    s1_min = cv2.getTrackbarPos("S1 Min", "HSV Controls")
    s1_max = cv2.getTrackbarPos("S1 Max", "HSV Controls")
    v1_min = cv2.getTrackbarPos("V1 Min", "HSV Controls")
    v1_max = cv2.getTrackbarPos("V1 Max", "HSV Controls")
    
    h2_min = cv2.getTrackbarPos("H2 Min", "HSV Controls")
    h2_max = cv2.getTrackbarPos("H2 Max", "HSV Controls")
    s2_min = cv2.getTrackbarPos("S2 Min", "HSV Controls")
    s2_max = cv2.getTrackbarPos("S2 Max", "HSV Controls")
    v2_min = cv2.getTrackbarPos("V2 Min", "HSV Controls")
    v2_max = cv2.getTrackbarPos("V2 Max", "HSV Controls")
    
    use_range2 = cv2.getTrackbarPos("Use Range2", "HSV Controls")
    min_area = cv2.getTrackbarPos("Min Area", "HSV Controls")
    
    return {
        'range1': (h1_min, h1_max, s1_min, s1_max, v1_min, v1_max),
        'range2': (h2_min, h2_max, s2_min, s2_max, v2_min, v2_max),
        'use_range2': bool(use_range2),
        'min_area': min_area
    }

def detect_hsv_shapes(frame, hsv_params):
    """使用HSV参数检测形状"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 第一个范围
    h1_min, h1_max, s1_min, s1_max, v1_min, v1_max = hsv_params['range1']
    lower1 = np.array([h1_min, s1_min, v1_min])
    upper1 = np.array([h1_max, s1_max, v1_max])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    
    # 如果启用第二个范围
    if hsv_params['use_range2']:
        h2_min, h2_max, s2_min, s2_max, v2_min, v2_max = hsv_params['range2']
        lower2 = np.array([h2_min, s2_min, v2_min])
        upper2 = np.array([h2_max, s2_max, v2_max])
        mask2 = cv2.inRange(hsv, lower2, upper2)
        # 合并两个掩码
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        mask = mask1
    
    # 形态学操作
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def identify_shape_by_area(contour):
    """
    通过顶点数和边长相似性识别形状 - 只识别圆形、三角形、正方形
    """
    area = cv2.contourArea(contour)
    if area < 20:  # 进一步降低最小面积要求
        return "Unknown"
    
    # 计算轮廓的周长
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return "Unknown"
    
    # 计算圆度 (4π×面积)/(周长²)，圆形接近1
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    # 计算边界矩形
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0
    
    # 多边形近似 - 使用较小的epsilon获得精确顶点
    epsilon = 0.02 * perimeter  # 使用较小的epsilon值
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = len(approx)
    
    # 打印调试信息
    print(f"    形状分析: 顶点数={vertices}, 圆度={circularity:.3f}, 长宽比={aspect_ratio:.3f}")
    
    # 简化的形状识别规则 - 主要基于顶点数
    if vertices == 3:
        print(f"    检测到三角形: {vertices}个顶点")
        return "Triangle"
    elif vertices == 4:
        # 对于四边形，检查边长相似性
        # 计算四条边的长度
        side_lengths = []
        for i in range(4):
            p1 = approx[i][0]
            p2 = approx[(i + 1) % 4][0]
            length = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            side_lengths.append(length)
        
        # 计算边长的标准差
        mean_length = np.mean(side_lengths)
        std_length = np.std(side_lengths)
        variation_coefficient = std_length / mean_length if mean_length > 0 else 1
        
        print(f"    四边形边长: {[f'{l:.1f}' for l in side_lengths]}, 变异系数: {variation_coefficient:.3f}")
        
        # 如果边长相似（变异系数小于0.3），认为是正方形
        if variation_coefficient < 0.3:
            print(f"    检测到正方形: {vertices}个顶点, 边长相似")
            return "Square"
        else:
            print(f"    四边形但边长不相似，不识别为正方形")
            return "Unknown"
    else:
        # 对于其他顶点数的形状，检查是否为圆形
        if circularity > 0.6:  # 圆度较高
            print(f"    检测到圆形: 圆度={circularity:.3f}")
            return "Circle"
        elif vertices >= 6 and circularity > 0.4:  # 多边形但较圆的
            print(f"    检测到圆形: {vertices}个顶点但圆度较高")
            return "Circle"
        else:
            print(f"    未识别形状: {vertices}个顶点")
            return "Unknown"

def main():
    """
    简化版黑色矩形检测器 - 解决摄像头卡住问题
    """
    print("🔍 启动黑色矩形检测器...")
    
    # 使用DirectShow后端避免卡住 (Windows)
    print("📹 初始化摄像头...")
    cap = None
    camera_found = False
    
    # USB摄像头通常在索引1-3，笔记本自带摄像头在索引0
    camera_indices = [1, 2, 3, 0]  # 优先检测USB摄像头
    
    for camera_index in camera_indices:
        print(f"🔍 尝试摄像头索引 {camera_index}...")
        try:
            # 先尝试DirectShow后端
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # 快速测试
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                print(f"✅ USB摄像头连接成功 (索引{camera_index})")
                camera_found = True
                break
            else:
                print(f"❌ 索引{camera_index}无法读取画面")
                cap.release()
                cap = None
        except Exception as e:
            print(f"❌ 索引{camera_index}初始化失败: {e}")
            if cap:
                cap.release()
                cap = None
    
    if not camera_found:
        print("❌ 未找到USB摄像头")
        print("💡 请检查：")
        print("   1. USB摄像头是否已连接")
        print("   2. 摄像头驱动是否正常")
        print("   3. 是否有其他程序占用摄像头")
        print("   4. 尝试重新插拔USB线")
        return
    
    # 设置摄像头参数（USB摄像头通常支持更高分辨率）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # 获取实际设置
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📐 USB摄像头配置: {actual_width}x{actual_height} @ {actual_fps:.1f}fps")
    
    print("✅ 摄像头就绪")
    print("操作: q-退出, s-保存, p-打印信息, h-显示/隐藏HSV控制")
    
    # 创建HSV控制滑条
    create_trackbars()
    
    cv2.namedWindow('Rectangle Detector', cv2.WINDOW_AUTOSIZE)
    
    frame_count = 0
    show_controls = True
    
    try:
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("❌ 读取帧失败")
                break
            
            frame_count += 1
            
            # 获取HSV参数
            hsv_params = get_hsv_values()
            
            # 使用HSV参数进行检测
            mask = detect_hsv_shapes(frame, hsv_params)
            
            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            rectangles_found = 0
            total_pixels = 0
            total_frame_pixels = frame.shape[0] * frame.shape[1]  # 计算总像素数
            black_pixels_count = np.sum(mask == 255)  # 黑色像素总数
            rectangle_info = []  # 存储矩形信息
            shape_stats = {"Square": 0, "Circle": 0, "Triangle": 0, "None": 0}  # 只统计支持的形状
            
            # 检测矩形
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < hsv_params['min_area']:  # 使用滑条设置的最小面积
                    continue
                
                # 多边形近似
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) == 4:  # 四边形（外框）
                    rectangles_found += 1
                    total_pixels += int(area)
                    
                    # 计算边界框获取宽高
                    rect = cv2.boundingRect(contour)
                    x, y, w, h = rect
                    
                    print(f"🔍 检测到外框矩形: 位置=({x},{y}), 尺寸={w}x{h}, 面积={area:.0f}")
                    
                    # 保存外框的四个角点
                    outer_corners = approx.reshape(-1, 2)
                    print(f"📍 外框四个角点: {outer_corners.tolist()}")
                    
                    # 裁剪出矩形区域
                    roi_frame = frame[y:y+h, x:x+w].copy()
                    
                    # 在裁剪区域中寻找内部的白色区域（非黑色）
                    roi_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                    
                    # 使用阈值分割找到白色区域（内部区域）
                    _, thresh = cv2.threshold(roi_gray, 100, 255, cv2.THRESH_BINARY)
                    
                    # 找到白色区域的轮廓
                    white_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    inner_shapes = []
                    shape_type = "None"
                    inner_black_w, inner_black_h = 0, 0
                    inner_area_pixels = 0
                    
                    if white_contours:
                        # 找到最大的白色区域（内部区域）
                        largest_white = max(white_contours, key=cv2.contourArea)
                        white_area = cv2.contourArea(largest_white)
                        
                        if white_area > 100:  # 白色区域足够大
                            # 获取白色区域的边界框
                            wx, wy, ww, wh = cv2.boundingRect(largest_white)
                            
                            print(f"🟦 内部白色区域: 位置=({wx},{wy}), 尺寸={ww}x{wh}, 面积={white_area:.0f}")
                            
                            # 在白色区域内寻找黑色图形
                            inner_roi = roi_frame[wy:wy+wh, wx:wx+ww].copy()
                            
                            # 对内部区域进行HSV检测，寻找黑色图形
                            inner_hsv = cv2.cvtColor(inner_roi, cv2.COLOR_BGR2HSV)
                            
                            # 使用相同的HSV参数检测黑色图形
                            h1_min, h1_max, s1_min, s1_max, v1_min, v1_max = hsv_params['range1']
                            lower1 = np.array([h1_min, s1_min, v1_min])
                            upper1 = np.array([h1_max, s1_max, v1_max])
                            inner_mask = cv2.inRange(inner_hsv, lower1, upper1)
                            
                            # 如果启用第二个HSV范围
                            if hsv_params['use_range2']:
                                h2_min, h2_max, s2_min, s2_max, v2_min, v2_max = hsv_params['range2']
                                lower2 = np.array([h2_min, s2_min, v2_min])
                                upper2 = np.array([h2_max, s2_max, v2_max])
                                inner_mask2 = cv2.inRange(inner_hsv, lower2, upper2)
                                inner_mask = cv2.bitwise_or(inner_mask, inner_mask2)
                            
                            # 形态学操作清理掩码
                            kernel = np.ones((3, 3), np.uint8)
                            inner_mask = cv2.morphologyEx(inner_mask, cv2.MORPH_OPEN, kernel)
                            inner_mask = cv2.morphologyEx(inner_mask, cv2.MORPH_CLOSE, kernel)
                            
                            # 在内部区域中查找黑色图形轮廓
                            shape_contours, _ = cv2.findContours(inner_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            if shape_contours:
                                print(f"🔍 在内部区域发现 {len(shape_contours)} 个黑色图形")
                                
                                # 找到最大的黑色图形
                                largest_shape = max(shape_contours, key=cv2.contourArea)
                                shape_area = cv2.contourArea(largest_shape)
                                
                                if shape_area > 50:  # 图形足够大
                                    print(f"🎯 最大黑色图形面积: {shape_area:.0f}")
                                    
                                    # 识别形状类型
                                    try:
                                        shape_type = identify_shape_by_area(largest_shape)
                                        print(f"🔍 形状识别结果: {shape_type}")
                                    except Exception as shape_error:
                                        print(f"⚠️ 形状识别错误: {shape_error}")
                                        shape_type = "Unknown"
                                    
                                    # 只处理支持的形状
                                    if shape_type in ["Circle", "Triangle", "Square"]:
                                        # 计算图形的尺寸和像素数
                                        inner_area_pixels = int(shape_area)
                                        
                                        if shape_type == "Circle":
                                            # 对于圆形，计算最小外接圆的直径
                                            (circle_x, circle_y), radius = cv2.minEnclosingCircle(largest_shape)
                                            diameter = int(2 * radius)
                                            inner_black_w = inner_black_h = diameter
                                        else:
                                            # 对于三角形和正方形，使用边界框
                                            shape_rect = cv2.boundingRect(largest_shape)
                                            inner_black_w, inner_black_h = shape_rect[2], shape_rect[3]
                                        
                                        # 存储形状信息
                                        inner_shapes.append({
                                            'contour': largest_shape,
                                            'shape_type': shape_type,
                                            'area': shape_area,
                                            'width': inner_black_w,
                                            'height': inner_black_h,
                                            'pixels': inner_area_pixels
                                        })
                                        
                                        print(f"✅ 成功识别内部图形: {shape_type}")
                                        print(f"📏 图形尺寸: {inner_black_w}x{inner_black_h}")
                                        print(f"📊 图形像素数: {inner_area_pixels}")
                                        
                                        # 在原图上绘制内部图形轮廓（转换坐标）
                                        shape_global = largest_shape.copy()
                                        shape_global[:, :, 0] += x + wx
                                        shape_global[:, :, 1] += y + wy
                                        
                                        # 根据形状类型用不同颜色绘制
                                        if shape_type == "Circle":
                                            cv2.polylines(frame, [shape_global], True, (255, 0, 255), 3)  # 紫色圆形
                                        elif shape_type == "Triangle":
                                            cv2.polylines(frame, [shape_global], True, (0, 255, 255), 3)  # 黄色三角形
                                        elif shape_type == "Square":
                                            cv2.polylines(frame, [shape_global], True, (255, 255, 0), 3)  # 青色正方形
                                    else:
                                        shape_type = "None"
                                        print(f"❌ 不支持的形状: {shape_type}")
                                else:
                                    print("❌ 黑色图形太小，忽略")
                            else:
                                print("❌ 内部区域未发现黑色图形")
                        else:
                            print("❌ 白色区域太小，忽略")
                    else:
                        print("❌ 未发现内部白色区域")
                    
                    # 绘制矩形轮廓
                    cv2.polylines(frame, [approx], True, (0, 0, 255), 2)
                    
                    # 如果检测到支持的内部形状，在原图上绘制轮廓
                    if inner_shapes and shape_type in ["Circle", "Triangle", "Square"]:
                        largest_inner_global = inner_shapes[0]['contour'].copy()
                        # 将相对坐标转换为全局坐标
                        largest_inner_global[:, :, 0] += x
                        largest_inner_global[:, :, 1] += y
                        
                        # 根据形状类型用不同颜色绘制内部形状
                        if shape_type == "Circle":
                            cv2.polylines(frame, [largest_inner_global], True, (255, 0, 255), 3)  # 紫色圆形
                        elif shape_type == "Triangle":
                            cv2.polylines(frame, [largest_inner_global], True, (0, 255, 255), 3)  # 黄色三角形
                        elif shape_type == "Square":
                            cv2.polylines(frame, [largest_inner_global], True, (255, 255, 0), 3)  # 青色正方形
                        
                        # 统计形状数量
                        shape_stats[shape_type] += 1
                    else:
                        shape_stats["None"] += 1
                    
                    # 计算中心点
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.putText(frame, f"#{rectangles_found}", (cx-15, cy-40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                        cv2.putText(frame, f"Outer:{w}x{h}", (cx-35, cy-20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        cv2.putText(frame, f"{shape_type}:{inner_black_w}x{inner_black_h}", (cx-35, cy), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                        cv2.putText(frame, f"{int(area)}px", (cx-20, cy+20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    # 存储矩形信息
                    rectangle_info.append({
                        'id': rectangles_found,
                        'area': int(area),
                        'outer_width': w,
                        'outer_height': h,
                        'inner_width': inner_black_w,
                        'inner_height': inner_black_h,
                        'shape_type': shape_type,
                        'center': (cx, cy) if M["m00"] != 0 else (x + w//2, y + h//2)
                    })
            
            # 计算占比
            frame_ratio = (total_pixels / total_frame_pixels) * 100 if total_frame_pixels > 0 else 0
            black_ratio = (black_pixels_count / total_frame_pixels) * 100 if total_frame_pixels > 0 else 0
            
            # 显示统计信息
            cv2.putText(frame, f"Rectangles: {rectangles_found}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Total Pixels: {total_pixels}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame Ratio: {frame_ratio:.2f}%", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Black Ratio: {black_ratio:.2f}%", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 显示HSV参数信息
            h1_min, h1_max, s1_min, s1_max, v1_min, v1_max = hsv_params['range1']
            cv2.putText(frame, f"HSV1: [{h1_min}-{h1_max},{s1_min}-{s1_max},{v1_min}-{v1_max}]", 
                       (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            if hsv_params['use_range2']:
                h2_min, h2_max, s2_min, s2_max, v2_min, v2_max = hsv_params['range2']
                cv2.putText(frame, f"HSV2: [{h2_min}-{h2_max},{s2_min}-{s2_max},{v2_min}-{v2_max}]", 
                           (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            cv2.putText(frame, f"Min Area: {hsv_params['min_area']}", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 显示形状统计
            y_offset = 150
            if shape_stats["Square"] > 0:
                cv2.putText(frame, f"Squares: {shape_stats['Square']}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                y_offset += 20
            if shape_stats["Circle"] > 0:
                cv2.putText(frame, f"Circles: {shape_stats['Circle']}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                y_offset += 20
            if shape_stats["Triangle"] > 0:
                cv2.putText(frame, f"Triangles: {shape_stats['Triangle']}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y_offset += 20
                
            cv2.putText(frame, f"Frame: {frame_count}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 计算FPS
            process_time = time.time() - start_time
            fps = 1.0 / process_time if process_time > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, y_offset+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 显示画面
            cv2.imshow('Rectangle Detector', frame)
            
            # 显示掩码（用于调试HSV参数）
            if show_controls:
                cv2.imshow('HSV Mask', mask)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("🛑 用户退出")
                break
            elif key == ord('s'):
                filename = f"detection_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 保存: {filename}")
            elif key == ord('h'):
                show_controls = not show_controls
                if show_controls:
                    cv2.namedWindow("HSV Controls", cv2.WINDOW_NORMAL)
                    cv2.imshow('HSV Mask', mask)
                    print("💡 HSV控制面板已显示")
                else:
                    cv2.destroyWindow("HSV Controls")
                    cv2.destroyWindow('HSV Mask')
                    print("💡 HSV控制面板已隐藏")
            elif key == ord('p'):
                print(f"📊 Frame {frame_count}: {rectangles_found} rectangles, {total_pixels} pixels")
                print(f"📏 Frame Ratio: {frame_ratio:.2f}%, Black Ratio: {black_ratio:.2f}%")
                print(f"🔺 Shape Statistics: Squares: {shape_stats['Square']}, Circles: {shape_stats['Circle']}, Triangles: {shape_stats['Triangle']}, None: {shape_stats['None']}")
                print(f"🎛️ HSV Range1: H{hsv_params['range1'][0]}-{hsv_params['range1'][1]}, S{hsv_params['range1'][2]}-{hsv_params['range1'][3]}, V{hsv_params['range1'][4]}-{hsv_params['range1'][5]}")
                if hsv_params['use_range2']:
                    print(f"🎛️ HSV Range2: H{hsv_params['range2'][0]}-{hsv_params['range2'][1]}, S{hsv_params['range2'][2]}-{hsv_params['range2'][3]}, V{hsv_params['range2'][4]}-{hsv_params['range2'][5]}")
                if rectangle_info:
                    print("📐 Rectangle Details:")
                    for rect in rectangle_info:
                        print(f"   Rectangle #{rect['id']}: Outer {rect['outer_width']}x{rect['outer_height']}, {rect['shape_type']} {rect['inner_width']}x{rect['inner_height']}, Area: {rect['area']}px")
    
    except KeyboardInterrupt:
        print("⚠️ 程序中断")
    except Exception as e:
        print(f"❌ 运行错误: {e}")
    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        print("✅ 程序退出")

if __name__ == "__main__":
    main()
