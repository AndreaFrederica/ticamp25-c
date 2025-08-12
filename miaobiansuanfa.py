import cv2
import numpy as np
import math

def binary_edge_detection_with_corners(image_path, output_path='output_edges.jpg'):
    """图像二值化+边缘检测+角点检测算法
    Args:
        image_path: 输入图像路径
        output_path: 输出描边图像保存路径（默认output_edges.jpg）
    Returns:
        dict: 包含edges(边缘图像), corners(角点坐标), result(最终结果图像)
    """
    # 1. 读取图像并转换为灰度图
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("图像读取失败，请检查路径是否正确")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 彩色转灰度
    
    # 2. 图像二值化（使用Otsu自动阈值）
    _, binary = cv2.threshold(
        gray, 
        0, 255, 
        cv2.THRESH_BINARY + cv2.THRESH_OTSU  # 大津法自动确定最佳阈值
    )
    
    # 3. 边缘检测（Canny算法）
    edges = cv2.Canny(
        binary,
        50, 150,  # 低阈值和高阈值
        apertureSize=3  # Sobel算子大小
    )
    
    # 4. 角点检测（提高质量阈值，寻找更明显的角点）
    corners = detect_corners(gray, max_corners=15, quality_level=0.05, min_distance=40)
    print(f"步骤1: 检测到 {len(corners)} 个角点")
    
    # 5. 去掉内角点，保留外角点
    outer_corners = filter_outer_corners(corners, binary)
    print(f"步骤2: 保留 {len(outer_corners)} 个外角点")
    
    # 6. 连接外角点并过滤黑线区域的连线
    valid_connections = connect_and_filter_outer_corners(outer_corners, binary)
    print(f"步骤3: 在黑线区域找到 {len(valid_connections)} 条有效连线")
    
    # 7. 找出最短边长
    shortest_edge = find_shortest_edge(valid_connections)
    if shortest_edge:
        shortest_length = shortest_edge[2]
        print(f"步骤4: 最短边长为 {shortest_length:.2f} 像素")
    else:
        shortest_length = 0
        print("步骤4: 未找到有效的最短边")
    
    # 8. 创建结果图像
    result = create_result_with_outer_corners(img, edges, corners, outer_corners, valid_connections, shortest_edge)
    
    # 9. 保存结果
    cv2.imwrite(output_path, result)
    
    return {
        'edges': edges,
        'corners': corners,
        'outer_corners': outer_corners,
        'valid_connections': valid_connections,
        'shortest_edge': shortest_edge,
        'shortest_length': shortest_length,
        'result': result,
        'original': img,
        'binary': binary
    }

def detect_corners(gray, max_corners=15, quality_level=0.05, min_distance=40):
    """检测图像中的角点（更严格的参数，寻找更明显的角点）
    Args:
        gray: 灰度图像
        max_corners: 最大角点数量
        quality_level: 角点质量阈值（提高以获得更明显的角点）
        min_distance: 角点间最小距离（增大以避免聚集）
    Returns:
        list: 角点坐标列表 [(x1,y1), (x2,y2), ...]
    """
    # 方法1: 使用Shi-Tomasi角点检测（更严格的参数）
    corners_st = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,  # 提高质量阈值
        minDistance=min_distance,    # 增大最小距离
        blockSize=5,                 # 增大块大小
        useHarrisDetector=False
    )
    
    # 方法2: 使用Harris角点检测作为补充（更严格的参数）
    harris = cv2.cornerHarris(gray, 3, 5, 0.06)  # 增大窗口大小和k值
    harris = cv2.dilate(harris, None)
    
    # 合并角点结果
    corners = []
    
    # 添加Shi-Tomasi角点
    if corners_st is not None:
        for corner in corners_st:
            x, y = corner.ravel()
            corners.append((int(x), int(y)))
    
    # 添加Harris角点（去重），使用更高的阈值
    threshold = 0.05 * harris.max()  # 提高阈值，只保留最强的角点
    for i in range(harris.shape[0]):
        for j in range(harris.shape[1]):
            if harris[i, j] > threshold:
                # 检查是否与已有角点重复
                is_duplicate = False
                for existing in corners:
                    dist = math.sqrt((j - existing[0])**2 + (i - existing[1])**2)
                    if dist < min_distance:
                        is_duplicate = True
                        break
                
                if not is_duplicate and len(corners) < max_corners:
                    corners.append((j, i))  # (x, y)
    
    return corners

def filter_outer_corners(corners, binary, boundary_threshold=10):
    """去掉内角点，保留外角点
    Args:
        corners: 所有角点列表
        binary: 二值图像
        boundary_threshold: 边界阈值
    Returns:
        list: 外角点列表
    """
    outer_corners = []
    h, w = binary.shape
    
    for corner in corners:
        x, y = corner
        
        # 检查角点是否靠近图像边界
        near_boundary = (x < boundary_threshold or x > w - boundary_threshold or 
                        y < boundary_threshold or y > h - boundary_threshold)
        
        # 检查角点周围的像素分布来判断是否为外角点
        is_outer = check_if_outer_corner(binary, x, y)
        
        if is_outer or near_boundary:
            outer_corners.append(corner)
    
    return outer_corners

def check_if_outer_corner(binary, x, y, radius=15):
    """检查角点是否为外角点
    Args:
        binary: 二值图像
        x, y: 角点坐标
        radius: 检查半径
    Returns:
        bool: 是否为外角点
    """
    h, w = binary.shape
    
    # 统计角点周围黑白像素的分布
    black_count = 0
    white_count = 0
    total_count = 0
    
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                total_count += 1
                if binary[ny, nx] == 0:  # 黑色像素
                    black_count += 1
                else:  # 白色像素
                    white_count += 1
    
    if total_count == 0:
        return False
    
    # 如果周围有足够的白色像素，认为是外角点
    white_ratio = white_count / total_count
    return white_ratio > 0.3  # 至少30%的白色像素

def connect_and_filter_outer_corners(outer_corners, binary):
    """连接外角点并过滤在黑线区域的连线
    Args:
        outer_corners: 外角点列表
        binary: 二值图像
    Returns:
        list: 有效连线列表 [((x1,y1), (x2,y2), length), ...]
    """
    if len(outer_corners) < 2:
        return []
    
    valid_connections = []
    
    # 连接所有外角点对
    for i in range(len(outer_corners)):
        for j in range(i + 1, len(outer_corners)):
            pt1, pt2 = outer_corners[i], outer_corners[j]
            
            # 计算连线长度
            length = math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
            
            # 检查连线是否主要在黑色区域
            if is_line_in_black_region(binary, pt1, pt2):
                valid_connections.append((pt1, pt2, length))
    
    return valid_connections

def is_line_in_black_region(binary, pt1, pt2, threshold=0.7):
    """检查连线是否主要在黑色区域
    Args:
        binary: 二值图像
        pt1, pt2: 连线两端点
        threshold: 黑色像素比例阈值
    Returns:
        bool: 是否主要在黑色区域
    """
    x1, y1 = pt1
    x2, y2 = pt2
    
    # 获取连线上的所有点
    line_points = get_line_points(x1, y1, x2, y2)
    
    if not line_points:
        return False
    
    # 统计黑色像素数量
    black_count = 0
    total_count = len(line_points)
    h, w = binary.shape
    
    for x, y in line_points:
        if 0 <= x < w and 0 <= y < h:
            if binary[y, x] == 0:  # 黑色像素
                black_count += 1
    
    # 如果大部分像素是黑色，认为连线在黑色区域
    return (black_count / total_count) >= threshold

def get_line_points(x1, y1, x2, y2):
    """使用Bresenham算法获取两点间直线上的所有像素点
    Args:
        x1, y1, x2, y2: 起点和终点坐标
    Returns:
        list: 直线上的像素点坐标列表
    """
    points = []
    
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    
    x_step = 1 if x1 < x2 else -1
    y_step = 1 if y1 < y2 else -1
    
    x, y = x1, y1
    
    if dx > dy:
        error = dx / 2
        while x != x2:
            points.append((x, y))
            error -= dy
            if error < 0:
                y += y_step
                error += dx
            x += x_step
        points.append((x2, y2))
    else:
        error = dy / 2
        while y != y2:
            points.append((x, y))
            error -= dx
            if error < 0:
                x += x_step
                error += dy
            y += y_step
        points.append((x2, y2))
    
    return points

def find_shortest_edge(valid_connections):
    """找出最短的连线
    Args:
        valid_connections: 有效连线列表
    Returns:
        tuple: 最短连线 (pt1, pt2, length) 或 None
    """
    if not valid_connections:
        return None
    
    # 按长度排序，返回最短的
    valid_connections.sort(key=lambda x: x[2])
    return valid_connections[0]

def create_result_with_outer_corners(original_img, edges, all_corners, outer_corners, valid_connections, shortest_edge):
    """创建包含外角点分析的结果图像
    Args:
        original_img: 原始图像
        edges: 边缘图像
        all_corners: 所有角点
        outer_corners: 外角点
        valid_connections: 有效连线
        shortest_edge: 最短边
    Returns:
        np.ndarray: 结果图像
    """
    result = original_img.copy()
    
    # 1. 绘制边缘（灰色）
    result[edges > 0] = [128, 128, 128]
    
    # 2. 绘制所有角点（蓝色小圆点）
    for corner in all_corners:
        x, y = corner
        cv2.circle(result, (x, y), 3, (255, 0, 0), -1)
    
    # 3. 绘制外角点（红色大圆点）
    for corner in outer_corners:
        x, y = corner
        cv2.circle(result, (x, y), 6, (0, 0, 255), -1)
        cv2.circle(result, (x, y), 8, (255, 255, 255), 1)
    
    # 4. 绘制有效连线（绿色）
    for connection in valid_connections:
        pt1, pt2, length = connection
        cv2.line(result, pt1, pt2, (0, 255, 0), 2)
    
    # 5. 绘制最短边（黄色，较粗）
    if shortest_edge:
        pt1, pt2, length = shortest_edge
        cv2.line(result, pt1, pt2, (0, 255, 255), 4)
        
        # 在最短边中点标注长度
        mid_x = (pt1[0] + pt2[0]) // 2
        mid_y = (pt1[1] + pt2[1]) // 2
        cv2.putText(result, f"{length:.1f}", (mid_x - 20, mid_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # 6. 添加信息文本
    cv2.putText(result, f"All Corners: {len(all_corners)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(result, f"Outer Corners: {len(outer_corners)}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(result, f"Valid Connections: {len(valid_connections)}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if shortest_edge:
        cv2.putText(result, f"Shortest Edge: {shortest_edge[2]:.2f}px", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.putText(result, "Blue: All Corners", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    cv2.putText(result, "Red: Outer Corners", (10, 135), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.putText(result, "Green: Valid Lines", (10, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(result, "Yellow: Shortest Edge", (10, 165), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    return result

def detect_complete_square_edges(edges, min_line_length=50, max_line_gap=10):
    """检测完整的正方形边缘
    Args:
        edges: 边缘图像
        min_line_length: 最小线段长度
        max_line_gap: 最大线段间隙
    Returns:
        dict: 包含完整边缘和不完整边缘的信息
    """
    # 使用霍夫变换检测直线
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                           minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    if lines is None:
        return {'complete_edges': [], 'incomplete_edges': [], 'all_lines': []}
    
    # 分析每条线段
    complete_edges = []
    incomplete_edges = []
    all_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        all_lines.append(((x1, y1), (x2, y2)))
        
        # 计算线段长度
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # 检查线段是否是完整的边缘
        is_complete = check_edge_completeness(edges, x1, y1, x2, y2)
        
        if is_complete and length > min_line_length:
            complete_edges.append(((x1, y1), (x2, y2), length))
        else:
            incomplete_edges.append(((x1, y1), (x2, y2), length))
    
    return {
        'complete_edges': complete_edges,
        'incomplete_edges': incomplete_edges,
        'all_lines': all_lines
    }

def check_edge_completeness(edges, x1, y1, x2, y2, tolerance=5):
    """检查边缘是否完整（两端是否有角点或连接）
    Args:
        edges: 边缘图像
        x1, y1, x2, y2: 线段端点
        tolerance: 检查范围
    Returns:
        bool: 是否为完整边缘
    """
    h, w = edges.shape
    
    # 检查线段两端的连接情况
    def check_endpoint_connection(x, y):
        if x < tolerance or x >= w - tolerance or y < tolerance or y >= h - tolerance:
            return False
        
        # 在端点周围查找连接的边缘
        connections = 0
        for dx in range(-tolerance, tolerance + 1):
            for dy in range(-tolerance, tolerance + 1):
                if 0 <= x + dx < w and 0 <= y + dy < h:
                    if edges[y + dy, x + dx] > 0:
                        connections += 1
        
        return connections > tolerance * 2  # 如果有足够多的连接点，认为是完整的
    
    # 检查线段是否与其他边缘形成角点
    start_connected = check_endpoint_connection(x1, y1)
    end_connected = check_endpoint_connection(x2, y2)
    
    return start_connected and end_connected

def create_result_with_complete_edges(original_img, edges, corners, edge_analysis):
    """创建包含完整和不完整边缘的结果图像
    Args:
        original_img: 原始图像
        edges: 边缘图像
        corners: 角点坐标列表
        edge_analysis: 边缘分析结果
    Returns:
        np.ndarray: 结果图像
    """
    result = original_img.copy()
    
    # 1. 绘制原始边缘（灰色，较细）
    result[edges > 0] = [128, 128, 128]  # 灰色边缘
    
    # 2. 绘制不完整边缘（红色）
    for edge in edge_analysis['incomplete_edges']:
        (x1, y1), (x2, y2), length = edge
        cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 红色线条
    
    # 3. 绘制完整边缘（绿色，较粗）
    for edge in edge_analysis['complete_edges']:
        (x1, y1), (x2, y2), length = edge
        cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 3)  # 绿色线条
    
    # 4. 绘制角点
    for corner in corners:
        x, y = corner
        cv2.circle(result, (x, y), 4, (255, 255, 0), -1)  # 黄色角点
    
    # 5. 添加信息文本
    cv2.putText(result, f"Complete Edges: {len(edge_analysis['complete_edges'])}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(result, f"Incomplete Edges: {len(edge_analysis['incomplete_edges'])}", (10, 55), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(result, "Gray: All Edges", (10, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
    cv2.putText(result, "Green: Complete Edges", (10, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(result, "Red: Incomplete Edges", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(result, "Yellow: Corners", (10, 140), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    return result

def create_result_with_corners(original_img, edges, corners):
    """创建包含边缘和角点的结果图像
    Args:
        original_img: 原始图像
        edges: 边缘图像
        corners: 角点坐标列表
    Returns:
        np.ndarray: 结果图像
    """
    # 创建结果图像（以原图为基础）
    result = original_img.copy()
    
    # 1. 绘制边缘（绿色）
    result[edges > 0] = [0, 255, 0]  # 绿色边缘
    
    # 2. 绘制角点（红色圆点）
    for corner in corners:
        x, y = corner
        cv2.circle(result, (x, y), 5, (0, 0, 255), -1)  # 红色实心圆
        cv2.circle(result, (x, y), 8, (255, 255, 255), 2)  # 白色外圈
    
    # 3. 连接相邻角点（蓝色线条）
    if len(corners) > 1:
        connected_pairs = connect_adjacent_corners(corners)
        for pair in connected_pairs:
            pt1, pt2 = pair
            cv2.line(result, pt1, pt2, (255, 0, 0), 3)  # 蓝色连线，线宽3
    
    # 4. 添加信息文本
    cv2.putText(result, f"Corners Found: {len(corners)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, "Green: Edges", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(result, "Red: Corners", (10, 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(result, "Blue: Corner Lines", (10, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    return result

# 使用示例
if __name__ == "__main__":
    input_image = "c:\\Users\\zengp\\Desktop\\Snipaste_2025-08-02_09-14-51.png"  # 替换为你的图片路径
    output_image = "edges_corners_result.jpg"
    
    # 执行边缘检测和角点检测
    results = binary_edge_detection_with_corners(input_image, output_image)
    
    print(f"处理完成！")
    print(f"所有角点: {len(results['corners'])}")
    print(f"外角点: {len(results['outer_corners'])}")
    print(f"有效连线: {len(results['valid_connections'])}")
    if results['shortest_edge']:
        print(f"最短边长: {results['shortest_length']:.2f} 像素")
    else:
        print("未找到有效的最短边")
    print(f"结果已保存到: {output_image}")
    
    # 显示结果
    cv2.imshow('Original', results['original'])
    cv2.imshow('Binary', results['binary'])
    cv2.imshow('Edges', results['edges'])
    cv2.imshow('Final Result', results['result'])
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()