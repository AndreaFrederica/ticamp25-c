import cv2
import numpy as np
import os

def extract_white_rectangles(image_path, output_dir="extracted_rectangles", min_area=500, save_images=True):
    """
    从图像中提取白色矩形区域（包括里面的文字）
    
    参数:
    - image_path: 输入图像路径
    - output_dir: 输出目录
    - min_area: 最小矩形面积阈值
    - save_images: 是否保存提取的图像
    
    返回:
    - rectangles: 提取的矩形列表，每个元素包含坐标和图像数据
    """
    
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法读取图像 {image_path}")
        return []
    
    print(f"原始图像尺寸: {img.shape}")
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 创建输出目录
    if save_images and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 方法1: 基于阈值的白色区域检测
    # rectangles = extract_by_threshold(img, gray, output_dir, min_area, save_images)
    
    # 方法2: 基于轮廓的矩形检测并矫正倾斜（推荐）
    rectangles = extract_by_contour_with_correction(img, gray, output_dir, min_area, save_images)
    
    # 方法3: 传统轮廓检测（备选）
    # rectangles = extract_by_contours(img, gray, output_dir, min_area, save_images)
    
    return rectangles

def extract_by_threshold(img, gray, output_dir, min_area, save_images):
    """基于阈值的白色矩形提取"""
    
    rectangles = []
    
    # 1. 二值化 - 提取白色区域
    # 使用高阈值来提取白色区域
    _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # 2. 形态学操作 - 填充和连接
    kernel = np.ones((3, 3), np.uint8)
    # 闭运算：连接近距离的白色区域
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # 开运算：去除噪点
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 3. 寻找连通域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(white_mask, connectivity=8)
    
    print(f"找到 {num_labels-1} 个连通域")
    
    # 4. 处理每个连通域
    for i in range(1, num_labels):  # 跳过背景(标签0)
        # 获取连通域统计信息
        x, y, w, h, area = stats[i]
        
        # 过滤太小的区域
        if area < min_area:
            continue
        
        # 检查是否为矩形形状（宽高比合理）
        aspect_ratio = w / h
        if aspect_ratio < 0.1 or aspect_ratio > 10:  # 过滤过于细长的区域
            continue
        
        # 扩展边界（包含边框）
        margin = 5
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img.shape[1], x + w + margin)
        y2 = min(img.shape[0], y + h + margin)
        
        # 提取矩形区域
        rect_img = img[y1:y2, x1:x2]
        
        # 保存结果
        rectangle_info = {
            'id': i,
            'bbox': (x1, y1, x2, y2),
            'area': area,
            'aspect_ratio': aspect_ratio,
            'image': rect_img,
            'center': centroids[i]
        }
        rectangles.append(rectangle_info)
        
        # 保存图像
        if save_images:
            filename = f"rectangle_{i:03d}_area{area}_pos{x}x{y}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, rect_img)
            print(f"保存矩形 {i}: {filepath} (面积: {area}, 位置: {x},{y}, 尺寸: {w}x{h})")
    
    # 保存调试图像
    if save_images:
        debug_img = img.copy()
        for rect in rectangles:
            x1, y1, x2, y2 = rect['bbox']
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(debug_img, f"ID:{rect['id']}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imwrite(os.path.join(output_dir, "debug_marked.png"), debug_img)
        cv2.imwrite(os.path.join(output_dir, "white_mask.png"), white_mask)
        print(f"保存调试图像到 {output_dir}")
    
    return rectangles

def extract_by_contour_with_correction(img, gray, output_dir, min_area, save_images):
    """基于轮廓提取并矫正倾斜的矩形"""
    
    rectangles = []
    
    # 1. 二值化
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # 2. 形态学操作
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 3. 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"找到 {len(contours)} 个轮廓")
    
    # 4. 处理每个轮廓
    for i, contour in enumerate(contours):
        # 计算轮廓面积
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # 轮廓近似，获取四边形
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 如果近似结果不是四边形，使用最小外接矩形
        if len(approx) != 4:
            # 获取最小外接矩形（可能是倾斜的）
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype=np.int32)  # 修复 np.int0 问题
            approx = box.reshape(-1, 1, 2)
        
        # 检查面积比例（轮廓面积与外接矩形面积的比例）
        rect_area = cv2.contourArea(approx)
        if rect_area == 0:
            continue
            
        area_ratio = area / rect_area
        if area_ratio < 0.5:  # 如果轮廓面积太小，可能不是矩形
            continue
        
        # 提取并矫正矩形
        corrected_img, transform_success = extract_and_correct_rectangle(img, approx.reshape(4, 2))
        
        if corrected_img is not None and transform_success:
            # 检查矫正后的图像尺寸
            h, w = corrected_img.shape[:2]
            if w < 20 or h < 20:  # 太小的图像跳过
                continue
                
            aspect_ratio = w / h
            if aspect_ratio < 0.1 or aspect_ratio > 10:  # 宽高比不合理
                continue
            
            rectangle_info = {
                'id': i,
                'contour_points': approx.reshape(4, 2),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'image': corrected_img,
                'original_contour': contour,
                'corrected_size': (w, h)
            }
            rectangles.append(rectangle_info)
            
            if save_images:
                filename = f"corrected_rect_{i:03d}_area{int(area)}.png"
                filepath = os.path.join(output_dir, filename)
                cv2.imwrite(filepath, corrected_img)
                print(f"保存矫正矩形 {i}: {filepath} (矫正后尺寸: {w}x{h})")
    
    # 保存调试图像
    if save_images and rectangles:
        debug_img = img.copy()
        for rect in rectangles:
            # 绘制原始轮廓
            cv2.drawContours(debug_img, [rect['original_contour']], -1, (0, 255, 0), 2)
            # 绘制四边形顶点
            points = rect['contour_points']
            points_int = np.array(points, dtype=np.int32)  # 确保是整数类型
            cv2.polylines(debug_img, [points_int], True, (255, 0, 0), 3)
            # 标记顶点
            for j, point in enumerate(points_int):
                cv2.circle(debug_img, tuple(point), 5, (0, 0, 255), -1)
                cv2.putText(debug_img, str(j), (point[0]+10, point[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            # 添加ID标签
            cv2.putText(debug_img, f"ID:{rect['id']}", tuple(points_int[0]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imwrite(os.path.join(output_dir, "contour_debug.png"), debug_img)
        cv2.imwrite(os.path.join(output_dir, "binary_mask.png"), binary)
        print(f"保存轮廓调试图像到 {output_dir}")
    
    return rectangles

def extract_and_correct_rectangle(img, points):
    """
    从四个顶点提取并矫正矩形
    
    参数:
    - img: 原始图像
    - points: 四个顶点坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    
    返回:
    - corrected_img: 矫正后的图像
    - success: 是否成功
    """
    
    try:
        # 确保points是正确的格式
        points = np.array(points, dtype=np.float32)
        if points.shape != (4, 2):
            return None, False
        
        # 对顶点进行排序：左上、右上、右下、左下
        ordered_points = order_points(points)
        
        # 计算矫正后的矩形尺寸
        width, height = calculate_corrected_size(ordered_points)
        
        if width < 20 or height < 20:
            return None, False
        
        # 定义目标矩形的四个顶点（矫正后的正矩形）
        dst_points = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
        
        # 计算透视变换矩阵
        transform_matrix = cv2.getPerspectiveTransform(ordered_points, dst_points)
        
        # 应用透视变换
        corrected_img = cv2.warpPerspective(img, transform_matrix, (int(width), int(height)))
        
        return corrected_img, True
        
    except Exception as e:
        print(f"矫正失败: {e}")
        return None, False

def order_points(pts):
    """
    对四个点进行排序：左上、右上、右下、左下
    """
    # 计算中心点
    center = np.mean(pts, axis=0)
    
    # 按照相对于中心点的角度排序
    def angle_from_center(point):
        return np.arctan2(point[1] - center[1], point[0] - center[0])
    
    # 按角度排序
    sorted_pts = sorted(pts, key=angle_from_center)
    
    # 确定四个角点：找到最左上的点作为起点
    dists_from_origin = [pt[0] + pt[1] for pt in sorted_pts]
    start_idx = np.argmin(dists_from_origin)
    
    # 重新排列，确保从左上开始顺时针
    ordered = []
    for i in range(4):
        ordered.append(sorted_pts[(start_idx + i) % 4])
    
    return np.array(ordered, dtype=np.float32)

def calculate_corrected_size(ordered_points):
    """
    根据四个顶点计算矫正后的矩形尺寸
    """
    # 计算上边和下边的长度
    top_width = np.linalg.norm(ordered_points[1] - ordered_points[0])
    bottom_width = np.linalg.norm(ordered_points[2] - ordered_points[3])
    
    # 计算左边和右边的长度
    left_height = np.linalg.norm(ordered_points[3] - ordered_points[0])
    right_height = np.linalg.norm(ordered_points[2] - ordered_points[1])
    
    # 取最大值作为矫正后的尺寸
    width = max(top_width, bottom_width)
    height = max(left_height, right_height)
    
    return int(width), int(height)

def extract_by_contours(img, gray, output_dir, min_area, save_images):
    """基于轮廓的矩形提取（备选方法）"""
    
    rectangles = []
    
    # 1. 二值化
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # 2. 形态学操作
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 3. 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"找到 {len(contours)} 个轮廓")
    
    # 4. 处理每个轮廓
    for i, contour in enumerate(contours):
        # 计算轮廓面积
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # 获取边界矩形
        x, y, w, h = cv2.boundingRect(contour)
        
        # 检查宽高比
        aspect_ratio = w / h
        if aspect_ratio < 0.1 or aspect_ratio > 10:
            continue
        
        # 扩展边界
        margin = 5
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img.shape[1], x + w + margin)
        y2 = min(img.shape[0], y + h + margin)
        
        # 提取矩形区域
        rect_img = img[y1:y2, x1:x2]
        
        rectangle_info = {
            'id': i,
            'bbox': (x1, y1, x2, y2),
            'area': area,
            'aspect_ratio': aspect_ratio,
            'image': rect_img,
            'contour': contour
        }
        rectangles.append(rectangle_info)
        
        if save_images:
            filename = f"contour_rect_{i:03d}_area{int(area)}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, rect_img)
            print(f"保存轮廓矩形 {i}: {filepath}")
    
    return rectangles

def enhance_white_detection(image_path, output_dir="enhanced_extraction"):
    """增强的白色矩形检测 - 使用轮廓矫正方法"""
    
    img = cv2.imread(image_path)
    if img is None:
        return []
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 使用新的轮廓矫正方法
    rectangles_all = extract_by_contour_with_correction(img, gray, output_dir, min_area=300, save_images=True)
    
    print(f"增强检测完成，提取了 {len(rectangles_all)} 个矫正矩形")
    return rectangles_all

def extract_white_rectangles_multi_method(image_path, output_dir="multi_method_extraction", min_area=500):
    """
    使用多种方法提取白色矩形，并进行效果对比
    """
    
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法读取图像 {image_path}")
        return []
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results = {}
    
    # 方法1: 阈值方法
    print("=== 方法1: 阈值检测 ===")
    threshold_dir = os.path.join(output_dir, "method1_threshold")
    os.makedirs(threshold_dir, exist_ok=True)
    results['threshold'] = extract_by_threshold(img, gray, threshold_dir, min_area, True)
    print(f"阈值方法提取了 {len(results['threshold'])} 个矩形")
    
    # 方法2: 轮廓矫正方法
    print("\n=== 方法2: 轮廓矫正 ===")
    contour_dir = os.path.join(output_dir, "method2_contour_correction")
    os.makedirs(contour_dir, exist_ok=True)
    results['contour_correction'] = extract_by_contour_with_correction(img, gray, contour_dir, min_area, True)
    print(f"轮廓矫正方法提取了 {len(results['contour_correction'])} 个矩形")
    
    # 方法3: 传统轮廓方法
    print("\n=== 方法3: 传统轮廓 ===")
    traditional_dir = os.path.join(output_dir, "method3_traditional_contour")
    os.makedirs(traditional_dir, exist_ok=True)
    results['traditional_contour'] = extract_by_contours(img, gray, traditional_dir, min_area, True)
    print(f"传统轮廓方法提取了 {len(results['traditional_contour'])} 个矩形")
    
    # 生成对比报告
    generate_comparison_report(results, output_dir)
    
    return results

def generate_comparison_report(results, output_dir):
    """生成方法对比报告"""
    
    report_path = os.path.join(output_dir, "comparison_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== 白色矩形提取方法对比报告 ===\n\n")
        
        for method_name, rectangles in results.items():
            f.write(f"方法: {method_name}\n")
            f.write(f"提取数量: {len(rectangles)} 个矩形\n")
            
            if rectangles:
                areas = [rect['area'] for rect in rectangles]
                f.write(f"面积范围: {min(areas):.0f} - {max(areas):.0f}\n")
                f.write(f"平均面积: {np.mean(areas):.0f}\n")
                
                if 'aspect_ratio' in rectangles[0]:
                    ratios = [rect['aspect_ratio'] for rect in rectangles]
                    f.write(f"宽高比范围: {min(ratios):.2f} - {max(ratios):.2f}\n")
                
                if 'corrected_size' in rectangles[0]:
                    sizes = [rect['corrected_size'] for rect in rectangles]
                    f.write(f"矫正后尺寸: {sizes}\n")
            
            f.write("\n" + "-"*50 + "\n\n")
        
        # 推荐
        best_method = max(results.keys(), key=lambda k: len(results[k]))
        f.write(f"推荐方法: {best_method} (提取数量最多)\n")
    
    print(f"对比报告已保存到: {report_path}")

def enhance_white_detection(image_path, output_dir="enhanced_extraction"):
    """增强的白色矩形检测"""
    
    img = cv2.imread(image_path)
    if img is None:
        return []
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 多种阈值方法组合
    rectangles_all = []
    
    # 方法1: 固定阈值
    _, mask1 = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    
    # 方法2: 自适应阈值
    mask2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # 方法3: Otsu阈值
    _, mask3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 组合所有mask
    combined_mask = cv2.bitwise_and(mask1, cv2.bitwise_and(mask2, mask3))
    
    # 形态学操作
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_mask, connectivity=8)
    
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        
        if area < 300:  # 最小面积
            continue
        
        aspect_ratio = w / h
        if aspect_ratio < 0.2 or aspect_ratio > 5:
            continue
        
        # 提取区域
        margin = 3
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img.shape[1], x + w + margin)
        y2 = min(img.shape[0], y + h + margin)
        
        rect_img = img[y1:y2, x1:x2]
        
        # 保存
        filename = f"enhanced_rect_{i:03d}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, rect_img)
        
        rectangles_all.append({
            'id': i,
            'bbox': (x1, y1, x2, y2),
            'area': area,
            'image': rect_img
        })
    
    # 保存调试图像
    cv2.imwrite(os.path.join(output_dir, "mask1_fixed.png"), mask1)
    cv2.imwrite(os.path.join(output_dir, "mask2_adaptive.png"), mask2)
    cv2.imwrite(os.path.join(output_dir, "mask3_otsu.png"), mask3)
    cv2.imwrite(os.path.join(output_dir, "combined_mask.png"), combined_mask)
    
    print(f"增强检测完成，提取了 {len(rectangles_all)} 个矩形")
    return rectangles_all

def main():
    """主函数"""
    
    # 输入图像路径
    image_path = input("请输入图像路径: ").strip()
    if not os.path.exists(image_path):
        print("图像文件不存在！")
        return
    
    print(f"处理图像: {image_path}")
    
    # 选择处理方法
    print("\n请选择处理方法:")
    print("1. 轮廓矫正方法 (推荐) - 沿轮廓提取并矫正倾斜")
    print("2. 阈值方法 - 基于连通域的传统方法")
    print("3. 多方法对比 - 同时使用所有方法并对比效果")
    
    choice = input("请输入选择 (1-3): ").strip()
    
    if choice == "1":
        print("\n=== 轮廓矫正方法 ===")
        rectangles = extract_white_rectangles(image_path, min_area=500)
        print(f"提取了 {len(rectangles)} 个矫正矩形")
        
        if rectangles:
            print("\n=== 提取结果统计 ===")
            for i, rect in enumerate(rectangles):
                points = rect['contour_points']
                size = rect.get('corrected_size', 'N/A')
                print(f"矩形 {rect['id']}: 四个顶点{points.tolist()}")
                print(f"  面积={rect['area']:.0f}, 矫正后尺寸={size}")
    
    elif choice == "2":
        print("\n=== 阈值方法 ===")
        # 临时切换到阈值方法
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rectangles = extract_by_threshold(img, gray, "extracted_rectangles", 500, True)
        print(f"提取了 {len(rectangles)} 个矩形")
        
        if rectangles:
            print("\n=== 提取结果统计 ===")
            for rect in rectangles:
                bbox = rect['bbox']
                print(f"矩形 {rect['id']}: 位置({bbox[0]},{bbox[1]})-({bbox[2]},{bbox[3]}), "
                      f"面积={rect['area']}, 宽高比={rect['aspect_ratio']:.2f}")
    
    elif choice == "3":
        print("\n=== 多方法对比 ===")
        results = extract_white_rectangles_multi_method(image_path, min_area=500)
        
        print("\n=== 对比结果 ===")
        for method_name, rectangles in results.items():
            print(f"{method_name}: {len(rectangles)} 个矩形")
    
    else:
        print("无效选择，使用默认的轮廓矫正方法")
        rectangles = extract_white_rectangles(image_path, min_area=500)
        print(f"提取了 {len(rectangles)} 个矫正矩形")

if __name__ == "__main__":
    # 示例用法
    # main()
    
    # 或者直接指定图像路径测试
    test_image = "./input/test.jfif"  # 替换为您的图像路径
    if os.path.exists(test_image):
        print(f"测试图像: {test_image}")
        
        # 使用新的轮廓矫正方法
        print("使用轮廓矫正方法...")
        rectangles = extract_white_rectangles(test_image, min_area=300)
        print(f"提取了 {len(rectangles)} 个白色矩形")
        
        if rectangles:
            print("\n矫正结果:")
            for rect in rectangles:
                size = rect.get('corrected_size', 'N/A')
                print(f"  矩形 {rect['id']}: 矫正后尺寸 {size}")
    else:
        print("请运行 main() 函数或提供正确的图像路径")
        main()
