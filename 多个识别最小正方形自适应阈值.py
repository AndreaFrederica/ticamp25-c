import cv2
import numpy as np
import os

# ---------------------- 工具函数 ----------------------
def _order_points(pts):
    """排序四边形角点为左上→右上→右下→左下"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下
    return rect

def _detect_outer_corners(img):
    """检测图像外框角点（用于透视变换）"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("未找到图像外框")
    
    # 筛选面积前3的轮廓，优先选择四边形
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
    largest = None
    for cnt in contours_sorted:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            largest = cnt
            break
    if largest is None:
        largest = contours_sorted[0]
        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
    
    if len(approx) != 4:
        raise ValueError("外框不是四边形")
    return _order_points(approx.reshape(4, 2))

def _judge_corner_type(bin_img, corner, radius=30):
    """使用圆形像素分布法判断角点类型（内角/外角）
    返回：True为外角点（绿色），False为内角点（红色）
    """
    h, w = bin_img.shape
    x, y = corner
    # 创建圆形掩码
    mask_circle = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_circle, (x, y), radius, 255, -1)
    # 计算圆内白色（目标）和黑色（背景）像素数量
    roi = bin_img & mask_circle
    total_pixels = cv2.countNonZero(mask_circle)
    white_pixels = cv2.countNonZero(roi)
    black_pixels = total_pixels - white_pixels
    # 外角点判定：白色像素少（目标外部）
    return white_pixels < black_pixels

def _calculate_distance(pt1, pt2):
    """计算两点间像素距离"""
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])** 2)

# ---------------------- 主函数 ----------------------
def process_with_area_threshold(image_path: str, output_dir: str = "output", min_area_threshold: int = 500):
    """
    新增功能：
    1. 二值化后添加连通域面积过滤，小于min_area_threshold的连通域将被抹除
    2. 可通过参数手动设定最小面积阈值（默认500像素）
    """
    # 初始化目录
    os.makedirs(output_dir, exist_ok=True)
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    filename = os.path.splitext(os.path.basename(image_path))[0]

    # 1. 读取图像与透视变换
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像：{image_path}")
    cv2.imwrite(os.path.join(debug_dir, f"{filename}_original.jpg"), img)

    outer_corners = _detect_outer_corners(img)
    dst_pts = np.float32([[0, 0], [2100, 0], [2100, 2970], [0, 2970]])
    M = cv2.getPerspectiveTransform(outer_corners, dst_pts)
    img_warped = cv2.warpPerspective(img, M, (2100, 2970))
    cv2.imwrite(os.path.join(debug_dir, f"{filename}_warped.jpg"), img_warped)

    # 2. 二值化处理
    gray_warped = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)
    _, binary_inv = cv2.threshold(gray_warped, int(gray_warped.mean()), 255, cv2.THRESH_BINARY_INV)  # 目标为白色
    h, w = binary_inv.shape
    # 去除背景
    for seed in [(0, 0), (w-20, 0), (0, h-20), (w-20, h-20)]:
        cv2.floodFill(binary_inv, None, seed, 0)
    cv2.imwrite(os.path.join(debug_dir, f"{filename}_binary_before_filter.jpg"), binary_inv)

    # 3. 连通域面积过滤（核心新增步骤）
    # 3.1 分析连通域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_inv)
    # 3.2 创建过滤后的二值图（初始全为背景）
    filtered_binary = np.zeros_like(binary_inv)
    # 3.3 保留面积大于等于阈值的连通域
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area_threshold:
            # 将符合条件的连通域保留（设置为白色）
            filtered_binary[labels == label_id] = 255
    # 保存过滤后的二值图
    cv2.imwrite(os.path.join(debug_dir, f"{filename}_binary_after_filter.jpg"), filtered_binary)
    print(f"已过滤面积小于 {min_area_threshold} 像素的连通域")

    # # 获取图像尺寸
    h, w = filtered_binary.shape

    # 定义四个角的像素位置（靠近边缘的像素）
    corners = [
        filtered_binary[150, 150],         # 左上角
        filtered_binary[150, w-150],       # 右上角
        filtered_binary[h-150, 150],       # 左下角
        filtered_binary[h-150, w-150]      # 右下角
    ]

    # 统计黑色像素的数量（值为0）
    black_count = sum(1 for val in corners if val < 30)

    # 如果黑色像素数量 >= 3，则 a=1，否则 a=0
    a = 100 if black_count >= 3 else 120

    # 4. 基于过滤后的二值图提取连通域
    num_labels_filtered, labels_filtered, stats_filtered, _ = cv2.connectedComponentsWithStats(filtered_binary)
    if num_labels_filtered < 2:
        raise ValueError(f"过滤后未检测到有效连通域（面积均小于 {min_area_threshold} 像素）")

    # 准备绘图
    result_img = img_warped.copy()
    corners_info = {}  # {连通域ID: [(角点坐标, 是否外角), ...]}
    valid_edges = []   # 存储有效线段：[(起点, 终点, 长度, 连通域ID), ...]

    # 5. 处理每个过滤后的连通域
    for label_id in range(1, num_labels_filtered):
        area = stats_filtered[label_id, cv2.CC_STAT_AREA]
        # 提取连通域掩码与轮廓
        mask = np.uint8(labels_filtered == label_id) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)

        # 提取角点
        peri = cv2.arcLength(cnt, True)
        epsilon = 0.01 * peri  # 轮廓逼近精度
        approx = cv2.approxPolyDP(cnt, epsilon, True).reshape(-1, 2)
        corners = [(int(x), int(y)) for x, y in approx]
        if len(corners) < 3:
            continue

        # 判断角点类型
        corner_types = [_judge_corner_type(filtered_binary, corner) for corner in corners]
        corners_info[label_id] = list(zip(corners, corner_types))
        print(f"\n连通域 {label_id} 处理：")
        print(f"  面积：{area:.1f}px，角点数量：{len(corners)} 个")

        # 6. 角点连线与有效线段筛选
        num_corners = len(corners)
        for i in range(num_corners):
            j = (i + 1) % num_corners
            pt1 = corners[i]
            pt2 = corners[j]
            type1 = corner_types[i]
            type2 = corner_types[j]

            # 绘制绿线（所有首尾连线）
            cv2.line(result_img, pt1, pt2, (0, 255, 0), 2)

            # 筛选有效线段（两端均为外角点）
            if type1 and type2:
                length = _calculate_distance(pt1, pt2)
                valid_edges.append((pt1, pt2, length, label_id))
                print(f"  有效线段 {label_id}-{i+1}→{label_id}-{j+1}：长度 {length:.1f}px")

    # 7. 找到并标注最短有效线段
    if not valid_edges:
        raise ValueError("未检测到有效线段（两端均为外角点的线段）")

    shortest_edge = min(valid_edges, key=lambda x: x[2])
    pt1, pt2, shortest_length, comp_id = shortest_edge
    print(f"\n===== 分析结果 =====")
    print(f"最短有效线段：连通域 {comp_id} 中的 {pt1}→{pt2}，长度 {shortest_length:.1f}px")

    # 用红线标注最短有效线段
    cv2.line(result_img, pt1, pt2, (0, 0, 255), 3)
    mid_pt = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
    cv2.putText(result_img, f"最短：{shortest_length:.1f}px", mid_pt,
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 8. 保存结果
    output_img_path = os.path.join(output_dir, f"{filename}_final_result.jpg")
    cv2.imwrite(output_img_path, result_img)


    # 转换为厘米
    print(a)
    #real_length = shortest / a
    diameter_cm = shortest_length / a
    #diameter_mm = diameter_px / 2100 * 20        # 2100 px 对应 20 mm
    #return diameter_cm

    return shortest_length, output_img_path,diameter_cm

# ---------------------- 运行示例 ----------------------
if __name__ == "__main__":
    input_path = "./image/1.png"  # 替换为你的图像路径
    output_dir = "./image/out.png"
    min_area = 60 # 可在此手动设定最小面积阈值（像素）

    try:
        shortest_len, result_path,diameter_cm = process_with_area_threshold(
            input_path, 
            output_dir, 
            min_area_threshold=min_area
        )
        print(f"\n处理完成！最短有效线段长度：{shortest_len:.2f}px")
        print(f"\n最小正方形边长：{diameter_cm:.2f}cm")
        print(f"结果图像保存至：{result_path}")
        print(f"过滤前后的二值图可在debug目录查看")
    except Exception as e:
        print(f"处理失败：{str(e)}")
