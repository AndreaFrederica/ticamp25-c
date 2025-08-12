import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

# 全局变量，用于存储最小正方形检测图像（与原 min_area_box 兼容）
min_square_images: List[np.ndarray] = []


def _judge_corner_type(bin_img: np.ndarray, corner: Tuple[int, int], radius: int = 30) -> bool:
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


def _calculate_distance(pt1: Tuple[int, int], pt2: Tuple[int, int]) -> float:
    """计算两点间像素距离"""
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)


def process_crop_for_minimum_square(
    crop_img: np.ndarray, 
    min_area_threshold: int = 500,
    physical_width_mm: float = 170.0,  # A4纸宽度210mm - 40mm边框 = 170mm
    physical_height_mm: float = 257.0  # A4纸高度297mm - 40mm边框 = 257mm
) -> Dict[str, Any]:
    """
    处理单个裁剪图像，检测最小正方形
    
    Args:
        crop_img: 已经裁剪好的图像
        min_area_threshold: 最小面积阈值（像素）
        physical_width_mm: A4纸宽度减去外框后的物理宽度（毫米，210-40=170mm）
        physical_height_mm: A4纸高度减去外框后的物理高度（毫米，297-40=257mm）
    
    Returns:
        包含检测结果的字典：
        {
            "success": bool,
            "shortest_edge_length_px": float,
            "shortest_edge_length_mm": float,
            "component_id": int,
            "start_point": tuple,
            "end_point": tuple,
            "all_valid_edges": list,
            "annotated_image": np.ndarray
        }
    """
    if crop_img is None or crop_img.size == 0:
        return {
            "success": False,
            "error": "输入图像为空",
            "shortest_edge_length_px": 0,
            "shortest_edge_length_mm": 0,
            "component_id": -1,
            "start_point": None,
            "end_point": None,
            "all_valid_edges": [],
            "annotated_image": None
        }

    # 1. 二值化处理
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    _, binary_inv = cv2.threshold(gray, int(gray.mean()), 255, cv2.THRESH_BINARY_INV)  # 目标为白色
    h, w = binary_inv.shape
    
    # 去除背景（从四个角开始填充）
    for seed in [(0, 0), (w-20, 0), (0, h-20), (w-20, h-20)]:
        if 0 <= seed[0] < w and 0 <= seed[1] < h:
            cv2.floodFill(binary_inv, None, seed, 0)

    # 2. 连通域面积过滤
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_inv)
    
    # 创建过滤后的二值图
    filtered_binary = np.zeros_like(binary_inv)
    
    # 保留面积大于等于阈值的连通域
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area_threshold:
            filtered_binary[labels == label_id] = 255

    # 3. 计算像素到毫米的转换比例
    # 基于crop图像的实际尺寸和A4纸的物理尺寸
    crop_height, crop_width = crop_img.shape[:2]
    
    # 计算像素到毫米的转换比例
    mm_per_pixel_x = physical_width_mm / crop_width
    mm_per_pixel_y = physical_height_mm / crop_height
    
    # 使用平均值作为最终的转换比例
    mm_per_pixel = (mm_per_pixel_x + mm_per_pixel_y) / 2

    # 4. 基于过滤后的二值图提取连通域
    num_labels_filtered, labels_filtered, stats_filtered, _ = cv2.connectedComponentsWithStats(filtered_binary)
    
    if num_labels_filtered < 2:
        return {
            "success": False,
            "error": f"过滤后未检测到有效连通域（面积均小于 {min_area_threshold} 像素）",
            "shortest_edge_length_px": 0,
            "shortest_edge_length_mm": 0,
            "component_id": -1,
            "start_point": None,
            "end_point": None,
            "all_valid_edges": [],
            "annotated_image": crop_img.copy()
        }

    # 准备绘图和结果收集
    result_img = crop_img.copy()
    valid_edges = []   # 存储有效线段：[(起点, 终点, 长度, 连通域ID), ...]

    # 5. 处理每个过滤后的连通域
    for label_id in range(1, num_labels_filtered):
        area = stats_filtered[label_id, cv2.CC_STAT_AREA]
        
        # 提取连通域掩码与轮廓
        mask = (labels_filtered == label_id).astype(np.uint8) * 255
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

    # 7. 找到并标注最短有效线段
    if not valid_edges:
        return {
            "success": False,
            "error": "未检测到有效线段（两端均为外角点的线段）",
            "shortest_edge_length_px": 0,
            "shortest_edge_length_mm": 0,
            "component_id": -1,
            "start_point": None,
            "end_point": None,
            "all_valid_edges": [],
            "annotated_image": result_img
        }

    shortest_edge = min(valid_edges, key=lambda x: x[2])
    pt1, pt2, shortest_length, comp_id = shortest_edge

    # 用红线标注最短有效线段
    cv2.line(result_img, pt1, pt2, (0, 0, 255), 3)
    mid_pt = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
    cv2.putText(result_img, f"最短：{shortest_length:.1f}px", mid_pt,
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 计算物理长度（毫米）
    diameter_mm = shortest_length * mm_per_pixel

    return {
        "success": True,
        "shortest_edge_length_px": shortest_length,
        "shortest_edge_length_mm": diameter_mm,
        "component_id": comp_id,
        "start_point": pt1,
        "end_point": pt2,
        "all_valid_edges": valid_edges,
        "annotated_image": result_img,
        "mm_per_pixel": mm_per_pixel,
        "filtered_components": num_labels_filtered - 1
    }


def process_multiple_crops(
    crops: List[np.ndarray], 
    min_area_threshold: int = 500,
    physical_width_mm: float = 170.0,  # A4纸宽度210mm - 40mm边框 = 170mm
    physical_height_mm: float = 257.0  # A4纸高度297mm - 40mm边框 = 257mm
) -> List[Dict[str, Any]]:
    """
    批量处理多个裁剪图像
    
    Args:
        crops: 裁剪图像列表
        min_area_threshold: 最小面积阈值（像素）
        physical_width_mm: A4纸宽度减去外框后的物理宽度（毫米，210-40=170mm）
        physical_height_mm: A4纸高度减去外框后的物理高度（毫米，297-40=257mm）
    
    Returns:
        每个crop的检测结果列表
    """
    results = []
    
    for i, crop in enumerate(crops):
        print(f"处理第 {i+1}/{len(crops)} 个裁剪图像...")
        result = process_crop_for_minimum_square(
            crop, 
            min_area_threshold=min_area_threshold,
            physical_width_mm=physical_width_mm,
            physical_height_mm=physical_height_mm
        )
        result["crop_index"] = i
        results.append(result)
        
        if result["success"]:
            print(f"  ✓ 最短边长: {result['shortest_edge_length_px']:.2f}px = {result['shortest_edge_length_mm']:.2f}mm")
        else:
            print(f"  ✗ 处理失败: {result['error']}")
    
    return results


def find_global_minimum_square(results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    从所有处理结果中找到全局最小的正方形
    
    Args:
        results: 所有crop的处理结果
    
    Returns:
        全局最小正方形的信息，如果没有则返回None
    """
    successful_results = [r for r in results if r["success"]]
    
    if not successful_results:
        return None
    
    # 按像素长度排序，找到最小的
    min_result = min(successful_results, key=lambda x: x["shortest_edge_length_px"])
    min_result["is_global_minimum"] = True
    
    return min_result


def _convert_detection_result_to_legacy_format(result: Dict[str, Any], crop_center: Tuple[float, float] = (0, 0)) -> Optional[Dict[str, Any]]:
    """
    将新的检测结果转换为与原 min_area_box 兼容的格式
    
    Args:
        result: process_crop_for_minimum_square 的返回结果
        crop_center: crop在原图中的中心坐标偏移
    
    Returns:
        兼容格式的正方形信息
    """
    if not result["success"]:
        return None
    
    # 计算中心点（使用最短边的中点）
    start_pt = result["start_point"]
    end_pt = result["end_point"]
    center_x = (start_pt[0] + end_pt[0]) / 2 + crop_center[0]
    center_y = (start_pt[1] + end_pt[1]) / 2 + crop_center[1]
    
    # 根据最短边长估算正方形信息
    side_length_px = result["shortest_edge_length_px"]
    side_length_mm = result["shortest_edge_length_mm"]
    area_px = side_length_px ** 2
    
    # 创建一个虚拟的box（基于中心点和边长）
    half_side = side_length_px / 2
    box = np.array([
        [center_x - half_side, center_y - half_side],
        [center_x + half_side, center_y - half_side],
        [center_x + half_side, center_y + half_side],
        [center_x - half_side, center_y + half_side]
    ], dtype=np.int32)
    
    return {
        "center": (center_x, center_y),
        "area": area_px,
        "side_length": side_length_px,
        "side_length_mm": side_length_mm,
        "aspect_ratio": 1.0,  # 假设是正方形
        "type": "minimum_square_detected",
        "box": box,
        "is_minimum": True
    }


def min_area_box_compatible(
    crops: List[np.ndarray], 
    stats: Dict[str, Any], 
    img: np.ndarray,
    min_area_threshold: int = 500,
    physical_width_mm: float = 170.0,
    physical_height_mm: float = 257.0
) -> None:
    """
    与原 min_area_box 函数完全兼容的入口函数
    
    使用新的最小正方形检测算法，但保持与原函数相同的行为：
    - 修改全局 min_square_images 列表
    - 在原图上绘制检测结果
    - 更新 stats 字典
    
    Args:
        crops: 裁剪图像列表
        stats: 统计信息字典（会被修改）
        img: 原始图像（会被修改，用于绘制）
        min_area_threshold: 最小面积阈值
        physical_width_mm: A4纸物理宽度（毫米）
        physical_height_mm: A4纸物理高度（毫米）
    """
    global min_square_images
    min_square_images = []  # 重置列表
    
    # 处理所有crops
    all_results = process_multiple_crops(
        crops, 
        min_area_threshold=min_area_threshold,
        physical_width_mm=physical_width_mm,
        physical_height_mm=physical_height_mm
    )
    
    # 找到全局最小正方形
    global_min_result = find_global_minimum_square(all_results)
    
    # 保存每个crop的处理图像到全局变量
    for result in all_results:
        if result["success"]:
            min_square_images.append(result["annotated_image"])
        else:
            # 如果处理失败，保存原crop
            crop_idx = result.get("crop_index", 0)
            if crop_idx < len(crops):
                min_square_images.append(crops[crop_idx].copy())
            else:
                min_square_images.append(np.zeros((100, 100, 3), dtype=np.uint8))
    
    # 在原图上绘制全局最小正方形（如果找到的话）
    if global_min_result and global_min_result["success"]:
        # 转换为兼容格式
        legacy_square = _convert_detection_result_to_legacy_format(global_min_result)
        
        if legacy_square:
            # 绘制最小正方形（模拟原函数的绘制风格）
            _draw_minimum_square_compatible(img, legacy_square)
            
            # 更新统计信息（与原格式完全兼容）
            stats["minimum_black_square"] = {
                "found": True,
                "center": [float(legacy_square["center"][0]), float(legacy_square["center"][1])],
                "area": float(legacy_square["area"]),
                "side_length": float(legacy_square["side_length"]),
                "side_length_mm": float(legacy_square["side_length_mm"]),
                "aspect_ratio": float(legacy_square["aspect_ratio"]),
                "type": legacy_square["type"],
            }
            
            # 兼容字段
            stats["black_squares"] = [
                {
                    "center": [float(legacy_square["center"][0]), float(legacy_square["center"][1])],
                    "area": float(legacy_square["area"]),
                    "side_length": float(legacy_square["side_length"]),
                    "side_length_mm": float(legacy_square["side_length_mm"]),
                    "aspect_ratio": float(legacy_square["aspect_ratio"]),
                    "is_minimum": True,
                }
            ]
        else:
            stats["minimum_black_square"] = {"found": False}
            stats["black_squares"] = []
    else:
        stats["minimum_black_square"] = {"found": False}
        stats["black_squares"] = []


def _draw_minimum_square_compatible(img: np.ndarray, min_square: Dict[str, Any]) -> None:
    """
    以与原 draw_minimum_square 兼容的方式绘制最小正方形
    """
    if min_square is None:
        return

    center = min_square["center"]
    box = min_square.get("box")
    
    if box is not None:
        # 绘制最小黑色正方形边框（青色，更粗的线条）
        cv2.drawContours(img, [box], -1, (255, 255, 0), 4)
    
    # 绘制中心点
    cv2.circle(img, (int(center[0]), int(center[1])), 12, (0, 255, 255), -1)

    # 添加最小正方形标记
    side_length = min_square.get("side_length", 0)
    side_length_mm = min_square.get("side_length_mm", 0)
    
    cv2.putText(
        img,
        f"MIN SQ:{int(side_length)}px",
        (int(center[0]) - 60, int(center[1]) - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )

    # 添加毫米信息
    cv2.putText(
        img,
        f"{side_length_mm:.1f}mm",
        (int(center[0]) - 30, int(center[1]) + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        2,
    )


# 使用示例
if __name__ == "__main__":
    # 这里是测试代码，实际使用时可以删除
    
    print("最小正方形检测器已准备就绪")
    print("\n=== 新API使用方法 ===")
    print("使用 process_crop_for_minimum_square() 处理单个裁剪图像")
    print("使用 process_multiple_crops() 批量处理多个裁剪图像")
    print("使用 find_global_minimum_square() 找到全局最小正方形")
    
    print("\n=== 兼容API使用方法 ===")
    print("使用 min_area_box_compatible() 替换原来的 min_area_box()")
    print("完全兼容原函数的接口和行为:")
    print("  - 接收相同的参数: crops, stats, img")
    print("  - 修改全局 min_square_images 列表")
    print("  - 在原图上绘制检测结果")
    print("  - 更新 stats 字典（包含毫米单位）")
    
    # 兼容性使用示例
    print("\n=== 替换示例 ===")
    print("原代码:")
    print("  min_area_box(crops, stats, img)")
    print("新代码:")
    print("  min_area_box_compatible(crops, stats, img)")
    print("  # 可选参数:")
    print("  # min_area_box_compatible(crops, stats, img, min_area_threshold=500)")
    
    # 单个图像处理示例（注释掉的代码保持不变）
    # if os.path.exists("test_crop.jpg"):
    #     test_img = cv2.imread("test_crop.jpg")
    #     result = process_crop_for_minimum_square(test_img, min_area_threshold=60)
    #     
    #     if result["success"]:
    #         print(f"检测成功！最短边长: {result['shortest_edge_length_mm']:.2f}mm")
    #         cv2.imwrite("result_annotated.jpg", result["annotated_image"])
    #     else:
    #         print(f"检测失败: {result['error']}")
