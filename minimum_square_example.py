"""
最小正方形检测器使用示例
展示如何在主程序中集成 minimum_square_detector.py
"""

import cv2
from minimum_square_detector import (
    process_crop_for_minimum_square,
    process_multiple_crops,
    find_global_minimum_square
)

def example_usage_with_crops(crops):
    """
    示例：如何使用最小正方形检测器处理已有的crops
    
    Args:
        crops: List[np.ndarray] - 已经裁剪好的图像列表
    """
    print(f"开始处理 {len(crops)} 个裁剪图像...")
    
    # 方法1: 批量处理所有crops
    results = process_multiple_crops(
        crops=crops,
        min_area_threshold=60,  # 最小面积阈值
        physical_width_mm=170.0,  # A4纸宽度210mm - 40mm边框 = 170mm
        physical_height_mm=257.0  # A4纸高度297mm - 40mm边框 = 257mm
    )
    
    # 打印每个crop的结果
    for result in results:
        crop_idx = result["crop_index"]
        if result["success"]:
            print(f"Crop {crop_idx}: 最短边长 {result['shortest_edge_length_mm']:.2f}mm")
        else:
            print(f"Crop {crop_idx}: 处理失败 - {result['error']}")
    
    # 找到全局最小的正方形
    global_min = find_global_minimum_square(results)
    if global_min:
        print(f"\n全局最小正方形在 Crop {global_min['crop_index']}")
        print(f"最短边长: {global_min['shortest_edge_length_mm']:.2f}mm")
        
        # 保存带标注的图像
        cv2.imwrite(f"global_minimum_crop_{global_min['crop_index']}.jpg", 
                   global_min["annotated_image"])
    else:
        print("\n未找到任何有效的正方形")
    
    return results, global_min


def example_single_crop_processing(crop_image):
    """
    示例：处理单个crop
    
    Args:
        crop_image: np.ndarray - 单个裁剪图像
    """
    result = process_crop_for_minimum_square(
        crop_img=crop_image,
        min_area_threshold=60,
        physical_width_mm=170.0,  # A4纸宽度210mm - 40mm边框 = 170mm
        physical_height_mm=257.0  # A4纸高度297mm - 40mm边框 = 257mm
    )
    
    if result["success"]:
        print("检测成功！")
        print(f"最短边长: {result['shortest_edge_length_px']:.2f}px")
        print(f"物理长度: {result['shortest_edge_length_mm']:.2f}mm")
        print(f"连通域ID: {result['component_id']}")
        print(f"起点: {result['start_point']}")
        print(f"终点: {result['end_point']}")
        print(f"所有有效边数: {len(result['all_valid_edges'])}")
        
        # 保存标注图像
        cv2.imwrite("single_crop_result.jpg", result["annotated_image"])
        
        return result
    else:
        print(f"检测失败: {result['error']}")
        return None


def integration_example_with_main_detector():
    """
    示例：如何与主检测程序集成
    这个函数展示了如何在你的 simple_detector_web.py 中使用
    """
    # 假设这是从你的主程序获取的crops
    # 在你的实际代码中，这些crops来自 create_a4_crops_from_contours() 函数
    
    print("=== 集成示例 ===")
    print("在你的 simple_detector_web.py 中，可以这样使用：")
    print("""
# 在 detect_and_annotate 函数中添加：
from minimum_square_detector import process_multiple_crops, find_global_minimum_square

def detect_and_annotate(img):
    # ...existing code...
    
    # 生成A4比例的裁剪图像
    crops = create_a4_crops_from_contours(img, hollow_rectangles)
    
    # 使用新的最小正方形检测器
    min_square_results = process_multiple_crops(crops, min_area_threshold=60)
    global_min_square = find_global_minimum_square(min_square_results)
    
    # 更新统计信息
    if global_min_square:
        stats["minimum_black_square"] = {
            "found": True,
            "crop_index": global_min_square["crop_index"],
            "side_length": global_min_square["shortest_edge_length_px"],
            "side_length_mm": global_min_square["shortest_edge_length_mm"],
            "center": [(global_min_square["start_point"][0] + global_min_square["end_point"][0]) // 2,
                      (global_min_square["start_point"][1] + global_min_square["end_point"][1]) // 2],
            "area": global_min_square["shortest_edge_length_px"] ** 2,
            "aspect_ratio": 1.0,
            "type": "minimum_square"
        }
    else:
        stats["minimum_black_square"] = {"found": False}
    
    # 保存结果图像到全局变量供API访问
    global min_square_images
    min_square_images = [result["annotated_image"] for result in min_square_results if result["success"]]
    
    # ...rest of existing code...
    return annotated_frame, mask, stats
""")


if __name__ == "__main__":
    # 模拟一些测试数据
    print("最小正方形检测器使用示例")
    print("=" * 50)
    
    # 如果有测试图像，可以取消注释下面的代码
    # test_crops = []
    # for i in range(1, 4):  # 假设有3个测试图像
    #     img_path = f"test_crop_{i}.jpg"
    #     if os.path.exists(img_path):
    #         test_crops.append(cv2.imread(img_path))
    # 
    # if test_crops:
    #     results, global_min = example_usage_with_crops(test_crops)
    # else:
    #     print("没有找到测试图像文件")
    
    integration_example_with_main_detector()
