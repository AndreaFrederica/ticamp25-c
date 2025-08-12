#!/usr/bin/env python3
"""
集成示例：如何在现有项目中替换原来的 min_area_box 函数

这个文件展示了如何使用新的 minimum_square_detector.py 中的
min_area_box_compatible() 函数来替换原来的 min_area_box() 函数。
"""

import cv2
import numpy as np
from minimum_square_detector import min_area_box_compatible, min_square_images


def example_integration_with_existing_code():
    """
    展示如何在现有代码中进行替换
    """
    print("=== 集成示例：替换 min_area_box 函数 ===\n")
    
    # 模拟现有代码中的数据结构
    # 假设你有一些裁剪好的图像
    crops = []
    stats = {}  # 原来的统计信息字典
    img = np.zeros((800, 600, 3), dtype=np.uint8)  # 模拟原图
    
    # 创建一些测试crops（实际使用时这些来自你的检测算法）
    for i in range(3):
        # 创建模拟的crop图像
        crop = np.ones((200, 150, 3), dtype=np.uint8) * 128
        # 添加一些模拟的黑色正方形
        cv2.rectangle(crop, (50, 50), (100, 100), (0, 0, 0), -1)
        cv2.rectangle(crop, (80, 80), (120, 120), (0, 0, 0), -1)
        crops.append(crop)
    
    print(f"准备处理 {len(crops)} 个crop图像...")
    
    # === 原来的代码（需要替换） ===
    # min_area_box(crops, stats, img)
    
    # === 新的代码（替换后） ===
    min_area_box_compatible(crops, stats, img)
    
    print("✓ 处理完成！")
    print(f"✓ 全局 min_square_images 列表包含 {len(min_square_images)} 个图像")
    print("✓ stats 字典已更新")
    
    # 检查结果
    print("\n=== 检测结果 ===")
    if stats.get("minimum_black_square", {}).get("found", False):
        min_sq = stats["minimum_black_square"]
        print("✓ 找到最小正方形:")
        print(f"  - 中心位置: ({min_sq['center'][0]:.1f}, {min_sq['center'][1]:.1f})")
        print(f"  - 面积: {min_sq['area']:.1f} 像素²")
        print(f"  - 边长: {min_sq['side_length']:.1f} 像素")
        if 'side_length_mm' in min_sq:
            print(f"  - 边长: {min_sq['side_length_mm']:.2f} 毫米")
        print(f"  - 长宽比: {min_sq['aspect_ratio']:.2f}")
        print(f"  - 类型: {min_sq['type']}")
    else:
        print("✗ 未找到最小正方形")
    
    # 检查兼容性字段
    if "black_squares" in stats:
        print(f"✓ 兼容性字段 'black_squares' 包含 {len(stats['black_squares'])} 个正方形")
    
    print("\n=== 集成说明 ===")
    print("1. 只需要将以下代码:")
    print("   min_area_box(crops, stats, img)")
    print("   替换为:")
    print("   min_area_box_compatible(crops, stats, img)")
    
    print("\n2. 新函数完全兼容原函数:")
    print("   ✓ 接受相同的参数")
    print("   ✓ 修改相同的全局变量 (min_square_images)")
    print("   ✓ 更新相同的 stats 字典结构")
    print("   ✓ 在原图上绘制检测结果")
    
    print("\n3. 额外的改进:")
    print("   ✓ 支持毫米单位的物理测量")
    print("   ✓ 更精确的角点检测算法")
    print("   ✓ 可配置的面积阈值")
    print("   ✓ 完整的类型注解")
    
    return stats, min_square_images


def show_parameter_options():
    """
    展示可选参数的使用方法
    """
    print("\n=== 可选参数配置 ===")
    
    # 基本使用（默认参数）
    print("1. 基本使用（默认参数）:")
    print("   min_area_box_compatible(crops, stats, img)")
    
    # 自定义面积阈值
    print("\n2. 自定义面积阈值:")
    print("   min_area_box_compatible(crops, stats, img, min_area_threshold=300)")
    
    # 自定义物理尺寸（如果你的A4纸边框不是40mm）
    print("\n3. 自定义物理尺寸:")
    print("   min_area_box_compatible(")
    print("       crops, stats, img,")
    print("       physical_width_mm=160.0,   # 如果边框是50mm")
    print("       physical_height_mm=247.0   # 297-50=247")
    print("   )")
    
    # 完整参数示例
    print("\n4. 完整参数示例:")
    print("   min_area_box_compatible(")
    print("       crops=crops,")
    print("       stats=stats,")
    print("       img=img,")
    print("       min_area_threshold=400,")
    print("       physical_width_mm=170.0,")
    print("       physical_height_mm=257.0")
    print("   )")


if __name__ == "__main__":
    # 运行集成示例
    example_integration_with_existing_code()
    
    # 显示参数选项
    show_parameter_options()
    
    print("\n=== 完成 ===")
    print("你现在可以在你的项目中使用 min_area_box_compatible() 函数了！")
