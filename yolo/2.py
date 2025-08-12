import cv2
import numpy as np
import os

def extract_white_rectangles_simple(image_path):
    """
    简单版本：从图像中提取白色矩形区域
    """
    
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法读取图像 {image_path}")
        return
    
    print(f"原始图像尺寸: {img.shape}")
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 二值化 - 提取白色区域 (阈值220以上为白色)
    _, white_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    
    # 形态学操作 - 连接断开的区域，去除噪点
    kernel = np.ones((3, 3), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 查找连通域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(white_mask, connectivity=8)
    
    print(f"找到 {num_labels-1} 个白色区域")
    
    # 创建输出目录
    output_dir = "white_rectangles"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 处理每个连通域
    extracted_count = 0
    debug_img = img.copy()
    
    for i in range(1, num_labels):  # 跳过背景(标签0)
        # 获取边界框信息
        x, y, w, h, area = stats[i]
        
        # 过滤条件
        min_area = 500  # 最小面积
        max_area = img.shape[0] * img.shape[1] * 0.8  # 最大面积（不超过图像80%）
        
        if area < min_area or area > max_area:
            continue
        
        # 检查宽高比（过滤过于细长的区域）
        aspect_ratio = w / h
        if aspect_ratio < 0.1 or aspect_ratio > 10:
            continue
        
        # 扩展边界（包含可能的边框）
        margin = 5
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img.shape[1], x + w + margin)
        y2 = min(img.shape[0], y + h + margin)
        
        # 提取矩形区域
        rect_img = img[y1:y2, x1:x2]
        
        # 保存提取的矩形
        filename = f"white_rect_{extracted_count+1:03d}_area{area}_pos{x}x{y}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, rect_img)
        
        # 在调试图像上标记
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(debug_img, f"{extracted_count+1}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        extracted_count += 1
        print(f"提取矩形 {extracted_count}: {filename}")
        print(f"  位置: ({x}, {y}), 尺寸: {w}x{h}, 面积: {area}")
    
    # 保存调试图像
    cv2.imwrite(os.path.join(output_dir, "debug_marked.png"), debug_img)
    cv2.imwrite(os.path.join(output_dir, "white_mask.png"), white_mask)
    
    print(f"\n完成！共提取了 {extracted_count} 个白色矩形")
    print(f"结果保存在: {output_dir}/")
    print("- debug_marked.png: 标记了检测区域的原图")
    print("- white_mask.png: 白色区域的二值化mask")
    print("- white_rect_*.png: 提取的各个矩形区域")

if __name__ == "__main__":
    # 获取用户输入的图像路径
    image_path = input("请输入图像路径: ").strip()
    
    # 去除可能的引号
    if image_path.startswith('"') and image_path.endswith('"'):
        image_path = image_path[1:-1]
    if image_path.startswith("'") and image_path.endswith("'"):
        image_path = image_path[1:-1]
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 图像文件不存在: {image_path}")
        exit(1)
    
    # 执行提取
    extract_white_rectangles_simple(image_path)