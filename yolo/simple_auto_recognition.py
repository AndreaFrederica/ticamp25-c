import cv2
import numpy as np
import requests
import json
import base64
import os

def simple_auto_recognition(image_path, yolo_api_url="http://localhost:8005"):
    """
    简化版自动识别：图像分割 + YOLO识别
    
    参数:
    - image_path: 输入图像路径
    - yolo_api_url: YOLO API地址
    
    返回:
    - 识别结果列表，包含文字和坐标信息
    """
    
    # 1. 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法读取图像 {image_path}")
        return []
    
    print(f"处理图像: {image_path} (尺寸: {img.shape})")
    
    # 2. 提取白色矩形
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # 形态学操作
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"找到 {len(contours)} 个轮廓")
    
    # 3. 处理每个轮廓
    all_results = []
    debug_img = img.copy()
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < 300:  # 最小面积过滤
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
        
        if rect_img.shape[0] < 20 or rect_img.shape[1] < 20:
            continue
        
        print(f"处理矩形 {i}: 位置({x1},{y1})-({x2},{y2}), 尺寸{w}x{h}")
        
        # 4. 发送到YOLO API
        try:
            # 转换为base64
            _, buffer = cv2.imencode('.jpg', rect_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # 发送请求
            response = requests.post(
                f"{yolo_api_url}/predict_base64",
                json={"image": img_base64},
                timeout=10
            )
            
            if response.status_code == 200:
                yolo_result = response.json()
                
                if yolo_result.get('success') and yolo_result.get('count', 0) > 0:
                    print(f"  YOLO识别到 {yolo_result['count']} 个数字")
                    
                    # 5. 转换坐标
                    for result in yolo_result['results']:
                        bbox = result['bbox']
                        
                        # 计算在矩形区域中的中心点
                        center_x_in_rect = (bbox['x1'] + bbox['x2']) / 2
                        center_y_in_rect = (bbox['y1'] + bbox['y2']) / 2
                        
                        # 转换到原图坐标
                        center_x_original = x1 + center_x_in_rect
                        center_y_original = y1 + center_y_in_rect
                        
                        recognition_result = {
                            'text': result['label'],
                            'confidence': result['confidence'],
                            'center_x': float(center_x_original),
                            'center_y': float(center_y_original),
                            'rectangle_id': i,
                            'rectangle_bbox': (x1, y1, x2, y2)
                        }
                        
                        all_results.append(recognition_result)
                        
                        # 在调试图上标记
                        center_x_int = int(center_x_original)
                        center_y_int = int(center_y_original)
                        cv2.circle(debug_img, (center_x_int, center_y_int), 5, (0, 255, 0), -1)
                        
                        label = f"{result['label']}:{result['confidence']:.2f}"
                        cv2.putText(debug_img, label, (center_x_int + 10, center_y_int - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    print(f"  矩形 {i}: 未识别到数字")
            else:
                print(f"  矩形 {i}: API请求失败 {response.status_code}")
                
        except Exception as e:
            print(f"  矩形 {i}: 处理失败 - {e}")
        
        # 绘制矩形边界
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(debug_img, f"R{i}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # 6. 保存调试图像
    if all_results:
        debug_path = "simple_recognition_debug.png"
        cv2.imwrite(debug_path, debug_img)
        print(f"调试图像已保存: {debug_path}")
        
        # 保存结果
        json_path = "simple_recognition_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存: {json_path}")
    
    return all_results

def test_api_connection(api_url="http://localhost:8005"):
    """测试API连接"""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

if __name__ == "__main__":
    print("=== 简化版自动图像识别 ===")
    
    # 检查API
    if not test_api_connection():
        print("❌ YOLO API不可用，请先启动服务:")
        print("python yolo_api.py")
        exit(1)
    else:
        print("✅ YOLO API连接正常")
    
    # 获取图像路径
    image_path = input("请输入图像路径: ").strip().strip('"').strip("'")
    
    if not os.path.exists(image_path):
        print(f"错误: 文件不存在 {image_path}")
        exit(1)
    
    # 进行识别
    results = simple_auto_recognition(image_path)
    
    # 显示结果
    if results:
        print(f"\n=== 识别结果 (共{len(results)}个) ===")
        for i, result in enumerate(results, 1):
            print(f"{i}. 文字: {result['text']}")
            print(f"   置信度: {result['confidence']:.3f}")
            print(f"   中心坐标: ({result['center_x']:.1f}, {result['center_y']:.1f})")
            print(f"   来源矩形: {result['rectangle_id']} {result['rectangle_bbox']}")
            print()
    else:
        print("未识别到任何数字")
