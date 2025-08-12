import cv2
import numpy as np
import requests
import json
import base64
import os
from typing import List, Dict, Tuple, Any
import tempfile

class AutoSegmentYOLO:
    """
    自动图像分割和YOLO识别器
    自动从图像中提取白色矩形区域，并逐个发送给YOLO API进行数字识别
    """
    
    def __init__(self, yolo_api_url="http://127.0.0.1:8005"):
        """
        初始化
        
        参数:
        - yolo_api_url: YOLO API服务地址
        """
        self.yolo_api_url = yolo_api_url
        self.results = []
        
    def extract_white_rectangles(self, img, min_area=300):
        """
        从图像中提取白色矩形区域
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 二值化
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangles = []
        
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
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.array(box, dtype=np.int32)
                approx = box.reshape(-1, 1, 2)
            
            # 检查面积比例
            rect_area = cv2.contourArea(approx)
            if rect_area == 0:
                continue
                
            area_ratio = area / rect_area
            if area_ratio < 0.5:
                continue
            
            # 提取并矫正矩形
            corrected_img, bbox_in_original = self.extract_and_correct_rectangle(
                img, approx.reshape(4, 2))
            
            if corrected_img is not None:
                h, w = corrected_img.shape[:2]
                if w >= 20 and h >= 20:
                    aspect_ratio = w / h
                    if 0.1 <= aspect_ratio <= 10:
                        rectangles.append({
                            'id': i,
                            'image': corrected_img,
                            'bbox_in_original': bbox_in_original,
                            'contour_points': approx.reshape(4, 2),
                            'area': area,
                            'corrected_size': (w, h)
                        })
        
        return rectangles
    
    def extract_and_correct_rectangle(self, img, points):
        """
        从四个顶点提取并矫正矩形
        """
        try:
            points = np.array(points, dtype=np.float32)
            if points.shape != (4, 2):
                return None, None
            
            # 对顶点进行排序
            ordered_points = self.order_points(points)
            
            # 计算矫正后的矩形尺寸
            width, height = self.calculate_corrected_size(ordered_points)
            
            if width < 20 or height < 20:
                return None, None
            
            # 定义目标矩形的四个顶点
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
            
            # 计算原图中的边界框（用于坐标转换）
            x_coords = ordered_points[:, 0]
            y_coords = ordered_points[:, 1]
            bbox_in_original = {
                'x_min': float(np.min(x_coords)),
                'y_min': float(np.min(y_coords)),
                'x_max': float(np.max(x_coords)),
                'y_max': float(np.max(y_coords)),
                'transform_matrix': transform_matrix,
                'original_points': ordered_points,
                'corrected_size': (int(width), int(height))
            }
            
            return corrected_img, bbox_in_original
            
        except Exception as e:
            print(f"矫正失败: {e}")
            return None, None
    
    def order_points(self, pts):
        """对四个点进行排序：左上、右上、右下、左下"""
        center = np.mean(pts, axis=0)
        
        def angle_from_center(point):
            return np.arctan2(point[1] - center[1], point[0] - center[0])
        
        sorted_pts = sorted(pts, key=angle_from_center)
        
        dists_from_origin = [pt[0] + pt[1] for pt in sorted_pts]
        start_idx = np.argmin(dists_from_origin)
        
        ordered = []
        for i in range(4):
            ordered.append(sorted_pts[(start_idx + i) % 4])
        
        return np.array(ordered, dtype=np.float32)
    
    def calculate_corrected_size(self, ordered_points):
        """根据四个顶点计算矫正后的矩形尺寸"""
        top_width = np.linalg.norm(ordered_points[1] - ordered_points[0])
        bottom_width = np.linalg.norm(ordered_points[2] - ordered_points[3])
        left_height = np.linalg.norm(ordered_points[3] - ordered_points[0])
        right_height = np.linalg.norm(ordered_points[2] - ordered_points[1])
        
        width = max(top_width, bottom_width)
        height = max(left_height, right_height)
        
        return int(width), int(height)
    
    def image_to_base64(self, img):
        """将OpenCV图像转换为base64编码"""
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64
    
    def send_to_yolo(self, img, confidence_threshold=0.5, nms_threshold=0.4):
        """
        将图像发送到YOLO API进行识别
        """
        try:
            # 将图像转换为base64
            img_base64 = self.image_to_base64(img)
            
            # 准备请求数据
            data = {"image": img_base64}
            
            # 发送POST请求
            response = requests.post(
                f"{self.yolo_api_url}/predict_base64",
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"YOLO API错误: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"发送到YOLO API失败: {e}")
            return None
    
    def convert_coordinates_to_original(self, yolo_results, bbox_info):
        """
        将YOLO识别结果的坐标转换回原图坐标系
        """
        if not yolo_results or not yolo_results.get('success'):
            return []
        
        converted_results = []
        transform_matrix = bbox_info['transform_matrix']
        original_points = bbox_info['original_points']
        corrected_size = bbox_info['corrected_size']
        
        # 计算逆变换矩阵
        try:
            inv_transform_matrix = cv2.invert(transform_matrix)[1]
        except:
            print("无法计算逆变换矩阵")
            return []
        
        for result in yolo_results.get('results', []):
            bbox = result['bbox']
            
            # 计算矫正图像中的中心点
            center_x_corrected = (bbox['x1'] + bbox['x2']) / 2
            center_y_corrected = (bbox['y1'] + bbox['y2']) / 2
            
            # 将矫正图像中的坐标转换回原图坐标
            corrected_point = np.array([[center_x_corrected, center_y_corrected]], dtype=np.float32)
            corrected_point = corrected_point.reshape(-1, 1, 2)
            
            # 应用逆变换
            original_point = cv2.perspectiveTransform(corrected_point, inv_transform_matrix)
            original_x, original_y = original_point[0][0]
            
            converted_results.append({
                'text': result['label'],
                'confidence': result['confidence'],
                'center_x': float(original_x),
                'center_y': float(original_y),
                'bbox_in_corrected': bbox,
                'bbox_in_original': {
                    'region_bounds': bbox_info
                }
            })
        
        return converted_results
    
    def process_image(self, image_path, confidence_threshold=0.5, nms_threshold=0.4, 
                     min_area=300, save_debug=True):
        """
        处理图像：自动分割 → YOLO识别 → 坐标转换
        
        参数:
        - image_path: 输入图像路径
        - confidence_threshold: YOLO置信度阈值
        - nms_threshold: YOLO NMS阈值
        - min_area: 最小矩形面积
        - save_debug: 是否保存调试图像
        
        返回:
        - 包含所有识别结果的列表，每个元素包含文字和在原图中的坐标
        """
        
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"错误: 无法读取图像 {image_path}")
            return []
        
        print(f"处理图像: {image_path}")
        print(f"原始图像尺寸: {img.shape}")
        
        # 1. 提取白色矩形区域
        print("正在提取白色矩形区域...")
        rectangles = self.extract_white_rectangles(img, min_area)
        print(f"提取到 {len(rectangles)} 个矩形区域")
        
        if len(rectangles) == 0:
            print("未找到符合条件的矩形区域")
            return []
        
        # 2. 逐个发送给YOLO进行识别
        all_results = []
        debug_img = img.copy()
        
        for i, rect in enumerate(rectangles):
            print(f"\n处理第 {i+1}/{len(rectangles)} 个矩形...")
            
            # 发送到YOLO API
            yolo_result = self.send_to_yolo(
                rect['image'], confidence_threshold, nms_threshold)
            
            if yolo_result and yolo_result.get('success'):
                print(f"YOLO识别到 {yolo_result['count']} 个数字")
                
                # 转换坐标到原图
                converted_results = self.convert_coordinates_to_original(
                    yolo_result, rect['bbox_in_original'])
                
                # 添加矩形区域信息
                for result in converted_results:
                    result['rectangle_id'] = rect['id']
                    result['rectangle_area'] = rect['area']
                    result['corrected_size'] = rect['corrected_size']
                
                all_results.extend(converted_results)
                
                # 在调试图像上标记结果
                if save_debug:
                    self.draw_results_on_image(debug_img, converted_results, rect)
            else:
                print(f"第 {i+1} 个矩形的YOLO识别失败")
        
        # 3. 保存调试图像
        if save_debug and all_results:
            debug_dir = "debug_auto_segment"
            os.makedirs(debug_dir, exist_ok=True)
            
            # 保存标记了结果的原图
            debug_path = os.path.join(debug_dir, "final_results.png")
            cv2.imwrite(debug_path, debug_img)
            print(f"调试图像已保存: {debug_path}")
            
            # 保存各个提取的矩形
            for i, rect in enumerate(rectangles):
                rect_path = os.path.join(debug_dir, f"extracted_rect_{i:03d}.png")
                cv2.imwrite(rect_path, rect['image'])
        
        print(f"\n=== 最终结果 ===")
        print(f"总共识别到 {len(all_results)} 个数字")
        
        return all_results
    
    def draw_results_on_image(self, img, results, rect_info):
        """在调试图像上绘制识别结果"""
        for result in results:
            center_x = int(result['center_x'])
            center_y = int(result['center_y'])
            text = result['text']
            confidence = result['confidence']
            
            # 绘制中心点
            cv2.circle(img, (center_x, center_y), 5, (0, 255, 0), -1)
            
            # 绘制文字和置信度
            label = f"{text}:{confidence:.2f}"
            cv2.putText(img, label, (center_x + 10, center_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 绘制矩形轮廓
        points = rect_info['contour_points'].astype(np.int32)
        cv2.polylines(img, [points], True, (255, 0, 0), 2)
    
    def save_results_to_json(self, results, output_path):
        """将结果保存为JSON文件"""
        # 清理结果中的NumPy数组，只保留基本数据类型
        clean_results = []
        for result in results:
            clean_result = {
                'text': result['text'],
                'confidence': float(result['confidence']),
                'center_x': float(result['center_x']),
                'center_y': float(result['center_y']),
                'rectangle_id': int(result['rectangle_id']),
                'rectangle_area': float(result['rectangle_area']),
                'corrected_size': [int(result['corrected_size'][0]), int(result['corrected_size'][1])],
                'bbox_in_corrected': {
                    'x1': float(result['bbox_in_corrected']['x1']),
                    'y1': float(result['bbox_in_corrected']['y1']),
                    'x2': float(result['bbox_in_corrected']['x2']),
                    'y2': float(result['bbox_in_corrected']['y2'])
                }
            }
            clean_results.append(clean_result)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {output_path}")

def main():
    """主函数"""
    
    # 初始化处理器
    processor = AutoSegmentYOLO(yolo_api_url="http://localhost:8005")
    
    # 获取输入图像路径
    image_path = input("请输入图像路径: ").strip()
    if image_path.startswith('"') and image_path.endswith('"'):
        image_path = image_path[1:-1]
    if image_path.startswith("'") and image_path.endswith("'"):
        image_path = image_path[1:-1]
    
    if not os.path.exists(image_path):
        print(f"错误: 图像文件不存在: {image_path}")
        return
    
    # 设置参数
    print("\n=== 参数设置 ===")
    confidence = float(input("YOLO置信度阈值 (默认0.5): ") or "0.5")
    nms = float(input("YOLO NMS阈值 (默认0.4): ") or "0.4")
    min_area = int(input("最小矩形面积 (默认300): ") or "300")
    
    # 处理图像
    print("\n=== 开始处理 ===")
    results = processor.process_image(
        image_path, 
        confidence_threshold=confidence,
        nms_threshold=nms,
        min_area=min_area,
        save_debug=True
    )
    
    # 显示结果
    if results:
        print(f"\n=== 识别结果 ===")
        for i, result in enumerate(results):
            print(f"数字 {i+1}:")
            print(f"  文字: {result['text']}")
            print(f"  置信度: {result['confidence']:.3f}")
            print(f"  中心坐标: ({result['center_x']:.1f}, {result['center_y']:.1f})")
            print(f"  来源矩形: ID {result['rectangle_id']}")
        
        # 保存结果到JSON
        output_json = "recognition_results.json"
        processor.save_results_to_json(results, output_json)
    else:
        print("未识别到任何数字")

def test_with_api():
    """测试API连通性"""
    try:
        response = requests.get("http://localhost:8005/health", timeout=5)
        if response.status_code == 200:
            result = response.json()
            if result.get("model_loaded"):
                print("✅ YOLO API连接正常，模型已加载")
                return True
            else:
                print("❌ YOLO API连接正常但模型未加载")
                return False
        else:
            print(f"❌ YOLO API响应异常: {response.status_code}")
            print(f"响应内容: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 无法连接到YOLO API: {e}")
        print("请确保服务运行在 http://localhost:8005")
        return False

if __name__ == "__main__":
    print("=== 自动图像分割和YOLO识别器 ===")
    
    # 检查API连接
    if not test_with_api():
        print("\n请先启动YOLO API服务:")
        print("cd /home/pi/Desktop/222/yolo")
        print("python yolo_api.py")
        exit(1)
    
    # 运行主程序
    main()
