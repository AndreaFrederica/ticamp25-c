import asyncio
from copy import copy
import cv2
from fastapi.staticfiles import StaticFiles
import numpy as np
import threading
import time
import math
import json
import base64
import os
import io
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union, Generator
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import easyocr
import requests
import concurrent.futures

# 导入自动切分YOLO类
import sys
sys.path.append('./yolo')
from auto_segment_yolo import AutoSegmentYOLO

# 导入YOLO相关模块
import onnxruntime as ort

from ina226 import INA226


vis: Optional[np.ndarray] = None
mask: Optional[np.ndarray] = None
watershed_img: Optional[np.ndarray] = None
min_square_images: List[np.ndarray] = []  # 存储每个crop的最小正方形检测图像

# 全局停止标志
shutdown_event: threading.Event = threading.Event()
capture_thread: Optional[threading.Thread] = None


app: FastAPI = FastAPI()
ina: INA226

print("=== INA226 Current Sensor Demo Program ===\n")

try:
    # 初始化INA226（假设使用0.1Ω分流电阻，最大电流3.2A）
    ina = INA226(i2c_bus=1, address=0x40, shunt_ohms=0.01, max_expected_amps=5.0)

    print("Starting continuous measurement...")
    print("Press Ctrl+C to stop measurement\n")

    measurements = ina.get_all_measurements()

    # 显示测量结果
    print(f"Shunt voltage: {measurements['shunt_voltage_mv']:8.3f} mV")
    print(f"Bus voltage:   {measurements['bus_voltage_v']:8.3f} V")
    print(f"Current:       {measurements['current_a'] * 1000:8.3f} mA")
    print(f"Power:         {measurements['power_w'] * 1000:8.3f} mW")
    print("-" * 40)


except Exception as e:
    print(f"Failed to initialize INA226: {e}")
    ina = None


app.add_middleware(
    CORSMiddleware,
    allow_origins=[],  # 留空
    allow_credentials=True,  # 允许携带 Cookie
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置文件路径
CONFIG_FILE: str = "config.json"
cap_o = True


OCR_LANGS = ["la"]  # 若需中文改 ['ch_sim', 'en']
OCR_ALLOWLIST = "0123456789"  # 只识别数字；改 '' 为“不限制字符”
USE_GPU = False  # 树莓派/CPU 设 False

# 延迟初始化 EasyOCR.Reader（耗时操作）
_reader = None
_auto_segment_yolo = None  # 添加AutoSegmentYOLO实例
_local_yolo = None  # 添加本地YOLO实例
raw_crops: List[Any]

class LocalYOLO:
    """本地YOLO识别器，避免HTTP调用"""
    
    def __init__(self, model_path="./yolo/best.onnx"):
        self.net = None
        self.model_h = 320
        self.model_w = 320
        self.nl = 3
        self.na = 3
        self.stride = [8., 16., 32.]
        self.anchor_grid = None
        self.dic_labels = {
            0: '1',
            1: '2',
            2: '3',
            3: '4',
            4: '5',
            5: '6',
            6: '7',
            7: '8'
            }
        self.model_path = model_path
        self.load_model()
    
    def _make_grid(self, nx, ny):
        """创建网格"""
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)
    
    def cal_outputs(self, outs, nl, na, model_w, model_h, anchor_grid, stride):
        """计算输出坐标"""
        row_ind = 0
        grid = [np.zeros(1)] * nl
        for i in range(nl):
            h, w = int(model_w / stride[i]), int(model_h / stride[i])
            length = int(na * h * w)
            if grid[i].shape[2:4] != (h, w):
                grid[i] = self._make_grid(w, h)

            outs[row_ind:row_ind + length, 0:2] = (outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + np.tile(
                grid[i], (na, 1))) * int(stride[i])
            outs[row_ind:row_ind + length, 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * np.repeat(
                anchor_grid[i], h * w, axis=0)
            row_ind += length
        return outs
    
    def post_process_opencv(self, outputs, model_h, model_w, img_h, img_w, thred_nms, thred_cond):
        """后处理"""
        conf = outputs[:, 4].tolist()
        c_x = outputs[:, 0] / model_w * img_w
        c_y = outputs[:, 1] / model_h * img_h
        w = outputs[:, 2] / model_w * img_w
        h = outputs[:, 3] / model_h * img_h
        p_cls = outputs[:, 5:]
        if len(p_cls.shape) == 1:
            p_cls = np.expand_dims(p_cls, 1)
        cls_id = np.argmax(p_cls, axis=1)

        p_x1 = np.expand_dims(c_x - w / 2, -1)
        p_y1 = np.expand_dims(c_y - h / 2, -1)
        p_x2 = np.expand_dims(c_x + w / 2, -1)
        p_y2 = np.expand_dims(c_y + h / 2, -1)
        areas = np.concatenate((p_x1, p_y1, p_x2, p_y2), axis=-1)

        areas = areas.tolist()
        ids = cv2.dnn.NMSBoxes(areas, conf, thred_cond, thred_nms)
        if len(ids) > 0:
            return np.array(areas)[ids], np.array(conf)[ids], cls_id[ids]
        else:
            return [], [], []
    
    def load_model(self):
        """加载模型"""
        try:
            so = ort.SessionOptions()
            self.net = ort.InferenceSession(self.model_path, so)
            
            # 初始化anchor_grid
            anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
            self.anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(self.nl, -1, 2)
            
            print("✅ 本地YOLO模型加载成功")
            return True
        except Exception as e:
            print(f"❌ 本地YOLO模型加载失败: {e}")
            return False
    
    def predict(self, img, thred_nms=0.4, thred_cond=0.5):
        """图像推理"""
        if self.net is None:
            print("模型未加载")
            return {"success": False, "count": 0, "results": []}
        
        try:
            # 图像预处理
            img_processed = cv2.resize(img, [self.model_w, self.model_h], interpolation=cv2.INTER_AREA)
            img_processed = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)
            img_processed = img_processed.astype(np.float32) / 255.0
            blob = np.expand_dims(np.transpose(img_processed, (2, 0, 1)), axis=0)

            # 模型推理
            outs = self.net.run(None, {self.net.get_inputs()[0].name: blob})[0].squeeze(axis=0)

            # 输出坐标矫正
            outs = self.cal_outputs(outs, self.nl, self.na, self.model_w, self.model_h, self.anchor_grid, self.stride)

            # 检测框计算
            img_h, img_w, _ = np.shape(img)
            boxes, confs, ids = self.post_process_opencv(outs, self.model_h, self.model_w, img_h, img_w, thred_nms, thred_cond)

            # 格式化结果
            results = []
            for box, conf, id in zip(boxes, confs, ids):
                result = {
                    "label": self.dic_labels[id],
                    "confidence": float(conf),
                    "bbox": {
                        "x1": float(box[0]),
                        "y1": float(box[1]),
                        "x2": float(box[2]),
                        "y2": float(box[3])
                    }
                }
                results.append(result)

            return {
                "success": True,
                "count": len(results),
                "results": results,
                "image_shape": {
                    "height": img.shape[0],
                    "width": img.shape[1],
                    "channels": img.shape[2]
                }
            }
        except Exception as e:
            print(f"YOLO推理失败: {e}")
            return {"success": False, "count": 0, "results": [], "error": str(e)}


def _get_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(OCR_LANGS, gpu=USE_GPU)
    return _reader


def _get_local_yolo():
    """获取本地YOLO实例"""
    global _local_yolo
    if _local_yolo is None:
        _local_yolo = LocalYOLO()
    return _local_yolo


def _get_auto_segment_yolo():
    """获取AutoSegmentYOLO实例 - 现在使用本地YOLO"""
    global _auto_segment_yolo
    if _auto_segment_yolo is None:
        try:
            # 创建一个修改过的AutoSegmentYOLO，使用本地YOLO而不是HTTP
            _auto_segment_yolo = LocalAutoSegmentYOLO()
            print("✅ 本地AutoSegmentYOLO 初始化成功")
        except Exception as e:
            print(f"❌ 本地AutoSegmentYOLO 初始化失败: {e}")
            _auto_segment_yolo = None
    return _auto_segment_yolo


class LocalAutoSegmentEasyOCR:
    """本地自动图像分割和EasyOCR识别器，使用相同的预处理逻辑"""
    
    def __init__(self):
        self.results = []
        
    def extract_white_rectangles(self, img, min_area=300):
        """
        从图像中提取白色矩形区域 - 与YOLO版本完全相同的算法
        """
        # 检查图像通道数，如果已经是灰度图像就不需要转换
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif len(img.shape) == 3 and img.shape[2] == 1:
            gray = img[:, :, 0]  # 提取单通道
        else:
            gray = img  # 已经是灰度图像
        
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
        从四个顶点提取并矫正矩形 - 与YOLO版本完全相同的算法
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
    
    def process_image(self, image_path, confidence_threshold=0.5, nms_threshold=0.4, min_area=300, save_debug=False):
        """处理图像 - 使用EasyOCR替代YOLO"""
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
        else:
            img = image_path
            
        if img is None:
            print(f"无法读取图像: {image_path}")
            return []
        
        print(f"处理图像: {image_path if isinstance(image_path, str) else 'numpy array'}")
        print(f"原始图像尺寸: {img.shape}")
        
        # 创建输出目录
        output_dir = "/tmp/out"
        os.makedirs(output_dir, exist_ok=True)
        
        # 提取白色矩形区域
        print("正在提取白色矩形区域...")
        rectangles = self.extract_white_rectangles(img, min_area)
        print(f"提取到 {len(rectangles)} 个矩形区域")
        
        all_results = []
        
        for i, rect_info in enumerate(rectangles):
            print(f"\n处理第 {i+1}/{len(rectangles)} 个矩形...")
            
            # 获取矫正后的图像
            rect_img = rect_info['image']
            bbox_in_original = rect_info['bbox_in_original']
            
            # 保存矩形区域到/tmp/out目录
            rect_filename = f"/tmp/out/rect_easyocr_{i+1:02d}.jpg"
            cv2.imwrite(rect_filename, rect_img)
            print(f"保存矩形区域到: {rect_filename}")
            
            # 使用EasyOCR进行识别
            try:
                ocr_results = run_ocr_on_frame(rect_img)
                
                if ocr_results:
                    print(f"第 {i+1} 个矩形识别到 {len(ocr_results)} 个数字")
                    
                    # 处理识别结果
                    for j, detection in enumerate(ocr_results):
                        print(f"  - 数字 {j+1}: '{detection['text']}' (置信度: {detection['conf']:.3f})")
                        
                        # 获取在矫正图像中的坐标 (EasyOCR返回四点坐标)
                        bbox_points = detection["bbox"]
                        
                        # 计算中心点
                        center_x_corrected = sum([p[0] for p in bbox_points]) / 4
                        center_y_corrected = sum([p[1] for p in bbox_points]) / 4
                        
                        # 将矫正图像中的坐标转换回原图坐标
                        transform_matrix = bbox_in_original['transform_matrix']
                        inverse_matrix = cv2.invert(transform_matrix)[1]
                        
                        # 转换中心点坐标
                        corrected_point = np.array([center_x_corrected, center_y_corrected, 1])
                        original_point = inverse_matrix @ corrected_point
                        
                        center_x = original_point[0] / original_point[2]
                        center_y = original_point[1] / original_point[2]
                        
                        # 转换bbox所有点的坐标
                        original_bbox = []
                        for point in bbox_points:
                            corrected_pt = np.array([point[0], point[1], 1])
                            original_pt = inverse_matrix @ corrected_pt
                            original_bbox.append([
                                original_pt[0] / original_pt[2],
                                original_pt[1] / original_pt[2]
                            ])
                        
                        result = {
                            'text': detection["text"],
                            'confidence': detection["conf"],
                            'center_x': center_x,
                            'center_y': center_y,
                            'rectangle_id': rect_info['id'],
                            'rectangle_area': rect_info['area'],
                            'corrected_size': rect_info['corrected_size'],
                            'bbox_in_corrected': bbox_points,
                            'bbox_in_original': original_bbox
                        }
                        
                        all_results.append(result)
                else:
                    print(f"第 {i+1} 个矩形的EasyOCR识别失败或无结果")
            except Exception as e:
                print(f"第 {i+1} 个矩形EasyOCR识别出错: {e}")
        
        # 保存原始图像到输出目录
        original_filename = f"/tmp/out/original_easyocr.jpg"
        cv2.imwrite(original_filename, img)
        print(f"保存原始图像到: {original_filename}")
        
        print(f"\n=== 最终结果 ===")
        print(f"总共识别到 {len(all_results)} 个数字")
        if all_results:
            print("识别到的数字:")
            for i, result in enumerate(all_results):
                print(f"  {i+1}. '{result['text']}' (置信度: {result['confidence']:.3f}, 矩形ID: {result['rectangle_id']})")
        print(f"所有文件保存在: {output_dir}")
        
        return all_results


class LocalAutoSegmentYOLO:
    """本地自动图像分割和YOLO识别器，不使用HTTP调用"""
    
    def __init__(self):
        self.yolo = _get_local_yolo()
        self.results = []
        
    def extract_white_rectangles(self, img, min_area=300):
        """
        从图像中提取白色矩形区域 - 完全照搬原始算法
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
        从四个顶点提取并矫正矩形 - 完全照搬原始算法
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
    
    def process_image(self, image_path, confidence_threshold=0.5, nms_threshold=0.4, min_area=300, save_debug=False):
        """处理图像 - 使用原始算法结构"""
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
        else:
            img = image_path
            
        if img is None:
            print(f"无法读取图像: {image_path}")
            return []
        
        print(f"处理图像: {image_path if isinstance(image_path, str) else 'numpy array'}")
        print(f"原始图像尺寸: {img.shape}")
        
        # 创建输出目录
        output_dir = "/tmp/out"
        os.makedirs(output_dir, exist_ok=True)
        
        # 提取白色矩形区域
        print("正在提取白色矩形区域...")
        rectangles = self.extract_white_rectangles(img, min_area)
        print(f"提取到 {len(rectangles)} 个矩形区域")
        
        all_results = []
        
        for i, rect_info in enumerate(rectangles):
            print(f"\n处理第 {i+1}/{len(rectangles)} 个矩形...")
            
            # 获取矫正后的图像
            rect_img = rect_info['image']
            bbox_in_original = rect_info['bbox_in_original']
            
            # 保存矩形区域到/tmp/out目录
            rect_filename = f"/tmp/out/rect_{i+1:02d}.jpg"
            cv2.imwrite(rect_filename, rect_img)
            print(f"保存矩形区域到: {rect_filename}")
            
            # 使用本地YOLO进行识别
            yolo_result = self.yolo.predict(rect_img, thred_nms=nms_threshold, thred_cond=confidence_threshold)
            
            if yolo_result.get("success") and yolo_result.get("count", 0) > 0:
                print(f"第 {i+1} 个矩形识别到 {yolo_result['count']} 个数字")
                
                # 处理识别结果
                for j, detection in enumerate(yolo_result["results"]):
                    print(f"  - 数字 {j+1}: '{detection['label']}' (置信度: {detection['confidence']:.3f})")
                    
                    # 获取在矫正图像中的坐标
                    bbox = detection["bbox"]
                    center_x_corrected = (bbox["x1"] + bbox["x2"]) / 2
                    center_y_corrected = (bbox["y1"] + bbox["y2"]) / 2
                    
                    # 将矫正图像中的坐标转换回原图坐标
                    transform_matrix = bbox_in_original['transform_matrix']
                    inverse_matrix = cv2.invert(transform_matrix)[1]
                    
                    # 转换中心点坐标
                    corrected_point = np.array([center_x_corrected, center_y_corrected, 1])
                    original_point = inverse_matrix @ corrected_point
                    
                    center_x = original_point[0] / original_point[2]
                    center_y = original_point[1] / original_point[2]
                    
                    result = {
                        'text': detection["label"],
                        'confidence': detection["confidence"],
                        'center_x': center_x,
                        'center_y': center_y,
                        'rectangle_id': rect_info['id'],
                        'rectangle_area': rect_info['area'],
                        'corrected_size': rect_info['corrected_size'],
                        'bbox_in_corrected': {
                            'x1': bbox["x1"],
                            'y1': bbox["y1"], 
                            'x2': bbox["x2"],
                            'y2': bbox["y2"]
                        }
                    }
                    
                    all_results.append(result)
            else:
                print(f"第 {i+1} 个矩形的YOLO识别失败或无结果")
        
        # 保存原始图像到输出目录
        original_filename = f"/tmp/out/original.jpg"
        cv2.imwrite(original_filename, img)
        print(f"保存原始图像到: {original_filename}")
        
        print(f"\n=== 最终结果 ===")
        print(f"总共识别到 {len(all_results)} 个数字")
        if all_results:
            print("识别到的数字:")
            for i, result in enumerate(all_results):
                print(f"  {i+1}. '{result['text']}' (置信度: {result['confidence']:.3f}, 矩形ID: {result['rectangle_id']})")
        print(f"所有文件保存在: {output_dir}")
        
        return all_results


def process_image_with_auto_segment(img_base64, idx):
    """在线程中处理图像的同步函数 - 使用本地YOLO"""
    try:
        print(f"线程中处理crop {idx}")
        
        # 获取本地AutoSegmentYOLO实例
        auto_segmenter = _get_auto_segment_yolo()
        if auto_segmenter is None:
            print("本地AutoSegmentYOLO实例获取失败")
            return []
        
        # 解码base64图片并保存临时文件
        image_data = base64.b64decode(img_base64)
        temp_path = f"/tmp/temp_crop_{idx}_{int(time.time())}.jpg"
        with open(temp_path, "wb") as f:
            f.write(image_data)
        
        print(f"处理临时图像: {temp_path}")
        
        # 直接调用处理方法
        results = auto_segmenter.process_image(
            temp_path,
            confidence_threshold=0.5,
            nms_threshold=0.4,
            min_area=300,
            save_debug=True
        )
        
        print(f"识别结果数量: {len(results)}")
        
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # 转换格式以兼容原有的OCR格式
        converted_results = []
        for item in results:
            # 将中心坐标转换为类似OCR的bbox格式
            center_x = item["center_x"]
            center_y = item["center_y"]
            
            # 估算一个合理的bbox（基于corrected_size）
            corrected_size = item.get("corrected_size", [50, 50])
            half_width = corrected_size[0] / 4  # 估算文字区域
            half_height = corrected_size[1] / 4
            
            bbox = [
                [center_x - half_width, center_y - half_height],
                [center_x + half_width, center_y - half_height],
                [center_x + half_width, center_y + half_height],
                [center_x - half_width, center_y + half_height]
            ]
            
            converted_results.append({
                "text": item["text"],
                "conf": item["confidence"],
                "bbox": bbox,
                "center": [center_x, center_y],
                "rectangle_id": item.get("rectangle_id", 0),
                "rectangle_area": item.get("rectangle_area", 0)
            })
        
        print(f"转换后的OCR结果数量: {len(converted_results)}")
        return converted_results
        
    except Exception as e:
        print(f"线程中本地AutoSegmentYOLO处理失败: {e}")
        import traceback
        traceback.print_exc()
        return []


def run_ocr_on_frame(frame: np.ndarray):
    """
    对单帧图像执行 OCR，返回列表，每项包含数字、置信度、四点坐标
    -------
    [{'text': '75', 'conf': 0.98,
      'bbox': [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}, ...]
    """
    if frame is None or frame.size == 0:
        return []

    reader = _get_reader()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = reader.readtext(
        rgb,
        allowlist=OCR_ALLOWLIST,
        detail=1,  # (bbox, text, conf)
    )

    parsed = []
    for bbox, text, conf in results:
        # 若只想要数字，可再次过滤
        if OCR_ALLOWLIST and any(c not in OCR_ALLOWLIST for c in text):
            continue
        parsed.append(
            {
                "text": text,
                "conf": float(conf),
                "bbox": [list(map(int, pt)) for pt in bbox],
            }
        )
    return parsed


def load_config() -> Dict[str, Any]:
    """加载配置文件"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return get_default_config()
    else:
        config = get_default_config()
        save_config(config)
        return config


def save_config(config: Optional[Dict[str, Any]] = None) -> None:
    """保存配置文件"""
    if config is None:
        # 构建当前配置
        config = {
            "camera": camera_config,
            "detection": params,
            "area_filter": area_filter_params,
            "perspective_correction": perspective_params,
            "black_detection": black_detection_params,
        }

    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        print("配置文件已保存")
    except Exception as e:
        print(f"保存配置文件失败: {e}")


def get_default_config() -> Dict[str, Any]:
    """获取默认配置"""
    return {
        "enable_ocr": True,
        "camera": {"index": 1, "width": 1920, "height": 1080},
        "detection": {
            "h1_min": 0,
            "h1_max": 179,
            "s1_min": 0,
            "s1_max": 255,
            "v1_min": 0,
            "v1_max": 85,
            "h2_min": 0,
            "h2_max": 179,
            "s2_min": 0,
            "s2_max": 255,
            "v2_min": 0,
            "v2_max": 85,
            "use_range2": False,
            "min_area": 200,
            "canny_min": 50,
            "canny_max": 150,
        },
        "area_filter": {
            "min_crop_area": 500000,
            "max_crop_area": 19000000,
            "enable_area_filter": True,
            "a4_ratio_tolerance": 0.3,
            "max_circularity": 0.7,
            "min_solidity": 0.8,
            "max_vertices": 8,
            "enable_a4_check": True,
        },
        "perspective_correction": {
            "enable": True,
            "target_width": 210,
            "target_height": 297,
            "a4_ratio": 1.414285714285714,
            "use_short_edge_for_measurement": True,
        },
        "black_detection": {
            "lower_h": 0,
            "lower_s": 0,
            "lower_v": 0,
            "upper_h": 255,
            "upper_s": 255,
            "upper_v": 80,
            "morph_kernel_size": 3,
        },
    }


# 全局常量
MIN_AREA: int = 10000

detection_params: Dict[str, Union[int, float]] = {
    "min_area": 500,
    "hollow_ratio": 0.1,  # 降低空心占比要求，更容易检测到空心框架
    "aspect_ratio_min": 0.2,
    "aspect_ratio_max": 5.0,
    "epsilon_factor": 0.02,
    "min_frame_thickness": 10,  # 最小框架厚度
    "min_vertices": 4,  # 多边形最少顶点数
    "max_vertices": 5,  # 多边形最多顶点数
}


# 加载配置
config: Dict[str, Any] = load_config()

# 全局参数：从配置文件读取
params: Dict[str, Any] = config["detection"].copy()
area_filter_params: Dict[str, Any] = config["area_filter"].copy()
perspective_params: Dict[str, Any] = config["perspective_correction"].copy()
black_detection_params: Dict[str, Any] = config["black_detection"].copy()
camera_config: Dict[str, Any] = config["camera"].copy()

# 从配置文件同步 detection_params 中的参数
if "min_vertices" in config["detection"]:
    detection_params["min_vertices"] = config["detection"]["min_vertices"]
if "max_vertices" in config["detection"]:
    detection_params["max_vertices"] = config["detection"]["max_vertices"]
cap: cv2.VideoCapture = cv2.VideoCapture(camera_config["index"])
cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config["width"])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config["height"])
if not cap.isOpened():
    raise RuntimeError("无法打开摄像头")

lock: threading.Lock = threading.Lock()
frame: Optional[np.ndarray] = None
latest_stats: Dict[str, Any] = {}
current_crops: List[np.ndarray] = []  # 存储当前检测到的裁剪区域
clients: List[WebSocket] = []

# 控制变量
show_all_rectangles: bool = False  # 控制是否显示所有检测到的矩形（红框标记）


# ------------------------------------------------------------------------------#
# 辅助函数 (完整替换后的版本)
# ------------------------------------------------------------------------------#


def dist(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    return math.hypot(q[0] - p[0], q[1] - p[1])


def midpoint(p: Tuple[int, int], q: Tuple[int, int]) -> Tuple[int, int]:
    return ((p[0] + q[0]) // 2, (p[1] + q[1]) // 2)


def order_pts(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    tr, bl = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.int32)


def is_a4_quad(pts: np.ndarray, a4_ratio: float = 297 / 210, tol: float = 0.2) -> bool:
    o = order_pts(pts)
    e = [dist(o[i], o[(i + 1) % 4]) for i in range(4)]
    long_e, short_e = (e[0] + e[2]) / 2, (e[1] + e[3]) / 2
    if short_e == 0:
        return False
    asp = long_e / short_e
    return abs(asp - a4_ratio) < tol or abs(asp - 1 / a4_ratio) < tol


# 保持原有的identify_shape_by_area不变
# def identify_shape_by_area(contour):
#     ...
def calculate_physical_dimensions(
    shape_result: Dict[str, Any], crop_width: int, crop_height: int
) -> Dict[str, Any]:
    """
    根据crop尺寸计算形状的物理尺寸

    Args:
        shape_result: 形状识别结果
        crop_width: crop图像宽度（像素）
        crop_height: crop图像高度（像素）

    Returns:
        包含物理尺寸信息的字典
    """
    # A4纸减去边框后的物理尺寸（毫米）
    physical_width_mm = 170  # 210 - 40
    physical_height_mm = 257  # 297 - 40

    # 计算像素到毫米的转换比例
    mm_per_pixel_x = physical_width_mm / crop_width
    mm_per_pixel_y = physical_height_mm / crop_height

    # 使用平均比例（更准确）
    mm_per_pixel = (mm_per_pixel_x + mm_per_pixel_y) / 2

    shape_type = shape_result["shape_type"]
    physical_info = {
        "mm_per_pixel": mm_per_pixel,
        "physical_width_mm": 0,
        "physical_height_mm": 0,
        "physical_diameter_mm": 0,
        "physical_side_lengths_mm": [],
        "physical_perimeter_mm": 0,
        "physical_area_mm2": 0,
        "measurement_type": "unknown",
    }

    # 计算物理面积
    physical_info["physical_area_mm2"] = shape_result["area"] * (mm_per_pixel**2)
    physical_info["physical_perimeter_mm"] = (
        shape_result.get("perimeter", 0) * mm_per_pixel
    )

    if shape_type == "Square":
        # 正方形：计算边长
        if shape_result.get("side_lengths"):
            physical_side_lengths = [
                length * mm_per_pixel for length in shape_result["side_lengths"]
            ]
            physical_info["physical_side_lengths_mm"] = physical_side_lengths
            avg_side_length = sum(physical_side_lengths) / len(physical_side_lengths)
            physical_info["physical_width_mm"] = avg_side_length
            physical_info["physical_height_mm"] = avg_side_length
            physical_info["measurement_type"] = "side_length"
        else:
            # 回退到宽高
            physical_info["physical_width_mm"] = shape_result["width"] * mm_per_pixel
            physical_info["physical_height_mm"] = shape_result["height"] * mm_per_pixel
            physical_info["measurement_type"] = "bounding_box"

    elif shape_type == "Circle":
        # 圆形：计算直径
        # 使用外接圆直径
        diameter_pixels = max(shape_result["width"], shape_result["height"])
        physical_info["physical_diameter_mm"] = diameter_pixels * mm_per_pixel
        physical_info["physical_width_mm"] = physical_info["physical_diameter_mm"]
        physical_info["physical_height_mm"] = physical_info["physical_diameter_mm"]
        physical_info["measurement_type"] = "diameter"

    elif shape_type == "Triangle":
        # TODO: 三角形当作等边处理
        if shape_result.get("side_lengths"):
            physical_side_lengths = [
                length * mm_per_pixel for length in shape_result["side_lengths"]
            ]
            physical_info["physical_side_lengths_mm"] = physical_side_lengths
            # 等边三角形处理
            avg_side_length = sum(physical_side_lengths) / len(physical_side_lengths)
            physical_info["physical_width_mm"] = avg_side_length
            physical_info["physical_height_mm"] = avg_side_length * 0.866  # sqrt(3)/2
            physical_info["measurement_type"] = (
                "equilateral_triangle"  # TODO: 只做了等边
            )
        else:
            physical_info["physical_width_mm"] = shape_result["width"] * mm_per_pixel
            physical_info["physical_height_mm"] = shape_result["height"] * mm_per_pixel
            physical_info["measurement_type"] = "bounding_box"
    else:
        # 其他形状：使用边界框
        physical_info["physical_width_mm"] = shape_result["width"] * mm_per_pixel
        physical_info["physical_height_mm"] = shape_result["height"] * mm_per_pixel
        if shape_result.get("side_lengths"):
            physical_side_lengths = [
                length * mm_per_pixel for length in shape_result["side_lengths"]
            ]
            physical_info["physical_side_lengths_mm"] = physical_side_lengths
        physical_info["measurement_type"] = "bounding_box"

    return physical_info


def identify_shape_by_area(
    contour: np.ndarray, area_thresh: int = 20
) -> Dict[str, Any]:
    """
    识别轮廓的形状，并返回详细信息
    返回值：
        {
            "shape_type": "Circle"/"Triangle"/"Square"/"Unknown",
            "area": float,
            "width": int,
            "height": int,
            "contour": numpy.array,
            "info": str,
            "side_lengths": List[float] (对于多边形),
            "mean_side_length": float (对于多边形),
            "perimeter": float
        }
    """
    area = cv2.contourArea(contour)
    if area < area_thresh:
        return {
            "shape_type": "Unknown",
            "area": area,
            "width": 0,
            "height": 0,
            "contour": contour,
            "info": "Too small area",
            "side_lengths": [],
            "mean_side_length": 0,
            "perimeter": 0,
        }

    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return {
            "shape_type": "Unknown",
            "area": area,
            "width": 0,
            "height": 0,
            "contour": contour,
            "info": "Zero perimeter",
            "side_lengths": [],
            "mean_side_length": 0,
            "perimeter": 0,
        }

    circularity = 4 * np.pi * area / (perimeter**2)
    x, y, w, h = cv2.boundingRect(contour)

    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = len(approx)

    shape_type = "Unknown"
    side_lengths = []
    mean_side_length = 0

    # 计算边长（对于多边形）
    if vertices >= 3:
        side_lengths = [
            float(np.linalg.norm(approx[i][0] - approx[(i + 1) % vertices][0]))
            for i in range(vertices)
        ]
        mean_side_length = float(np.mean(side_lengths))

    info = f"Vertices={vertices}, Circ={circularity:.3f}, Ratio={w / h:.3f}"

    if vertices == 3:
        shape_type = "Triangle"
        info += f"; Triangle, Sides={[f'{s:.1f}' for s in side_lengths]}"
    elif vertices == 4:
        var_coeff = (
            np.std(side_lengths) / mean_side_length if mean_side_length > 0 else 1
        )
        info += (
            f"; Var_coeff={var_coeff:.3f}, Sides={[f'{s:.1f}' for s in side_lengths]}"
        )

        # 判定为正方形：边长相似且长宽比接近 1
        if var_coeff < 0.4:
            shape_type = "Square"
            # 将宽高设置为平均边长
            w = h = int(mean_side_length)
            info += "; Square"
        else:
            info += "; Quad but not square"
    elif circularity > 0.6:
        shape_type = "Circle"
        info += "; Circle"
    elif vertices >= 6 and circularity > 0.4:
        shape_type = "Circle"
        info += "; Near circle"
    else:
        info += "; Unknown shape"

    return {
        "shape_type": shape_type,
        "area": area,
        "width": w
        if shape_type != "Circle"
        else int(cv2.minEnclosingCircle(contour)[1] * 2),
        "height": h
        if shape_type != "Circle"
        else int(cv2.minEnclosingCircle(contour)[1] * 2),
        "contour": contour,
        "info": info,
        "side_lengths": side_lengths,
        "mean_side_length": mean_side_length,
        "perimeter": perimeter,
    }


# ------------------------------------------------------------------------------#
# 阶段 1：生成二值掩码（双 HSV 范围 + 形态学）
# ------------------------------------------------------------------------------#
def build_mask(hsv: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    lower1 = np.array([params["h1_min"], params["s1_min"], params["v1_min"]])
    upper1 = np.array([params["h1_max"], params["s1_max"], params["v1_max"]])
    mask1 = cv2.inRange(hsv, lower1, upper1)

    if params["use_range2"]:
        lower2 = np.array([params["h2_min"], params["s2_min"], params["v2_min"]])
        upper2 = np.array([params["h2_max"], params["s2_max"], params["v2_max"]])
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        mask = mask1

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


# ------------------------------------------------------------------------------#
# 阶段 2：轮廓检测与过滤（只保留近似 A4 的四边形）
# ------------------------------------------------------------------------------#
def find_a4_quads(mask: np.ndarray, params: Dict[str, Any]) -> List[np.ndarray]:
    edges = cv2.Canny(mask, params["canny_min"], params["canny_max"])
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quads = []
    for cnt in cnts:
        if cv2.contourArea(cnt) < MIN_AREA:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and is_a4_quad(approx.reshape(4, 2)):
            quads.append(approx.reshape(4, 2))
    return quads


def preprocess_image(frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    global params, detection_params
    """图像预处理 - 使用HSV色彩空间"""
    # 转换为HSV色彩空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 分离HSV通道
    h, s, v = cv2.split(hsv)

    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 自适应阈值处理
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # 使用实时调整的HSV阈值处理
    lower_hsv = np.array([params["h1_min"], params["s1_min"], params["v1_min"]])
    upper_hsv = np.array([params["h1_max"], params["s1_max"], params["v1_max"]])
    hsv_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # 结合灰度阈值和HSV掩码
    combined = cv2.bitwise_or(cv2.bitwise_not(thresh), hsv_mask)

    # 形态学操作，去除噪点
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    # 返回处理结果和调试信息
    debug_info = {
        "hsv": hsv,
        "h_channel": h,
        "s_channel": s,
        "v_channel": v,
        "gray": gray,
        "blurred": blurred,
        "thresh": thresh,
        "hsv_mask": hsv_mask,
        "combined": combined,
        "hsv_params": params.copy(),  # 添加当前HSV参数用于显示
        "detection_params": detection_params.copy(),
    }

    return cleaned, debug_info


def find_hollow_rectangles(
    processed_img: np.ndarray, debug_info: Dict[str, Any]
) -> List[Dict[str, Any]]:
    global detection_params, params, area_filter_params
    """检测空心矩形框架"""
    # 使用RETR_LIST来获取所有轮廓，包括内外轮廓
    contours, _ = cv2.findContours(
        processed_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    hollow_rectangles = []
    all_contours = []  # 保存所有轮廓用于调试

    for contour in contours:
        # 计算轮廓面积，过滤太小的轮廓
        area = cv2.contourArea(contour)
        all_contours.append({"contour": contour, "area": area, "is_valid": False})

        # 使用实时调整的最小面积阈值
        if area < detection_params["min_area"]:
            continue

        # 轮廓近似 - 使用更小的epsilon来保持更多细节
        epsilon = detection_params["epsilon_factor"] * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 检查是否接近四边形（允许可配置的顶点数范围，因为矩形框架可能有些变形）
        if detection_params["min_vertices"] <= len(approx) <= detection_params["max_vertices"]:
            # 计算轮廓的边界框
            x, y, w, h = cv2.boundingRect(approx)

            # 过滤太小的矩形
            if (
                w < detection_params["min_frame_thickness"] * 3
                or h < detection_params["min_frame_thickness"] * 3
            ):
                continue

            # 使用实时调整的长宽比范围
            aspect_ratio = float(w) / h
            if (
                detection_params["aspect_ratio_min"]
                < aspect_ratio
                < detection_params["aspect_ratio_max"]
            ):
                # 检查是否为空心矩形框架
                if is_hollow_rectangle(processed_img, contour):
                    # 在添加到结果之前进行面积过滤
                    if area_filter_params["enable_area_filter"]:
                        if (
                            area < area_filter_params["min_crop_area"]
                            or area > area_filter_params["max_crop_area"]
                        ):
                            continue  # 跳过不符合面积要求的矩形

                    hollow_rectangles.append(
                        {
                            "contour": approx,
                            "bbox": (x, y, w, h),
                            "area": area,
                            "aspect_ratio": aspect_ratio,
                        }
                    )
                    # 标记为有效轮廓
                    all_contours[-1]["is_valid"] = True

    # 按面积排序，优先选择大的空心框架
    hollow_rectangles.sort(key=lambda x: x["area"], reverse=True)

    # 添加调试信息
    debug_info["all_contours"] = all_contours
    debug_info["valid_rectangles"] = len(hollow_rectangles)

    return hollow_rectangles


def is_hollow_rectangle(processed_img: np.ndarray, contour: np.ndarray) -> bool:
    global detection_params, params
    """判断是否为空心矩形框架"""
    # 获取轮廓的边界框
    x, y, w, h = cv2.boundingRect(contour)

    # 创建轮廓掩码
    mask = np.zeros(processed_img.shape, np.uint8)
    cv2.fillPoly(mask, [contour], 255)

    # 创建内部区域掩码（稍微缩小一点，避开边框）
    frame_thickness = detection_params["min_frame_thickness"]
    inner_x = x + frame_thickness
    inner_y = y + frame_thickness
    inner_w = w - 2 * frame_thickness
    inner_h = h - 2 * frame_thickness

    # 确保内部区域有效
    if inner_w <= 0 or inner_h <= 0:
        return False

    # 创建内部矩形掩码
    inner_mask = np.zeros(processed_img.shape, np.uint8)
    cv2.rectangle(
        inner_mask, (inner_x, inner_y), (inner_x + inner_w, inner_y + inner_h), 255, -1
    )

    # 只检查轮廓内部的内部矩形区域
    combined_mask = cv2.bitwise_and(mask, inner_mask)

    # 计算内部区域的像素
    total_inner_pixels = cv2.countNonZero(combined_mask)
    if total_inner_pixels == 0:
        return False

    # 计算内部区域中的白色像素（背景/空心部分）
    inner_region = cv2.bitwise_and(processed_img, combined_mask)
    white_pixels = cv2.countNonZero(inner_region)

    # 计算空心比例
    hollow_ratio = white_pixels / total_inner_pixels if total_inner_pixels > 0 else 0

    # 同时检查轮廓边界是否有足够的黑色像素（框架部分）
    # 创建边框区域掩码
    frame_mask = cv2.bitwise_and(mask, cv2.bitwise_not(inner_mask))
    frame_region = cv2.bitwise_and(processed_img, frame_mask)
    frame_total = cv2.countNonZero(frame_mask)
    frame_black = frame_total - cv2.countNonZero(frame_region) if frame_total > 0 else 0
    frame_ratio = frame_black / frame_total if frame_total > 0 else 0

    # 同时满足：内部空心 + 边框有实体
    is_hollow = hollow_ratio > detection_params["hollow_ratio"]
    has_frame = frame_ratio > 0.3  # 边框至少30%是实体

    return is_hollow and has_frame


def draw_annotations(
    frame: np.ndarray, hollow_rectangles: List[Dict[str, Any]]
) -> np.ndarray:
    global detection_params, params
    """绘制标注"""
    annotated_frame = frame.copy()

    for i, rect in enumerate(hollow_rectangles):
        contour = rect["contour"]
        x, y, w, h = rect["bbox"]
        area = rect["area"]
        aspect_ratio = rect.get("aspect_ratio", 0)

        # 绘制轮廓 - 绿色
        cv2.drawContours(annotated_frame, [contour], -1, (0, 255, 0), 3)

        # 绘制边界框 - 蓝色
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 绘制中心点 - 红色
        center_x = x + w // 2
        center_y = y + h // 2
        cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1)

        # 如果有4个或更多点，计算并绘制长边中点连线
        if len(contour) >= 4:
            # 获取四个主要顶点
            src_pts = contour.reshape(-1, 2).astype(np.float32)
            if len(src_pts) > 4:
                hull = cv2.convexHull(src_pts.astype(np.int32))
                epsilon = 0.02 * cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, epsilon, True)
                if len(approx) >= 4:
                    src_pts = approx[:4].reshape(-1, 2).astype(np.float32)
                else:
                    src_pts = src_pts[:4]

            # 排序顶点
            ordered_pts = order_points(src_pts)

            # 计算四条边的长度
            edge1, edge2, edge3, edge4 = calculate_edge_lengths(src_pts)
            horizontal_avg = (edge1 + edge3) / 2  # 上边和下边平均
            vertical_avg = (edge2 + edge4) / 2  # 左边和右边平均

            # 直接比较哪个是长边
            if horizontal_avg > vertical_avg:
                # 水平方向是长边，短边是垂直方向，计算左右边中点
                left_mid = midpoint(
                    (int(ordered_pts[0][0]), int(ordered_pts[0][1])),
                    (int(ordered_pts[3][0]), int(ordered_pts[3][1])),
                )
                right_mid = midpoint(
                    (int(ordered_pts[1][0]), int(ordered_pts[1][1])),
                    (int(ordered_pts[2][0]), int(ordered_pts[2][1])),
                )
                new_length = dist(left_mid, right_mid)
                mid1, mid2 = left_mid, right_mid
            else:
                # 垂直方向是长边，短边是水平方向，计算上下边中点
                top_mid = midpoint(
                    (int(ordered_pts[0][0]), int(ordered_pts[0][1])),
                    (int(ordered_pts[1][0]), int(ordered_pts[1][1])),
                )
                bottom_mid = midpoint(
                    (int(ordered_pts[3][0]), int(ordered_pts[3][1])),
                    (int(ordered_pts[2][0]), int(ordered_pts[2][1])),
                )
                new_length = dist(top_mid, bottom_mid)
                mid1, mid2 = top_mid, bottom_mid

            # 绘制红色连线（短边中点连线）
            cv2.line(annotated_frame, mid1, mid2, (0, 0, 255), 3)

            # 在中点画小圆圈
            cv2.circle(annotated_frame, mid1, 6, (0, 0, 255), -1)
            cv2.circle(annotated_frame, mid2, 6, (0, 0, 255), -1)

            # 绘制四个顶点
            colors = [
                (255, 0, 0),
                (0, 255, 0),
                (0, 0, 255),
                (255, 255, 0),
            ]  # 蓝、绿、红、黄
            labels = ["TL", "TR", "BR", "BL"]

            for j, (pt, color, label) in enumerate(zip(ordered_pts, colors, labels)):
                cv2.circle(annotated_frame, (int(pt[0]), int(pt[1])), 8, color, -1)
                cv2.putText(
                    annotated_frame,
                    label,
                    (int(pt[0]) + 10, int(pt[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
        else:
            new_length = 0
            horizontal_avg = vertical_avg = 0

        # 添加文字标注
        label = f"Hollow Rect {i + 1}"
        info = f"Area: {int(area)}"
        size_info = f"Size: {w}x{h}"
        ratio_info = f"Ratio: {aspect_ratio:.2f}"
        edge_info = f"H_avg:{horizontal_avg:.0f} V_avg:{vertical_avg:.0f}"
        new_length_info = f"Short Edge Length: {new_length:.0f}px"

        # 绘制标注文字
        text_lines = [label, info, size_info, ratio_info, edge_info, new_length_info]
        for idx, text in enumerate(text_lines):
            y_offset = -70 + (idx * 15)
            cv2.putText(
                annotated_frame,
                text,
                (x, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        # 绘制四个角点
        for point in contour:
            px, py = point[0]
            cv2.circle(annotated_frame, (px, py), 3, (255, 255, 0), -1)

    return annotated_frame


def annotate_inner(
    crops: List[np.ndarray], params: Dict[str, Any], stats: Dict[str, Any]
) -> None:
    """内部标注 - 在裁剪图像中检测形状"""
    hsv1_lower = np.array([params["h1_min"], params["s1_min"], params["v1_min"]])
    hsv1_upper = np.array([params["h1_max"], params["s1_max"], params["v1_max"]])
    hsv2_lower = (
        np.array([params["h2_min"], params["s2_min"], params["v2_min"]])
        if params["use_range2"]
        else None
    )
    hsv2_upper = (
        np.array([params["h2_max"], params["s2_max"], params["v2_max"]])
        if params["use_range2"]
        else None
    )

    for idx, roi in enumerate(crops):
        if idx >= len(stats["rects"]):
            continue

        # 转换为灰度图
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 二值化处理找到白色区域
        _, white = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        w_cnts, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not w_cnts:
            continue

        # 找到最大的白色区域作为内部区域
        wx, wy, ww, wh = cv2.boundingRect(max(w_cnts, key=cv2.contourArea))
        inner = roi[wy : wy + wh, wx : wx + ww]

        # 转换为HSV进行颜色检测
        hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv1_lower, hsv1_upper)
        if params["use_range2"] and hsv2_lower is not None and hsv2_upper is not None:
            mask2 = cv2.inRange(hsv, hsv2_lower, hsv2_upper)
            mask = cv2.bitwise_or(mask, mask2)

        # 形态学操作
        k = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        # 查找轮廓
        s_cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not s_cnts:
            continue

        # 找到所有符合条件的形状，而不仅仅是最大的
        detected_shapes = []
        for shape in s_cnts:
            # 过滤太小的轮廓
            shape_area = cv2.contourArea(shape)
            if shape_area < 50:  # 最小面积阈值
                continue

            shape_result = identify_shape_by_area(shape)
            detected_shapes.append(shape_result)

        # 如果没有检测到任何形状，跳过
        if not detected_shapes:
            continue

        # 更新统计信息 - 包含所有检测到的形状
        all_shapes_info = []
        total_inner_area = 0

        for shape_result in detected_shapes:
            total_inner_area += shape_result["area"]

            # 计算物理尺寸
            physical_info = calculate_physical_dimensions(
                shape_result, roi.shape[1], roi.shape[0]
            )

            # 计算形状的中心点
            shape = shape_result["contour"]
            M = cv2.moments(shape)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"]) + wx
                cy = int(M["m01"] / M["m00"]) + wy
                center = [cx, cy]
            else:
                # 如果无法计算质心，使用边界框中心
                x, y, w, h = cv2.boundingRect(shape)
                center = [x + w // 2 + wx, y + h // 2 + wy]

            # 计算边界框
            x, y, w, h = cv2.boundingRect(shape)
            bbox = [x + wx, y + wy, w, h]  # 调整坐标到原图

            shape_info = {
                "shape_type": shape_result["shape_type"],
                "width": int(shape_result["width"]),
                "height": int(shape_result["height"]),
                "area": int(shape_result["area"]),
                "info": shape_result["info"],
                "contour": shape_result["contour"].reshape(-1, 2).tolist(),
                # 像素尺寸信息
                "side_lengths": [
                    float(length) for length in shape_result.get("side_lengths", [])
                ],
                "mean_side_length": float(shape_result.get("mean_side_length", 0)),
                "perimeter": float(shape_result.get("perimeter", 0)),
                # 物理尺寸信息
                "physical_info": physical_info,
                # 添加位置信息
                "position": {
                    "center": center,
                    "bbox": bbox,
                    "contour_points": shape_result["contour"].reshape(-1, 2).tolist(),
                },
            }
            all_shapes_info.append(shape_info)

        # 选择最大的形状作为主要形状（用于兼容性）
        main_shape = max(detected_shapes, key=lambda x: x["area"])
        main_physical_info = calculate_physical_dimensions(
            main_shape, roi.shape[1], roi.shape[0]
        )

        stats["rects"][idx].update(
            {
                "shape_type": main_shape["shape_type"],
                "inner_width": int(main_shape["width"]),
                "inner_height": int(main_shape["height"]),
                "inner_area": int(main_shape["area"]),
                "inner_info": main_shape["info"],
                "inner_contour": main_shape["contour"].reshape(-1, 2).tolist(),
                # 主要形状的像素尺寸信息
                "inner_side_lengths": [
                    float(length) for length in main_shape.get("side_lengths", [])
                ],
                "inner_mean_side_length": float(main_shape.get("mean_side_length", 0)),
                "inner_perimeter": float(main_shape.get("perimeter", 0)),
                # 主要形状的物理尺寸信息
                "inner_physical_info": main_physical_info,
                # 新增：所有检测到的形状信息（包含物理尺寸）
                "all_shapes": all_shapes_info,
                "shapes_count": len(detected_shapes),
                "total_inner_area": int(total_inner_area),
            }
        )

        # 在裁剪图像上绘制所有检测到的形状
        for i, shape_result in enumerate(detected_shapes):
            shape = shape_result["contour"]

            # 绘制轮廓 - 不同的颜色
            colors = [
                (0, 255, 0),
                (255, 0, 0),
                (0, 0, 255),
                (255, 255, 0),
                (255, 0, 255),
                (0, 255, 255),
            ]
            color = colors[i % len(colors)]
            cv2.drawContours(roi, [shape], -1, color, 2)

            # 计算形状的中心点
            M = cv2.moments(shape)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"]) + wx
                cy = int(M["m01"] / M["m00"]) + wy
                cv2.circle(roi, (cx, cy), 5, color, -1)

                # 添加标注文字
                cv2.putText(
                    roi,
                    f"{shape_result['shape_type']} #{i + 1}",
                    (cx - 40, cy - 30 - i * 20),  # 错开显示位置
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
                cv2.putText(
                    roi,
                    f"Area: {int(shape_result['area'])}px",
                    (cx - 40, cy - 10 - i * 25),  # 错开显示位置
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )

                # 计算并显示物理尺寸
                physical_info = calculate_physical_dimensions(
                    shape_result, roi.shape[1], roi.shape[0]
                )

                # 根据形状类型显示相应的物理尺寸
                if shape_result["shape_type"] == "Square":
                    if physical_info["physical_side_lengths_mm"]:
                        avg_side = sum(physical_info["physical_side_lengths_mm"]) / len(
                            physical_info["physical_side_lengths_mm"]
                        )
                        cv2.putText(
                            roi,
                            f"Side: {avg_side:.1f}mm",
                            (cx - 40, cy + 10 - i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            color,
                            1,
                        )
                elif shape_result["shape_type"] == "Circle":
                    cv2.putText(
                        roi,
                        f"Dia: {physical_info['physical_diameter_mm']:.1f}mm",
                        (cx - 40, cy + 10 - i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                    )
                elif shape_result["shape_type"] == "Triangle":
                    if physical_info["physical_side_lengths_mm"]:
                        avg_side = sum(physical_info["physical_side_lengths_mm"]) / len(
                            physical_info["physical_side_lengths_mm"]
                        )
                        cv2.putText(
                            roi,
                            f"Side: {avg_side:.1f}mm (eq.tri)",  # TODO: 等边处理
                            (cx - 40, cy + 10 - i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            color,
                            1,
                        )
                else:
                    cv2.putText(
                        roi,
                        f"Size: {physical_info['physical_width_mm']:.1f}x{physical_info['physical_height_mm']:.1f}mm",
                        (cx - 40, cy + 10 - i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        color,
                        1,
                    )

                # 显示像素边长信息（如果有的话）- 更小的字体显示在底部
                if (
                    shape_result.get("side_lengths")
                    and len(shape_result["side_lengths"]) > 0
                ):
                    side_lengths = shape_result["side_lengths"]
                    if len(side_lengths) == 4:  # 四边形
                        sides_text = f"Px: {side_lengths[0]:.0f},{side_lengths[1]:.0f},{side_lengths[2]:.0f},{side_lengths[3]:.0f}"
                    elif len(side_lengths) == 3:  # 三角形
                        sides_text = f"Px: {side_lengths[0]:.0f},{side_lengths[1]:.0f},{side_lengths[2]:.0f}"
                    else:
                        sides_text = (
                            f"Px: {shape_result.get('mean_side_length', 0):.0f}"
                        )

                    cv2.putText(
                        roi,
                        sides_text,
                        (cx - 40, cy + 30 - i * 25),  # 错开显示位置
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        color,
                        1,
                    )


def order_points(pts: np.ndarray) -> np.ndarray:
    """按照左上、右上、右下、左下的顺序排列四个点"""
    # 计算中心点
    center = np.mean(pts, axis=0)

    # 按角度排序
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    sorted_indices = np.argsort(angles)

    # 重新排列点，确保按左上、右上、右下、左下顺序
    sorted_pts = pts[sorted_indices]

    # 找到最上面的两个点
    top_indices = np.argsort(sorted_pts[:, 1])[:2]
    top_points = sorted_pts[top_indices]

    # 找到最下面的两个点
    bottom_indices = np.argsort(sorted_pts[:, 1])[-2:]
    bottom_points = sorted_pts[bottom_indices]

    # 对上面的点按x坐标排序（左上、右上）
    if top_points[0][0] > top_points[1][0]:
        top_points = top_points[::-1]

    # 对下面的点按x坐标排序（左下、右下）
    if bottom_points[0][0] > bottom_points[1][0]:
        bottom_points = bottom_points[::-1]

    return np.array(
        [top_points[0], top_points[1], bottom_points[1], bottom_points[0]],
        dtype=np.float32,
    )


def calculate_edge_lengths(pts: np.ndarray) -> Tuple[float, float, float, float]:
    """计算四边形四条边的长度"""
    ordered_pts = order_points(pts)

    edge1 = np.linalg.norm(ordered_pts[1] - ordered_pts[0])  # 上边
    edge2 = np.linalg.norm(ordered_pts[2] - ordered_pts[1])  # 右边
    edge3 = np.linalg.norm(ordered_pts[3] - ordered_pts[2])  # 下边
    edge4 = np.linalg.norm(ordered_pts[0] - ordered_pts[3])  # 左边

    return edge1, edge2, edge3, edge4


def create_a4_crops_from_contours(
    img: np.ndarray, hollow_rectangles: List[Dict[str, Any]]
) -> List[np.ndarray]:
    """根据绿色轮廓创建拉伸到A4比例的裁剪图像 - 自适应尺寸版本"""
    crops = []

    # A4纸比例 (297mm x 210mm)，减去各边2cm后为 257mm x 170mm
    target_width_mm = 170  # 210 - 40
    target_height_mm = 257  # 297 - 40
    a4_ratio = target_height_mm / target_width_mm  # 约1.51

    for rect in hollow_rectangles:
        contour_points = rect["contour"]

        # 确保有4个点
        if len(contour_points) < 4:
            continue

        # 如果点数超过4个，使用convex hull获取凸包，然后近似为4个点
        if len(contour_points) > 4:
            hull = cv2.convexHull(contour_points)
            epsilon = 0.02 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            if len(approx) >= 4:
                contour_points = approx[:4]
            else:
                contour_points = contour_points[:4]

        # 转换为正确的格式
        src_pts = contour_points.reshape(-1, 2).astype(np.float32)

        # 排序源点（左上、右上、右下、左下）
        ordered_src = order_points(src_pts)

        # 计算检测到的四边形的实际尺寸
        top_width = np.linalg.norm(ordered_src[1] - ordered_src[0])
        bottom_width = np.linalg.norm(ordered_src[3] - ordered_src[2])
        left_height = np.linalg.norm(ordered_src[0] - ordered_src[3])
        right_height = np.linalg.norm(ordered_src[2] - ordered_src[1])

        # 使用平均值作为实际检测到的宽高
        detected_width = (top_width + bottom_width) / 2
        detected_height = (left_height + right_height) / 2

        # 关键修改：强制使用A4比例，但保持检测到的尺寸特征
        # 方案：使用检测到的较长边作为参考，按A4比例计算另一边

        if detected_width >= detected_height:
            # 检测到的形状偏向横向，以宽度为准
            target_width = int(detected_width)
            target_height = int(detected_width * a4_ratio)  # 强制A4比例
        else:
            # 检测到的形状偏向纵向，以高度为准
            target_height = int(detected_height)
            target_width = int(detected_height / a4_ratio)  # 强制A4比例

        # 移除最小尺寸限制 - 完全自适应

        # 目标点（矩形），强制A4比例
        dst_pts = np.array(
            [
                [0, 0],  # 左上
                [target_width - 1, 0],  # 右上
                [target_width - 1, target_height - 1],  # 右下
                [0, target_height - 1],  # 左下
            ],
            dtype=np.float32,
        )

        try:
            # 计算透视变换矩阵
            matrix = cv2.getPerspectiveTransform(ordered_src, dst_pts)

            # 应用透视变换
            warped = cv2.warpPerspective(img, matrix, (target_width, target_height))

            crops.append(warped)

            # 调试信息
            # print(f"Crop {len(crops)-1}: 检测尺寸({detected_width:.1f}x{detected_height:.1f}) -> 目标尺寸({target_width}x{target_height}) 比例:{target_height/target_width:.3f}")

        except Exception as e:
            print(f"透视变换失败: {e}")
            continue

    return crops


def draw_additional_annotations(
    annotated_frame: np.ndarray,
    hollow_rectangles: List[Dict[str, Any]],
    crops: List[np.ndarray],
    debug_info: Dict[str, Any],
) -> None:
    """在标注图像上绘制额外的信息：顶点、边长、统计信息等"""
    # 在每个轮廓上绘制四个顶点和边长信息
    for i, rect in enumerate(hollow_rectangles):
        contour = rect["contour"]
        if len(contour) >= 4:
            # 获取四个主要顶点
            src_pts = contour.reshape(-1, 2).astype(np.float32)
            if len(src_pts) > 4:
                hull = cv2.convexHull(src_pts.astype(np.int32))
                epsilon = 0.02 * cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, epsilon, True)
                if len(approx) >= 4:
                    src_pts = approx[:4].reshape(-1, 2).astype(np.float32)
                else:
                    src_pts = src_pts[:4]

            # 计算边长
            edge1, edge2, edge3, edge4 = calculate_edge_lengths(src_pts)
            horizontal_avg = (edge1 + edge3) / 2
            vertical_avg = (edge2 + edge4) / 2

            # 绘制四个顶点
            ordered_pts = order_points(src_pts)
            colors = [
                (255, 0, 0),
                (0, 255, 0),
                (0, 0, 255),
                (255, 255, 0),
            ]  # 蓝、绿、红、黄
            labels = ["TL", "TR", "BR", "BL"]

            for j, (pt, color, label) in enumerate(zip(ordered_pts, colors, labels)):
                cv2.circle(annotated_frame, (int(pt[0]), int(pt[1])), 8, color, -1)
                cv2.putText(
                    annotated_frame,
                    label,
                    (int(pt[0]) + 10, int(pt[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            # 显示边长信息
            x, y, w, h = rect["bbox"]
            edge_info = f"H_avg:{horizontal_avg:.0f} V_avg:{vertical_avg:.0f}"
            cv2.putText(
                annotated_frame,
                edge_info,
                (x, y + h + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

    # 显示检测到的空心矩形数量和调试信息
    info_text = f"Detected: {len(hollow_rectangles)} Crops: {len(crops)}"
    cv2.putText(
        annotated_frame,
        info_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    debug_text = f"Total Contours: {len(debug_info.get('all_contours', []))}"
    cv2.putText(
        annotated_frame,
        debug_text,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )


def detect_and_annotate(
    img: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    global watershed_img, current_crops, raw_crops
    stats = {
        "count": 0,
        "rects": [],
        "area_sum": 0,
        "inner_rectangles": [],
        "inner_count": 0,
        "crops_count": 0,
        "black_squares": [],
        "minimum_black_square": {"found": False},
    }

    # 图像预处理
    processed, debug_info = preprocess_image(img)

    mask = processed.copy()

    # 检测空心矩形
    hollow_rectangles = find_hollow_rectangles(processed, debug_info)

    # 生成A4比例的裁剪图像
    crops = create_a4_crops_from_contours(img, hollow_rectangles)

    # 更新全局crops变量
    current_crops = crops
    raw_crops = [crop.copy() for crop in crops]

    # 绘制标注
    annotated_frame = draw_annotations(img, hollow_rectangles)

    # 绘制额外的标注信息（顶点、边长、统计信息等）
    draw_additional_annotations(annotated_frame, hollow_rectangles, crops, debug_info)

    # 更新统计信息
    stats["count"] = len(hollow_rectangles)
    stats["inner_count"] = len(hollow_rectangles)
    stats["crops_count"] = len(crops)

    # 将hollow_rectangles转换为适合stats的格式
    for i, rect in enumerate(hollow_rectangles):
        x, y, w, h = rect["bbox"]
        area = rect["area"]

        # 计算边长信息
        contour_points = rect["contour"].reshape(-1, 2).astype(np.float32)
        if len(contour_points) >= 4:
            edge1, edge2, edge3, edge4 = calculate_edge_lengths(contour_points[:4])
            horizontal_avg = (edge1 + edge3) / 2
            vertical_avg = (edge2 + edge4) / 2

            # 计算新长度（短边中点连线长度）
            ordered_pts = order_points(contour_points[:4])
            if horizontal_avg > vertical_avg:
                # 水平方向是长边，短边是垂直方向，计算左右边中点
                left_mid = (
                    (ordered_pts[0][0] + ordered_pts[3][0]) / 2,
                    (ordered_pts[0][1] + ordered_pts[3][1]) / 2,
                )
                right_mid = (
                    (ordered_pts[1][0] + ordered_pts[2][0]) / 2,
                    (ordered_pts[1][1] + ordered_pts[2][1]) / 2,
                )
                new_length = dist(left_mid, right_mid)
            else:
                # 垂直方向是长边，短边是水平方向，计算上下边中点
                top_mid = (
                    (ordered_pts[0][0] + ordered_pts[1][0]) / 2,
                    (ordered_pts[0][1] + ordered_pts[1][1]) / 2,
                )
                bottom_mid = (
                    (ordered_pts[3][0] + ordered_pts[2][0]) / 2,
                    (ordered_pts[3][1] + ordered_pts[2][1]) / 2,
                )
                new_length = dist(top_mid, bottom_mid)
        else:
            horizontal_avg = vertical_avg = new_length = 0

        rect_info = {
            "id": i + 1,
            "outer_width": int(w),
            "outer_height": int(h),
            "area": int(area),
            "position": (int(x), int(y)),
            "aspect_ratio": float(rect.get("aspect_ratio", 0)),
            "horizontal_avg": float(horizontal_avg),
            "vertical_avg": float(vertical_avg),
            "new_long_px": float(new_length),  # 新长度
            "crop_width": crops[i].shape[1] if i < len(crops) else 0,
            "crop_height": crops[i].shape[0] if i < len(crops) else 0,
        }
        stats["rects"].append(rect_info)

        # 同时添加到inner_rectangles
        inner_info = {
            "id": i + 1,
            "bbox": (x, y, w, h),
            "area": int(area),
            "aspect_ratio": float(rect.get("aspect_ratio", 0)),
            "center": [x + w // 2, y + h // 2],
            "width": int(w),
            "height": int(h),
            "horizontal_avg": float(horizontal_avg),
            "vertical_avg": float(vertical_avg),
            "new_long_px": float(new_length),
            "crop_generated": i < len(crops),
        }
        stats["inner_rectangles"].append(inner_info)

    # 内部标注，使用裁剪后的图像
    annotate_inner(crops, params, stats)
    # min_area_box(crops, stats, img)
    neo_crops = [crop.copy() for crop in raw_crops]
    min_area_box_compatible(neo_crops, stats, img)

    return annotated_frame, mask, stats


def _judge_corner_type(
    bin_img: np.ndarray, corner: Tuple[int, int], radius: int = 30
) -> bool:
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
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def process_crop_for_minimum_square(
    crop_img: np.ndarray,
    min_area_threshold: int = 500,
    physical_width_mm: float = 170.0,  # A4纸宽度210mm - 40mm边框 = 170mm
    physical_height_mm: float = 257.0,  # A4纸高度297mm - 40mm边框 = 257mm
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
            "annotated_image": None,
        }

    # 1. 二值化处理
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    _, binary_inv = cv2.threshold(
        gray, int(gray.mean()), 255, cv2.THRESH_BINARY_INV
    )  # 目标为白色
    h, w = binary_inv.shape

    # 去除背景（从四个角开始填充）
    for seed in [(0, 0), (w - 20, 0), (0, h - 20), (w - 20, h - 20)]:
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
    num_labels_filtered, labels_filtered, stats_filtered, _ = (
        cv2.connectedComponentsWithStats(filtered_binary)
    )

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
            "annotated_image": crop_img.copy(),
        }

    # 准备绘图和结果收集
    result_img = crop_img.copy()
    valid_edges = []  # 存储有效线段：[(起点, 终点, 长度, 连通域ID), ...]

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
        corner_types = [
            _judge_corner_type(filtered_binary, corner) for corner in corners
        ]

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
            "annotated_image": result_img,
        }

    shortest_edge = min(valid_edges, key=lambda x: x[2])
    pt1, pt2, shortest_length, comp_id = shortest_edge

    # 用红线标注最短有效线段
    cv2.line(result_img, pt1, pt2, (0, 0, 255), 3)
    mid_pt = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
    cv2.putText(
        result_img,
        f"MIN：{shortest_length:.1f}px",
        mid_pt,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )

    # 计算物理长度（毫米）
    diameter_mm = shortest_length * mm_per_pixel

    # 调试：显示详细的检测信息
    # print(f"[DEBUG CROP] 检测到最短边: {shortest_length:.2f}px -> {diameter_mm:.2f}mm (转换比例: {mm_per_pixel:.4f}mm/px)")
    # print(f"[DEBUG CROP] Crop尺寸: {crop_width}x{crop_height}px, 物理尺寸: {physical_width_mm}x{physical_height_mm}mm")

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
        "filtered_components": num_labels_filtered - 1,
    }


def process_multiple_crops(
    crops: List[np.ndarray],
    min_area_threshold: int = 10,
    physical_width_mm: float = 170.0,  # A4纸宽度210mm - 40mm边框 = 170mm
    physical_height_mm: float = 257.0,  # A4纸高度297mm - 40mm边框 = 257mm
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
        # print(f"处理第 {i+1}/{len(crops)} 个裁剪图像...")
        result = process_crop_for_minimum_square(
            crop,
            min_area_threshold=min_area_threshold,
            physical_width_mm=physical_width_mm,
            physical_height_mm=physical_height_mm,
        )
        result["crop_index"] = i
        results.append(result)

        if result["success"]:
            # print(f"  ✓ 最短边长: {result['shortest_edge_length_px']:.2f}px = {result['shortest_edge_length_mm']:.2f}mm")
            pass
        else:
            print(f"  ✗ 处理失败: {result['error']}")

    return results


def find_global_minimum_square(
    results: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    从所有处理结果中找到全局最小的边

    Args:
        results: 所有crop的处理结果

    Returns:
        全局最小边的信息，如果没有则返回None
    """
    successful_results = [r for r in results if r["success"]]

    if not successful_results:
        return None

    # 按像素长度排序，找到最小的
    min_result = min(successful_results, key=lambda x: x["shortest_edge_length_px"])
    min_result["is_global_minimum"] = True

    return min_result


def _convert_detection_result_to_legacy_format(
    result: Dict[str, Any], crop_center: Tuple[float, float] = (0, 0)
) -> Optional[Dict[str, Any]]:
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
    area_px = side_length_px**2

    # 创建一个虚拟的box（基于中心点和边长）
    half_side = side_length_px / 2
    box = np.array(
        [
            [center_x - half_side, center_y - half_side],
            [center_x + half_side, center_y - half_side],
            [center_x + half_side, center_y + half_side],
            [center_x - half_side, center_y + half_side],
        ],
        dtype=np.int32,
    )

    return {
        "center": (center_x, center_y),
        "area": area_px,
        "side_length": side_length_px,
        "side_length_mm": side_length_mm,
        "aspect_ratio": 1.0,  # 假设是正方形
        "type": "minimum_square_detected",
        "box": box,
        "is_minimum": True,
    }


def min_area_box_compatible(
    crops: List[np.ndarray],
    stats: Dict[str, Any],
    img: np.ndarray,
    min_area_threshold: int = 500,
    physical_width_mm: float = 170.0,
    physical_height_mm: float = 257.0,
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
        physical_height_mm=physical_height_mm,
    )

    # 调试输出：显示所有检测结果
    # print(f"[DEBUG] 共处理 {len(crops)} 个crops:")
    # for i, result in enumerate(all_results):
    #     if result["success"]:
    #         print(f"  Crop {i}: {result['shortest_edge_length_px']:.2f}px = {result['shortest_edge_length_mm']:.2f}mm")
    #     else:
    #         print(f"  Crop {i}: 检测失败 - {result['error']}")

    # 找到全局最小边
    global_min_result = find_global_minimum_square(all_results)

    # if global_min_result:
    #     print(f"[DEBUG] 全局最小来自 Crop {global_min_result.get('crop_index')} : {global_min_result['shortest_edge_length_px']:.2f}px = {global_min_result['shortest_edge_length_mm']:.2f}mm")

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

    # 在原图上绘制全局最小边（如果找到的话）
    if global_min_result and global_min_result["success"]:
        # 调试输出
        # print(f"[DEBUG] 新算法检测到最小边: {global_min_result['shortest_edge_length_px']:.2f}px = {global_min_result['shortest_edge_length_mm']:.2f}mm")

        # 直接使用最小边的信息，不转换为正方形格式
        edge_length_px = global_min_result["shortest_edge_length_px"]
        edge_length_mm = global_min_result["shortest_edge_length_mm"]
        start_pt = global_min_result["start_point"]
        end_pt = global_min_result["end_point"]

        # 计算边的中心点用于绘制
        center_x = (start_pt[0] + end_pt[0]) / 2
        center_y = (start_pt[1] + end_pt[1]) / 2

        # 绘制最小边
        _draw_minimum_edge_compatible(
            img, center_x, center_y, edge_length_px, edge_length_mm
        )

        # 更新统计信息（最小边信息）
        stats["minimum_black_square"] = {
            "found": True,
            "center": [float(center_x), float(center_y)],
            "edge_length_px": float(edge_length_px),
            "edge_length_mm": float(edge_length_mm),
            "start_point": [float(start_pt[0]), float(start_pt[1])],
            "end_point": [float(end_pt[0]), float(end_pt[1])],
            "type": "minimum_edge_detected",
        }

        # 调试：显示返回给API的数据
        # print(f"[DEBUG] API返回数据: edge_length={edge_length_px:.2f}px, edge_length_mm={edge_length_mm:.2f}mm")

        # 兼容字段 - 保持原有的格式但使用边长信息
        stats["black_squares"] = [
            {
                "center": [float(center_x), float(center_y)],
                "edge_length_px": float(edge_length_px),
                "edge_length_mm": float(edge_length_mm),
                "is_minimum": True,
                "type": "minimum_edge_detected",
            }
        ]
    else:
        stats["minimum_black_square"] = {"found": False}
        stats["black_squares"] = []


def _draw_minimum_edge_compatible(
    img: np.ndarray,
    center_x: float,
    center_y: float,
    edge_length_px: float,
    edge_length_mm: float,
) -> None:
    """
    绘制最小边检测结果
    """
    # 绘制中心点
    cv2.circle(img, (int(center_x), int(center_y)), 12, (0, 255, 255), -1)

    # 添加最小边长标记
    cv2.putText(
        img,
        f"MIN EDGE:{int(edge_length_px)}px",
        (int(center_x) - 60, int(center_y) - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )

    # 添加毫米信息
    cv2.putText(
        img,
        f"{edge_length_mm:.1f}mm",
        (int(center_x) - 30, int(center_y) + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        2,
    )


def _draw_minimum_square_compatible(
    img: np.ndarray, min_square: Dict[str, Any]
) -> None:
    """
    以与原 draw_minimum_square 兼容的方式绘制最小边检测结果
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

    # 添加最小边长标记
    side_length = min_square.get("side_length", 0)
    side_length_mm = min_square.get("side_length_mm", 0)

    cv2.putText(
        img,
        f"MIN EDGE:{int(side_length)}px",
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


def min_area_box(
    crops: List[np.ndarray],
    stats: Dict[str, Any],
    img: np.ndarray,
) -> None:
    global min_square_images
    min_square_images = []  # 重置列表

    for idx, crop in enumerate(crops):
        # 检测黑色正方形
        all_black_squares, black_mask, watershed_img = detect_black_squares_hsv(crop)

        # # 过滤出在A4框内的黑色正方形
        # filtered_black_squares = filter_black_squares_in_a4_frames(all_black_squares, quads)

        # 找到最小的黑色正方形
        minimum_square = find_minimum_square(all_black_squares)

        # 在图像上绘制最小的黑色正方形
        draw_minimum_square(img, minimum_square)

        # 保存当前crop的处理图像（包含分水岭和检测结果）
        min_square_images.append(
            watershed_img.copy() if watershed_img is not None else crop.copy()
        )

        # 将最小黑色正方形信息添加到统计数据中
        if minimum_square:
            stats["minimum_black_square"] = {
                "found": True,
                "center": [
                    float(minimum_square["center"][0]),
                    float(minimum_square["center"][1]),
                ],
                "area": float(minimum_square["area"]),
                "side_length": float(minimum_square["side_length"]),
                "aspect_ratio": float(minimum_square["aspect_ratio"]),
                "type": minimum_square["type"],
            }
            # 为了兼容，也保留black_squares字段
            stats["black_squares"] = [
                {
                    "center": [
                        float(minimum_square["center"][0]),
                        float(minimum_square["center"][1]),
                    ],
                    "area": float(minimum_square["area"]),
                    "side_length": float(minimum_square["side_length"]),
                    "aspect_ratio": float(minimum_square["aspect_ratio"]),
                    "is_minimum": True,
                }
            ]
        else:
            stats["minimum_black_square"] = {"found": False}
            stats["black_squares"] = []


# 分水岭算法
def detect_black_squares_hsv(
    image: np.ndarray, debug_image: Optional[np.ndarray] = None
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    """
    使用HSV检测黑色正方形 - 简化版本，支持重叠正方形分离
    """
    if debug_image is None:
        debug_image = image.copy()

    # 转换为HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 黑色的HSV范围（更宽松的参数）
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 80])  # 提高V值上限

    # 创建掩码
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # 形态学操作
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 分离重叠的正方形
    separated_squares, watershed_debug = separate_overlapping_squares_simple(
        mask, debug_image
    )

    return separated_squares, mask, watershed_debug


def separate_overlapping_squares_simple(
    mask: np.ndarray, debug_image: np.ndarray
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    使用距离变换和分水岭算法分离重叠的正方形 - 简化版
    """
    # 距离变换
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # 找到局部最大值作为种子点
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 标记连通组件
    _, markers = cv2.connectedComponents(sure_fg)

    # 应用分水岭算法
    # 需要3通道图像用于分水岭算法
    watershed_input = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(watershed_input, markers)

    # 创建分水岭调试图像
    watershed_debug = np.zeros_like(debug_image)
    # 显示分水岭边界
    watershed_debug[markers == -1] = [0, 0, 255]  # 边界为红色

    # 为不同区域分配不同颜色
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]
    for i in range(1, markers.max() + 1):
        color = colors[(i - 1) % len(colors)]
        watershed_debug[markers == i] = color

    black_squares = []

    # 为每个分离的区域检测正方形
    for marker_id in range(1, markers.max() + 1):
        # 创建当前区域的掩码
        region_mask = (markers == marker_id).astype(np.uint8) * 255

        # 在当前区域查找轮廓
        contours, _ = cv2.findContours(
            region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)

            # 过滤小面积
            if area < 400:
                continue

            # 计算最小外接矩形
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            width, height = rect[1]
            if width > 0 and height > 0:
                aspect_ratio = max(width, height) / min(width, height)

                # 检查是否接近正方形
                if aspect_ratio < 2.5:
                    black_squares.append(
                        {
                            "type": "black_square",
                            "contour": contour,
                            "box": box,
                            "area": area,
                            "aspect_ratio": aspect_ratio,
                            "center": rect[0],
                            "side_length": math.sqrt(area),
                        }
                    )

                    # 绘制分离后的黑色正方形
                    cv2.drawContours(debug_image, [box], -1, (255, 0, 255), 3)
                    cv2.circle(
                        debug_image,
                        (int(rect[0][0]), int(rect[0][1])),
                        8,
                        (255, 255, 0),
                        -1,
                    )
                    cv2.putText(
                        debug_image,
                        f"SEP-{marker_id}:{int(math.sqrt(area))}",
                        (int(rect[0][0]) - 40, int(rect[0][1]) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 0, 255),
                        2,
                    )

    # 如果分水岭算法没有找到多个区域，回退到原始方法
    if len(black_squares) == 0:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < 400:
                continue

            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            width, height = rect[1]
            if width > 0 and height > 0:
                aspect_ratio = max(width, height) / min(width, height)

                if aspect_ratio < 2.5:
                    black_squares.append(
                        {
                            "type": "black_square",
                            "contour": contour,
                            "box": box,
                            "area": area,
                            "aspect_ratio": aspect_ratio,
                            "center": rect[0],
                            "side_length": math.sqrt(area),
                        }
                    )

                    cv2.drawContours(debug_image, [box], -1, (255, 0, 255), 3)
                    cv2.circle(
                        debug_image,
                        (int(rect[0][0]), int(rect[0][1])),
                        8,
                        (255, 255, 0),
                        -1,
                    )
                    cv2.putText(
                        debug_image,
                        f"ORIG:{int(math.sqrt(area))}",
                        (int(rect[0][0]) - 30, int(rect[0][1]) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 0, 255),
                        2,
                    )

    return black_squares, watershed_debug


def filter_black_squares_in_a4_frames(
    black_squares: List[Dict[str, Any]], quads: List[np.ndarray]
) -> List[Dict[str, Any]]:
    """
    过滤出在A4框内的黑色正方形
    """
    if not quads or not black_squares:
        return []

    filtered_squares = []

    for square in black_squares:
        center = square["center"]

        # 检查每个A4框
        for quad in quads:
            # 将四边形转换为轮廓格式
            quad_contour = quad.reshape(-1, 1, 2)

            # 检查黑色正方形的中心是否在A4框内
            if cv2.pointPolygonTest(quad_contour, center, False) >= 0:
                # 添加所属A4框信息
                square["parent_quad"] = quad.tolist()
                filtered_squares.append(square)
                break  # 找到一个包含的框就够了

    return filtered_squares


def find_minimum_square(
    filtered_squares: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    从过滤后的黑色正方形中找到最小的一个
    返回最小正方形的信息，如果没有则返回None
    """
    if not filtered_squares:
        return None

    # 按面积排序，选择最小的
    min_square = min(filtered_squares, key=lambda x: x["area"])

    # 添加最小标记
    min_square["is_minimum"] = True

    return min_square


def draw_minimum_square(img: np.ndarray, min_square: Optional[Dict[str, Any]]) -> None:
    """
    在图像上绘制最小的黑色正方形，使用特殊标记
    """
    if min_square is None:
        return

    box = min_square["box"]
    center = min_square["center"]

    # 绘制最小黑色正方形边框（青色，更粗的线条）
    cv2.drawContours(img, [box], -1, (255, 255, 0), 4)
    cv2.circle(img, (int(center[0]), int(center[1])), 12, (0, 255, 255), -1)

    # 添加最小正方形标记
    cv2.putText(
        img,
        f"MIN BLACK:{int(min_square['side_length'])}",
        (int(center[0]) - 60, int(center[1]) - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )

    # 添加面积信息
    cv2.putText(
        img,
        f"Area:{int(min_square['area'])}",
        (int(center[0]) - 40, int(center[1]) + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 255),
        2,
    )

    # 添加箭头指向最小正方形
    arrow_end = (int(center[0]), int(center[1]) - 50)
    cv2.arrowedLine(
        img,
        arrow_end,
        (int(center[0]), int(center[1]) - 15),
        (0, 255, 255),
        3,
        tipLength=0.3,
    )
    cv2.putText(
        img,
        "MINIMUM",
        (int(center[0]) - 35, int(center[1]) - 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        2,
    )


def extract_minimum_square_side_length(min_square: Optional[Dict[str, Any]]) -> int:
    """
    提取最小正方形的边长
    返回边长数值，如果没有找到正方形则返回0
    """
    if min_square is None:
        return 0

    return int(min_square["side_length"])


# 捕获循环
prev_time: float = time.time()


def _capture_loop() -> None:
    global frame, latest_stats, prev_time, vis, mask, shutdown_event, cap_o
    while not shutdown_event.is_set():
        if not cap_o:
            time.sleep(1)
            continue
        try:
            ret, img = cap.read()
            if not ret:
                if shutdown_event.is_set():
                    break
                time.sleep(0.01)
                continue

            with lock:
                frame = img.copy()
            vis, mask, stats = detect_and_annotate(img.copy())
            total_pixels = img.shape[0] * img.shape[1]
            black_pixels = int((mask == 255).sum())
            now = time.time()
            fps = 1 / (now - prev_time) if now != prev_time else 0
            prev_time = now
            frame_ratio = stats["area_sum"] / total_pixels * 100
            black_ratio = black_pixels / total_pixels * 100

            # 提取最小正方形边长
            min_square_info = stats.get("minimum_black_square", {"found": False})
            min_side_length = (
                int(min_square_info.get("side_length", 0))
                if min_square_info.get("found", False)
                else 0
            )

            latest_stats = {
                "count": stats["count"],
                "total_pixels": int(total_pixels),
                "frame_ratio": int(frame_ratio),
                "black_ratio": int(black_ratio),
                "fps": int(fps),
                "rects": stats["rects"],
                # 新增内框检测相关统计
                "inner_rectangles": stats.get("inner_rectangles", []),
                "inner_count": stats.get("inner_count", 0),
                "inner_total_area": stats.get("area_sum", 0),
                "crops_count": stats.get("crops_count", 0),  # 裁剪区域数量
                # 保持原有黑色正方形相关统计
                "black_squares": stats.get("black_squares", []),
                "black_squares_count": len(stats.get("black_squares", [])),
                "minimum_black_square": stats.get(
                    "minimum_black_square", {"found": False}
                ),
                "minimum_side_length": min_side_length,
            }

            # 检查是否需要停止
            if shutdown_event.wait(0.03):
                break

        except Exception as e:
            if not shutdown_event.is_set():
                print(f"摄像头捕获错误: {e}")
                time.sleep(0.1)
            break

    print("摄像头捕获线程已停止")


capture_thread = threading.Thread(target=_capture_loop, daemon=True)
capture_thread.start()


# MJPEG
async def broadcast_loop() -> None:
    global shutdown_event
    while not shutdown_event.is_set():
        try:
            status = latest_stats.copy()
            # new = []
            if "rects" in status.keys():
                for item in status["rects"]:
                    try:
                        item.pop("outer_pts")
                        item.pop("midpoints")
                        item.pop("inner_contour")
                    # new.append(item)
                    except Exception:
                        pass

            text = json.dumps(status)
            bad = []
            for ws in clients:
                try:
                    await ws.send_text(text)
                except WebSocketDisconnect:
                    bad.append(ws)
                except Exception as e:
                    if not shutdown_event.is_set():
                        print(f"WebSocket发送错误: {e}")
                    bad.append(ws)
            for ws in bad:
                clients.remove(ws)

            # 使用asyncio.sleep并检查停止事件
            try:
                await asyncio.wait_for(asyncio.sleep(0.1), timeout=0.1)
            except asyncio.TimeoutError:
                pass

        except Exception as e:
            if not shutdown_event.is_set():
                print(f"广播循环错误: {e}")
            break

    print("WebSocket广播循环已停止")


def mjpeg_generator(proc: str) -> Generator[bytes, None, None]:
    global watershed_img, mask, vis, shutdown_event
    while not shutdown_event.is_set():
        with lock:
            img = frame.copy() if frame is not None else None
        if img is None:
            if shutdown_event.wait(0.01):
                break
            continue
        try:
            if proc == "vis":
                tgt = vis
            elif proc == "mask":
                tgt = mask
            elif proc == "watershed_img":
                if watershed_img is not None:
                    tgt = watershed_img
                else:
                    if shutdown_event.wait(0.03):
                        break
                    continue
            else:
                if shutdown_event.wait(0.03):
                    break
                continue

            if tgt is not None:
                _, buf = cv2.imencode(".jpg", tgt)
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + buf.tobytes()
                    + b"\r\n"
                )

        except Exception as e:
            if not shutdown_event.is_set():
                print(f"MJPEG生成错误: {e}")
            break

        if shutdown_event.wait(0.03):
            break

    print(f"MJPEG生成器 ({proc}) 已停止")


# routes


@app.get("/ocr", tags=["OCR识别"])
async def detect_numbers():
    """
    OCR文字识别接口

    返回每个裁剪区域中识别到的文字信息
    """
    global raw_crops, cap_o
    cap_o = False
    ocr_result = dict()
    if raw_crops and params["enable_ocr"]:
        for idx, image in enumerate(raw_crops):
            ocr_result[idx] = run_ocr_on_frame(image)
    cap_o = True
    return ocr_result


# ------------------------------------------------------------------------------#
# INA226 功率监控路由节点
# ------------------------------------------------------------------------------#


@app.get("/api/ina226/measurements", tags=["功率监控"])
async def get_ina226_status_api():
    """
    获取INA226传感器状态信息
    """
    global ina
    return {
        'data': ina.get_all_measurements()
    }


# ------------------------------------------------------------------------------#


@app.get("/api/ocr_measurement_analysis", tags=["OCR识别"])
async def get_ocr_measurement_analysis():
    """
    结合OCR和物理测量的分析接口
    """
    global latest_stats, raw_crops, cap_o
    start_time = time.time()  # 开始计时
    cap_o = False

    # 1. 获取物理测量数据
    measurement_response = await get_physical_measurements()
    if not measurement_response["success"]:
        return {
            "success": False,
            "error": "Failed to get physical measurements",
            "analysis": [],
        }

    # 2. 获取OCR结果
    ocr_results = {}
    if raw_crops and params["enable_ocr"]:
        for idx, image in enumerate(raw_crops):
            ocr_results[idx] = run_ocr_on_frame(image)

    # 3. 合并结果
    analysis = []
    for measurement in measurement_response["measurements"]:
        crop_idx = measurement["crop_index"]

        # 复制测量数据的基本结构
        crop_analysis = {
            "crop_index": crop_idx,
            "target": measurement["target"],
            "shapes": [],
            "ocr_raw_data": ocr_results.get(crop_idx, []),  # 添加原始OCR数据
        }

        # 处理每个形状
        for shape in measurement["shapes"]:
            # 获取OCR数据
            ocr_data = {"detected": False, "text": "", "confidence": 0.0, "bbox": []}

            # 如果有OCR结果，尝试匹配
            if crop_idx in ocr_results:
                shape_bbox = shape.get("pixel_dimensions", {})
                shape_center = shape.get("position", {}).get("center", [0, 0])

                # 如果形状中心点为[0,0]，尝试从其他数据源获取中心点
                if shape_center == [0, 0]:
                    # 尝试从bbox计算中心点
                    bbox = shape.get("position", {}).get("bbox", [0, 0, 0, 0])
                    if bbox != [0, 0, 0, 0] and len(bbox) == 4:
                        shape_center = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]
                    else:
                        # 如果还是没有，跳过匹配
                        shape_center = [0, 0]

                # 如果形状中心点不是[0,0]，进行位置匹配
                if shape_center != [0, 0]:
                    shape_width_half = shape_bbox.get("width", 0) / 2
                    shape_height_half = shape_bbox.get("height", 0) / 2

                    # 寻找最佳匹配的OCR结果
                    for ocr_result in ocr_results[crop_idx]:
                        ocr_bbox = ocr_result["bbox"]
                        ocr_center = [
                            (ocr_bbox[0][0] + ocr_bbox[2][0]) / 2,
                            (ocr_bbox[0][1] + ocr_bbox[2][1]) / 2,
                        ]

                        distance_x = abs(ocr_center[0] - shape_center[0])
                        distance_y = abs(ocr_center[1] - shape_center[1])

                        # 检查是否在形状范围内
                        if (
                            distance_x < shape_width_half
                            and distance_y < shape_height_half
                        ):
                            ocr_data = {
                                "detected": True,
                                "text": ocr_result["text"],
                                "confidence": ocr_result["conf"],
                                "bbox": ocr_result["bbox"],
                            }
                            break  # 找到匹配后立即退出循环

            # 组合形状数据和OCR结果
            if "position" not in shape:
                shape["position"] = {}
            shape["position"]["contour_points"] = shape.get("contour_points", [])

            shape_analysis = {
                **shape,  # 保留所有原有的形状数据
                "ocr_data": ocr_data,
            }

            crop_analysis["shapes"].append(shape_analysis)

        analysis.append(crop_analysis)

    elapsed_time = time.time() - start_time  # 计算耗时
    cap_o = True
    return {
        "success": True,
        "analysis": analysis,
        "total_crops": len(analysis),
        "references": measurement_response.get("a4_reference", {}),
        "elapsed_seconds": round(elapsed_time, 3),  # 添加耗时信息，保留3位小数
    }


@app.get("/api/ocr_masked_analysis", tags=["OCR识别"])
async def get_ocr_masked_analysis():
    """
    对mask处理过的图像进行OCR识别
    """
    global raw_crops, cap_o
    start_time = time.time()  # 开始计时
    cap_o = False

    # 检查是否有crops数据
    if not raw_crops:
        return {
            "success": False,
            "error": "No crops data available",
            "ocr_results": {},
        }

    # 对每个crop应用mask处理并进行OCR
    ocr_results = {}
    
    if params["enable_ocr"]:
        for idx, crop in enumerate(raw_crops):
            # 应用mask处理
            masked_crop = apply_mask_to_crop(crop)
            
            # 统一缩放到840*1118
            resized_crop = cv2.resize(masked_crop, (840, 1118))
            
            # 进行OCR识别
            ocr_results[idx] = run_ocr_on_frame(resized_crop)

    elapsed_time = time.time() - start_time  # 计算耗时
    cap_o = True
    return {
        "success": True,
        "ocr_results": ocr_results,
        "total_crops": len(ocr_results),
        "elapsed_seconds": round(elapsed_time, 3),  # 添加耗时信息，保留3位小数
    }


@app.get("/api/ocr_masked_measurement_analysis", tags=["OCR识别"])
async def get_ocr_masked_measurement_analysis():
    """
    结合OCR和物理测量的分析接口 - 使用mask处理过的图像
    """
    global latest_stats, raw_crops, cap_o
    start_time = time.time()  # 开始计时
    cap_o = False

    # 1. 获取物理测量数据
    measurement_response = await get_physical_measurements()
    if not measurement_response["success"]:
        return {
            "success": False,
            "error": "Failed to get physical measurements",
            "analysis": [],
        }

    # 2. 获取OCR结果 - 使用mask处理过的图像
    ocr_results = {}
    scale_factors = {}  # 存储每个crop的缩放因子
    
    if raw_crops and params["enable_ocr"]:
        for idx, crop in enumerate(raw_crops):
            # 应用mask处理
            masked_crop = apply_mask_to_crop(crop)
            
            # 记录原始尺寸
            original_height, original_width = masked_crop.shape[:2]
            
            # 统一缩放到840*1118
            resized_crop = cv2.resize(masked_crop, (840, 1118))
            
            # 计算缩放因子
            scale_x = 840 / original_width
            scale_y = 1118 / original_height
            scale_factors[idx] = {"scale_x": scale_x, "scale_y": scale_y}
            
            # 进行OCR识别
            ocr_results[idx] = run_ocr_on_frame(resized_crop)

    # 3. 合并结果
    analysis = []
    for measurement in measurement_response["measurements"]:
        crop_idx = measurement["crop_index"]

        # 复制测量数据的基本结构
        crop_analysis = {
            "crop_index": crop_idx,
            "target": measurement["target"],
            "shapes": [],
            "ocr_raw_data": ocr_results.get(crop_idx, []),  # 添加原始OCR数据
            "scale_factors": scale_factors.get(crop_idx, {"scale_x": 1.0, "scale_y": 1.0}),  # 添加缩放因子
        }

        # 处理每个形状
        for shape in measurement["shapes"]:
            # 获取OCR数据
            ocr_data = {"detected": False, "text": "", "confidence": 0.0, "bbox": []}

            # 如果有OCR结果，尝试匹配
            if crop_idx in ocr_results:
                shape_bbox = shape.get("pixel_dimensions", {})
                shape_center = shape.get("position", {}).get("center", [0, 0])

                # 如果形状中心点为[0,0]，尝试从其他数据源获取中心点
                if shape_center == [0, 0]:
                    # 尝试从bbox计算中心点
                    bbox = shape.get("position", {}).get("bbox", [0, 0, 0, 0])
                    if bbox != [0, 0, 0, 0] and len(bbox) == 4:
                        shape_center = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]
                    else:
                        # 如果还是没有，跳过匹配
                        shape_center = [0, 0]

                # 如果形状中心点不是[0,0]，进行位置匹配
                if shape_center != [0, 0] and crop_idx in scale_factors:
                    # 将原始坐标转换为缩放后的坐标
                    scale_x = scale_factors[crop_idx]["scale_x"]
                    scale_y = scale_factors[crop_idx]["scale_y"]
                    
                    scaled_center_x = shape_center[0] * scale_x
                    scaled_center_y = shape_center[1] * scale_y
                    scaled_width_half = shape_bbox.get("width", 0) * scale_x / 2
                    scaled_height_half = shape_bbox.get("height", 0) * scale_y / 2

                    # 寻找最佳匹配的OCR结果
                    for ocr_result in ocr_results[crop_idx]:
                        ocr_bbox = ocr_result["bbox"]
                        ocr_center = [
                            (ocr_bbox[0][0] + ocr_bbox[2][0]) / 2,
                            (ocr_bbox[0][1] + ocr_bbox[2][1]) / 2,
                        ]

                        distance_x = abs(ocr_center[0] - scaled_center_x)
                        distance_y = abs(ocr_center[1] - scaled_center_y)

                        # 检查是否在形状范围内（使用缩放后的坐标）
                        if (
                            distance_x < scaled_width_half
                            and distance_y < scaled_height_half
                        ):
                            # 将OCR的bbox坐标转换回原始坐标系
                            original_bbox = []
                            for point in ocr_bbox:
                                original_point = [
                                    point[0] / scale_x,
                                    point[1] / scale_y
                                ]
                                original_bbox.append(original_point)
                            
                            ocr_data = {
                                "detected": True,
                                "text": ocr_result["text"],
                                "confidence": ocr_result["conf"],
                                "bbox": original_bbox,  # 转换回原始坐标系的bbox
                                "scaled_bbox": ocr_result["bbox"],  # 保留缩放后的bbox用于调试
                            }
                            break  # 找到匹配后立即退出循环

            # 组合形状数据和OCR结果
            if "position" not in shape:
                shape["position"] = {}
            shape["position"]["contour_points"] = shape.get("contour_points", [])

            shape_analysis = {
                **shape,  # 保留所有原有的形状数据
                "ocr_data": ocr_data,
            }

            crop_analysis["shapes"].append(shape_analysis)

        analysis.append(crop_analysis)

    elapsed_time = time.time() - start_time  # 计算耗时
    cap_o = True
    return {
        "success": True,
        "analysis": analysis,
        "total_crops": len(analysis),
        "references": measurement_response.get("a4_reference", {}),
        "elapsed_seconds": round(elapsed_time, 3),  # 添加耗时信息，保留3位小数
    }


@app.get("/api/ocr_scaled_measurement_analysis", tags=["OCR识别"])
async def get_ocr_scaled_measurement_analysis():
    """
    结合OCR和物理测量的分析接口 - 使用缩放但不mask的图像
    """
    global latest_stats, raw_crops, cap_o
    start_time = time.time()  # 开始计时
    cap_o = False

    # 1. 获取物理测量数据
    measurement_response = await get_physical_measurements()
    if not measurement_response["success"]:
        return {
            "success": False,
            "error": "Failed to get physical measurements",
            "analysis": [],
        }

    # 2. 获取OCR结果 - 使用缩放但不mask的图像
    ocr_results = {}
    scale_factors = {}  # 存储每个crop的缩放因子
    
    if raw_crops and params["enable_ocr"]:
        for idx, crop in enumerate(raw_crops):
            # 记录原始尺寸
            original_height, original_width = crop.shape[:2]
            
            # 统一缩放到840*1118，不应用mask
            resized_crop = cv2.resize(crop, (840, 1118))
            
            # 计算缩放因子
            scale_x = 840 / original_width
            scale_y = 1118 / original_height
            scale_factors[idx] = {"scale_x": scale_x, "scale_y": scale_y}
            
            # 进行OCR识别
            ocr_results[idx] = run_ocr_on_frame(resized_crop)

    # 3. 合并结果
    analysis = []
    for measurement in measurement_response["measurements"]:
        crop_idx = measurement["crop_index"]

        # 复制测量数据的基本结构
        crop_analysis = {
            "crop_index": crop_idx,
            "target": measurement["target"],
            "shapes": [],
            "ocr_raw_data": ocr_results.get(crop_idx, []),  # 添加原始OCR数据
            "scale_factors": scale_factors.get(crop_idx, {"scale_x": 1.0, "scale_y": 1.0}),  # 添加缩放因子
        }

        # 处理每个形状
        for shape in measurement["shapes"]:
            # 获取OCR数据
            ocr_data = {"detected": False, "text": "", "confidence": 0.0, "bbox": []}

            # 如果有OCR结果，尝试匹配
            if crop_idx in ocr_results:
                shape_bbox = shape.get("pixel_dimensions", {})
                shape_center = shape.get("position", {}).get("center", [0, 0])

                # 如果形状中心点为[0,0]，尝试从其他数据源获取中心点
                if shape_center == [0, 0]:
                    # 尝试从bbox计算中心点
                    bbox = shape.get("position", {}).get("bbox", [0, 0, 0, 0])
                    if bbox != [0, 0, 0, 0] and len(bbox) == 4:
                        shape_center = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]
                    else:
                        # 如果还是没有，跳过匹配
                        shape_center = [0, 0]

                # 如果形状中心点不是[0,0]，进行位置匹配
                if shape_center != [0, 0] and crop_idx in scale_factors:
                    # 将原始坐标转换为缩放后的坐标
                    scale_x = scale_factors[crop_idx]["scale_x"]
                    scale_y = scale_factors[crop_idx]["scale_y"]
                    
                    scaled_center_x = shape_center[0] * scale_x
                    scaled_center_y = shape_center[1] * scale_y
                    scaled_width_half = shape_bbox.get("width", 0) * scale_x / 2
                    scaled_height_half = shape_bbox.get("height", 0) * scale_y / 2

                    # 寻找最佳匹配的OCR结果
                    for ocr_result in ocr_results[crop_idx]:
                        ocr_bbox = ocr_result["bbox"]
                        ocr_center = [
                            (ocr_bbox[0][0] + ocr_bbox[2][0]) / 2,
                            (ocr_bbox[0][1] + ocr_bbox[2][1]) / 2,
                        ]

                        distance_x = abs(ocr_center[0] - scaled_center_x)
                        distance_y = abs(ocr_center[1] - scaled_center_y)

                        # 检查是否在形状范围内（使用缩放后的坐标）
                        if (
                            distance_x < scaled_width_half
                            and distance_y < scaled_height_half
                        ):
                            # 将OCR的bbox坐标转换回原始坐标系
                            original_bbox = []
                            for point in ocr_bbox:
                                original_point = [
                                    point[0] / scale_x,
                                    point[1] / scale_y
                                ]
                                original_bbox.append(original_point)
                            
                            ocr_data = {
                                "detected": True,
                                "text": ocr_result["text"],
                                "confidence": ocr_result["conf"],
                                "bbox": original_bbox,  # 转换回原始坐标系的bbox
                                "scaled_bbox": ocr_result["bbox"],  # 保留缩放后的bbox用于调试
                            }
                            break  # 找到匹配后立即退出循环

            # 组合形状数据和OCR结果
            if "position" not in shape:
                shape["position"] = {}
            shape["position"]["contour_points"] = shape.get("contour_points", [])

            shape_analysis = {
                **shape,  # 保留所有原有的形状数据
                "ocr_data": ocr_data,
            }

            crop_analysis["shapes"].append(shape_analysis)

        analysis.append(crop_analysis)

    elapsed_time = time.time() - start_time  # 计算耗时
    cap_o = True
    return {
        "success": True,
        "analysis": analysis,
        "total_crops": len(analysis),
        "references": measurement_response.get("a4_reference", {}),
        "elapsed_seconds": round(elapsed_time, 3),  # 添加耗时信息，保留3位小数
    }


@app.get("/api/scaled_crops/{crop_index}", tags=["图像获取"])
async def get_scaled_crop(crop_index: int):
    """
    获取指定索引的缩放但不mask的crop图像，用于检查OCR数据源
    """
    global raw_crops
    
    if not raw_crops or crop_index < 0 or crop_index >= len(raw_crops):
        raise HTTPException(status_code=404, detail="Crop not found")
    
    crop = raw_crops[crop_index]
    
    # 统一缩放到840*1118，不应用mask
    resized_crop = cv2.resize(crop, (840, 1118))
    
    # 将图像编码为JPEG格式
    _, buffer = cv2.imencode('.jpg', resized_crop)
    
    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg",
        headers={"Content-Disposition": f"inline; filename=scaled_crop_{crop_index}.jpg"}
    )


@app.get("/api/masked_crops/{crop_index}", tags=["图像获取"])
async def get_masked_crop(crop_index: int):
    """
    获取指定索引的mask处理过的crop图像，用于检查OCR数据源
    """
    global raw_crops
    
    if not raw_crops or crop_index < 0 or crop_index >= len(raw_crops):
        raise HTTPException(status_code=404, detail="Crop not found")
    
    crop = raw_crops[crop_index]
    
    # 应用mask处理
    masked_crop = apply_mask_to_crop(crop)
    
    # 统一缩放到840*1118
    resized_crop = cv2.resize(masked_crop, (840, 1118))
    
    # 将图像编码为JPEG格式
    _, buffer = cv2.imencode('.jpg', resized_crop)
    
    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg",
        headers={"Content-Disposition": f"inline; filename=masked_crop_{crop_index}.jpg"}
    )


def apply_mask_to_crop(crop: np.ndarray) -> np.ndarray:
    """
    对crop图像应用mask处理
    """
    # 应用与主处理流程相同的mask处理
    processed, _ = preprocess_image(crop)
    
    # 直接返回预处理后的mask，与Mask Stream保持一致
    return processed


@app.get("/video/processed", tags=["视频流"])
def video_processed() -> StreamingResponse:
    """
    获取处理后的视频流
    """
    return StreamingResponse(
        mjpeg_generator("vis"), media_type="multipart/x-mixed-replace;boundary=frame"
    )


@app.get("/video/mask", tags=["视频流"])
def video_mask() -> StreamingResponse:
    """
    获取掩码视频流
    """
    return StreamingResponse(
        mjpeg_generator("mask"), media_type="multipart/x-mixed-replace;boundary=frame"
    )


@app.get("/video/watershed_img", tags=["视频流"])
def video_watershed_img() -> StreamingResponse:
    """
    获取分水岭处理后的视频流
    """
    return StreamingResponse(
        mjpeg_generator("watershed_img"),
        media_type="multipart/x-mixed-replace;boundary=frame",
    )


@app.post("/control/hsv", tags=["参数控制"])
async def set_hsv(req: Request) -> Dict[str, Any]:
    """
    设置HSV颜色空间参数
    """
    d = await req.json()
    for k in [
        "h1_min",
        "h1_max",
        "s1_min",
        "s1_max",
        "v1_min",
        "v1_max",
        "h2_min",
        "h2_max",
        "s2_min",
        "s2_max",
        "v2_min",
        "v2_max",
    ]:
        if k not in d:
            raise HTTPException(400)
        params[k] = int(d[k])
    params["use_range2"] = bool(d.get("use_range2", 0))
    params["min_area"] = int(d.get("min_area", 200))
    return {"params": params}


@app.post("/control/canny", tags=["参数控制"])
async def set_canny(req: Request) -> Dict[str, Any]:
    """
    设置Canny边缘检测参数
    """
    d = await req.json()
    for k in ["canny_min", "canny_max"]:
        if k not in d:
            raise HTTPException(400)
        params[k] = int(d[k])
    return {"params": params}


@app.post("/control/show_rectangles")
async def toggle_show_rectangles(req: Request) -> Dict[str, bool]:
    """
    控制是否显示所有检测到的矩形（红框标记）
    """
    global show_all_rectangles
    d = await req.json()
    show_all_rectangles = bool(d.get("show", False))
    return {"show_all_rectangles": show_all_rectangles}


@app.get("/control/show_rectangles")
async def get_show_rectangles() -> Dict[str, bool]:
    """
    获取当前红框显示状态
    """
    return {"show_all_rectangles": show_all_rectangles}


@app.get("/api/minimum_square", tags=["形状检测"])
async def get_minimum_square() -> Dict[str, Any]:
    """
    获取当前检测到的最小黑色正方形信息
    """
    return latest_stats.get("minimum_black_square", {"found": False})


@app.get("/api/minimum_square/side_length", tags=["形状检测"])
async def get_minimum_square_side_length() -> Dict[str, int]:
    """
    只返回最小正方形的边长数值
    """
    min_square_info = latest_stats.get("minimum_black_square", {"found": False})
    if min_square_info.get("found", False):
        return {"side_length": int(min_square_info.get("side_length", 0))}
    else:
        return {"side_length": 0}


@app.get("/api/inner_rectangles")
async def get_inner_rectangles() -> Dict[str, Any]:
    """
    获取当前检测到的内框信息
    """
    return {
        "inner_count": latest_stats.get("inner_count", 0),
        "inner_rectangles": latest_stats.get("inner_rectangles", []),
        "inner_total_area": latest_stats.get("inner_total_area", 0),
    }


@app.get("/api/inner_rectangles/count")
async def get_inner_rectangles_count() -> Dict[str, int]:
    """
    只返回检测到的内框数量
    """
    return {"count": latest_stats.get("inner_count", 0)}


@app.get("/api/inner_rectangles/crops")
async def get_inner_crops() -> Dict[str, Any]:
    """获取内框裁剪区域图像数据"""
    with lock:
        crops = current_crops
        crops_data = []

        for i, crop in enumerate(crops):
            # 将裁剪图像转换为base64
            _, buffer = cv2.imencode(".jpg", crop)
            image_base64 = base64.b64encode(buffer).decode("utf-8")

            crops_data.append(
                {"index": i, "shape": crop.shape, "image_base64": image_base64}
            )

    return {
        "crops_count": len(crops_data),
        "crops": crops_data,
        "timestamp": datetime.now().isoformat(),
    }


@app.websocket("/ws")
async def ws_ep(ws: WebSocket) -> None:
    await ws.accept()
    clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        clients.remove(ws)


@app.on_event("startup")
async def startup() -> None:
    print("应用程序启动中...")
    asyncio.create_task(broadcast_loop())


@app.on_event("shutdown")
async def shutdown() -> None:
    print("应用程序关闭中...")
    global shutdown_event, capture_thread, cap

    # 设置停止标志
    shutdown_event.set()

    # 等待摄像头线程结束
    if capture_thread and capture_thread.is_alive():
        print("等待摄像头线程结束...")
        capture_thread.join(timeout=2.0)
        if capture_thread.is_alive():
            print("摄像头线程未在超时时间内结束")

    # 释放摄像头资源
    if cap and cap.isOpened():
        print("释放摄像头资源...")
        cap.release()

    # 关闭所有WebSocket连接
    for ws in clients.copy():
        try:
            await ws.close()
        except Exception:
            pass
    clients.clear()

    print("应用程序已完全关闭")


# @app.get("/", response_class=HTMLResponse)
# def index() -> HTMLResponse:
#     with open("simple_detector.html", "r", encoding="utf-8") as f:
#         return HTMLResponse(f.read())


@app.get("/debug/area", response_class=HTMLResponse)
def area_config() -> HTMLResponse:
    with open("area_filter_control.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/debug/area2", response_class=HTMLResponse)
def area_config2() -> HTMLResponse:
    with open("a4_measurement_control.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/config")
async def get_config() -> Dict[str, Any]:
    """获取当前配置"""
    return {
        "detection_params": params,
        "algorithm_params": detection_params,  # 添加算法参数
        "area_filter_params": area_filter_params,
        "perspective_params": perspective_params,
        "black_detection_params": black_detection_params,
        "camera_params": camera_config,
    }


@app.post("/config/detection")
async def update_detection_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """更新检测参数"""
    global params
    try:
        for key, value in data.items():
            if key in params:
                params[key] = value
        save_config()
        return {
            "success": True,
            "message": "检测参数已更新",
            "detection_params": params,
        }
    except Exception as e:
        return {"success": False, "message": f"更新失败: {str(e)}"}


@app.post("/config/black_detection")
async def update_black_detection_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """更新黑色检测参数"""
    global black_detection_params
    try:
        for key, value in data.items():
            if key in black_detection_params:
                black_detection_params[key] = value
        save_config()
        return {
            "success": True,
            "message": "黑色检测参数已更新",
            "black_detection_params": black_detection_params,
        }
    except Exception as e:
        return {"success": False, "message": f"更新失败: {str(e)}"}


@app.post("/config/detection_params")
async def update_detection_params_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """更新检测算法参数"""
    global detection_params
    try:
        for key, value in data.items():
            if key in detection_params:
                detection_params[key] = value
        # 同时更新配置文件中的对应参数
        config = load_config()
        if "min_vertices" in data:
            config["detection"]["min_vertices"] = data["min_vertices"]
        if "max_vertices" in data:
            config["detection"]["max_vertices"] = data["max_vertices"]
        save_config(config)
        return {
            "success": True,
            "message": "检测算法参数已更新",
            "detection_params": detection_params,
        }
    except Exception as e:
        return {"success": False, "message": f"更新失败: {str(e)}"}


@app.post("/config/area_filter")
async def update_area_filter_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """更新面积过滤参数"""
    global area_filter_params
    try:
        for key, value in data.items():
            if key in area_filter_params:
                area_filter_params[key] = value
        save_config()
        return {
            "success": True,
            "message": "面积过滤参数已更新",
            "area_filter_params": area_filter_params,
        }
    except Exception as e:
        return {"success": False, "message": f"更新失败: {str(e)}"}


@app.post("/config/perspective")
async def update_perspective_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """更新梯形校正参数"""
    global perspective_params
    try:
        for key, value in data.items():
            if key in perspective_params:
                perspective_params[key] = value
        save_config()
        return {
            "success": True,
            "message": "梯形校正参数已更新",
            "perspective_params": perspective_params,
        }
    except Exception as e:
        return {"success": False, "message": f"更新失败: {str(e)}"}


@app.post("/config/camera")
async def update_camera_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """更新摄像头参数"""
    global camera_config
    try:
        for key, value in data.items():
            if key in camera_config:
                camera_config[key] = value
        save_config()
        return {
            "success": True,
            "message": "摄像头参数已更新",
            "camera_params": camera_config,
        }
    except Exception as e:
        return {"success": False, "message": f"更新失败: {str(e)}"}


@app.post("/config/custom_string")
async def save_custom_string(request: Request) -> Dict[str, Any]:
    """
    保存自定义配置到配置文件

    请求体格式：
    {
        "key": "配置项名称",
        "value": "配置项值"
    }

    或者批量更新：
    {
        "configs": [
            {"key": "配置项1", "value": "值1"},
            {"key": "配置项2", "value": "值2"}
        ]
    }
    """
    try:
        data = await request.json()

        # 确保custom_config字典存在
        if "custom_config" not in params:
            params["custom_config"] = {}

        # 处理单个配置项
        if "key" in data and "value" in data:
            key = str(data["key"])
            params["custom_config"][key] = data["value"]

        # 处理批量配置
        elif "configs" in data and isinstance(data["configs"], list):
            for item in data["configs"]:
                if isinstance(item, dict) and "key" in item and "value" in item:
                    key = str(item["key"])
                    params["custom_config"][key] = item["value"]
                else:
                    raise ValueError("Invalid config format in batch update")
        else:
            raise ValueError("Invalid request format")

        # 保存到配置文件
        save_config()

        return {
            "success": True,
            "message": "自定义配置已保存",
            "custom_config": params["custom_config"],
        }
    except Exception as e:
        return {"success": False, "message": f"保存失败: {str(e)}"}


@app.get("/config/custom_string")
async def get_custom_string() -> Dict[str, Any]:
    """
    获取保存的自定义配置

    可以通过query参数指定key获取特定配置项：
    GET /config/custom_string?key=配置项名称

    不指定key则返回所有配置
    """
    try:
        if "custom_config" not in params:
            params["custom_config"] = {}

        return {"success": True, "custom_config": params["custom_config"]}
    except Exception as e:
        return {"success": False, "message": f"获取失败: {str(e)}"}


# 获取裁剪图像的接口
@app.get("/crops")
async def get_crops() -> Dict[str, Any]:
    """获取当前的裁剪图像信息"""
    global current_crops

    if not current_crops:
        return {"crops_count": 0, "crops": []}

    crops_info = []
    for i, crop in enumerate(current_crops):
        height, width = crop.shape[:2]
        crops_info.append(
            {
                "index": i,
                "width": width,
                "height": height,
                "channels": crop.shape[2] if len(crop.shape) > 2 else 1,
            }
        )

    return {"crops_count": len(current_crops), "crops": crops_info}


@app.get("/crop/{crop_index}")
async def get_crop_image(crop_index: int) -> Response:
    """获取指定索引的裁剪图像"""
    global current_crops

    if not current_crops or crop_index < 0 or crop_index >= len(current_crops):
        raise HTTPException(status_code=404, detail="裁剪图像不存在")

    crop = current_crops[crop_index]

    # 编码为JPEG
    _, buffer = cv2.imencode(".jpg", crop)

    return Response(
        content=buffer.tobytes(),
        media_type="image/jpeg",
        headers={"Content-Disposition": f"inline; filename=crop_{crop_index}.jpg"},
    )


@app.get("/rawcrop/{crop_index}")
async def get_rawcrop_image(crop_index: int) -> Response:
    """获取指定索引的裁剪图像"""
    global raw_crops

    if not raw_crops or crop_index < 0 or crop_index >= len(raw_crops):
        raise HTTPException(status_code=404, detail="裁剪图像不存在")

    crop = raw_crops[crop_index]

    # 编码为JPEG
    _, buffer = cv2.imencode(".jpg", crop)

    return Response(
        content=buffer.tobytes(),
        media_type="image/jpeg",
        headers={"Content-Disposition": f"inline; filename=crop_{crop_index}.jpg"},
    )


@app.get("/crop/min/{crop_index}")
async def get_min_square_image(crop_index: int) -> Response:
    """获取指定索引的最小正方形检测图像"""
    global min_square_images

    if not min_square_images or crop_index < 0 or crop_index >= len(min_square_images):
        raise HTTPException(status_code=404, detail="最小正方形检测图像不存在")

    min_image = min_square_images[crop_index]

    # 编码为JPEG
    _, buffer = cv2.imencode(".jpg", min_image)

    return Response(
        content=buffer.tobytes(),
        media_type="image/jpeg",
        headers={
            "Content-Disposition": f"inline; filename=min_square_{crop_index}.jpg"
        },
    )


@app.get("/api/physical_measurements")
async def get_physical_measurements():
    """获取所有形状的物理尺寸信息

    返回格式：
    {
        "success": bool,
        "measurements": [
            {
                "crop_index": int,
                "target": {
                    "id": int,
                    "bbox": [x, y, w, h],
                    "area": int,
                    "aspect_ratio": float,
                    "crop_width": int,
                    "crop_height": int,
                    "position": [x, y],
                    "horizontal_avg": float,
                    "vertical_avg": float,
                    "new_long_px": float
                },
                "shapes": [
                    {
                        "shape_index": int,
                        "shape_type": str,  # "Square", "Circle", "Triangle", etc.
                        "pixel_dimensions": {
                            "width": int, "height": int, "area": int,
                            "side_lengths": [float], "mean_side_length": float, "perimeter": float
                        },
                        "physical_dimensions": {
                            "width_mm": float, "height_mm": float, "area_mm2": float,
                            "diameter_mm": float,  # for circles
                            "side_lengths_mm": [float], "mean_side_length_mm": float, "perimeter_mm": float,
                            "measurement_type": str,  # "side_length", "diameter", "bounding_box", etc.
                            "mm_per_pixel": float
                        }
                    }
                ]
            }
        ]
    }
    """
    global latest_stats

    if not latest_stats or "rects" not in latest_stats:
        return {
            "success": False,
            "error": "No measurement data available",
            "measurements": [],
        }

    measurements = []

    for crop_idx, rect_data in enumerate(latest_stats["rects"]):
        crop_measurements = {
            "crop_index": crop_idx,
            "target": {
                "id": rect_data["id"],
                "bbox": rect_data.get(
                    "bbox",
                    [
                        rect_data["position"][0],
                        rect_data["position"][1],
                        rect_data["outer_width"],
                        rect_data["outer_height"],
                    ],
                ),
                "area": rect_data["area"],
                "aspect_ratio": rect_data["aspect_ratio"],
                "crop_width": rect_data["crop_width"],
                "crop_height": rect_data["crop_height"],
                "position": rect_data["position"],
                "horizontal_avg": rect_data["horizontal_avg"],
                "vertical_avg": rect_data["vertical_avg"],
                "new_long_px": rect_data["new_long_px"],
            },
            "shapes": [],
        }

        # 处理所有形状
        if "all_shapes" in rect_data and rect_data["all_shapes"]:
            for shape_idx, shape_data in enumerate(rect_data["all_shapes"]):
                if "physical_info" in shape_data:
                    physical_info = shape_data["physical_info"]

                    shape_measurement = {
                        "shape_index": shape_idx,
                        "shape_type": shape_data["shape_type"],
                        "pixel_dimensions": {
                            "width": shape_data["width"],
                            "height": shape_data["height"],
                            "area": shape_data["area"],
                            "side_lengths": shape_data.get("side_lengths", []),
                            "mean_side_length": shape_data.get("mean_side_length", 0),
                            "perimeter": shape_data.get("perimeter", 0),
                        },
                        "physical_dimensions": {
                            "width_mm": physical_info["physical_width_mm"],
                            "height_mm": physical_info["physical_height_mm"],
                            "area_mm2": physical_info["physical_area_mm2"],
                            "diameter_mm": physical_info["physical_diameter_mm"],
                            "side_lengths_mm": physical_info[
                                "physical_side_lengths_mm"
                            ],
                            "perimeter_mm": physical_info["physical_perimeter_mm"],
                            "measurement_type": physical_info["measurement_type"],
                            "mm_per_pixel": physical_info["mm_per_pixel"],
                        },
                        "position": shape_data.get(
                            "position",
                            {
                                "center": [0, 0],
                                "bbox": [0, 0, 0, 0],
                                "contour_points": [],
                            },
                        ),
                    }
                    crop_measurements["shapes"].append(shape_measurement)

        # 无论是否有形状，都添加到结果中，但只返回必要的深度信息
        measurements.append(crop_measurements)

    return {
        "success": True,
        "measurements": measurements,
        "total_crops": len(measurements),
        "a4_reference": {
            "physical_width_mm": 170,  # 210 - 40
            "physical_height_mm": 257,  # 297 - 40
            "note": "A4 paper minus 20mm border on each side",
        },
    }


@app.get("/api/physical_measurements/{crop_index}")
async def get_crop_physical_measurements(crop_index: int):
    """获取指定crop的物理尺寸信息"""
    global latest_stats

    if (
        not latest_stats
        or "rects" not in latest_stats
        or crop_index < 0
        or crop_index >= len(latest_stats["rects"])
    ):
        raise HTTPException(status_code=404, detail="Crop index not found")

    rect_data = latest_stats["rects"][crop_index]

    crop_measurements = {"crop_index": crop_index, "shapes": []}
    # 处理所有形状
    if "all_shapes" in rect_data and rect_data["all_shapes"]:
        for shape_idx, shape_data in enumerate(rect_data["all_shapes"]):
            if "physical_info" in shape_data:
                physical_info = shape_data["physical_info"]

                shape_measurement = {
                    "shape_index": shape_idx,
                    "shape_type": shape_data["shape_type"],
                    "pixel_dimensions": {
                        "width": shape_data["width"],
                        "height": shape_data["height"],
                        "area": shape_data["area"],
                        "side_lengths": shape_data.get("side_lengths", []),
                        "mean_side_length": shape_data.get("mean_side_length", 0),
                        "perimeter": shape_data.get("perimeter", 0),
                    },
                    "physical_dimensions": {
                        "width_mm": physical_info["physical_width_mm"],
                        "height_mm": physical_info["physical_height_mm"],
                        "area_mm2": physical_info["physical_area_mm2"],
                        "diameter_mm": physical_info["physical_diameter_mm"],
                        "side_lengths_mm": physical_info["physical_side_lengths_mm"],
                        "perimeter_mm": physical_info["physical_perimeter_mm"],
                        "measurement_type": physical_info["measurement_type"],
                        "mm_per_pixel": physical_info["mm_per_pixel"],
                    },
                }
                crop_measurements["shapes"].append(shape_measurement)

    return {
        "success": True,
        "crop_measurements": crop_measurements,
        "a4_reference": {
            "physical_width_mm": 170,  # 210 - 40
            "physical_height_mm": 257,  # 297 - 40
            "note": "A4 paper minus 20mm border on each side",
        },
    }


# @app.get("/api/minimum_square_measurements")
# async def get_minimum_square_measurements():
#     """获取所有裁剪区域中的最小正方形测量信息
#
#     返回格式：
#     {
#         "success": bool,
#         "measurements": [
#             {
#                 "crop_index": int,
#                 "target": {
#                     "id": int,
#                     "bbox": [x, y, w, h],
#                     "area": int,
#                     "aspect_ratio": float,
#                     "crop_width": int,
#                     "crop_height": int,
#                     "position": [x, y],
#                     "horizontal_avg": float,
#                     "vertical_avg": float,
#                     "new_long_px": float
#                 },
#                 "squares": [
#                     {
#                         "shape_index": int,
#                         "found": bool,
#                         "center": [x, y],
#                         "area": float,
#                         "side_length": float,
#                         "aspect_ratio": float,
#                         "type": str,
#                         "pixel_dimensions": {
#                             "width": int,
#                             "height": int,
#                             "perimeter": float
#                         },
#                         "physical_dimensions": {
#                             "width_mm": float,
#                             "height_mm": float,
#                             "area_mm2": float,
#                             "perimeter_mm": float,
#                             "side_length_mm": float,
#                             "mm_per_pixel": float
#                         }
#                     }
#                 ]
#             }
#         ]
#     }
#     """
@app.get("/api/minimum_square_measurements")
async def get_minimum_square_measurements():
    """获取所有裁剪区域中的最小边测量信息"""
    global latest_stats

    if not latest_stats or "rects" not in latest_stats:
        return {
            "success": False,
            "error": "No measurement data available",
            "measurements": [],
        }

    measurements = []

    for crop_idx, rect_data in enumerate(latest_stats["rects"]):
        crop_measurement = {
            "crop_index": crop_idx,
            "target": {
                "id": rect_data["id"],
                "bbox": [
                    rect_data["position"][0],
                    rect_data["position"][1],
                    rect_data["outer_width"],
                    rect_data["outer_height"],
                ],
                "area": rect_data["area"],
                "aspect_ratio": rect_data["aspect_ratio"],
                "crop_width": rect_data["crop_width"],
                "crop_height": rect_data["crop_height"],
                "position": rect_data["position"],
                "horizontal_avg": rect_data["horizontal_avg"],
                "vertical_avg": rect_data["vertical_avg"],
                "new_long_px": rect_data["new_long_px"],
            },
            "edges": [],  # 改为edges而不是squares
        }

        # 从全局状态获取最小边信息
        min_square_info = latest_stats.get("minimum_black_square", {"found": False})

        if min_square_info.get("found", False):
            # 计算物理尺寸 (基于A4纸比例)
            edge_length_px = min_square_info["edge_length_px"]
            edge_length_mm = min_square_info["edge_length_mm"]

            # A4纸物理尺寸（减去边距）
            a4_width_mm = 170  # 210 - 40
            # 根据裁剪图像尺寸计算mm/像素比例
            crop_width = rect_data["crop_width"]
            if crop_width > 0:
                mm_per_pixel = a4_width_mm / crop_width

                square_measurement = {
                    "shape_index": 0,
                    "found": True,
                    "center": min_square_info["center"],
                    "edge_length_px": edge_length_px,
                    "edge_length_mm": edge_length_mm,
                    "type": min_square_info.get("type", "minimum_edge_detected"),
                    "start_point": min_square_info.get("start_point", [0, 0]),
                    "end_point": min_square_info.get("end_point", [0, 0]),
                    "pixel_dimensions": {
                        "edge_length": edge_length_px,
                        "note": "This is the shortest detected edge, not a complete square",
                    },
                    "physical_dimensions": {
                        "edge_length_mm": edge_length_mm,
                        "mm_per_pixel": mm_per_pixel,
                        "note": "Physical measurement of the shortest detected edge",
                    },
                }
                crop_measurement["edges"].append(square_measurement)
        else:
            crop_measurement["edges"].append({"shape_index": 0, "found": False})

        measurements.append(crop_measurement)

    return {
        "success": True,
        "measurements": measurements,
        "total_crops": len(measurements),
        "a4_reference": {
            "physical_width_mm": 170,  # 210 - 40
            "physical_height_mm": 257,  # 297 - 40
            "note": "A4 paper minus 20mm border on each side",
        },
    }


@app.get("/api/ocr_auto_segment_easyocr_analysis", tags=["OCR识别"])
async def get_ocr_auto_segment_easyocr_analysis():
    """
    结合自动切分EasyOCR识别和物理测量的分析接口
    """
    global latest_stats, raw_crops, cap_o
    cap_o = False  # 暂停采集循环
    start_time = time.time()  # 开始计时
    
    # 1. 获取物理测量数据（这会生成raw_crops）
    measurement_response = await get_physical_measurements()
    if not measurement_response["success"]:
        cap_o = True  # 恢复采集循环
        return {
            "success": False,
            "error": "Failed to get physical measurements",
            "analysis": [],
        }
    
    # 2. 使用自动切分EasyOCR识别替代OCR
    auto_segment_results = {}
    scale_factors = {}  # 存储每个crop的缩放因子
    
    print(f"调试信息: raw_crops数量={len(raw_crops) if raw_crops else 0}, enable_ocr={params.get('enable_ocr', False)}")
    
    if raw_crops and params["enable_ocr"]:
        print(f"开始处理 {len(raw_crops)} 个crop区域")
        for idx, crop in enumerate(raw_crops):
            print(f"处理第 {idx+1} 个crop...")
            try:
                # 应用mask处理
                masked_crop = apply_mask_to_crop(crop)
                
                # 记录原始尺寸
                original_height, original_width = masked_crop.shape[:2]
                
                # 统一缩放到840*1118
                resized_crop = cv2.resize(masked_crop, (840, 1118))
                
                # 计算缩放因子
                scale_x = 840 / original_width
                scale_y = 1118 / original_height
                scale_factors[idx] = {"scale_x": scale_x, "scale_y": scale_y}
                
                print(f"=== 使用EasyOCR自动分割处理 ===")
                print(f"原始尺寸: {original_width}x{original_height}")
                print(f"缩放后尺寸: 840x1118")
                print(f"缩放因子: x={scale_x:.3f}, y={scale_y:.3f}")
                
                # 使用EasyOCR自动分割处理器处理缩放后的图像
                auto_segment_processor = LocalAutoSegmentEasyOCR()
                easyocr_results = auto_segment_processor.process_image(
                    resized_crop, 
                    confidence_threshold=0.5,
                    save_debug=True
                )
                
                # 转换结果格式以匹配YOLO版本的格式
                converted_results = []
                for result in easyocr_results:
                    converted_result = {
                        'text': result['text'],
                        'conf': result['confidence'],
                        'center': [result['center_x'], result['center_y']],
                        'bbox': result['bbox_in_original'],
                        'rectangle_id': result.get('rectangle_id', 0),
                        'rectangle_area': result.get('rectangle_area', 0)
                    }
                    converted_results.append(converted_result)
                
                auto_segment_results[idx] = converted_results
                print(f"EasyOCR识别到 {len(converted_results)} 个结果")
                
            except Exception as e:
                print(f"处理crop {idx} 时出错: {e}")
                auto_segment_results[idx] = []
    else:
        print("跳过自动切分处理：raw_crops为空或OCR功能被禁用")

    # 3. 合并结果（与YOLO版本完全相同的逻辑）
    analysis = []
    for measurement in measurement_response["measurements"]:
        crop_idx = measurement["crop_index"]

        # 复制测量数据的基本结构
        crop_analysis = {
            "crop_index": crop_idx,
            "target": measurement["target"],
            "shapes": [],
            "ocr_raw_data": auto_segment_results.get(crop_idx, []),  # 添加原始自动切分数据
            "scale_factors": scale_factors.get(crop_idx, {"scale_x": 1.0, "scale_y": 1.0}),  # 添加缩放因子
        }

        # 处理每个形状
        for shape in measurement["shapes"]:
            # 获取识别数据
            ocr_data = {"detected": False, "text": "", "conf": 0.0, "bbox": [], "center": []}

            # 如果有自动切分结果，尝试匹配
            if crop_idx in auto_segment_results:
                shape_bbox = shape.get("pixel_dimensions", {})
                shape_center = shape.get("position", {}).get("center", [0, 0])

                # 如果形状中心点为[0,0]，尝试从其他数据源获取中心点
                if shape_center == [0, 0]:
                    # 尝试从bbox计算中心点
                    bbox = shape.get("position", {}).get("bbox", [0, 0, 0, 0])
                    if bbox != [0, 0, 0, 0] and len(bbox) == 4:
                        shape_center = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]
                    else:
                        # 如果还是没有，跳过匹配
                        shape_center = [0, 0]

                # 如果形状中心点不是[0,0]，进行位置匹配
                if shape_center != [0, 0] and crop_idx in scale_factors:
                    # 将原始坐标转换为缩放后的坐标
                    scale_x = scale_factors[crop_idx]["scale_x"]
                    scale_y = scale_factors[crop_idx]["scale_y"]
                    
                    scaled_center_x = shape_center[0] * scale_x
                    scaled_center_y = shape_center[1] * scale_y
                    scaled_width_half = shape_bbox.get("width", 0) * scale_x / 2
                    scaled_height_half = shape_bbox.get("height", 0) * scale_y / 2

                    # 寻找最佳匹配的识别结果
                    best_match = None
                    min_distance = float('inf')
                    
                    for result in auto_segment_results[crop_idx]:
                        result_center = result["center"]
                        
                        distance_x = abs(result_center[0] - scaled_center_x)
                        distance_y = abs(result_center[1] - scaled_center_y)
                        total_distance = (distance_x ** 2 + distance_y ** 2) ** 0.5

                        # 检查是否在形状范围内（使用缩放后的坐标）
                        if (
                            distance_x < scaled_width_half
                            and distance_y < scaled_height_half
                            and total_distance < min_distance
                        ):
                            min_distance = total_distance
                            best_match = result

                    # 如果找到最佳匹配
                    if best_match:
                        # 将识别结果的bbox坐标转换回原始坐标系
                        original_bbox = []
                        for point in best_match["bbox"]:
                            original_point = [
                                point[0] / scale_x,
                                point[1] / scale_y
                            ]
                            original_bbox.append(original_point)
                        
                        # 转换中心点坐标
                        original_center = [
                            best_match["center"][0] / scale_x,
                            best_match["center"][1] / scale_y
                        ]
                        
                        ocr_data = {
                            "detected": True,
                            "text": best_match["text"],
                            "conf": best_match["conf"],
                            "bbox": original_bbox,
                            "center": original_center,
                            "scaled_bbox": best_match["bbox"],
                            "scaled_center": best_match["center"],
                            "rectangle_id": best_match.get("rectangle_id", 0),
                            "rectangle_area": best_match.get("rectangle_area", 0),
                            "match_distance": min_distance,
                        }

            # 组合形状数据和识别结果
            if "position" not in shape:
                shape["position"] = {}
            shape["position"]["contour_points"] = shape.get("contour_points", [])

            shape_analysis = {
                **shape,  # 保留所有原有的形状数据
                "ocr_data": ocr_data,  # 使用ocr_data替代recognition_data
            }

            crop_analysis["shapes"].append(shape_analysis)

        analysis.append(crop_analysis)

    elapsed_time = time.time() - start_time  # 计算耗时
    cap_o = True
    return {
        "success": True,
        "analysis": analysis,
        "total_crops": len(analysis),
        "references": measurement_response.get("a4_reference", {}),
        "elapsed_seconds": round(elapsed_time, 3),  # 添加耗时信息，保留3位小数
    }


@app.get("/api/ocr_auto_segment_analysis", tags=["OCR识别"])
async def get_ocr_auto_segment_analysis():
    """
    结合自动切分YOLO识别和物理测量的分析接口
    """
    global latest_stats, raw_crops, cap_o
    start_time = time.time()  # 开始计时
    cap_o = False

    # 1. 获取物理测量数据
    measurement_response = await get_physical_measurements()
    if not measurement_response["success"]:
        return {
            "success": False,
            "error": "Failed to get physical measurements",
            "analysis": [],
        }

    # 2. 使用自动切分YOLO识别替代OCR
    auto_segment_results = {}
    scale_factors = {}  # 存储每个crop的缩放因子
    
    print(f"调试信息: raw_crops数量={len(raw_crops) if raw_crops else 0}, enable_ocr={params.get('enable_ocr', False)}")
    
    if raw_crops and params["enable_ocr"]:
        print(f"开始处理 {len(raw_crops)} 个crop区域")
        for idx, crop in enumerate(raw_crops):
            print(f"处理第 {idx+1} 个crop...")
            try:
                # 应用mask处理
                masked_crop = apply_mask_to_crop(crop)
                
                # 记录原始尺寸
                original_height, original_width = masked_crop.shape[:2]
                
                # 统一缩放到840*1118
                resized_crop = cv2.resize(masked_crop, (840, 1118))
                
                # 计算缩放因子
                scale_x = 840 / original_width
                scale_y = 1118 / original_height
                scale_factors[idx] = {"scale_x": scale_x, "scale_y": scale_y}
                
                # 将图像转换为base64
                _, buffer = cv2.imencode('.jpg', resized_crop)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                print(f"=== 异步调用本地AutoSegmentYOLO ===")
                print(f"图像数据长度: {len(img_base64)} 字符")
                
                # 使用线程池执行同步的AutoSegmentYOLO调用
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    try:
                        # 在线程池中执行同步函数
                        converted_results = await loop.run_in_executor(
                            executor, 
                            process_image_with_auto_segment, 
                            img_base64, 
                            idx
                        )
                        auto_segment_results[idx] = converted_results
                        
                    except Exception as e:
                        print(f"异步AutoSegmentYOLO处理失败: {e}")
                        import traceback
                        traceback.print_exc()
                        auto_segment_results[idx] = []
                    
            except Exception as e:
                print(f"处理crop {idx} 时出错: {e}")
                auto_segment_results[idx] = []
    else:
        print("跳过自动切分处理：raw_crops为空或OCR功能被禁用")

    # 3. 合并结果
    analysis = []
    for measurement in measurement_response["measurements"]:
        crop_idx = measurement["crop_index"]

        # 复制测量数据的基本结构
        crop_analysis = {
            "crop_index": crop_idx,
            "target": measurement["target"],
            "shapes": [],
            "ocr_raw_data": auto_segment_results.get(crop_idx, []),  # 添加原始自动切分数据
            "scale_factors": scale_factors.get(crop_idx, {"scale_x": 1.0, "scale_y": 1.0}),  # 添加缩放因子
        }

        # 处理每个形状
        for shape in measurement["shapes"]:
            # 获取识别数据
            ocr_data = {"detected": False, "text": "", "conf": 0.0, "bbox": [], "center": []}

            # 如果有自动切分结果，尝试匹配
            if crop_idx in auto_segment_results:
                shape_bbox = shape.get("pixel_dimensions", {})
                shape_center = shape.get("position", {}).get("center", [0, 0])

                # 如果形状中心点为[0,0]，尝试从其他数据源获取中心点
                if shape_center == [0, 0]:
                    # 尝试从bbox计算中心点
                    bbox = shape.get("position", {}).get("bbox", [0, 0, 0, 0])
                    if bbox != [0, 0, 0, 0] and len(bbox) == 4:
                        shape_center = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]
                    else:
                        # 如果还是没有，跳过匹配
                        shape_center = [0, 0]

                # 如果形状中心点不是[0,0]，进行位置匹配
                if shape_center != [0, 0] and crop_idx in scale_factors:
                    # 将原始坐标转换为缩放后的坐标
                    scale_x = scale_factors[crop_idx]["scale_x"]
                    scale_y = scale_factors[crop_idx]["scale_y"]
                    
                    scaled_center_x = shape_center[0] * scale_x
                    scaled_center_y = shape_center[1] * scale_y
                    scaled_width_half = shape_bbox.get("width", 0) * scale_x / 2
                    scaled_height_half = shape_bbox.get("height", 0) * scale_y / 2

                    # 寻找最佳匹配的识别结果
                    best_match = None
                    min_distance = float('inf')
                    
                    for result in auto_segment_results[crop_idx]:
                        result_center = result["center"]
                        
                        distance_x = abs(result_center[0] - scaled_center_x)
                        distance_y = abs(result_center[1] - scaled_center_y)
                        total_distance = (distance_x ** 2 + distance_y ** 2) ** 0.5

                        # 检查是否在形状范围内（使用缩放后的坐标）
                        if (
                            distance_x < scaled_width_half
                            and distance_y < scaled_height_half
                            and total_distance < min_distance
                        ):
                            min_distance = total_distance
                            best_match = result

                    # 如果找到最佳匹配
                    if best_match:
                        # 将识别结果的bbox坐标转换回原始坐标系
                        original_bbox = []
                        for point in best_match["bbox"]:
                            original_point = [
                                point[0] / scale_x,
                                point[1] / scale_y
                            ]
                            original_bbox.append(original_point)
                        
                        # 转换中心点坐标
                        original_center = [
                            best_match["center"][0] / scale_x,
                            best_match["center"][1] / scale_y
                        ]
                        
                        ocr_data = {
                            "detected": True,
                            "text": best_match["text"],
                            "conf": best_match["conf"],
                            "bbox": original_bbox,
                            "center": original_center,
                            "scaled_bbox": best_match["bbox"],
                            "scaled_center": best_match["center"],
                            "rectangle_id": best_match.get("rectangle_id", 0),
                            "rectangle_area": best_match.get("rectangle_area", 0),
                            "match_distance": min_distance,
                        }

            # 组合形状数据和识别结果
            if "position" not in shape:
                shape["position"] = {}
            shape["position"]["contour_points"] = shape.get("contour_points", [])

            shape_analysis = {
                **shape,  # 保留所有原有的形状数据
                "ocr_data": ocr_data,  # 使用ocr_data替代recognition_data
            }

            crop_analysis["shapes"].append(shape_analysis)

        analysis.append(crop_analysis)

    elapsed_time = time.time() - start_time  # 计算耗时
    cap_o = True
    return {
        "success": True,
        "analysis": analysis,
        "total_crops": len(analysis),
        "references": measurement_response.get("a4_reference", {}),
        "elapsed_seconds": round(elapsed_time, 3),  # 添加耗时信息，保留3位小数
    }


static_dir = os.path.join(os.path.dirname(__file__), "spa")
print(static_dir)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="ui")
