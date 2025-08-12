import cv2
import numpy as np
import requests
import json
import base64
import os
import tempfile
import time
from typing import List, Dict, Tuple, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from auto_segment_yolo import AutoSegmentYOLO

app = FastAPI(
    title="自动分割识别API", 
    description="基于YOLO的自动图像分割和数字识别服务", 
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局处理器
processor = None

@app.on_event("startup")
async def startup_event():
    """启动时初始化处理器"""
    global processor
    try:
        processor = AutoSegmentYOLO(yolo_api_url="http://localhost:8005")
        # 测试YOLO API连接
        response = requests.get("http://localhost:8005/health", timeout=5)
        if response.status_code != 200:
            raise Exception("YOLO API不可用")
        print("✅ 自动分割识别API启动成功")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        raise Exception("无法连接到YOLO API服务")

@app.get("/", tags=["根目录"])
async def root():
    """根路径 - 返回上传页面"""
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>自动分割识别API</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Arial', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 20px;
            }
            
            .container {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                padding: 40px;
                max-width: 900px;
                width: 100%;
                margin: 20px;
            }
            
            h1 {
                text-align: center;
                color: #333;
                margin-bottom: 30px;
                font-size: 2.5em;
                font-weight: 300;
            }
            
            .upload-area {
                border: 3px dashed #ddd;
                border-radius: 15px;
                padding: 40px;
                text-align: center;
                margin-bottom: 30px;
                transition: all 0.3s ease;
                cursor: pointer;
            }
            
            .upload-area:hover {
                border-color: #667eea;
                background-color: #f8f9ff;
            }
            
            .upload-area.dragover {
                border-color: #667eea;
                background-color: #f0f4ff;
            }
            
            #file-input {
                display: none;
            }
            
            .upload-text {
                color: #666;
                font-size: 1.2em;
                margin-bottom: 20px;
            }
            
            .upload-btn {
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 25px;
                cursor: pointer;
                font-size: 1.1em;
                transition: transform 0.3s ease;
            }
            
            .upload-btn:hover {
                transform: translateY(-2px);
            }
            
            .params {
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                gap: 20px;
                margin-bottom: 30px;
            }
            
            .param-group {
                display: flex;
                flex-direction: column;
            }
            
            .param-group label {
                margin-bottom: 5px;
                color: #555;
                font-weight: 500;
            }
            
            .param-group input {
                padding: 10px;
                border: 2px solid #ddd;
                border-radius: 8px;
                font-size: 1em;
            }
            
            .param-group input:focus {
                outline: none;
                border-color: #667eea;
            }
            
            .predict-btn {
                width: 100%;
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
                border: none;
                padding: 15px;
                border-radius: 10px;
                font-size: 1.2em;
                cursor: pointer;
                margin-bottom: 30px;
                transition: all 0.3s ease;
            }
            
            .predict-btn:hover:not(:disabled) {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
            }
            
            .predict-btn:disabled {
                background: #ccc;
                cursor: not-allowed;
            }
            
            .preview {
                text-align: center;
                margin-bottom: 20px;
            }
            
            .preview img {
                max-width: 100%;
                max-height: 400px;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            
            .results {
                background: #f8f9fa;
                border-radius: 10px;
                padding: 20px;
                margin-top: 20px;
            }
            
            .result-item {
                background: white;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 10px;
                border-left: 4px solid #667eea;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }
            
            .result-item:last-child {
                margin-bottom: 0;
            }
            
            .loading {
                display: none;
                text-align: center;
                padding: 20px;
            }
            
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto 10px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .error {
                background: #f8d7da;
                color: #721c24;
                padding: 15px;
                border-radius: 8px;
                margin-top: 20px;
                border-left: 4px solid #dc3545;
            }
            
            .success {
                background: #d4edda;
                color: #155724;
                padding: 15px;
                border-radius: 8px;
                margin-top: 20px;
                border-left: 4px solid #28a745;
            }
            
            .debug-section {
                margin-top: 20px;
                padding: 15px;
                background: #e9ecef;
                border-radius: 8px;
            }
            
            .download-btn {
                background: #28a745;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
                cursor: pointer;
                margin: 5px;
                text-decoration: none;
                display: inline-block;
            }
            
            .download-btn:hover {
                background: #218838;
            }
            
            @media (max-width: 600px) {
                .params {
                    grid-template-columns: 1fr;
                }
                
                .container {
                    padding: 20px;
                    margin: 10px;
                }
                
                h1 {
                    font-size: 2em;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 自动分割识别</h1>
            
            <div class="upload-area" onclick="document.getElementById('file-input').click()" 
                 ondrop="dropHandler(event);" ondragover="dragOverHandler(event);" ondragleave="dragLeaveHandler(event);">
                <div class="upload-text">
                    📁 点击选择图片或拖拽图片到此处<br>
                    <small>系统会自动分割白色矩形区域并逐个识别数字</small>
                </div>
                <button type="button" class="upload-btn">选择图片</button>
                <input type="file" id="file-input" accept="image/*" onchange="handleFileSelect(event)">
            </div>
            
            <div class="params">
                <div class="param-group">
                    <label for="confidence">YOLO置信度阈值:</label>
                    <input type="number" id="confidence" min="0" max="1" step="0.1" value="0.5">
                </div>
                <div class="param-group">
                    <label for="nms">YOLO NMS阈值:</label>
                    <input type="number" id="nms" min="0" max="1" step="0.1" value="0.4">
                </div>
                <div class="param-group">
                    <label for="min_area">最小矩形面积:</label>
                    <input type="number" id="min_area" min="100" max="10000" step="100" value="300">
                </div>
            </div>
            
            <button class="predict-btn" id="predict-btn" onclick="predict()" disabled>
                🚀 开始自动分割识别
            </button>
            
            <div class="preview" id="preview" style="display: none;">
                <img id="preview-img" src="" alt="预览图片">
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <div>正在自动分割和识别中...</div>
            </div>
            
            <div id="results"></div>
        </div>
        
        <script>
            let selectedFile = null;
            let lastTaskId = null;
            
            function handleFileSelect(event) {
                const file = event.target.files[0];
                if (file) {
                    selectedFile = file;
                    showPreview(file);
                    document.getElementById('predict-btn').disabled = false;
                }
            }
            
            function showPreview(file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const preview = document.getElementById('preview');
                    const img = document.getElementById('preview-img');
                    img.src = e.target.result;
                    preview.style.display = 'block';
                };
                reader.readAsDataURL(file);
            }
            
            function dragOverHandler(event) {
                event.preventDefault();
                event.currentTarget.classList.add('dragover');
            }
            
            function dragLeaveHandler(event) {
                event.currentTarget.classList.remove('dragover');
            }
            
            function dropHandler(event) {
                event.preventDefault();
                event.currentTarget.classList.remove('dragover');
                
                const files = event.dataTransfer.files;
                if (files.length > 0 && files[0].type.startsWith('image/')) {
                    selectedFile = files[0];
                    showPreview(files[0]);
                    document.getElementById('predict-btn').disabled = false;
                }
            }
            
            async function predict() {
                if (!selectedFile) {
                    showError('请先选择图片');
                    return;
                }
                
                const confidence = document.getElementById('confidence').value;
                const nms = document.getElementById('nms').value;
                const minArea = document.getElementById('min_area').value;
                
                document.getElementById('loading').style.display = 'block';
                document.getElementById('results').innerHTML = '';
                document.getElementById('predict-btn').disabled = true;
                
                try {
                    const formData = new FormData();
                    formData.append('file', selectedFile);
                    formData.append('confidence_threshold', confidence);
                    formData.append('nms_threshold', nms);
                    formData.append('min_area', minArea);
                    
                    const response = await fetch('/auto_segment_recognize', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        lastTaskId = result.task_id;
                        showResults(result);
                    } else {
                        showError(result.detail || '识别失败');
                    }
                } catch (error) {
                    showError('网络错误: ' + error.message);
                } finally {
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('predict-btn').disabled = false;
                }
            }
            
            function showResults(result) {
                const resultsDiv = document.getElementById('results');
                
                if (result.count === 0) {
                    resultsDiv.innerHTML = '<div class="success">✅ 处理完成，但未识别到数字</div>';
                    return;
                }
                
                let html = `<div class="success">✅ 自动分割识别完成，检测到 ${result.count} 个数字</div>`;
                html += '<div class="results">';
                
                result.results.forEach((item, index) => {
                    html += `
                        <div class="result-item">
                            <h4>数字 ${index + 1}: ${item.text}</h4>
                            <p><strong>置信度:</strong> ${(item.confidence * 100).toFixed(1)}%</p>
                            <p><strong>中心坐标:</strong> (${item.center_x.toFixed(1)}, ${item.center_y.toFixed(1)})</p>
                            <p><strong>来源矩形:</strong> ID ${item.rectangle_id} (面积: ${item.rectangle_area.toFixed(0)})</p>
                            <p><strong>矫正尺寸:</strong> ${item.corrected_size[0]} × ${item.corrected_size[1]} 像素</p>
                        </div>
                    `;
                });
                
                html += '</div>';
                
                // 添加参数信息
                html += `
                    <div style="margin-top: 15px; padding: 10px; background: #e9ecef; border-radius: 5px; font-size: 0.9em;">
                        <strong>处理参数:</strong> 置信度阈值 ${result.parameters.confidence_threshold}, 
                        NMS阈值 ${result.parameters.nms_threshold}, 
                        最小面积 ${result.parameters.min_area}
                    </div>
                `;
                
                // 添加下载链接
                if (result.debug_image_url) {
                    html += `
                        <div class="debug-section">
                            <h4>调试文件下载:</h4>
                            <a href="${result.debug_image_url}" class="download-btn" target="_blank">📷 下载调试图像</a>
                            <a href="${result.json_url}" class="download-btn" target="_blank">📄 下载JSON结果</a>
                        </div>
                    `;
                }
                
                resultsDiv.innerHTML = html;
            }
            
            function showError(message) {
                const resultsDiv = document.getElementById('results');
                resultsDiv.innerHTML = `<div class="error">❌ ${message}</div>`;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health", tags=["健康检查"])
async def health_check():
    """健康检查"""
    global processor
    yolo_status = False
    try:
        response = requests.get("http://localhost:8005/health", timeout=3)
        yolo_status = response.status_code == 200
    except:
        pass
    
    return {
        "status": "healthy",
        "processor_loaded": processor is not None,
        "yolo_api_available": yolo_status
    }

@app.post("/auto_segment_recognize", tags=["自动分割识别"])
async def auto_segment_recognize(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5),
    nms_threshold: float = Form(0.4),
    min_area: int = Form(300)
):
    """
    自动分割和识别图像中的数字
    
    参数:
    - file: 上传的图像文件
    - confidence_threshold: YOLO置信度阈值 (0.0-1.0)
    - nms_threshold: YOLO NMS阈值 (0.0-1.0) 
    - min_area: 最小矩形面积 (像素)
    
    返回:
    - 识别结果，包含所有检测到的数字及其坐标信息
    """
    global processor
    
    if not processor:
        raise HTTPException(status_code=500, detail="处理器未初始化")
    
    # 检查文件类型
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="请上传图片文件")
    
    # 生成任务ID
    task_id = f"task_{int(time.time())}_{os.getpid()}"
    
    try:
        # 读取图片
        contents = await file.read()
        
        # 保存临时文件
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_path = os.path.join(temp_dir, f"{task_id}_{file.filename}")
        with open(temp_path, "wb") as temp_file:
            temp_file.write(contents)
        
        # 处理图像
        results = processor.process_image(
            temp_path,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            min_area=min_area,
            save_debug=True
        )
        
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
        
        # 保存结果到JSON
        json_path = f"output/results_{task_id}.json"
        processor.save_results_to_json(results, json_path)
        
        # 查找调试图像
        debug_image_path = "debug_auto_segment/final_results.png"
        debug_image_url = None
        json_url = None
        
        if os.path.exists(debug_image_path):
            debug_image_url = f"/download_debug_image/{task_id}"
        if os.path.exists(json_path):
            json_url = f"/download_json/{task_id}"
        
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return {
            "success": True,
            "task_id": task_id,
            "count": len(clean_results),
            "results": clean_results,
            "parameters": {
                "confidence_threshold": confidence_threshold,
                "nms_threshold": nms_threshold,
                "min_area": min_area
            },
            "debug_image_url": debug_image_url,
            "json_url": json_url,
            "processing_time": time.time()
        }
        
    except Exception as e:
        # 清理临时文件
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

@app.post("/auto_segment_base64", tags=["自动分割识别"])
async def auto_segment_base64(data: dict):
    """
    使用base64编码的图片进行自动分割识别
    
    请求格式: 
    {
        "image": "base64编码的图片数据",
        "confidence_threshold": 0.5,
        "nms_threshold": 0.4,
        "min_area": 300
    }
    """
    global processor
    
    if not processor:
        raise HTTPException(status_code=500, detail="处理器未初始化")
    
    if "image" not in data:
        raise HTTPException(status_code=400, detail="请提供image字段")
    
    # 获取参数
    confidence_threshold = data.get("confidence_threshold", 0.5)
    nms_threshold = data.get("nms_threshold", 0.4)
    min_area = data.get("min_area", 300)
    
    # 生成任务ID
    task_id = f"task_{int(time.time())}_{os.getpid()}"
    
    try:
        # 解码base64图片
        image_data = base64.b64decode(data["image"])
        
        # 保存临时文件
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_path = os.path.join(temp_dir, f"{task_id}.jpg")
        with open(temp_path, "wb") as temp_file:
            temp_file.write(image_data)
        
        # 处理图像
        results = processor.process_image(
            temp_path,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            min_area=min_area,
            save_debug=True
        )
        
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
        
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return {
            "success": True,
            "task_id": task_id,
            "count": len(clean_results),
            "results": clean_results,
            "parameters": {
                "confidence_threshold": confidence_threshold,
                "nms_threshold": nms_threshold,
                "min_area": min_area
            },
            "processing_time": time.time()
        }
        
    except Exception as e:
        # 清理临时文件
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

@app.get("/download_debug_image/{task_id}", tags=["文件下载"])
async def download_debug_image(task_id: str):
    """下载调试图像"""
    debug_path = "debug_auto_segment/final_results.png"
    if os.path.exists(debug_path):
        return FileResponse(
            debug_path,
            media_type="image/png",
            filename=f"debug_result_{task_id}.png"
        )
    else:
        raise HTTPException(status_code=404, detail="调试图像不存在")

@app.get("/download_json/{task_id}", tags=["文件下载"])
async def download_json(task_id: str):
    """下载JSON结果"""
    json_path = f"results_{task_id}.json"
    if os.path.exists(json_path):
        return FileResponse(
            json_path,
            media_type="application/json",
            filename=f"results_{task_id}.json"
        )
    else:
        # 尝试默认文件名
        default_json = "recognition_results.json"
        if os.path.exists(default_json):
            return FileResponse(
                default_json,
                media_type="application/json",
                filename=f"results_{task_id}.json"
            )
        else:
            raise HTTPException(status_code=404, detail="JSON结果文件不存在")

if __name__ == "__main__":
    print("启动自动分割识别API服务...")
    print("Web界面: http://localhost:8006")
    print("API文档: http://localhost:8006/docs")
    print("请确保YOLO API服务运行在 http://localhost:8005")
    uvicorn.run(app, host="0.0.0.0", port=8006)
