import cv2
import numpy as np
import onnxruntime as ort
import base64
import io
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from PIL import Image

app = FastAPI(title="YOLO数字识别API", description="基于YOLO模型的数字识别服务", version="1.0.0")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
net = None
model_h = 320
model_w = 320
nl = 3
na = 3
stride = [8., 16., 32.]
anchor_grid = None

# 标签字典
dic_labels = {
    0: '1', 1: '2', 2: '3', 3: '4',
    4: '5', 5: '6', 6: '7', 7: '8'
}

def _make_grid(nx, ny):
    """创建网格"""
    xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
    return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

def cal_outputs(outs, nl, na, model_w, model_h, anchor_grid, stride):
    """计算输出坐标"""
    row_ind = 0
    grid = [np.zeros(1)] * nl
    for i in range(nl):
        h, w = int(model_w / stride[i]), int(model_h / stride[i])
        length = int(na * h * w)
        if grid[i].shape[2:4] != (h, w):
            grid[i] = _make_grid(w, h)

        outs[row_ind:row_ind + length, 0:2] = (outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + np.tile(
            grid[i], (na, 1))) * int(stride[i])
        outs[row_ind:row_ind + length, 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * np.repeat(
            anchor_grid[i], h * w, axis=0)
        row_ind += length
    return outs

def post_process_opencv(outputs, model_h, model_w, img_h, img_w, thred_nms, thred_cond):
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

def infer_img(img0, thred_nms=0.4, thred_cond=0.5):
    """图像推理"""
    global net, model_h, model_w, nl, na, stride, anchor_grid
    
    # 图像预处理
    img = cv2.resize(img0, [model_w, model_h], interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

    # 模型推理
    outs = net.run(None, {net.get_inputs()[0].name: blob})[0].squeeze(axis=0)

    # 输出坐标矫正
    outs = cal_outputs(outs, nl, na, model_w, model_h, anchor_grid, stride)

    # 检测框计算
    img_h, img_w, _ = np.shape(img0)
    boxes, confs, ids = post_process_opencv(outs, model_h, model_w, img_h, img_w, thred_nms, thred_cond)

    return boxes, confs, ids

def load_model():
    """加载模型"""
    global net, anchor_grid
    try:
        model_pb_path = "best.onnx"
        so = ort.SessionOptions()
        net = ort.InferenceSession(model_pb_path, so)
        
        # 初始化anchor_grid
        anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(nl, -1, 2)
        
        print("模型加载成功")
        return True
    except Exception as e:
        print(f"模型加载失败: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    if not load_model():
        raise Exception("模型加载失败，无法启动服务")

@app.get("/", tags=["根目录"])
async def root():
    """根路径 - 返回上传页面"""
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>YOLO数字识别API</title>
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
                max-width: 800px;
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
                grid-template-columns: 1fr 1fr;
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
            <h1>🔢 YOLO数字识别</h1>
            
            <div class="upload-area" onclick="document.getElementById('file-input').click()" 
                 ondrop="dropHandler(event);" ondragover="dragOverHandler(event);" ondragleave="dragLeaveHandler(event);">
                <div class="upload-text">
                    📁 点击选择图片或拖拽图片到此处
                </div>
                <button type="button" class="upload-btn">选择图片</button>
                <input type="file" id="file-input" accept="image/*" onchange="handleFileSelect(event)">
            </div>
            
            <div class="params">
                <div class="param-group">
                    <label for="confidence">置信度阈值:</label>
                    <input type="number" id="confidence" min="0" max="1" step="0.1" value="0.5">
                </div>
                <div class="param-group">
                    <label for="nms">NMS阈值:</label>
                    <input type="number" id="nms" min="0" max="1" step="0.1" value="0.4">
                </div>
            </div>
            
            <button class="predict-btn" id="predict-btn" onclick="predict()" disabled>
                🚀 开始识别
            </button>
            
            <div class="preview" id="preview" style="display: none;">
                <img id="preview-img" src="" alt="预览图片">
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <div>正在识别中...</div>
            </div>
            
            <div id="results"></div>
        </div>
        
        <script>
            let selectedFile = null;
            
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
                
                document.getElementById('loading').style.display = 'block';
                document.getElementById('results').innerHTML = '';
                document.getElementById('predict-btn').disabled = true;
                
                try {
                    const formData = new FormData();
                    formData.append('file', selectedFile);
                    
                    const response = await fetch(`/predict_with_params?confidence_threshold=${confidence}&nms_threshold=${nms}`, {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
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
                    resultsDiv.innerHTML = '<div class="success">✅ 识别完成，但未检测到数字</div>';
                    return;
                }
                
                let html = `<div class="success">✅ 识别完成，检测到 ${result.count} 个数字</div>`;
                html += '<div class="results">';
                
                result.results.forEach((item, index) => {
                    const bbox = item.bbox;
                    html += `
                        <div class="result-item">
                            <h4>数字 ${index + 1}: ${item.label}</h4>
                            <p><strong>置信度:</strong> ${(item.confidence * 100).toFixed(1)}%</p>
                            <p><strong>位置:</strong> (${bbox.x1.toFixed(0)}, ${bbox.y1.toFixed(0)}) - (${bbox.x2.toFixed(0)}, ${bbox.y2.toFixed(0)})</p>
                            <p><strong>大小:</strong> ${(bbox.x2 - bbox.x1).toFixed(0)} × ${(bbox.y2 - bbox.y1).toFixed(0)} 像素</p>
                        </div>
                    `;
                });
                
                html += '</div>';
                
                if (result.parameters) {
                    html += `
                        <div style="margin-top: 15px; padding: 10px; background: #e9ecef; border-radius: 5px; font-size: 0.9em;">
                            <strong>参数:</strong> 置信度阈值 ${result.parameters.confidence_threshold}, NMS阈值 ${result.parameters.nms_threshold}
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
    return {"status": "healthy", "model_loaded": net is not None}

@app.post("/predict", tags=["预测"])
async def predict_image(file: UploadFile = File(...)):
    """
    上传图片进行数字识别
    """
    if not net:
        raise HTTPException(status_code=500, detail="模型未加载")
    
    # 检查文件类型
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="请上传图片文件")
    
    try:
        # 读取图片
        contents = await file.read()
        
        # 转换为OpenCV格式
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="无法解析图片")
        
        # 进行推理
        boxes, confs, ids = infer_img(img, thred_nms=0.4, thred_cond=0.5)
        
        # 格式化结果
        results = []
        for box, conf, id in zip(boxes, confs, ids):
            result = {
                "label": dic_labels[id],
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
        raise HTTPException(status_code=500, detail=f"处理图片时出错: {str(e)}")

@app.post("/predict_base64", tags=["预测"])
async def predict_base64(data: dict):
    """
    使用base64编码的图片进行数字识别
    请求格式: {"image": "base64编码的图片数据"}
    """
    if not net:
        raise HTTPException(status_code=500, detail="模型未加载")
    
    if "image" not in data:
        raise HTTPException(status_code=400, detail="请提供image字段")
    
    try:
        # 解码base64图片
        image_data = base64.b64decode(data["image"])
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="无法解析base64图片")
        
        # 进行推理
        boxes, confs, ids = infer_img(img, thred_nms=0.4, thred_cond=0.5)
        
        # 格式化结果
        results = []
        for box, conf, id in zip(boxes, confs, ids):
            result = {
                "label": dic_labels[id],
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
        raise HTTPException(status_code=500, detail=f"处理图片时出错: {str(e)}")

@app.post("/predict_with_params", tags=["预测"])
async def predict_with_params(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5,
    nms_threshold: float = 0.4
):
    """
    带参数的图片识别
    """
    if not net:
        raise HTTPException(status_code=500, detail="模型未加载")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="请上传图片文件")
    
    try:
        # 读取图片
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="无法解析图片")
        
        # 进行推理
        boxes, confs, ids = infer_img(img, thred_nms=nms_threshold, thred_cond=confidence_threshold)
        
        # 格式化结果
        results = []
        for box, conf, id in zip(boxes, confs, ids):
            result = {
                "label": dic_labels[id],
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
            "parameters": {
                "confidence_threshold": confidence_threshold,
                "nms_threshold": nms_threshold
            },
            "image_shape": {
                "height": img.shape[0],
                "width": img.shape[1],
                "channels": img.shape[2]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理图片时出错: {str(e)}")

if __name__ == "__main__":
    print("启动YOLO数字识别API服务...")
    print("API文档地址: http://localhost:8005/docs")
    uvicorn.run(app, host="0.0.0.0", port=8005)
