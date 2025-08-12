#!/bin/bash

echo "=== 自动分割识别API服务启动脚本 ==="

# 检查YOLO API是否运行
echo "检查YOLO API服务状态..."
if curl -s http://localhost:8005/health > /dev/null; then
    echo "✅ YOLO API (端口8005) 运行正常"
else
    echo "❌ YOLO API (端口8005) 不可用"
    echo "请先启动YOLO API服务:"
    echo "python yolo_api.py"
    exit 1
fi

# 启动自动分割识别API
echo "启动自动分割识别API服务 (端口8006)..."
python auto_segment_api.py
