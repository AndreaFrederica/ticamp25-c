import requests
import base64
import json

def test_api():
    """测试API的简单客户端"""
    
    # API地址
    base_url = "http://localhost:8005"
    
    print("=== YOLO数字识别API测试客户端 ===\n")
    
    # 1. 健康检查
    print("1. 健康检查...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"状态: {response.status_code}")
        print(f"响应: {response.json()}\n")
    except Exception as e:
        print(f"健康检查失败: {e}\n")
        return
    
    # 2. 测试文件上传
    print("2. 测试文件上传...")
    image_path = input("请输入图片路径 (或按Enter跳过): ").strip()
    
    if image_path:
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(f"{base_url}/predict", files=files)
            
            print(f"状态: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"检测到 {result['count']} 个数字:")
                for i, res in enumerate(result['results']):
                    print(f"  {i+1}. 数字: {res['label']}, 置信度: {res['confidence']:.3f}")
                    bbox = res['bbox']
                    print(f"     位置: ({bbox['x1']:.1f}, {bbox['y1']:.1f}) - ({bbox['x2']:.1f}, {bbox['y2']:.1f})")
            else:
                print(f"错误: {response.text}")
        except Exception as e:
            print(f"文件上传测试失败: {e}")
    
    print()
    
    # 3. 测试base64
    print("3. 测试base64...")
    if image_path:
        try:
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            data = {"image": image_data}
            response = requests.post(f"{base_url}/predict_base64", json=data)
            
            print(f"状态: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"检测到 {result['count']} 个数字:")
                for i, res in enumerate(result['results']):
                    print(f"  {i+1}. 数字: {res['label']}, 置信度: {res['confidence']:.3f}")
            else:
                print(f"错误: {response.text}")
        except Exception as e:
            print(f"base64测试失败: {e}")
    
    print()
    
    # 4. 测试带参数的预测
    print("4. 测试带参数的预测...")
    if image_path:
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                params = {
                    'confidence_threshold': 0.3,  # 降低置信度阈值
                    'nms_threshold': 0.4
                }
                response = requests.post(f"{base_url}/predict_with_params", files=files, params=params)
            
            print(f"状态: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"检测到 {result['count']} 个数字 (置信度阈值: 0.3):")
                for i, res in enumerate(result['results']):
                    print(f"  {i+1}. 数字: {res['label']}, 置信度: {res['confidence']:.3f}")
            else:
                print(f"错误: {response.text}")
        except Exception as e:
            print(f"参数测试失败: {e}")

def test_with_curl():
    """提供curl命令示例"""
    print("\n=== CURL命令示例 ===")
    print("1. 健康检查:")
    print("curl http://localhost:8005/health")
    print()
    
    print("2. 上传图片:")
    print("curl -X POST -F 'file=@your_image.jpg' http://localhost:8005/predict")
    print()
    
    print("3. base64图片:")
    print('curl -X POST -H "Content-Type: application/json" \\')
    print('     -d \'{"image":"base64编码的图片数据"}\' \\')
    print('     http://localhost:8005/predict_base64')
    print()
    
    print("4. 带参数的预测:")
    print("curl -X POST -F 'file=@your_image.jpg' \\")
    print("     'http://localhost:8005/predict_with_params?confidence_threshold=0.3&nms_threshold=0.4'")

if __name__ == "__main__":
    test_api()
    test_with_curl()
