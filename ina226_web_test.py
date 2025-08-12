#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试INA226 Web API路由节点
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_ina226_routes():
    """测试所有INA226相关的路由"""
    print("=== INA226 Web API 路由测试 ===\n")
    
    # 1. 测试状态接口
    print("1. 测试状态接口...")
    try:
        response = requests.get(f"{BASE_URL}/api/ina226/status")
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"状态接口测试失败: {e}")
    print()
    
    # 2. 测试初始化接口
    print("2. 测试初始化接口...")
    try:
        response = requests.post(f"{BASE_URL}/api/ina226/init")
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"初始化接口测试失败: {e}")
    print()
    
    # 等待初始化完成
    time.sleep(1)
    
    # 3. 测试基本测量数据接口
    print("3. 测试基本测量数据接口...")
    try:
        response = requests.get(f"{BASE_URL}/api/ina226/measurements")
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        else:
            print(f"错误响应: {response.text}")
    except Exception as e:
        print(f"基本测量数据接口测试失败: {e}")
    print()
    
    # 4. 测试详细测量数据接口
    print("4. 测试详细测量数据接口...")
    try:
        response = requests.get(f"{BASE_URL}/api/ina226/measurements/detailed")
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        else:
            print(f"错误响应: {response.text}")
    except Exception as e:
        print(f"详细测量数据接口测试失败: {e}")
    print()
    
    # 5. 测试格式化测量数据接口
    print("5. 测试格式化测量数据接口...")
    try:
        response = requests.get(f"{BASE_URL}/api/ina226/measurements/formatted")
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        else:
            print(f"错误响应: {response.text}")
    except Exception as e:
        print(f"格式化测量数据接口测试失败: {e}")
    print()
    
    # 6. 测试摘要数据接口
    print("6. 测试摘要数据接口...")
    try:
        response = requests.get(f"{BASE_URL}/api/ina226/summary")
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            # 只显示部分数据，避免输出过长
            print("响应包含以下键:")
            print(f"  success: {data.get('success')}")
            if 'data' in data and data['data']:
                print(f"  data keys: {list(data['data'].keys())}")
        else:
            print(f"错误响应: {response.text}")
    except Exception as e:
        print(f"摘要数据接口测试失败: {e}")
    print()
    
    # 7. 测试监控数据接口
    print("7. 测试监控数据接口 (5次采样，0.1秒间隔)...")
    try:
        response = requests.get(f"{BASE_URL}/api/ina226/monitoring?samples=5&interval=0.1")
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("响应包含以下键:")
            print(f"  success: {data.get('success')}")
            if 'data' in data and data['data']:
                print(f"  data keys: {list(data['data'].keys())}")
                if 'statistics' in data['data']:
                    print(f"  statistics keys: {list(data['data']['statistics'].keys())}")
        else:
            print(f"错误响应: {response.text}")
    except Exception as e:
        print(f"监控数据接口测试失败: {e}")
    print()
    
    # 8. 测试配置更新接口
    print("8. 测试配置更新接口...")
    try:
        config_data = {
            "max_expected_amps": 3.0  # 更新最大预期电流
        }
        response = requests.post(f"{BASE_URL}/api/ina226/config", json=config_data)
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"配置更新接口测试失败: {e}")
    print()
    
    print("=== 测试完成 ===")

def continuous_monitoring_test():
    """连续监控测试"""
    print("=== INA226 连续监控测试 ===")
    print("每2秒获取一次格式化数据，按Ctrl+C停止\n")
    
    try:
        while True:
            try:
                response = requests.get(f"{BASE_URL}/api/ina226/measurements/formatted")
                if response.status_code == 200:
                    data = response.json()
                    if data['success']:
                        measurements = data['data']
                        print(f"时间: {measurements.get('timestamp', 'N/A')}")
                        print(f"电压: {measurements.get('bus_voltage', 'N/A')}")
                        print(f"电流: {measurements.get('current', 'N/A')}")
                        print(f"功率: {measurements.get('power', 'N/A')}")
                        print("-" * 40)
                    else:
                        print("获取数据失败")
                else:
                    print(f"HTTP错误: {response.status_code}")
                
                time.sleep(2)
                
            except Exception as e:
                print(f"请求错误: {e}")
                time.sleep(2)
                
    except KeyboardInterrupt:
        print("\n监控已停止")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "monitor":
        continuous_monitoring_test()
    else:
        test_ina226_routes()
        print("\n运行 'python ina226_web_test.py monitor' 进行连续监控测试")
