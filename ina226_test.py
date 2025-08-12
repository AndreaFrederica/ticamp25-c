#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INA226字典格式计量数据测试程序
演示各种获取字典格式数据的方法
"""

import json
import time
from ina226 import INA226

def test_measurement_functions():
    """测试各种测量函数"""
    print("=== INA226 字典格式计量数据测试 ===\n")
    
    try:
        # 初始化INA226
        print("初始化 INA226...")
        ina = INA226(i2c_bus=1, address=0x40, shunt_ohms=0.01, max_expected_amps=5.0)
        print("初始化完成!\n")
        
        # 1. 基本测量数据
        print("1. 基本测量数据 (get_all_measurements):")
        basic_data = ina.get_all_measurements()
        print(json.dumps(basic_data, indent=2, ensure_ascii=False))
        print()
        
        # 2. 带单位信息的测量数据
        print("2. 带单位信息的测量数据 (get_measurements_with_units):")
        detailed_data = ina.get_measurements_with_units()
        print(json.dumps(detailed_data, indent=2, ensure_ascii=False))
        print()
        
        # 3. 格式化的测量数据
        print("3. 格式化的测量数据 (get_formatted_measurements):")
        formatted_data = ina.get_formatted_measurements()
        print(json.dumps(formatted_data, indent=2, ensure_ascii=False))
        print()
        
        # 4. 摘要数据
        print("4. 摘要数据 (get_summary_dict):")
        summary_data = ina.get_summary_dict()
        print(json.dumps(summary_data, indent=2, ensure_ascii=False))
        print()
        
        # 5. 监控数据（多次采样）
        print("5. 监控数据 - 5次采样，间隔0.2秒 (get_monitoring_data):")
        monitoring_data = ina.get_monitoring_data(samples=5, interval=0.2)
        print(json.dumps(monitoring_data, indent=2, ensure_ascii=False))
        print()
        
        print("=== 所有测试完成 ===")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
    finally:
        try:
            ina.close()
        except Exception:
            pass

def continuous_monitoring_demo():
    """连续监控演示"""
    print("=== 连续监控演示 ===")
    print("每秒获取一次格式化的测量数据")
    print("按 Ctrl+C 停止\n")
    
    try:
        ina = INA226(i2c_bus=1, address=0x40, shunt_ohms=0.01, max_expected_amps=5.0)
        
        while True:
            # 获取格式化数据
            data = ina.get_formatted_measurements()
            
            # 打印表格形式的数据
            print(f"时间: {data['timestamp']}")
            print(f"分流电压: {data['shunt_voltage']}")
            print(f"总线电压: {data['bus_voltage']}")
            print(f"电流:     {data['current']}")
            print(f"功率:     {data['power']}")
            print("-" * 40)
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n监控已停止")
    except Exception as e:
        print(f"监控过程中发生错误: {e}")
    finally:
        try:
            ina.close()
        except Exception:
            pass

def save_data_to_file_demo():
    """将数据保存到文件的演示"""
    print("=== 数据保存演示 ===")
    
    try:
        ina = INA226(i2c_bus=1, address=0x40, shunt_ohms=0.01, max_expected_amps=5.0)
        
        # 获取详细的测量数据
        data = ina.get_measurements_with_units()
        
        # 保存到JSON文件
        filename = f"ina226_data_{int(time.time())}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"数据已保存到文件: {filename}")
        
        # 获取监控数据并保存
        monitoring_data = ina.get_monitoring_data(samples=10, interval=0.1)
        monitoring_filename = f"ina226_monitoring_{int(time.time())}.json"
        with open(monitoring_filename, 'w', encoding='utf-8') as f:
            json.dump(monitoring_data, f, indent=2, ensure_ascii=False)
        
        print(f"监控数据已保存到文件: {monitoring_filename}")
        
    except Exception as e:
        print(f"保存数据时发生错误: {e}")
    finally:
        try:
            ina.close()
        except Exception:
            pass

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "monitor":
            continuous_monitoring_demo()
        elif sys.argv[1] == "save":
            save_data_to_file_demo()
        else:
            print("用法:")
            print("  python ina226_test.py        # 运行所有测试")
            print("  python ina226_test.py monitor # 连续监控")
            print("  python ina226_test.py save   # 保存数据")
    else:
        test_measurement_functions()
