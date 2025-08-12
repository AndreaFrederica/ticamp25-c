#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速演示INA226字典格式数据功能
这个脚本使用模拟数据来演示功能，不需要真实的INA226硬件
"""

import json
import time

# 模拟INA226类的简化版本用于演示
class MockINA226:
    """模拟INA226类用于演示"""
    
    def __init__(self):
        self.address = 0x40
        self.shunt_ohms = 0.01
        self.max_expected_amps = 5.0
        self.current_lsb = 0.0001525
        self.power_lsb = 0.003815
        self._start_time = time.time()
    
    def get_shunt_voltage(self):
        """模拟分流电压"""
        return 2.5  # mV
    
    def get_bus_voltage(self):
        """模拟总线电压"""
        return 12.0  # V
    
    def get_current(self):
        """模拟电流"""
        return 0.25  # A
    
    def get_power(self):
        """模拟功率"""
        return 3.0  # W
    
    def get_all_measurements(self):
        """获取所有测量值"""
        return {
            'shunt_voltage_mv': self.get_shunt_voltage(),
            'bus_voltage_v': self.get_bus_voltage(),
            'current_a': self.get_current(),
            'power_w': self.get_power()
        }
    
    def get_measurements_with_units(self):
        """获取带单位信息的测量值字典"""
        shunt_mv = self.get_shunt_voltage()
        bus_v = self.get_bus_voltage()
        current_a = self.get_current()
        power_w = self.get_power()
        
        return {
            'timestamp': time.time(),
            'measurements': {
                'shunt_voltage': {'value': shunt_mv, 'unit': 'mV'},
                'bus_voltage': {'value': bus_v, 'unit': 'V'},
                'current': {'value': current_a, 'unit': 'A'},
                'current_ma': {'value': current_a * 1000, 'unit': 'mA'},
                'power': {'value': power_w, 'unit': 'W'},
                'power_mw': {'value': power_w * 1000, 'unit': 'mW'}
            },
            'device_info': {
                'address': f'0x{self.address:02X}',
                'shunt_ohms': self.shunt_ohms,
                'max_expected_amps': self.max_expected_amps,
                'current_lsb': self.current_lsb,
                'power_lsb': self.power_lsb
            }
        }
    
    def get_formatted_measurements(self):
        """获取格式化的测量值字典"""
        measurements = self.get_all_measurements()
        
        return {
            'shunt_voltage': f"{measurements['shunt_voltage_mv']:.3f} mV",
            'bus_voltage': f"{measurements['bus_voltage_v']:.3f} V",
            'current': f"{measurements['current_a']*1000:.3f} mA",
            'power': f"{measurements['power_w']*1000:.3f} mW",
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

def demo():
    """演示字典格式数据功能"""
    print("=== INA226 字典格式计量数据演示 ===\n")
    
    # 创建模拟传感器实例
    ina = MockINA226()
    
    # 1. 基本测量数据
    print("1. 基本测量数据:")
    basic_data = ina.get_all_measurements()
    print(json.dumps(basic_data, indent=2, ensure_ascii=False))
    print()
    
    # 2. 带单位信息的测量数据
    print("2. 带单位信息的测量数据:")
    detailed_data = ina.get_measurements_with_units()
    print(json.dumps(detailed_data, indent=2, ensure_ascii=False))
    print()
    
    # 3. 格式化的测量数据
    print("3. 格式化的测量数据:")
    formatted_data = ina.get_formatted_measurements()
    print(json.dumps(formatted_data, indent=2, ensure_ascii=False))
    print()
    
    print("=== 演示完成 ===")
    print("\n这些功能在实际的INA226类中都已实现！")
    print("要使用真实硬件，请运行: python ina226_test.py")

if __name__ == "__main__":
    demo()
