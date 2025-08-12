#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INA226电流/功率监控传感器驱动程序
适用于树莓派5的I2C接口
"""

import smbus2
import time

class INA226:
    """INA226电流/功率监控传感器类"""
    
    # 寄存器地址
    REG_CONFIG = 0x00
    REG_SHUNT_VOLTAGE = 0x01
    REG_BUS_VOLTAGE = 0x02
    REG_POWER = 0x03
    REG_CURRENT = 0x04
    REG_CALIBRATION = 0x05
    REG_MASK_ENABLE = 0x06
    REG_ALERT_LIMIT = 0x07
    REG_MANUFACTURER_ID = 0xFE
    REG_DIE_ID = 0xFF
    
    # 配置寄存器位定义
    CONFIG_RESET = 0x8000
    CONFIG_AVG_1 = 0x0000
    CONFIG_AVG_4 = 0x0200
    CONFIG_AVG_16 = 0x0400
    CONFIG_AVG_64 = 0x0600
    CONFIG_AVG_128 = 0x0800
    CONFIG_AVG_256 = 0x0A00
    CONFIG_AVG_512 = 0x0C00
    CONFIG_AVG_1024 = 0x0E00
    
    CONFIG_VBUS_CT_140US = 0x0000
    CONFIG_VBUS_CT_204US = 0x0040
    CONFIG_VBUS_CT_332US = 0x0080
    CONFIG_VBUS_CT_588US = 0x00C0
    CONFIG_VBUS_CT_1100US = 0x0100
    CONFIG_VBUS_CT_2116US = 0x0140
    CONFIG_VBUS_CT_4156US = 0x0180
    CONFIG_VBUS_CT_8244US = 0x01C0
    
    CONFIG_VSHUNT_CT_140US = 0x0000
    CONFIG_VSHUNT_CT_204US = 0x0008
    CONFIG_VSHUNT_CT_332US = 0x0010
    CONFIG_VSHUNT_CT_588US = 0x0018
    CONFIG_VSHUNT_CT_1100US = 0x0020
    CONFIG_VSHUNT_CT_2116US = 0x0028
    CONFIG_VSHUNT_CT_4156US = 0x0030
    CONFIG_VSHUNT_CT_8244US = 0x0038
    
    CONFIG_MODE_POWER_DOWN = 0x0000
    CONFIG_MODE_SHUNT_TRIG = 0x0001
    CONFIG_MODE_BUS_TRIG = 0x0002
    CONFIG_MODE_SHUNT_BUS_TRIG = 0x0003
    CONFIG_MODE_POWER_DOWN2 = 0x0004
    CONFIG_MODE_SHUNT_CONT = 0x0005
    CONFIG_MODE_BUS_CONT = 0x0006
    CONFIG_MODE_SHUNT_BUS_CONT = 0x0007
    
    def __init__(self, i2c_bus=1, address=0x40, shunt_ohms=0.01, max_expected_amps=3.2):
        """
        初始化INA226传感器
        Args:
            i2c_bus: I2C总线号
            address: I2C设备地址
            shunt_ohms: 分流电阻值（欧姆）
            max_expected_amps: 最大预期电流（安培）
        """
        self.bus = smbus2.SMBus(i2c_bus)
        self.address = address
        self.shunt_ohms = shunt_ohms
        self.max_expected_amps = max_expected_amps
        self._start_time = time.time()  # 记录启动时间
        
        # 初始化传感器
        self._initialize()
    
    def _initialize(self):
        """初始化传感器配置"""
        try:
            # 软重置
            self._write_register(self.REG_CONFIG, self.CONFIG_RESET)
            time.sleep(0.01)  # 等待重置完成
            
            # 验证制造商ID和设备ID
            manufacturer_id = self._read_register(self.REG_MANUFACTURER_ID)
            die_id = self._read_register(self.REG_DIE_ID)
            
            if manufacturer_id != 0x5449:  # "TI"的ASCII码
                raise ValueError(f"Invalid manufacturer ID: 0x{manufacturer_id:04X}")
            
            if die_id != 0x2260:  # INA226的设备ID
                raise ValueError(f"Invalid device ID: 0x{die_id:04X}")
            
            print(f"INA226 initialized successfully (address: 0x{self.address:02X})")
            
            # 配置传感器
            self._configure()
            
        except Exception as e:
            raise RuntimeError(f"INA226 initialization failed: {e}")
    
    def _configure(self):
        """配置传感器参数"""
        # 设置配置寄存器
        config = (self.CONFIG_AVG_16 |          # 平均16次采样
                 self.CONFIG_VBUS_CT_1100US |    # 总线电压转换时间
                 self.CONFIG_VSHUNT_CT_1100US |  # 分流电压转换时间
                 self.CONFIG_MODE_SHUNT_BUS_CONT) # 连续测量模式
        
        self._write_register(self.REG_CONFIG, config)
        
        # 计算并设置校准寄存器
        self._calibrate()
    
    def _calibrate(self):
        """校准传感器"""
        # 计算LSB (Least Significant Bit)
        self.current_lsb = self.max_expected_amps / 32768.0  # 15位ADC
        
        # 计算校准值
        # CAL = 0.00512 / (Current_LSB * Rshunt)
        cal_value = int(0.00512 / (self.current_lsb * self.shunt_ohms))
        
        # 确保校准值在有效范围内
        if cal_value > 0x7FFF:
            cal_value = 0x7FFF
        elif cal_value < 1:
            cal_value = 1
        
        self._write_register(self.REG_CALIBRATION, cal_value)
        
        # 功率LSB是电流LSB的25倍
        self.power_lsb = self.current_lsb * 25
        
        print("Calibration completed:")
        print(f"  Current LSB: {self.current_lsb*1000:.3f} mA/bit")
        print(f"  Power LSB: {self.power_lsb*1000:.3f} mW/bit")
        print(f"  Calibration value: 0x{cal_value:04X}")
    
    def _read_register(self, register):
        """读取16位寄存器值"""
        try:
            # SMBus使用小端序，但INA226使用大端序
            data = self.bus.read_word_data(self.address, register)
            # 转换字节序
            return ((data & 0xFF) << 8) | ((data >> 8) & 0xFF)
        except Exception as e:
            raise IOError(f"Failed to read register 0x{register:02X}: {e}")
    
    def _write_register(self, register, value):
        """写入16位寄存器值"""
        try:
            # 转换字节序
            swapped = ((value & 0xFF) << 8) | ((value >> 8) & 0xFF)
            self.bus.write_word_data(self.address, register, swapped)
        except Exception as e:
            raise IOError(f"Failed to write register 0x{register:02X}: {e}")
    
    def get_shunt_voltage(self):
        """
        获取分流电压（毫伏）
        Returns:
            float: 分流电压值（mV）
        """
        raw_value = self._read_register(self.REG_SHUNT_VOLTAGE)
        # 处理有符号数
        if raw_value > 32767:
            raw_value -= 65536
        # 分流电压LSB = 2.5μV
        return raw_value * 0.0025
    
    def get_bus_voltage(self):
        """
        获取总线电压（伏特）
        Returns:
            float: 总线电压值（V）
        """
        raw_value = self._read_register(self.REG_BUS_VOLTAGE)
        # 总线电压LSB = 1.25mV
        return raw_value * 0.00125
    
    def get_current(self):
        """
        获取电流（安培）
        Returns:
            float: 电流值（A）
        """
        raw_value = self._read_register(self.REG_CURRENT)
        # 处理有符号数
        if raw_value > 32767:
            raw_value -= 65536
        return raw_value * self.current_lsb
    
    def get_power(self):
        """
        获取功率（瓦特）
        Returns:
            float: 功率值（W）
        """
        raw_value = self._read_register(self.REG_POWER)
        return raw_value * self.power_lsb
    
    def get_all_measurements(self):
        """
        获取所有测量值
        Returns:
            dict: 包含所有测量值的字典
        """
        return {
            'shunt_voltage_mv': self.get_shunt_voltage(),
            'bus_voltage_v': self.get_bus_voltage(),
            'current_a': self.get_current(),
            'power_w': self.get_power()
        }
    
    def get_measurements_with_units(self):
        """
        获取带单位信息的测量值字典
        Returns:
            dict: 包含测量值和单位的字典
        """
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
        """
        获取格式化的测量值字典（便于显示）
        Returns:
            dict: 包含格式化字符串的字典
        """
        measurements = self.get_all_measurements()
        
        return {
            'shunt_voltage': f"{measurements['shunt_voltage_mv']:.3f} mV",
            'bus_voltage': f"{measurements['bus_voltage_v']:.3f} V",
            'current': f"{measurements['current_a']*1000:.3f} mA",
            'power': f"{measurements['power_w']*1000:.3f} mW",
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def get_summary_dict(self):
        """
        获取测量摘要字典（包含计算的额外信息）
        Returns:
            dict: 包含摘要信息的字典
        """
        measurements = self.get_all_measurements()
        current_a = measurements['current_a']
        bus_v = measurements['bus_voltage_v']
        power_w = measurements['power_w']
        
        # 计算电阻（如果电流不为零）
        load_resistance = bus_v / current_a if current_a != 0 else float('inf')
        
        # 计算功率效率（假设理想情况下的功率）
        theoretical_power = bus_v * current_a
        efficiency = (power_w / theoretical_power * 100) if theoretical_power != 0 else 0
        
        return {
            'basic_measurements': measurements,
            'calculated_values': {
                'load_resistance_ohms': load_resistance,
                'power_efficiency_percent': efficiency,
                'energy_1h_wh': power_w,  # 1小时的能耗（瓦时）
                'energy_24h_wh': power_w * 24,  # 24小时的能耗（瓦时）
            },
            'status': {
                'is_consuming_power': power_w > 0.001,  # 大于1mW认为在消耗功率
                'is_high_current': abs(current_a) > self.max_expected_amps * 0.8,  # 超过80%最大电流
                'measurement_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'uptime_seconds': time.time() - getattr(self, '_start_time', time.time())
            }
        }
    
    def get_monitoring_data(self, samples=1, interval=0.1):
        """
        获取多次采样的监控数据
        Args:
            samples: 采样次数
            interval: 采样间隔（秒）
        Returns:
            dict: 包含统计信息的字典
        """
        if samples < 1:
            samples = 1
            
        measurements_list = []
        
        for i in range(samples):
            measurements_list.append(self.get_all_measurements())
            if i < samples - 1:  # 最后一次不需要等待
                time.sleep(interval)
        
        # 计算统计信息
        currents = [m['current_a'] for m in measurements_list]
        voltages = [m['bus_voltage_v'] for m in measurements_list]
        powers = [m['power_w'] for m in measurements_list]
        
        return {
            'sample_info': {
                'samples': samples,
                'interval_seconds': interval,
                'total_time_seconds': (samples - 1) * interval
            },
            'statistics': {
                'current': {
                    'min_a': min(currents),
                    'max_a': max(currents),
                    'avg_a': sum(currents) / len(currents),
                    'min_ma': min(currents) * 1000,
                    'max_ma': max(currents) * 1000,
                    'avg_ma': sum(currents) / len(currents) * 1000
                },
                'voltage': {
                    'min_v': min(voltages),
                    'max_v': max(voltages),
                    'avg_v': sum(voltages) / len(voltages)
                },
                'power': {
                    'min_w': min(powers),
                    'max_w': max(powers),
                    'avg_w': sum(powers) / len(powers),
                    'min_mw': min(powers) * 1000,
                    'max_mw': max(powers) * 1000,
                    'avg_mw': sum(powers) / len(powers) * 1000
                }
            },
            'raw_data': measurements_list
        }
    
    def close(self):
        """关闭I2C连接"""
        self.bus.close()

def demo():
    """Demo program"""
    print("=== INA226 Current Sensor Demo Program ===\n")
    
    try:
        # 初始化INA226（假设使用0.1Ω分流电阻，最大电流3.2A）
        ina = INA226(i2c_bus=1, address=0x40, shunt_ohms=0.01, max_expected_amps=5.0)
        
        print("Starting continuous measurement...")
        print("Press Ctrl+C to stop measurement\n")
        
        while True:
            # 获取所有测量值
            measurements = ina.get_all_measurements()
            
            # 显示测量结果
            print(f"Shunt voltage: {measurements['shunt_voltage_mv']:8.3f} mV")
            print(f"Bus voltage:   {measurements['bus_voltage_v']:8.3f} V")
            print(f"Current:       {measurements['current_a']*1000:8.3f} mA")
            print(f"Power:         {measurements['power_w']*1000:8.3f} mW")
            print("-" * 40)
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nMeasurement stopped")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            ina.close()
        except Exception:
            pass

if __name__ == "__main__":
    demo()