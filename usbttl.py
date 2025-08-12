#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#持续接收  start_receive()
#单次接收  receive_usb_ttl(port='/dev/ttyUSB0', baudrate=115200, timeout=5)
#发送 send_usb_ttl('Hello World!\n')
"""
树莓派 USB 转 TTL 收发数据工具函数 - 简化版
支持常开接收模式，接口简单易用
"""

import serial
import time
import threading
from queue import Queue

# 全局变量用于常开接收
_global_receiver = None
_global_port = '/dev/ttyUSB0'
_global_baudrate = 115200

def send_usb_ttl(data, port=None, baudrate=None):
    """
    简化的USB转TTL发送函数
    :param data: 要发送的字符串或字节
    :param port: 串口设备名（可选，默认使用全局配置）
    :param baudrate: 波特率（可选，默认使用全局配置）
    :return: True/False
    """
    global _global_port, _global_baudrate
    
    # 使用全局配置或传入的参数
    use_port = port or _global_port
    use_baudrate = baudrate or _global_baudrate
    
    try:
        ser = serial.Serial(port=use_port, baudrate=use_baudrate, timeout=1)
        if isinstance(data, str):
            data = data.encode('utf-8')
        ser.write(data)
        ser.flush()
        print(f"发送: {data!r}")
        ser.close()
        return True
    except Exception as e:
        print(f"发送失败: {e}")
        return False

def start_receive(port=None, baudrate=None, callback=None):
    """
    启动常开接收模式（全局接收器）
    :param port: 串口设备名（可选）
    :param baudrate: 波特率（可选）
    :param callback: 接收到数据时的回调函数（可选）
    """
    global _global_receiver, _global_port, _global_baudrate
    
    if _global_receiver and _global_receiver.running:
        print("常开接收已启动")
        return True
    
    # 使用传入参数或全局配置
    use_port = port or _global_port
    use_baudrate = baudrate or _global_baudrate
    
    # 更新全局配置
    _global_port = use_port
    _global_baudrate = use_baudrate
    
    _global_receiver = GlobalReceiver(use_port, use_baudrate, callback)
    return _global_receiver.start()

def stop_receive():
    """
    停止常开接收模式
    """
    global _global_receiver
    
    if _global_receiver:
        _global_receiver.stop()
        _global_receiver = None
        print("常开接收已停止")
    else:
        print("常开接收未启动")

def get_received_data():
    """
    获取接收到的数据（从全局接收器）
    :return: 接收到的数据列表
    """
    global _global_receiver
    
    if _global_receiver:
        return _global_receiver.get_data()
    else:
        return []

def is_receiving():
    """
    检查是否正在接收
    :return: True/False
    """
    global _global_receiver
    return _global_receiver and _global_receiver.running

def set_config(port='/dev/ttyUSB0', baudrate=115200):
    """
    设置全局配置
    :param port: 串口设备名
    :param baudrate: 波特率
    """
    global _global_port, _global_baudrate
    _global_port = port
    _global_baudrate = baudrate
    print(f"配置已更新: {port} @ {baudrate}")

class GlobalReceiver:
    """
    全局接收器类
    """
    def __init__(self, port, baudrate, callback=None):
        self.port = port
        self.baudrate = baudrate
        self.callback = callback
        self.serial = None
        self.running = False
        self.receive_queue = Queue()
        self.receive_thread = None
    
    def start(self):
        """
        启动接收
        """
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.1,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            
            self.running = True
            self.receive_thread = threading.Thread(target=self._receive_worker, daemon=True)
            self.receive_thread.start()
            
            print(f"✓ 常开接收已启动: {self.port} @ {self.baudrate}")
            return True
            
        except Exception as e:
            print(f"✗ 启动常开接收失败: {e}")
            return False
    
    def stop(self):
        """
        停止接收
        """
        self.running = False
        if self.receive_thread:
            self.receive_thread.join(timeout=1)
        if self.serial and self.serial.is_open:
            self.serial.close()
    
    def _receive_worker(self):
        """
        接收工作线程
        """
        while self.running:
            try:
                if self.serial and self.serial.is_open and self.serial.in_waiting > 0:
                    data = self.serial.read(self.serial.in_waiting)
                    self.receive_queue.put(data)
                    
                    # 实时打印接收到的数据
                    current_time = time.strftime('%H:%M:%S')
                    try:
                        text_data = data.decode('utf-8', errors='ignore').strip()
                        if text_data:
                            print(f"[{current_time}] 收到: '{text_data}'")
                        
                        # 显示16进制格式
                        hex_str = ' '.join([f'0x{b:02X}' for b in data])
                        print(f"[{current_time}] 数据: [{hex_str}]")
                        
                        # 如果是标准棋盘数据格式，解析并显示
                        if len(data) == 4 and data[0] == 0xAA and data[3] == 0xBB:
                            black_byte = data[1]
                            white_byte = data[2]
                            black_binary = format(black_byte, '08b')
                            white_binary = format(white_byte, '08b')
                            black_grid = black_binary[-9:] if len(black_binary) >= 9 else black_binary.zfill(9)
                            white_grid = white_binary[-9:] if len(white_binary) >= 9 else white_binary.zfill(9)
                            print(f"[{current_time}] 棋盘: 黑棋:{black_grid}, 白棋:{white_grid}")
                            
                    except Exception:
                        hex_str = ' '.join([f'0x{b:02X}' for b in data])
                        print(f"[{current_time}] 数据: [{hex_str}]")
                    
                    # 调用回调函数
                    if self.callback:
                        try:
                            self.callback(data)
                        except Exception as e:
                            print(f"回调函数错误: {e}")
                            
            except Exception as e:
                if self.running:
                    print(f"接收错误: {e}")
            
            time.sleep(0.01)  # 10ms间隔
    
    def get_data(self):
        """
        获取接收到的数据
        """
        received_data = []
        while not self.receive_queue.empty():
            received_data.append(self.receive_queue.get())
        return received_data

def receive_usb_ttl(port='/dev/ttyUSB0', baudrate=115200, timeout=5):
    """
    单次接收数据（兼容性函数）
    :param port: 串口设备名
    :param baudrate: 波特率
    :param timeout: 接收超时时间（秒）
    :return: 接收到的数据或None
    """
    try:
        ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
        print(f"等待接收数据从 {port} @ {baudrate} (超时: {timeout}秒)...")
        
        received_data = b""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting)
                received_data += data
                print(f"接收到: {data!r}")
                
                # 如果收到换行符，认为是一条完整消息
                if b'\n' in data:
                    break
            time.sleep(0.01)  # 10ms间隔
        
        ser.close()
        
        if received_data:
            print(f"接收完成，总共收到 {len(received_data)} 字节")
            return received_data
        else:
            print("接收超时，未收到数据")
            return None
            
    except Exception as e:
        print(f"接收失败: {e}")
        return None

class USBTTLTransceiver:
    """
    USB转TTL收发器类，支持同时收发数据
    """
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.running = False
        self.receive_queue = Queue()
        self.receive_thread = None
    
    def connect(self):
        """
        连接串口
        """
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.1,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            print(f"✓ USB-TTL连接成功: {self.port} @ {self.baudrate}")
            return True
        except Exception as e:
            print(f"✗ USB-TTL连接失败: {e}")
            return False
    
    def disconnect(self):
        """
        断开连接
        """
        self.stop_receiving()
        if self.serial and self.serial.is_open:
            self.serial.close()
            print("USB-TTL连接已关闭")
    
    def send_data(self, data):
        """
        发送数据
        """
        if not self.serial or not self.serial.is_open:
            print("串口未连接")
            return False
        
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            bytes_sent = self.serial.write(data)
            self.serial.flush()
            print(f"发送: {data!r} ({bytes_sent} bytes)")
            return True
        except Exception as e:
            print(f"发送失败: {e}")
            return False
    
    def start_receiving(self):
        """
        开始后台接收数据
        """
        if self.running:
            return
        
        self.running = True
        
        def receive_worker():
            while self.running:
                try:
                    if self.serial and self.serial.is_open and self.serial.in_waiting > 0:
                        data = self.serial.read(self.serial.in_waiting)
                        self.receive_queue.put(data)
                        print(f"接收: {data!r}")
                except Exception as e:
                    print(f"接收错误: {e}")
                time.sleep(0.01)
        
        self.receive_thread = threading.Thread(target=receive_worker, daemon=True)
        self.receive_thread.start()
        print("开始后台接收数据...")
    
    def stop_receiving(self):
        """
        停止接收数据
        """
        self.running = False
        if self.receive_thread:
            self.receive_thread.join(timeout=1)
        print("停止接收数据")
    
    def get_received_data(self):
        """
        获取接收到的数据
        """
        received_data = []
        while not self.receive_queue.empty():
            received_data.append(self.receive_queue.get())
        return received_data
    
    def send_and_wait_response(self, data, timeout=5):
        """
        发送数据并等待回应
        """
        # 清空接收队列
        while not self.receive_queue.empty():
            self.receive_queue.get()
        
        # 发送数据
        if not self.send_data(data):
            return None
        
        # 等待回应
        start_time = time.time()
        received_data = b""
        
        while time.time() - start_time < timeout:
            while not self.receive_queue.empty():
                data_chunk = self.receive_queue.get()
                received_data += data_chunk
                
                # 如果收到换行符，认为是完整回应
                if b'\n' in data_chunk:
                    return received_data
            
            time.sleep(0.01)
        
        return received_data if received_data else None

# 示例用法
if __name__ == "__main__":
    print("USB-TTL收发测试 - 简化版")
    print("=" * 40)
    
    # 设置配置
    set_config(port='/dev/ttyUSB0', baudrate=115200)
    
    # 测试1: 启动常开接收
    print("\n1. 启动常开接收:")
    start_receive()
    
    # 测试2: 简单发送（使用全局配置）
    print("\n2. 测试发送功能:")
    send_usb_ttl('Hello World!\n')
    send_usb_ttl('测试中文\n')
    
    # 发送字节数据
    chess_data = bytearray([0xAA, 0x01, 0x02, 0xBB])
    send_usb_ttl(chess_data)
    
    # 等待一段时间看接收效果
    print("\n3. 等待接收数据（5秒）...")
    time.sleep(5)
    
    # 获取接收到的数据
    received_data = get_received_data()
    if received_data:
        print(f"\n4. 获取到的数据: {len(received_data)} 条")
        for i, data in enumerate(received_data):
            print(f"  数据{i+1}: {data!r}")
    else:
        print("\n4. 未获取到缓存数据")
    
    # 测试5: 兼容性函数测试
    print("\n5. 测试兼容性接收函数:")
    received = receive_usb_ttl(timeout=2)
    if received:
        print(f"兼容性函数接收到: {received}")
    
    # 停止接收
    print("\n6. 停止常开接收:")
    stop_receive()
    
    print("\n测试完成!")
    print("\n简化使用方法:")
    print("1. start_receive()        # 启动常开接收")
    print("2. send_usb_ttl('data')   # 发送数据")
    print("3. get_received_data()    # 获取接收数据")
    print("4. stop_receive()         # 停止接收")