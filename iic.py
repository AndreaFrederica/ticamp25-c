import fcntl
import time
import sys

# Linux I2C控制码 - 可供其他模块使用
I2C_SLAVE = 0x0703        # ioctl命令：设置I2C从机地址
I2C_SMBUS = 0x0720        # ioctl命令：SMBus功能控制

# 配置参数 - 可供其他模块使用和修改
DEV_PATH = "/dev/i2c-1"   # I2C总线设备路径（树莓派通常为i2c-1）
DEV_ADDRESS = 0x33        # 从机设备地址（STM32配置的地址）
SEND_INTERVAL = 0.01         # 发送间隔（秒）

# 要发送的数据 - 可供其他模块使用和修改
DATA_TO_SEND = b"\xAA\x55\x01\x02\x03\x04\x05\x55"  # 默认发送数据

# 全局I2C设备句柄（避免重复打开关闭）
_i2c_device = None
_current_address = None

def disable_smbus(device):
    """显式禁用SMBus模式（针对某些内核版本）"""
    try:
        # 使用ioctl设置I2C功能标志，移除非必要功能
        funcs = fcntl.ioctl(device, I2C_SMBUS, 0)
        disable_flags = funcs & ~(1 << 0 | 1 << 1)  # 禁用SMBus Quick Command等功能
        fcntl.ioctl(device, I2C_SMBUS, disable_flags)
    except IOError:
        pass  # 如果失败，继续尝试基础I2C操作

def send_i2c_data():
    """使用原始I2C协议发送数据"""
    try:
        # 1. 打开I2C设备（读写模式）
        with open(DEV_PATH, "wb+", buffering=0) as i2c_dev:  # 非缓冲模式
            
            # 2. 设置从机地址
            fcntl.ioctl(i2c_dev, I2C_SLAVE, DEV_ADDRESS)
            
            # 3. 显式禁用SMBus功能（可选）
            disable_smbus(i2c_dev)
            
            # 4. 发送原始I2C数据
            bytes_written = i2c_dev.write(DATA_TO_SEND)
            
            print(f"已发送数据: {DATA_TO_SEND.hex(' ', 1)} -> 地址 {hex(DEV_ADDRESS)}")
            return bytes_written
            
    except IOError as e:
        print(f"❌ I2C发送失败: {str(e)}")
        return 0
    except Exception as e:
        print(f"❌ 未知错误: {str(e)}")
        return 0

def send_custom_i2c_data(data, address=None):
    """发送自定义I2C数据（优化版本 - 减少延迟）
    
    Args:
        data (bytes): 要发送的数据
        address (int, optional): 目标设备地址，如果不指定则使用默认地址
    
    Returns:
        int: 发送的字节数
    """
    try:
        target_address = address if address is not None else DEV_ADDRESS
        
        # 1. 打开I2C设备（读写模式，优化缓冲设置）
        with open(DEV_PATH, "wb", buffering=0) as i2c_dev:  # 只写模式，非缓冲
            
            # 2. 设置从机地址
            fcntl.ioctl(i2c_dev, I2C_SLAVE, target_address)
            
            # 3. 跳过SMBus禁用步骤以提高速度（如果不需要的话）
            # disable_smbus(i2c_dev)  # 注释掉这行可以稍微提速
            
            # 4. 直接发送数据，不做额外检查
            bytes_written = i2c_dev.write(data)
            i2c_dev.flush()  # 强制刷新缓冲区
            
            # 5. 减少打印输出（打印会增加延迟）
            # print(f"已发送自定义数据: {data.hex(' ', 1)} -> 地址 {hex(target_address)}")
            return bytes_written
            
    except IOError as e:
        # 简化错误处理
        return 0
    except Exception as e:
        return 0

def init_i2c_device():
    """初始化I2C设备（一次性操作）"""
    global _i2c_device, _current_address
    try:
        _i2c_device = open(DEV_PATH, "wb", buffering=0)
        fcntl.ioctl(_i2c_device, I2C_SLAVE, DEV_ADDRESS)
        _current_address = DEV_ADDRESS
        return True
    except:
        return False

def send_fast_i2c_data(data):
    """超快速I2C发送（使用全局设备句柄）"""
    global _i2c_device
    try:
        if _i2c_device is None:
            if not init_i2c_device():
                return 0
        
        bytes_written = _i2c_device.write(data)
        _i2c_device.flush()
        return bytes_written
    except:
        return 0

def close_i2c_device():
    """关闭I2C设备"""
    global _i2c_device
    if _i2c_device:
        _i2c_device.close()
        _i2c_device = None

if __name__ == "__main__":
    try:
        print("🚀 开始原始I2C数据传输... (Ctrl+C停止)")
        while True:
            if send_i2c_data() != len(DATA_TO_SEND):
                time.sleep(0.001)  # 失败后暂停0.01秒避免爆屏
            else:
                time.sleep(SEND_INTERVAL)
                
    except KeyboardInterrupt:
        print("\n🛑 用户中断，停止发送")