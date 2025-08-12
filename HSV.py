import cv2
import numpy as np

def nothing(val):
    """滑动条回调函数（空函数）"""
    pass

def hsv_color_picker():
    """HSV颜色调试工具 - 实时调节HSV参数并显示检测效果"""
    
    print("正在初始化HSV颜色调试工具...")
    
    # 检查OpenCV版本
    try:
        print(f"OpenCV版本: {cv2.__version__}")
    except Exception as e:
        print(f"OpenCV导入错误: {e}")
        return
    
    # 初始化摄像头
    print("正在尝试打开摄像头...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ 错误：无法打开摄像头 (设备索引 0)")
        print("尝试其他摄像头索引...")
        for i in range(1, 5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"✅ 成功打开摄像头 (设备索引 {i})")
                break
            cap.release()
        else:
            print("❌ 错误：找不到可用的摄像头设备")
            print("请检查：")
            print("1. 摄像头是否已连接")
            print("2. 摄像头是否被其他程序占用")
            print("3. 权限是否足够")
            return
    else:
        print("✅ 成功打开摄像头 (设备索引 0)")
    
    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 创建主窗口
    print("正在创建显示窗口...")
    try:
        cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
        cv2.namedWindow('HSV Controls', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
        
        # 调整窗口大小和位置
        cv2.resizeWindow('HSV Controls', 600, 400)
        cv2.moveWindow('Original', 50, 50)
        cv2.moveWindow('HSV Controls', 700, 50)
        cv2.moveWindow('Mask', 50, 400)
        cv2.moveWindow('Result', 700, 400)
        print("✅ 窗口创建成功")
    except Exception as e:
        print(f"❌ 窗口创建失败: {e}")
        cap.release()
        return
    
    # 初始HSV范围值（紫色为例）
    initial_values = {
        'h_min': 130, 'h_max': 160,
        's_min': 50, 's_max': 255,
        'v_min': 50, 'v_max': 255,
        'h_min2': 0, 'h_max2': 0,  # 第二范围（用于红色等跨越色轮的颜色）
        'use_dual': 0,  # 是否使用双范围
        'morphology': 7  # 形态学核大小
    }
    
    # 创建HSV调节滑动条
    print("正在创建调节滑动条...")
    try:
        cv2.createTrackbar('H Min', 'HSV Controls', initial_values['h_min'], 179, nothing)
        cv2.createTrackbar('H Max', 'HSV Controls', initial_values['h_max'], 179, nothing)
        cv2.createTrackbar('S Min', 'HSV Controls', initial_values['s_min'], 255, nothing)
        cv2.createTrackbar('S Max', 'HSV Controls', initial_values['s_max'], 255, nothing)
        cv2.createTrackbar('V Min', 'HSV Controls', initial_values['v_min'], 255, nothing)
        cv2.createTrackbar('V Max', 'HSV Controls', initial_values['v_max'], 255, nothing)
        
        # 双范围支持（主要用于红色）
        cv2.createTrackbar('Use Dual Range', 'HSV Controls', initial_values['use_dual'], 1, nothing)
        cv2.createTrackbar('H Min2', 'HSV Controls', initial_values['h_min2'], 179, nothing)
        cv2.createTrackbar('H Max2', 'HSV Controls', initial_values['h_max2'], 179, nothing)
        
        # 形态学处理
        cv2.createTrackbar('Morphology Kernel', 'HSV Controls', initial_values['morphology'], 15, nothing)
        print("✅ 滑动条创建成功")
    except Exception as e:
        print(f"❌ 滑动条创建失败: {e}")
        cap.release()
        cv2.destroyAllWindows()
        return
    
    # 预设颜色按钮（通过键盘切换）
    presets = {
        '1': {'name': '红色', 'h_min': 0, 'h_max': 10, 's_min': 50, 'v_min': 50, 'use_dual': 1, 'h_min2': 170, 'h_max2': 179},
        '2': {'name': '绿色', 'h_min': 40, 'h_max': 80, 's_min': 50, 'v_min': 50, 'use_dual': 0},
        '3': {'name': '蓝色', 'h_min': 100, 'h_max': 130, 's_min': 50, 'v_min': 50, 'use_dual': 0},
        '4': {'name': '黄色', 'h_min': 20, 'h_max': 30, 's_min': 50, 'v_min': 50, 'use_dual': 0},
        '5': {'name': '橙色', 'h_min': 10, 'h_max': 20, 's_min': 50, 'v_min': 50, 'use_dual': 0},
        '6': {'name': '紫色', 'h_min': 130, 'h_max': 160, 's_min': 50, 'v_min': 50, 'use_dual': 0},
    }
    
    print("=== HSV颜色调试工具 ===")
    print("功能说明：")
    print("1. 实时调节HSV参数，查看检测效果")
    print("2. 支持双范围检测（用于红色等跨越色轮的颜色）")
    print("3. 形态学处理去除噪点")
    print("4. 显示检测到的轮廓和中心点")
    print("\n键盘快捷键：")
    print("1-6: 快速切换预设颜色")
    print("S: 保存当前HSV参数到文件")
    print("R: 重置为默认值")
    print("C: 复制当前参数到剪贴板")
    print("ESC/Q: 退出程序")
    print("\n预设颜色：")
    for key, preset in presets.items():
        print(f"  {key}: {preset['name']}")
    print("\n开始检测...")
    
    current_preset = "自定义"
    
    try:
        frame_count = 0
        print("开始主循环...")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ 无法读取摄像头画面")
                break
            
            frame_count += 1
            if frame_count == 1:
                print("✅ 成功读取摄像头画面")
                actual_width = frame.shape[1]
                actual_height = frame.shape[0]
                print(f"摄像头分辨率: {actual_width}x{actual_height}")
            
            # 每100帧打印一次状态（避免输出过多）
            if frame_count % 100 == 0:
                print(f"正在处理第 {frame_count} 帧...")
            
            # 获取当前滑动条的值
            h_min = cv2.getTrackbarPos('H Min', 'HSV Controls')
            h_max = cv2.getTrackbarPos('H Max', 'HSV Controls')
            s_min = cv2.getTrackbarPos('S Min', 'HSV Controls')
            s_max = cv2.getTrackbarPos('S Max', 'HSV Controls')
            v_min = cv2.getTrackbarPos('V Min', 'HSV Controls')
            v_max = cv2.getTrackbarPos('V Max', 'HSV Controls')
            use_dual = cv2.getTrackbarPos('Use Dual Range', 'HSV Controls')
            h_min2 = cv2.getTrackbarPos('H Min2', 'HSV Controls')
            h_max2 = cv2.getTrackbarPos('H Max2', 'HSV Controls')
            morphology_size = cv2.getTrackbarPos('Morphology Kernel', 'HSV Controls')
            
            # 转换为HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 创建掩码
            lower_bound = np.array([h_min, s_min, v_min])
            upper_bound = np.array([h_max, s_max, v_max])
            mask1 = cv2.inRange(hsv, lower_bound, upper_bound)
            
            # 如果启用双范围
            if use_dual:
                lower_bound2 = np.array([h_min2, s_min, v_min])
                upper_bound2 = np.array([h_max2, s_max, v_max])
                mask2 = cv2.inRange(hsv, lower_bound2, upper_bound2)
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = mask1
            
            # 形态学处理
            if morphology_size > 0:
                kernel_size = max(3, morphology_size)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # 应用掩码
            result = cv2.bitwise_and(frame, frame, mask=mask)
            
            # 查找轮廓并绘制
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 在原图上绘制检测结果
            display_frame = frame.copy()
            detected_objects = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # 过滤小轮廓
                    # 绘制轮廓
                    cv2.drawContours(display_frame, [contour], -1, (0, 255, 0), 2)
                    
                    # 计算中心点
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.circle(display_frame, (cx, cy), 8, (255, 0, 0), -1)
                        cv2.putText(display_frame, f"({cx},{cy})", (cx-30, cy-15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        detected_objects += 1
            
            # 显示参数信息
            info_text = [
                f"预设: {current_preset}",
                f"H: {h_min}-{h_max}" + (f", {h_min2}-{h_max2}" if use_dual else ""),
                f"S: {s_min}-{s_max}, V: {v_min}-{v_max}",
                f"形态学核: {morphology_size}",
                f"检测到: {detected_objects} 个对象"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(display_frame, text, (10, 30 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # 显示图像
            cv2.imshow('Original', display_frame)
            cv2.imshow('Mask', mask)
            cv2.imshow('Result', result)
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            
            # 退出
            if key == 27 or key == ord('q') or key == ord('Q'):
                break
            
            # 预设颜色切换
            elif chr(key) in presets:
                preset = presets[chr(key)]
                current_preset = preset['name']
                print(f"切换到预设: {current_preset}")
                
                cv2.setTrackbarPos('H Min', 'HSV Controls', preset['h_min'])
                cv2.setTrackbarPos('H Max', 'HSV Controls', preset['h_max'])
                cv2.setTrackbarPos('S Min', 'HSV Controls', preset['s_min'])
                cv2.setTrackbarPos('S Max', 'HSV Controls', 255)
                cv2.setTrackbarPos('V Min', 'HSV Controls', preset['v_min'])
                cv2.setTrackbarPos('V Max', 'HSV Controls', 255)
                cv2.setTrackbarPos('Use Dual Range', 'HSV Controls', preset.get('use_dual', 0))
                if 'h_min2' in preset:
                    cv2.setTrackbarPos('H Min2', 'HSV Controls', preset['h_min2'])
                    cv2.setTrackbarPos('H Max2', 'HSV Controls', preset['h_max2'])
            
            # 保存参数
            elif key == ord('s') or key == ord('S'):
                params = {
                    'name': current_preset,
                    'h_min': h_min, 'h_max': h_max,
                    's_min': s_min, 's_max': s_max,
                    'v_min': v_min, 'v_max': v_max,
                    'use_dual': bool(use_dual),
                    'h_min2': h_min2, 'h_max2': h_max2,
                    'morphology': morphology_size
                }
                
                # 保存到文件
                with open('hsv_params.txt', 'w', encoding='utf-8') as f:
                    f.write(f"# HSV参数 - {current_preset}\n")
                    f.write(f"# 生成时间: {cv2.getTickCount()}\n\n")
                    
                    if use_dual:
                        f.write("# 双范围模式（适用于红色等）\n")
                        f.write(f"lower1 = [{h_min}, {s_min}, {v_min}]\n")
                        f.write(f"upper1 = [{h_max}, 255, 255]\n")
                        f.write(f"lower2 = [{h_min2}, {s_min}, {v_min}]\n")
                        f.write(f"upper2 = [{h_max2}, 255, 255]\n")
                        f.write("use_dual_range = True\n")
                    else:
                        f.write("# 单范围模式\n")
                        f.write(f"lower = [{h_min}, {s_min}, {v_min}]\n")
                        f.write(f"upper = [{h_max}, 255, 255]\n")
                        f.write("use_dual_range = False\n")
                    
                    f.write(f"\n# 形态学处理核大小\n")
                    f.write(f"morphology_kernel = {morphology_size}\n")
                
                print(f"参数已保存到 hsv_params.txt")
                print(f"当前参数: H({h_min}-{h_max}), S({s_min}-{s_max}), V({v_min}-{v_max})")
                if use_dual:
                    print(f"第二范围: H({h_min2}-{h_max2})")
            
            # 复制参数到控制台
            elif key == ord('c') or key == ord('C'):
                if use_dual:
                    param_str = f"# {current_preset} - 双范围\n"
                    param_str += f"'lower1': [{h_min}, {s_min}, {v_min}], 'upper1': [{h_max}, 255, 255],\n"
                    param_str += f"'lower2': [{h_min2}, {s_min}, {v_min}], 'upper2': [{h_max2}, 255, 255], 'use_dual_range': True"
                else:
                    param_str = f"# {current_preset} - 单范围\n"
                    param_str += f"'lower1': [{h_min}, {s_min}, {v_min}], 'upper1': [{h_max}, 255, 255], 'use_dual_range': False"
                
                print("\n=== 复制以下参数到代码中 ===")
                print(param_str)
                print("================================\n")
            
            # 重置参数
            elif key == ord('r') or key == ord('R'):
                current_preset = "默认"
                print("重置为默认参数")
                cv2.setTrackbarPos('H Min', 'HSV Controls', 0)
                cv2.setTrackbarPos('H Max', 'HSV Controls', 179)
                cv2.setTrackbarPos('S Min', 'HSV Controls', 0)
                cv2.setTrackbarPos('S Max', 'HSV Controls', 255)
                cv2.setTrackbarPos('V Min', 'HSV Controls', 0)
                cv2.setTrackbarPos('V Max', 'HSV Controls', 255)
                cv2.setTrackbarPos('Use Dual Range', 'HSV Controls', 0)
                cv2.setTrackbarPos('Morphology Kernel', 'HSV Controls', 7)
    
    except Exception as e:
        print(f"程序错误: {e}")
    
    finally:
        # 清理资源
        try:
            cap.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except:
            pass
        
        print("HSV调试工具已退出")

if __name__ == "__main__":
    try:
        print("启动HSV颜色调试工具...")
        hsv_color_picker()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序运行错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("程序结束")
