#!/usr/bin/env python3

"""
舵机PID和Deadband参数调节脚本
用于解决teleop中的jittering问题
"""

import time
import sys
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

def test_motor_parameters(port, motor_name, motor_id, motor_model):
    """测试单个电机的参数"""
    print(f"\n=== 测试电机: {motor_name} (ID: {motor_id}) ===")
    
    try:
        # 创建单个电机的总线
        bus = FeetechMotorsBus(
            port=port,
            motors={motor_name: Motor(motor_id, motor_model, MotorNormMode.RANGE_M100_100)},
        )
        
        bus.connect()
        print(f"✓ {motor_name} 连接成功")
        
        # 读取当前PID参数
        print(f"\n当前PID参数:")
        p_coeff = bus.read("P_Coefficient", motor_name, normalize=False)
        i_coeff = bus.read("I_Coefficient", motor_name, normalize=False)
        d_coeff = bus.read("D_Coefficient", motor_name, normalize=False)
        print(f"  P_Coefficient: {p_coeff}")
        print(f"  I_Coefficient: {i_coeff}")
        print(f"  D_Coefficient: {d_coeff}")
        
        # 读取当前Deadband参数
        print(f"\n当前Deadband参数:")
        cw_deadzone = bus.read("CW_Dead_Zone", motor_name, normalize=False)
        ccw_deadzone = bus.read("CCW_Dead_Zone", motor_name, normalize=False)
        print(f"  CW_Dead_Zone: {cw_deadzone}")
        print(f"  CCW_Dead_Zone: {ccw_deadzone}")
        
        # 读取当前位置
        current_pos = bus.read("Present_Position", motor_name, normalize=False)
        print(f"\n当前位置: {current_pos}")
        
        bus.disconnect()
        return True
        
    except Exception as e:
        print(f"✗ {motor_name} 测试失败: {e}")
        return False

def set_motor_parameters(port, motor_name, motor_id, motor_model, p_coeff, i_coeff, d_coeff, cw_deadzone, ccw_deadzone):
    """设置单个电机的参数"""
    print(f"\n=== 设置电机参数: {motor_name} ===")
    
    try:
        bus = FeetechMotorsBus(
            port=port,
            motors={motor_name: Motor(motor_id, motor_model, MotorNormMode.RANGE_M100_100)},
        )
        
        bus.connect()
        
        # 禁用扭矩以安全设置参数
        bus.disable_torque()
        
        # 设置PID参数
        print(f"设置PID参数: P={p_coeff}, I={i_coeff}, D={d_coeff}")
        bus.write("P_Coefficient", motor_name, p_coeff)
        bus.write("I_Coefficient", motor_name, i_coeff)
        bus.write("D_Coefficient", motor_name, d_coeff)
        
        # 设置Deadband参数
        print(f"设置Deadband参数: CW={cw_deadzone}, CCW={ccw_deadzone}")
        bus.write("CW_Dead_Zone", motor_name, cw_deadzone)
        bus.write("CCW_Dead_Zone", motor_name, ccw_deadzone)
        
        # 重新启用扭矩
        bus.enable_torque()
        
        print(f"✓ {motor_name} 参数设置完成")
        bus.disconnect()
        return True
        
    except Exception as e:
        print(f"✗ {motor_name} 参数设置失败: {e}")
        return False

def interactive_tuning(port):
    """交互式参数调节"""
    print("\n=== 交互式舵机参数调节 ===")
    print("这个工具可以帮助您调节舵机的PID和Deadband参数来解决jittering问题")
    
    # SO101 follower 的电机配置
    motors_config = {
        "shoulder_pan": (1, "sts3215"),
        "shoulder_lift": (2, "sts3215"), 
        "elbow_flex": (3, "sts3215"),
        "wrist_flex": (4, "sts3215"),
        "wrist_roll": (5, "sts3215"),
        "gripper": (6, "sts3215"),
    }
    
    while True:
        print("\n" + "="*50)
        print("选择操作:")
        print("1. 查看所有电机当前参数")
        print("2. 调节单个电机参数")
        print("3. 应用推荐的抗抖动参数")
        print("4. 退出")
        
        choice = input("\n请输入选择 (1-4): ").strip()
        
        if choice == "1":
            print("\n查看所有电机参数...")
            for motor_name, (motor_id, motor_model) in motors_config.items():
                test_motor_parameters(port, motor_name, motor_id, motor_model)
                
        elif choice == "2":
            print("\n可用的电机:")
            for i, motor_name in enumerate(motors_config.keys(), 1):
                print(f"{i}. {motor_name}")
            
            try:
                motor_idx = int(input("选择电机编号: ")) - 1
                motor_names = list(motors_config.keys())
                if 0 <= motor_idx < len(motor_names):
                    motor_name = motor_names[motor_idx]
                    motor_id, motor_model = motors_config[motor_name]
                    
                    print(f"\n调节 {motor_name} 的参数:")
                    p_coeff = int(input("P_Coefficient (推荐: 8-16): ") or "12")
                    i_coeff = int(input("I_Coefficient (推荐: 0-4): ") or "0")
                    d_coeff = int(input("D_Coefficient (推荐: 16-32): ") or "24")
                    cw_deadzone = int(input("CW_Dead_Zone (推荐: 2-8): ") or "4")
                    ccw_deadzone = int(input("CCW_Dead_Zone (推荐: 2-8): ") or "4")
                    
                    set_motor_parameters(port, motor_name, motor_id, motor_model, 
                                       p_coeff, i_coeff, d_coeff, cw_deadzone, ccw_deadzone)
                else:
                    print("无效的电机编号")
            except ValueError:
                print("请输入有效的数字")
                
        elif choice == "3":
            print("\n应用推荐的抗抖动参数...")
            # 推荐的抗抖动参数
            recommended_params = {
                "P_Coefficient": 12,  # 降低P值减少振荡
                "I_Coefficient": 0,   # 保持I为0避免积分饱和
                "D_Coefficient": 24,  # 增加D值提供阻尼
                "CW_Dead_Zone": 4,    # 增加死区减少抖动
                "CCW_Dead_Zone": 2,
            }
            
            for motor_name, (motor_id, motor_model) in motors_config.items():
                if motor_name != "gripper":  # 跳过夹爪
                    set_motor_parameters(port, motor_name, motor_id, motor_model,
                                       recommended_params["P_Coefficient"],
                                       recommended_params["I_Coefficient"], 
                                       recommended_params["D_Coefficient"],
                                       recommended_params["CW_Dead_Zone"],
                                       recommended_params["CCW_Dead_Zone"])
            print("✓ 推荐参数已应用到所有电机（除夹爪外）")
            
        elif choice == "4":
            print("退出参数调节工具")
            break
            
        else:
            print("无效选择，请重新输入")

def main():
    port = "/dev/ttyACM1"  # 您的follower arm端口
    
    print("舵机PID和Deadband参数调节工具")
    print(f"使用端口: {port}")
    print("\n注意: 调节参数前请确保机器人处于安全位置")
    
    # 检查端口权限
    try:
        import os
        if not os.access(port, os.R_OK | os.W_OK):
            print(f"\n⚠️  警告: 端口 {port} 权限不足")
            print("请运行: sudo chmod 666 /dev/ttyACM1")
            return
    except Exception as e:
        print(f"端口检查失败: {e}")
        return
    
    interactive_tuning(port)

if __name__ == "__main__":
    main() 