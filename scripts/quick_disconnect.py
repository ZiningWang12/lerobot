#!/usr/bin/env python3
"""
快速断开电机连接脚本
用于在record中断后快速关闭电机扭矩

使用方法:
    python scripts/quick_disconnect.py --port=/dev/ttyACM1
"""

import argparse
import logging
import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots import make_robot_from_config

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def disconnect_so101(port: str):
    """断开SO101 follower连接并关闭扭矩"""
    # 创建SO101 follower配置
    config = SO101FollowerConfig(port=port, disable_torque_on_disconnect=True)
    
    # 创建机器人实例
    robot = make_robot_from_config(config)
    logger.info(f"连接到SO101 follower: {port}")
    
    robot.connect()
    robot.is_connected = True
    
    # 断开连接（这会自动关闭扭矩）
    logger.info("断开机器人连接...")
    robot.disconnect()
    logger.info("✅ SO101 follower已断开连接，扭矩已关闭！")
    return True


def main():
    parser = argparse.ArgumentParser(description="快速断开SO101 follower连接")
    parser.add_argument("--port", type=str, default="/dev/ttyACM1",
                       help="串口路径 (默认: /dev/ttyACM1)")
    
    args = parser.parse_args()
    
    logger.info(f"开始断开SO101 follower连接...")
    logger.info(f"串口: {args.port}")
    
    if disconnect_so101(args.port):
        logger.info("🎉 断开连接成功！现在关节应该可以自由移动了。")
    else:
        logger.error("❌ 断开连接失败。请检查串口和连接状态。")

if __name__ == "__main__":
    main() 