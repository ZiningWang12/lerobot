#!/bin/bash

# HILSERL强化学习训练完整启动脚本
# 基于ENVIRONMENT_SETUP.md的环境配置

set -e

echo "🚀 HILSERL强化学习训练启动"
echo "时间: $(date)"
echo "基于: ENVIRONMENT_SETUP.md环境配置"

# 1. 激活LeRobot环境
echo ""
echo "=== 步骤1: 激活LeRobot环境 ==="
if [[ -z "$VIRTUAL_ENV" ]] || [[ ! "$VIRTUAL_ENV" == *"lerobot_env"* ]]; then
    echo "激活LeRobot虚拟环境..."
    source lerobot_env/bin/activate
    echo "✅ 环境已激活: $VIRTUAL_ENV"
else
    echo "✅ LeRobot环境已激活: $VIRTUAL_ENV"
fi

# 2. 设置环境变量
echo ""
echo "=== 步骤2: 设置环境变量 ==="
export HF_HUB_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0
echo "✅ 环境变量已设置"

# 3. 验证环境
echo ""
echo "=== 步骤3: 验证环境 ==="
echo "检查Python版本..."
python --version

echo "检查PyTorch和GPU..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo "检查LeRobot..."
python -c "import lerobot; print('LeRobot版本:', lerobot.__version__)"

# 4. 检查配置文件
echo ""
echo "=== 步骤4: 检查配置文件 ==="
CONFIG_FILE="src/lerobot/configs/train_config_hilserl_smolvla_ring.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 错误: 配置文件 $CONFIG_FILE 不存在"
    exit 1
fi
echo "✅ 配置文件存在: $CONFIG_FILE"

# 5. 检查模型文件
echo ""
echo "=== 步骤5: 检查模型文件 ==="
REWARD_MODEL_PATH="outputs/reward_classifier_resnet10_ring_224_v2/checkpoints/006000/pretrained_model"
SMOLVLA_MODEL_PATH="outputs/train/2025-08-04/11-43-08_smolvla/checkpoints/008000/pretrained_model"

if [ ! -d "$REWARD_MODEL_PATH" ]; then
    echo "❌ 错误: Reward model路径不存在: $REWARD_MODEL_PATH"
    echo "请确保已训练好reward classifier"
    exit 1
fi
echo "✅ Reward model存在: $REWARD_MODEL_PATH"

if [ ! -d "$SMOLVLA_MODEL_PATH" ]; then
    echo "❌ 错误: smolVLA预训练模型路径不存在: $SMOLVLA_MODEL_PATH"
    echo "请确保已训练好smolVLA模型"
    exit 1
fi
echo "✅ smolVLA模型存在: $SMOLVLA_MODEL_PATH"

# 6. 检查设备文件
echo ""
echo "=== 步骤6: 检查设备文件 ==="
DEVICES=(
    "/dev/ttyACM0:Leader机器人串口"
    "/dev/ttyACM1:Follower机器人串口"
    "/dev/video4:Handeye摄像头"
    "/dev/video6:Global摄像头"
)

for device_info in "${DEVICES[@]}"; do
    device="${device_info%%:*}"
    desc="${device_info##*:}"
    if [ -e "$device" ]; then
        echo "✅ $desc: $device"
    else
        echo "❌ $desc: $device (不存在)"
        echo "请检查设备连接"
        exit 1
    fi
done

# 7. 设置设备权限
echo ""
echo "=== 步骤7: 设置设备权限 ==="
echo "设置串口权限..."
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
echo "✅ 设备权限已设置"

# 8. 创建输出目录
echo ""
echo "=== 步骤8: 创建输出目录 ==="
OUTPUT_DIR="outputs/train/hilserl_smolvla_ring"
mkdir -p "$OUTPUT_DIR/logs"
echo "✅ 输出目录已创建: $OUTPUT_DIR"

# 9. 启动训练
echo ""
echo "=== 步骤9: 启动HILSERL训练 ==="
echo "启动Learner服务器..."
echo "Learner将在后台运行，日志保存在 $OUTPUT_DIR/logs/learner_hilserl_smolvla_ring.log"

# 启动learner服务器
python -c "
import lerobot.robots.so101_follower
import lerobot.robots.so100_follower
import lerobot.teleoperators.so101_leader
import lerobot.teleoperators.so100_leader
import lerobot.teleoperators.gamepad
import lerobot.cameras.opencv
import lerobot.cameras.realsense
print('All modules imported successfully')
" && python -m lerobot.scripts.rl.learner \
    --config_path "$CONFIG_FILE" \
    > "$OUTPUT_DIR/logs/learner_hilserl_smolvla_ring.log" 2>&1 &

LEARNER_PID=$!
echo "Learner进程ID: $LEARNER_PID"

# 等待learner启动
echo "等待learner服务器启动..."
sleep 10

# 检查learner是否正常启动
if ! kill -0 $LEARNER_PID 2>/dev/null; then
    echo "❌ 错误: Learner服务器启动失败"
    echo "请检查日志文件: $OUTPUT_DIR/logs/learner_hilserl_smolvla_ring.log"
    exit 1
fi

echo "✅ Learner服务器启动成功"

echo ""
echo "=== 步骤10: 启动Actor服务器 ==="
echo "Actor将在前台运行，您可以通过游戏手柄进行人工干预"
echo "按Ctrl+C停止训练"
echo ""
echo "🎮 人工干预指南:"
echo "  - 当机器人行为不安全或无效时进行干预"
echo "  - 初期频繁干预，随着策略改进逐渐减少"
echo "  - 按下游戏手柄右上角扳机键接管控制"
echo ""

# 启动actor服务器
python -m lerobot.scripts.rl.actor \
    --config_path "$CONFIG_FILE"

echo ""
echo "=== 训练完成 ==="
echo "时间: $(date)"
echo "模型保存在: $OUTPUT_DIR/"
echo "日志文件: $OUTPUT_DIR/logs/" 