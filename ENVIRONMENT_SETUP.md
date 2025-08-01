# LeRobot 环境启动指南

## 快速启动

### 1. 激活环境
```bash
# 激活虚拟环境
source lerobot_env/bin/activate

# 设置网络镜像（国内用户）
export HF_ENDPOINT=https://hf-mirror.com
```

### 2. 验证环境
```bash
# 检查Python版本
python --version  # 应为 Python 3.10.x

# 检查PyTorch和GPU支持
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 检查LeRobot安装
python -c "import lerobot; print('LeRobot版本:', lerobot.__version__)"
```

## 核心功能验证

### 3. 数据集可视化
```bash
# 可视化仿真数据集
python -m lerobot.scripts.visualize_dataset --repo-id lerobot/pusht --episode-index 0

# 可视化真机数据集
python -m lerobot.scripts.visualize_dataset --repo-id lerobot/aloha_static_coffee --episode-index 0
```

### 4. 预训练模型推理
```bash
# Diffusion Policy (GPU推理)
python -m lerobot.scripts.eval --policy.path=lerobot/diffusion_pusht --env.type=pusht --eval.batch_size=1 --eval.n_episodes=1 --policy.use_amp=true --policy.device=cuda

# ACT Policy (GPU推理)
python -m lerobot.scripts.eval --policy.path=lerobot/act_aloha_sim_insertion_human --env.type=aloha --eval.batch_size=1 --eval.n_episodes=1 --policy.use_amp=true --policy.device=cuda
```

### 5. 运行示例代码
```bash
# 加载数据集示例
python examples/1_load_lerobot_dataset.py

# 评估预训练模型示例
python examples/2_evaluate_pretrained_policy.py

# 训练模型示例
python examples/3_train_policy.py
```

### 6. 实时TeleOp调试界面

#### 重要参数说明
- **`--fps=30`**: 控制循环频率，每秒执行30次控制循环
  - 30fps: 平衡性能和稳定性（推荐）
  - 60fps: 更高响应速度，但增加系统负载
  - 10-15fps: 适合调试或资源有限时
- **`--display_data=true`**: 启用实时数据可视化（Rerun界面）
- **`--dataset.repo_id`**: 数据集存储的HuggingFace仓库ID
- **`--dataset.num_episodes`**: 录制的任务次数
- **`--dataset.episode_time_s`**: 每个任务的录制时长（秒）

#### 带双摄像头的TeleOp调试（HandEye + Global）
```bash
# 双臂+双摄像头实时调试（HandEye + Global）
python -m lerobot.teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=znw_arm_f1 \
    --robot.cameras='{"handeye": {"type": "opencv", "index_or_path": "/dev/video4", "width": 800, "height": 600, "fps": 25}, "global": {"type": "opencv", "index_or_path": "/dev/video6", "width": 800, "height": 600, "fps": 25}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=znw_arm_l1 \
    --display_data=true \
    --fps=60
```

#### 电机校准GUI界面
```bash
# 电机校准和调试界面
python -m lerobot.calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=znw_arm_f1


python -m lerobot.calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=znw_arm_l1
```

#### 摄像头测试
```bash
# 测试HandEye摄像头连接和性能
python tests/cameras/test_real_opencv.py

#### 舵机参数调节（解决jittering问题）
```bash
# 交互式调节舵机PID和Deadband参数
python motor_tuning_script.py
```

#### 数据录制（双摄像头）
```bash
# 录制双臂+双摄像头数据集（HandEye + Global）
python -m lerobot.record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=znw_arm_f1 \
    --robot.cameras='{"handeye": {"type": "opencv", "index_or_path": "/dev/video4", "width": 800, "height": 600, "fps": 25}, "global": {"type": "opencv", "index_or_path": "/dev/video6", "width": 800, "height": 600, "fps": 25}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=znw_arm_l1 \
    --dataset.repo_id=wzn12/teleop_ring \
    --dataset.single_task="pick ring" \
    --dataset.num_episodes=20 \
    --dataset.episode_time_s=20 \
    --dataset.reset_time_s=15 \
    --dataset.push_to_hub=true \
    --dataset.private=true \
    --display_data=true 
```

#### 数据回放（查看录制的数据,会启动机器人）
```bash
# 回放录制的数据集
python -m lerobot.replay \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=znw_arm_f1 \
    --dataset.repo_id=your_username/your_dataset_name \
    --dataset.episode=0
```

#### 模型inference
```bash
python -m lerobot.record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.cameras='{"handeye": {"type": "opencv", "index_or_path": "/dev/video4", "width": 800, "height": 600, "fps": 25}, "global": {"type": "opencv", "index_or_path": "/dev/video6", "width": 800, "height": 600, "fps": 25}}' \
    --robot.id=znw_arm_l1 \
    --dataset.repo_id=wzn12/ring-smolVLA-test \
    --dataset.num_episodes=2 \
    --dataset.single_task="pick up the black ring" \
    --dataset.episode_time_s=60 \
    # <- Teleop optional if you want to teleoperate to record or in between episodes with a policy \
    # --teleop.type=so100_leader \
    # --teleop.port=/dev/tty.usbmodem58760431551 \
    # --teleop.id=blue \
    # <- Policy optional if you want to record with a policy \
    --policy.path=lerobot/smolvla_base \
    --policy.device=cuda \
    --policy.use_amp=true \
    --display_data=true
```

### 7. 设备端口配置

#### 当前设备配置
- **Follower Arm**: `/dev/ttyACM1` (ID: znw_arm_f1)
- **Leader Arm**: `/dev/ttyACM0` (ID: znw_arm_l1)  
- **HandEye Camera**: `/dev/video4` (800x600@25fps)
- **Global Camera**: `/dev/video6` (800x600@25fps)

#### 端口权限设置
```bash
# 设置串口权限（每次重启后需要重新设置）
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1

```

#### 查找可用端口
```bash
# 查找可用的串口设备
python -m lerobot.find_port

# 查找可用的摄像头设备
ls /dev/video*
```

## 环境信息

### 已安装的关键组件
- **Python**: 3.10.x
- **PyTorch**: 2.7.1+cu128 (支持RTX 5080)
- **CUDA**: 12.8
- **LeRobot**: 最新版本
- **仿真环境**: gym-pusht, gym-aloha
- **可视化**: rerun.io

### 支持的模型类型
- ACT (Action Chunking Transformer)
- Diffusion Policy
- TDMPC (Temporal Difference Model Predictive Control)
- VQ-BeT (Vector Quantized Behavior Transformer)
- PI0 (Policy Iteration Zero)
- SmolVLA (Small Vision-Language-Action)
- SAC (Soft Actor-Critic)

## 故障排除

### 常见问题
1. **网络下载慢**: 已配置 `HF_ENDPOINT=https://hf-mirror.com`
2. **GPU兼容性**: 已安装支持RTX 5080的PyTorch 2.7.1+cu128
3. **环境依赖**: 已降级pymunk到6.11.1解决兼容性问题

### 性能优化
- 使用 `--policy.use_amp=true` 启用自动混合精度
- 使用 `--policy.device=cuda` 启用GPU加速
- 调整 `--eval.batch_size` 根据GPU内存调整批次大小

## 下一步

环境配置完成后，您可以：
1. 运行真机数据集收集
2. 训练自定义策略
3. 部署到实际机器人
4. 参与社区贡献

更多详细信息请参考 [LeRobot官方文档](https://huggingface.co/docs/lerobot)。