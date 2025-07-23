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

### 支持的机器人平台
- **仿真**: pusht, aloha, xarm
- **真机**: koch, aloha, so100, so101, stretch3, viperx
- **摄像头**: opencv, intelrealsense
- **电机**: dynamixel, feetech

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