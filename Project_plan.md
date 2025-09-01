## 2025-08-08 上午
我现在希望做真机强化学习的第一步，训练一个reward classifier,前天我已经录制了teleop数据：
dataset.repo_id=wzn12/teleop_ring
以及标注了reward label：
data/teleop_ring_labeled
但是我不知道这个数据该怎么按照官方的教程用。因此需要：
- 组织数据和label，按照官方的教程，给出reward model数据的训练使用方案(尽量复用官方已有接口与代码) （DONE）
- 先使用官方的resnet10训练，看下能训练到多少 （DONE）
    - 问题：原版为128*128,但对于我的应用来说太小了，有没有可能扩大一些到比如256*256甚至保持原版800*600？ 
    - 结论：测试了128和224的validation结果，确实是224更好，4.5k acc能有98.8%
    - 参考：lerobot.scripts.train --config_path src/lerobot/configs/reward_classifier_resnet10_ring_224.json 
- 但我觉得官方的resnet10上限太低，后续我希望尝试更换GroundingDino作为reward model，之前已经配置好了GroundingSAM，先分析下如何替换 


## 2025-08-08 下午
目前224*224的4.5k的resnet10 reward model可用了，接下来我要进一步进行真机强化学习训练：https://huggingface.co/docs/lerobot/hilserl：
- 使用smolVLA作为policy，不要像HILSERL一样train from scratch，模型路径：--policy.path=outputs/train/2025-08-04/11-43-08_smolvla/checkpoints/008000/pretrained_model
- 上面的模型是一个很好的initial，我测试下来"pick up the black ring"可以有合理的traj，只不过成功率还不高
- 给出实现方案和计划，完成真机强化学习训练的验证 （DONE）

### HILSERL强化学习验证完成情况
**第0阶段：Reward model训练** ✅
reward model训练完成:使用--config_path src/lerobot/configs/reward_classifier_resnet10_ring_224.json 训出来的模型已上传
https://huggingface.co/wzn12/reward_classifier_resnet10_ring_224
#### 遗留问题
HILSERL里面的SAC的policy跟smolVLA作为policy的结构差不少（smolVLA不在原生HILSERL架构里面），需要详细分析，给出smolVLA嵌入RL训练流程的解决方案


## 2025-08-11 上午
#### 解决思路：综合来看，先采取方案A：smolVLA + 独立Critic（推荐）
核心思想：保持smolVLA作为Actor，添加独立的Critic网络

**第一阶段：SmolVLASACPolicy改造** ✅
- 创建SmolVLASACPolicy
- 实现基本的SAC接口
- 验证训练流程
- 需要先使用inference模式提前验证smolVLA policy的实现的正确性，这里有一个很好的模型，使用它进行推理应该可以在不进行RL训练的情况下完成任务，--policy.path=outputs/train/2025-08-04/11-43-08_smolvla/checkpoints/008000/pretrained_model。即使用smolVLASACPolicy的推理结果，应该跟使用lerobot.record加载smolVLA进行直接推理的结果一致

**第一阶段完成情况：**
- test_smolvla_sac_inference.py 已验证可正常加载预训练参数，且推理结果一致
- SmolVLASACPolicy已完整实现，包含独立Critic网络

**第二阶段：Critic/Value Model复用-初始化-warmup** ✅
- 创建独立的Critic Model，可复用HILSERL/SAC的critic模型结构和权重初始化
- 在与policy连训之前，对Critic Model进行warmup，使用--repo_id=wzn12/teleop_ring_labeled数据进行offline训练
- Critic模型是独立的模型，现不使用smolVLA的feature
## 2025-08-13 下午
**第二阶段完成情况：** ✅
- 创建了critic_warmup.py脚本，用于Critic模型的offline训练
- 实现了独立Critic网络的初始化和warmup流程
- 支持使用标注数据进行预训练
- 支持WandB日志记录和HuggingFace模型上传
- 配置文件：src/lerobot/configs/critic_warmup_smolvla_ring.json
- 启动命令：cd /home/znw/projects/lerobot/lerobot && source lerobot_env/bin/activate && python scripts/critic_warmup.py 
- 训练结果：/home/znw/projects/lerobot/lerobot/outputs/train/critic_warmup_smolvla_ring，看起来epoch 19的val loss最小，当然实际上到epoch4之后就不怎么变了（这个TD loss的scale还算挺大的）

**第三阶段：真机强化学习连训** ✅
- https://huggingface.co/docs/lerobot/hilserl 参考官方教程，进行正式的，基于smolVLA policy和独立Critic的真机强化学习训练验证
- 之前有一个脚本start_hilserl_training.py，看能不能改造后使用
## 2025-08-14 下午
#### 已完成的工作
1. **配置文件创建** ✅
   - 创建了 `src/lerobot/configs/train_config_hilserl_smolvla_ring.json`
   - 配置了基于smolVLA的HILSERL训练参数
   - 集成了reward classifier和预训练模型路径

2. **训练脚本准备** ✅
   - 创建了 `scripts/start_hilserl_training.py` 自动化启动脚本

3. **框架集成与加载路径打通** ✅
   - 在 `src/lerobot/policies/factory.py` 注册 `smolvla_sac`，可被actor/learner直接实例化
   - 在 `src/lerobot/envs/configs.py` 为 `gym_manipulator` 增加默认 `features`/`features_map`（手眼/全局双目800×600，state/action为6D），便于从env推导policy特征
   - 在 HIL 环境中通过 `env.reward_classifier_pretrained_path` 自动加载 Reward Classifier，并封装到 `RewardWrapper`
   - `SmolVLASACPolicy` 支持用 `policy.critic_init_state_path` 注入离线 warmup 的 critic 权重
   - 启动脚本 `start_hilserl_training.py` 从配置读取路径，自动把 `env.pretrained_policy_name_or_path` 作为 `--policy.path` 传给 learner/actor

## 2025-08-18 👆
   - 目前看来第三阶段不太行，smolVLA和HILSERL的框架不兼容的地方太多了

## 2025-08-27
   - 目前来看 $reward = label \in \{0,1\} $ 太sparse了，在VLA没有学好的情况下onlineRL很难优化需要设计一个
   $$reward(t)=label(t) + 0.5 \times (1 - \min\{1, \left \Vert a(t)-a^* \right \Vert_2\})$$
   其中$a^*$是第一帧$label=1$时机器人joint的弧度制读数，这样去牵引pick任务接近目标

#### 参考（11-43-08_smolvla的训练与测试）
#### 模型training
```bash
export HF_HUB_OFFLINE=1
python -m lerobot.scripts.train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=wzn12/teleop_ring \
    --batch_size=56 \
    --steps=8000 \
    --policy.repo_id=wzn12/my_smolvla_model \
    --policy.device=cuda \
    --policy.use_amp=false \
    --wandb.enable=true \
    --save_freq=2000 \
    --num_workers=8  # 从4增加到8

```

#### 模型inference（实时机器人控制）
```bash
export HF_HUB_OFFLINE=1
python -m lerobot.record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=znw_arm_f1 \
    --robot.cameras='{"handeye": {"type": "opencv", "index_or_path": "/dev/video4", "width": 800, "height": 600, "fps": 25}, "global": {"type": "opencv", "index_or_path": "/dev/video6", "width": 800, "height": 600, "fps": 25}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=znw_arm_l1 \
    --dataset.single_task="Pick the screw driver" \
    --dataset.episode_time_s=100 \
    --dataset.reset_time_s=15 \
    --dataset.num_episodes=10 \
    --policy.path=outputs/train/2025-08-04/11-43-08_smolvla/checkpoints/008000/pretrained_model \
    --policy.device=cuda \
    --policy.use_amp=false \
    --display_data=true \
    --dataset.repo_id=wzn12/eval_ring-smolVLA_test1
```

#### HILSERL强化学习训练启动（推荐使用 .py 启动器）
```bash
# 一键启动（推荐）
python src/lerobot/scripts/start_hilserl_training.py

# 或手动启动
# 终端1: Learner服务器
source lerobot_env/bin/activate
export HF_HUB_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
python -m lerobot.scripts.rl.learner \
    --config_path src/lerobot/configs/train_config_hilserl_smolvla_ring.json \
    --policy.path outputs/train/2025-08-04/11-43-08_smolvla/checkpoints/008000/pretrained_model

# 终端2: Actor服务器  
source lerobot_env/bin/activate
export HF_HUB_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
python -m lerobot.scripts.rl.actor \
    --config_path src/lerobot/configs/train_config_hilserl_smolvla_ring.json \
    --policy.path outputs/train/2025-08-04/11-43-08_smolvla/checkpoints/008000/pretrained_model
```

关键配置（位于 `src/lerobot/configs/train_config_hilserl_smolvla_ring.json`）
- **env.pretrained_policy_name_or_path**: 预训练的 smolVLA 模型目录（自动传给 `--policy.path`）
- **env.reward_classifier_pretrained_path**: 奖励分类器的 `pretrained_model` 路径
- **policy.critic_init_state_path**: 离线 warmup 产生的 critic checkpoint（如 `outputs/train/critic_warmup_smolvla_ring/best_critic_checkpoint/best_critic.pth`）

#### 验证和监控
```bash
# 验证配置
python3 scripts/verify_hilserl_setup.py

# 监控训练日志
tail -f outputs/train/hilserl_smolvla_ring/logs/learner_*.log
tail -f outputs/train/hilserl_smolvla_ring/logs/actor_*.log

# 检查进程状态
ps aux | grep python | grep learner
ps aux | grep python | grep actor
```
