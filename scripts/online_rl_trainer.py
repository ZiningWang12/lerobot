#!/usr/bin/env python3
"""
渐进式混合强化学习训练脚本
基于现有LeRobot组件，实现轻量级在线训练

特点:
- 渐进式混合训练：按照episode数量自动切换离线/在线数据比例
- 人工标注Reward集成：支持人工标注reward和自动reward model两种模式
- 混合ReplayBuffer：支持离线+在线数据混合采样
- 增强可视化：复用record.py机制，添加reward和Q-value可视化
- 复用现有组件：最大化代码复用，减少重复
"""

import os
import sys
import time
import json
import logging
import wandb
from pathlib import Path
from typing import Dict, Any, Optional

# 使用EasyDict替代MockConfig，支持深层嵌套的点号访问
from easydict import EasyDict

def create_lerobot_compatible_config(config_dict: Dict[str, Any]) -> EasyDict:
    """创建兼容LeRobot的配置对象 - 使用EasyDict支持深层嵌套"""
    return EasyDict(config_dict)


def save_policy_bundle(save_path: str, policy):
    """以与加载逻辑兼容且简洁的方式保存策略：
    - 保存 actor 为 HF 目录（SmolVLAPolicy.from_pretrained 可直接加载）
    - 保存 critic_state_dict 到独立文件（critic_init_state_path 可加载）
    - 写入 resume_overrides.json 便于直接复现加载配置
    """
    os.makedirs(save_path, exist_ok=True)

    # 保存 actor（SmolVLA）为 HF 目录
    actor_save_dir = os.path.join(save_path, "actor")
    os.makedirs(actor_save_dir, exist_ok=True)
    policy.smolvla.save_pretrained(actor_save_dir)

    # 保存 critic 状态
    critic_state = {
        'critic_state_dict': policy.critic.state_dict(),
    }
    torch.save(critic_state, os.path.join(save_path, 'critic_state.pth'))

    # resume 覆盖配置
    resume_overrides = {
        'policy': {
            'pretrained_path': actor_save_dir,
            'critic_init_state_path': os.path.join(save_path, 'critic_state.pth'),
        }
    }
    with open(os.path.join(save_path, 'resume_overrides.json'), 'w') as f:
        json.dump(resume_overrides, f, indent=2)

    return {
        'actor_dir': actor_save_dir,
        'critic_state': os.path.join(save_path, 'critic_state.pth'),
        'resume_overrides': os.path.join(save_path, 'resume_overrides.json'),
    }

def setup_torch_dynamo(config: EasyDict) -> None:
    """根据配置设置TorchDynamo"""
    if hasattr(config, 'torch_dynamo') and config.torch_dynamo.disable:
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
        reason = getattr(config.torch_dynamo, 'reason', 'Unknown reason')
        print(f"🔧 已禁用TorchDynamo: {reason}")
    else:
        print("🔧 TorchDynamo保持启用状态")

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np

# 直接导入critic_warmup.py中的组件
from scripts.critic_warmup import PerformanceMonitor, CriticWarmupConfig

# 导入新创建的模块
from scripts.reward_model_integration import create_reward_model_integrator
from scripts.hybrid_replay_buffer import create_hybrid_replay_buffer, MixingStage
from scripts.online_rl_utils import create_online_dataset, create_robot_config, create_teleop_config, create_data_collection_components

# 复用现有组件
from lerobot.policies.smolvla_sac.modeling_smolvla_sac import SmolVLASACPolicy
from lerobot.utils.utils import get_safe_torch_device, init_logging
from lerobot.utils.random_utils import set_seed
from lerobot.utils.wandb_utils import WandBLogger

# 解耦gym_manipulator，直接使用record.py中的机器人控制
from lerobot.robots import make_robot_from_config, RobotConfig
from lerobot.teleoperators import make_teleoperator_from_config, TeleoperatorConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 直接导入record.py中的可视化函数
from lerobot.utils.visualization_utils import log_rerun_data, _init_rerun

# 导入rerun用于增强可视化
import rerun as rr
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 导入必要的函数
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def log_enhanced_rerun_data(observation: Dict, action: Dict, reward: float, 
                           q_values: Optional[torch.Tensor] = None, 
                           episode: int = 0, step: int = 0):
    """增强的Rerun可视化，包括reward和Q-value"""
    
    # 基础可视化：observation和action
    log_rerun_data(observation, action)
    
    # 添加reward可视化 - 使用正确的rerun API
    rr.log("training/reward", rr.Scalar(reward))
    rr.log(f"episodes/episode_{episode}/reward", rr.Scalar(reward))
    
    # 添加Q-value可视化（如果提供）
    if q_values is not None:
        if isinstance(q_values, torch.Tensor):
            q_values = q_values.detach().cpu().numpy()
        
        # 记录Q-value统计信息
        if len(q_values) > 0:
            rr.log("training/q_value_mean", rr.Scalar(float(np.mean(q_values))))
            rr.log("training/q_value_max", rr.Scalar(float(np.max(q_values))))
            rr.log("training/q_value_min", rr.Scalar(float(np.min(q_values))))
            rr.log(f"episodes/episode_{episode}/q_value_mean", rr.Scalar(float(np.mean(q_values))))
    
    # 添加episode和step信息
    rr.log("training/episode", rr.Scalar(episode))
    rr.log("training/step", rr.Scalar(step))


class HumanInterventionInterface:
    """人工干预接口 - 预留功能"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.intervention_active = False
        print("🎮 人工干预接口已预留")
    
    def check_intervention_signal(self) -> bool:
        """检查干预信号 - 预留接口"""
        return False
    
    def record_intervention(self, observation: Dict, action: torch.Tensor, 
                           human_action: torch.Tensor):
        """记录干预数据 - 预留接口"""
        print("📝 记录人工干预数据")


def run_comprehensive_test(config: Dict[str, Any]) -> bool:
    """运行核心测试，验证关键组件"""
    from hybrid_replay_buffer import test_hybrid_replay_buffer
    test_hybrid_replay_buffer()
    print("🧪 No Test Yet. TODO...")
    
    return True


def train_progressive_hybrid_rl(config: Dict[str, Any]) -> bool:
    """渐进式混合强化学习训练主函数"""
    # 初始化性能监控器 - 直接使用critic_warmup.py中的实现
    monitor = PerformanceMonitor()
    
    # 设置设备
    device = get_safe_torch_device("cuda", log=True)
    print(f"🔧 使用设备: {device}")
    
    # 清理Rerun进程，防止相机接口冲突
    print("🧹 清理Rerun进程，防止相机接口冲突...")
    import subprocess
    import time
    
    try:
        # 尝试关闭所有Rerun进程
        subprocess.run(["pkill", "-f", "rerun"], capture_output=True, check=False)
        # 等待接口释放
        print("   ⏳ 等待相机接口释放...")
        time.sleep(3)
        
    except Exception as e:
        print(f"   ⚠️ Rerun进程清理过程中出现警告: {e}")    
    # 创建机器人 - 解耦gym_manipulator，直接使用record.py的方式
    print("🤖 创建机器人...")
    robot_config = create_robot_config(config)
    robot = make_robot_from_config(robot_config)
    robot.connect()
    print("✅ 机器人创建并连接成功")
    
    # 创建遥操作器（可选）
    teleop = None
    if config.get("teleop"): # Changed from config.teleop to config.get("teleop")
        print("🎮 创建遥操作器...")
        teleop_config = create_teleop_config(config)
        teleop = make_teleoperator_from_config(teleop_config)
        teleop.connect()
        print("✅ 遥操作器创建并连接成功")
    
    # 加载离线数据集
    print("📚 加载离线数据到ReplayBuffer...")
    offline_dataset = LeRobotDataset(
        repo_id=config.dataset.repo_id,
        root=config.dataset.root,  # 从配置文件读取数据集路径
        download_videos=False  # 使用本地数据集，不需要下载
    )
    
    print(f"✅ 数据集加载成功，包含 {len(offline_dataset)} 个episodes")
    print(f"🔍 数据集meta: {offline_dataset.meta}")
    
    # 直接使用from_pretrained，传递必要的参数
    from lerobot.policies.smolvla_sac.modeling_smolvla_sac import SmolVLASACPolicy
    
    policy = SmolVLASACPolicy.from_pretrained(
        pretrained_name_or_path=config.policy.pretrained_path,
        dataset_stats=offline_dataset.meta.stats
    )
    
    # 调试：检查策略配置
    print(f"🔍 策略配置: {policy.config}")
    print(f"🔍 n_action_steps: {getattr(policy.config, 'n_action_steps', 'N/A')}")
    print(f"🔍 chunk_size: {getattr(policy.config, 'chunk_size', 'N/A')}")
    
    policy = policy.to(device)
    policy.train()
    
    # 创建Reward Model集成器
    print("🎯 创建Reward Model集成器...")
    reward_integrator = create_reward_model_integrator(config.env.reward_model)
    
    # 创建混合ReplayBuffer
    hybrid_buffer = create_hybrid_replay_buffer({
        "offline_capacity": config.replay_buffer.offline_capacity,
        "online_capacity": config.replay_buffer.online_capacity,
        "device": device
    })
    
    print("📚 加载离线数据到ReplayBuffer...")
    offline_data_count = hybrid_buffer.load_offline_data(
        dataset=offline_dataset,
        max_episodes=config.replay_buffer.max_offline_episodes
    )
    
    # 设置渐进式混合阶段
    if config.policy.progressive_mixing.enable:
        stages = config.policy.progressive_mixing.stages
        mixing_stages = {}
        for stage_name, stage_config in stages.items():
            episodes = stage_config.episodes
            offline_ratio = stage_config.offline_ratio
            description = stage_config.description
            
            mixing_stages[stage_name] = MixingStage(
                name=stage_name,
                episodes=episodes,
                offline_ratio=offline_ratio,
                description=description
            )
        
        hybrid_buffer.set_mixing_stages(mixing_stages)
    
    print(f"✅ 混合ReplayBuffer创建成功，离线容量: {config.replay_buffer.offline_capacity}, 在线容量: {config.replay_buffer.online_capacity}")
    
    # 创建online数据集用于记录
    print("📊 创建online数据集...")
    online_dataset = create_online_dataset(config, robot)
    data_collector, teleop_appender, episode_manager = create_data_collection_components(
        config=config,
        robot=robot,
        policy=policy,
        teleop=teleop,
        device=device,
        online_dataset=online_dataset,
        hybrid_buffer=hybrid_buffer
    )
    print("✅ 数据采集组件创建成功")
    
    # 初始化Rerun可视化（如果启用）
    if config.visualization.enable:
        print("🎨 初始化Rerun可视化...")
        _init_rerun(session_name="progressive_hybrid_rl")
    # 检查SmolVLA参数是否可训练
    print("🔍 检查SmolVLA参数状态...")
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    smolvla_params = sum(p.numel() for p in policy.smolvla.parameters())
    smolvla_trainable = sum(p.numel() for p in policy.smolvla.parameters() if p.requires_grad)
    
    print(f"   总参数数量: {total_params:,}")
    print(f"   可训练参数数量: {trainable_params:,}")
    print(f"   SmolVLA总参数: {smolvla_params:,}")
    print(f"   SmolVLA可训练参数: {smolvla_trainable:,}")
    print(f"   SmolVLA训练比例: {smolvla_trainable/smolvla_params*100:.1f}%")
    
    # 记录初始SmolVLA参数（用于后续比较）
    initial_smolvla_params = {}
    for name, param in policy.smolvla.named_parameters():
        if param.requires_grad:
            initial_smolvla_params[name] = param.data.clone()
    
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=10000, 
        gamma=0.9
    )
    print("✅ 优化器创建成功")
    
    # 初始化人工干预接口
    intervention_interface = HumanInterventionInterface(config)
    
    # 训练统计
    episode_rewards = []
    training_losses = []
    stage_transitions = []
    
    print(f"🎯 开始训练，总episode数: {config.training.total_episodes}")
    
    # 主训练循环 - 基于episode的渐进式混合训练
    for episode in range(config.training.total_episodes):
        import psutil
        import gc
        mem_before = psutil.Process().memory_info().rss / 1024 / 1024
        print(f"🔍 Episode {episode} 开始前: 内存{mem_before:.1f}MB")
        
        # 使用EpisodeManager运行完整的episode
        episode_stats = episode_manager.run_episode(episode)
        
        # 🔍 内存监控：episode数据采集完成后
        mem_after_collect = psutil.Process().memory_info().rss / 1024 / 1024
        mem_increase_collect = mem_after_collect - mem_before
        print(f"🔍 Episode {episode} 数据采集后: 内存{mem_after_collect:.1f}MB(+{mem_increase_collect:.1f}MB)")
        
        episode_reward = episode_stats['episode_reward']
        episode_steps = episode_stats['total_steps']
        
        # 人工标注reward处理
        if config.env.reward_model.human_labeling.enable:
            # 处理整个episode的标注流程：启动标注、等待完成、更新reward
            local_dataset_path = online_dataset.root
            success = reward_integrator.process_episode_labeling(
                episode_idx=episode,
                dataset_repo_id=local_dataset_path,  # 使用本地路径
                output_repo_id=config.env.reward_model.human_labeling.output_repo_id,
                online_dataset=online_dataset
            )
            
            if not success:
                continue  # 跳过这个episode

        # 检查是否有足够的数据进行训练
        if (len(hybrid_buffer.online_buffer) >= config.replay_buffer.min_samples_for_training and 
            episode >= config.training.learning_starts):
            
            # 计算本episode需要进行的训练步数
            # 使用UTD ratio：每个episode进行多次训练更新
            utd_ratio = config.training.utd_ratio
            training_steps = max(1, episode_steps // config.training.train_freq)
            
            for train_step in range(training_steps):
                # 前utd_ratio-1次更新
                for update_idx in range(utd_ratio - 1):
                    # 混合采样经验批次
                    batch, sampling_info = hybrid_buffer.sample_mixed(config.training.batch_size, episode)
                    
                    # 记录混合训练信息
                    stage_transitions.append({
                        'episode': episode + 1,
                        'train_step': train_step,
                        'update_idx': update_idx,
                        'stage': sampling_info['stage'],
                        'offline_ratio': sampling_info['offline_ratio'],
                        'offline_size': sampling_info['offline_size'],
                        'online_size': sampling_info['online_size']
                    })
                    
                    # SmolVLASACPolicy的训练逻辑
                    loss_critic = policy.compute_loss_critic(
                        observations=batch["state"],
                        actions=batch["action"],
                        rewards=batch["reward"],
                        next_observations=batch["next_state"],
                        done=batch["done"]
                    )
                    
                    # 获取Q-value用于可视化
                    with torch.no_grad():
                        q_values = policy.critic_forward(
                            observations=batch["state"],
                            actions=batch["action"],
                            use_target=False
                        )
                    
                    # 只更新critic (前utd_ratio-1次)
                    optimizer.zero_grad()
                    loss_critic.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(
                        policy.parameters(), 
                        max_norm=config.optimizer.grad_clip_norm
                    )
                    
                    # 优化器更新
                    optimizer.step()
                    
                    training_losses.append(loss_critic.item())
                
                # 最后一次更新 (第utd_ratio次) - 完整更新
                batch, sampling_info = hybrid_buffer.sample_mixed(config.training.batch_size, episode)
                
                # 记录混合训练信息
                stage_transitions.append({
                    'episode': episode + 1,
                    'train_step': train_step,
                    'update_idx': utd_ratio - 1,
                    'stage': sampling_info['stage'],
                    'offline_ratio': sampling_info['offline_ratio'],
                    'offline_size': sampling_info['offline_size'],
                    'online_size': sampling_info['online_size']
                })
                
                # 最后一次更新：完整的SAC更新
                loss_critic = policy.compute_loss_critic(
                    observations=batch["state"],
                    actions=batch["action"],
                    rewards=batch["reward"],
                    next_observations=batch["next_state"],
                    done=batch["done"]
                )
                
                loss_actor = policy.compute_loss_actor(
                    observations=batch["state"]
                )
                
                loss_temperature = policy.compute_loss_temperature(
                    observations=batch["state"]
                )
                
                # 获取Q-value用于可视化
                with torch.no_grad():
                    q_values = policy.critic_forward(
                        observations=batch["state"],
                        actions=batch["action"],
                        use_target=False
                    )
                
                # 总损失
                total_loss = loss_critic + loss_actor + loss_temperature
                
                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), 
                    max_norm=config.optimizer.grad_clip_norm
                )
                
                # 优化器更新
                optimizer.step()
                
                training_losses.append(total_loss.item())
                
                # 检查SmolVLA参数是否真的更新了（每10步检查一次）
                if train_step % 10 == 0:
                    param_changes = []
                    for name, param in policy.smolvla.named_parameters():
                        if param.requires_grad and name in initial_smolvla_params:
                            change = torch.norm(param.data - initial_smolvla_params[name]).item()
                            param_changes.append(change)
                    
                    if param_changes:
                        avg_change = sum(param_changes) / len(param_changes)
                        max_change = max(param_changes)
                        print(f"   🔍 SmolVLA参数变化 - 平均: {avg_change:.6f}, 最大: {max_change:.6f}")
                    else:
                        print(f"   ⚠️ SmolVLA参数未检测到变化！")
                
                print(f"训练步骤 {train_step + 1}/{training_steps},总损失: {total_loss.item():.4f}，Critic: {loss_critic.item():.4f}, Actor: {loss_actor.item():.4f}, Temp: {loss_temperature.item():.4f}")
                #monitor.end_training()
                # 更新目标网络 (每次UTD循环后)
                policy.update_target_networks()
                
        else:
            print(f"   ⏳ 跳过训练: 在线数据不足({len(hybrid_buffer.online_buffer)} < {config.replay_buffer.min_samples_for_training}) 或未达到学习开始条件(episode {episode} < {config.training.learning_starts})")
        
        # 更新学习率（每个episode后）
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"   📈 学习率更新: {current_lr:.6f}")
        
        # 结束episode（数据已在前面保存）
        episode_rewards.append(episode_reward)
        
        # 显示episode统计
        print(f"🎬 Episode {episode + 1} 完成")
        print(f"   总奖励: {episode_reward:.4f}, 总步数: {episode_steps}")
        
        # 显示缓冲区统计
        buffer_stats = hybrid_buffer.get_buffer_stats()
        print(f"   当前阶段: {buffer_stats['current_stage']},离线数据: {buffer_stats['offline_buffer']['size']}/{buffer_stats['offline_buffer']['capacity']},在线数据: {buffer_stats['online_buffer']['size']}/{buffer_stats['online_buffer']['capacity']}")
        
        # 保存检查点
        if (episode + 1) % config.logging.save_freq == 0:
            save_path = os.path.join(config.output_dir, f"checkpoint_episode_{episode+1}")
            save_policy_bundle(save_path=save_path, policy=policy)
            # 保存缓冲区状态
            hybrid_buffer.save_buffer_state(save_path)
        
        # 性能监控 - 使用critic_warmup.py中的方法
        #if (episode + 1) % 5 == 0:
        #    monitor.print_performance_summary()
    
    # 检查最终SmolVLA参数变化
    print("🔍 检查最终SmolVLA参数变化...")
    final_param_changes = []
    for name, param in policy.smolvla.named_parameters():
        if param.requires_grad and name in initial_smolvla_params:
            change = torch.norm(param.data - initial_smolvla_params[name]).item()
            final_param_changes.append((name, change))
    
    if final_param_changes:
        final_param_changes.sort(key=lambda x: x[1], reverse=True)
        print(f"   SmolVLA参数变化统计:")
        print(f"   平均变化: {sum(c[1] for c in final_param_changes)/len(final_param_changes):.6f}")
        print(f"   最大变化: {final_param_changes[0][1]:.6f} ({final_param_changes[0][0]})")
        print(f"   最小变化: {final_param_changes[-1][1]:.6f} ({final_param_changes[-1][0]})")
        print(f"   有变化的参数: {len([c for c in final_param_changes if c[1] > 1e-8])}/{len(final_param_changes)}")
    else:
        print("   ⚠️ SmolVLA参数完全没有变化！")
    
    # 保存最终模型
    final_save_path = os.path.join(config.output_dir, "final_model")
    os.makedirs(final_save_path, exist_ok=True)
    save_policy_bundle(save_path=final_save_path, policy=policy)
    
    # 保存缓冲区状态
    hybrid_buffer.save_buffer_state(final_save_path)
    
    # 保存online数据集到HuggingFace Hub
    try:
        print("🚀 上传online数据集到HuggingFace Hub...")
        online_dataset.push_to_hub(
            tags=["online_rl", "progressive_hybrid", "smolvla_sac"],
            private=False
        )
        print("✅ Online数据集上传成功")
    except Exception as e:
        print(f"⚠️ Online数据集上传失败: {e}")
    
    # 关闭连接
    robot.disconnect()
    if teleop:
        teleop.disconnect()
    
    print(f"✅ 渐进式混合强化学习训练完成！")
    print(f"   总episode数: {config.training.total_episodes}")
    print(f"   总奖励: {sum(episode_rewards):.2f}")
    print(f"   平均奖励: {np.mean(episode_rewards):.4f}")
    print(f"   模型保存在: {config.output_dir}")
    print(f"   Online数据集保存在: {online_dataset.root}")
    
    return True


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="渐进式混合强化学习训练")
    parser.add_argument(
        "--config_path", 
        type=str, 
        default="scripts/online_rl_config.json",
        help="配置文件路径"
    )
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="仅运行测试，不进行训练"
    )
    args = parser.parse_args()
    
    # 初始化日志
    init_logging()
    
    # 加载配置
    with open(args.config_path, 'r') as f:
        config_dict = json.load(f)
    
    # 处理时间戳占位符
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")  # 年月日-时分格式
    config_dict_str = json.dumps(config_dict)
    config_dict_str = config_dict_str.replace("${timestamp}", timestamp)
    config_dict = json.loads(config_dict_str)
    
    # 创建兼容LeRobot的配置对象
    config = create_lerobot_compatible_config(config_dict)
    
    # 设置TorchDynamo（必须在导入torch之前）
    setup_torch_dynamo(config)
    
    print(f"✅ 配置加载成功: {config.job_name}")
    print(f"   策略类型: {config.policy.type} (支持渐进式混合训练)")
    print(f"   机器人类型: {config.env.robot.type}")
    print(f"   启用渐进式混合: {config.policy.progressive_mixing.enable if hasattr(config.policy, 'progressive_mixing') else False}")
    print(f"   WandB启用: {config.wandb.enable}")
    print(f"   WandB项目: {config.wandb.project}")
    
    # 设置随机种子
    set_seed(config.seed if config.seed is not None else 42)
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    '''
    # 初始化WandB - 使用我们的配置
    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=config.job_name,
        notes=config.wandb.notes,
        tags=config.wandb.tags if hasattr(config.wandb, 'tags') else [],
        config=config_dict,
        dir=config.output_dir
    )
    print(f"✅ WandB初始化成功")
    print(f"🔗 运行地址: {wandb.run.get_url()}")
    
    # 创建简化的logger接口，与critic_warmup.py保持一致
    class SimpleWandBLogger:
        def log_dict(self, d, step):
            wandb.log(d, step=step)
    wandb_logger = SimpleWandBLogger()
    '''
    
    # 根据参数决定运行模式
    if args.test_only:
        print("🧪 运行测试模式...")
        success = run_comprehensive_test(config)
    else:
        print("🚀 运行训练模式...")
        success = train_progressive_hybrid_rl(config)
        if success:
            print("\n🎉 训练完成！")
            return 0
        else:
            print("\n❌ 训练失败")
            return 1


if __name__ == "__main__":
    exit(main()) 