#!/usr/bin/env python3
"""
Critic模型Warmup脚本
使用标注数据进行offline训练，为后续的强化学习训练做准备

这个脚本实现第二阶段的Critic/Value Model复用-初始化-warmup：
- 创建独立的Critic Model，可复用HILSERL/SAC的critic模型结构和权重初始化
- 在与policy连训之前，对Critic Model进行warmup，使用标注数据进行offline训练
- Critic模型是独立的模型，不使用smolVLA的feature
"""

import os
import sys
import time
import psutil
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
import logging
from dataclasses import dataclass, asdict
from termcolor import colored

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lerobot.policies.smolvla_sac.modeling_smolvla_sac import SmolVLASACPolicy
from lerobot.policies.smolvla_sac.configuration_smolvla_sac import SmolVLASACConfig
from datasets import load_dataset
from lerobot.configs.types import PolicyFeature, FeatureType
from lerobot.utils.wandb_utils import WandBLogger
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging


@dataclass
class CriticWarmupConfig:
    """Critic Warmup训练配置"""
    # 任务配置
    job_name: str = "critic_warmup_smolvla_ring"
    output_dir: str = "outputs/train/critic_warmup_smolvla_ring"
    seed: int = 42
    
    # 数据集配置
    dataset_repo_id: str = "wzn12/teleop_ring_labeled"
    max_samples: int = 1000
    train_test_split: float = 0.8
    
    # 训练配置
    batch_size: int = 48
    epochs: int = 20
    num_workers: int = 4
    
    # 优化器配置
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    
    # 调度器配置
    scheduler_step_size: int = 5
    scheduler_gamma: float = 0.9
    
    # 记录配置
    log_freq: int = 10
    save_freq: int = 5
    eval_freq: int = 2
    
    # WandB配置
    wandb_enable: bool = True
    wandb_project: str = "lerobot_critic_warmup"
    wandb_entity: str = None
    wandb_notes: str = "Critic offline warmup training for SmolVLA-SAC policy"
    wandb_tags: list = None
    
    # HuggingFace配置
    push_to_hub: bool = True
    repo_id: str = "wzn12/critic_warmup_smolvla_ring"
    
    def __post_init__(self):
        if self.wandb_tags is None:
            self.wandb_tags = ["critic", "warmup", "smolvla", "ring"]
    
    @classmethod
    def from_json(cls, config_path: str):
        """从JSON配置文件加载配置"""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # 从嵌套配置中提取参数
        kwargs = {
            'job_name': config_dict.get('job_name', 'critic_warmup_smolvla_ring'),
            'output_dir': config_dict.get('output_dir', 'outputs/train/critic_warmup_smolvla_ring'),
            'seed': config_dict.get('seed', 42),
            'dataset_repo_id': config_dict['dataset']['repo_id'],
            'max_samples': config_dict.get('max_samples', 1000),
            'train_test_split': config_dict.get('train_test_split', 0.8),
            'batch_size': config_dict.get('batch_size', 48),
            'epochs': config_dict.get('epochs', 20),
            'num_workers': config_dict.get('num_workers', 4),
            'learning_rate': config_dict.get('optimizer', {}).get('lr', 3e-4),
            'weight_decay': config_dict.get('optimizer', {}).get('weight_decay', 1e-4),
            'grad_clip_norm': config_dict.get('optimizer', {}).get('grad_clip_norm', 1.0),
            'scheduler_step_size': config_dict.get('scheduler', {}).get('step_size', 5),
            'scheduler_gamma': config_dict.get('scheduler', {}).get('gamma', 0.9),
            'log_freq': config_dict.get('log_freq', 10),
            'save_freq': config_dict.get('save_freq', 5),
            'eval_freq': config_dict.get('eval_freq', 2),
            'wandb_enable': config_dict.get('wandb', {}).get('enable', True),
            'wandb_project': config_dict.get('wandb', {}).get('project', 'lerobot_critic_warmup'),
            'wandb_entity': config_dict.get('wandb', {}).get('entity'),
            'wandb_notes': config_dict.get('wandb', {}).get('notes', 'Critic offline warmup training'),
            'wandb_tags': config_dict.get('wandb', {}).get('tags', ['critic', 'warmup', 'smolvla', 'ring']),
            'push_to_hub': config_dict.get('policy', {}).get('push_to_hub', True),
            'repo_id': config_dict.get('policy', {}).get('repo_id', 'wzn12/critic_warmup_smolvla_ring'),
        }
        return cls(**kwargs)


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.epoch_times = []
        self.batch_times = []
        self.data_loading_times = []
        self.forward_times = []
        self.backward_times = []
        self.optimizer_times = []
        self.gpu_memory_usage = []
        self.cpu_memory_usage = []
        self.gpu_utilization = []  # 新增GPU利用率监控
        
        # 获取GPU信息
        if torch.cuda.is_available():
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            print(f"🔧 GPU监控: {self.gpu_name}, 总内存: {self.gpu_memory_total:.1f}GB")
        else:
            self.gpu_name = "CPU"
            self.gpu_memory_total = 0
    
    def start_epoch(self):
        self.epoch_start = time.time()
    
    def end_epoch(self):
        epoch_time = time.time() - self.epoch_start
        self.epoch_times.append(epoch_time)
    
    def start_batch(self):
        self.batch_start = time.time()
    
    def end_batch(self):
        batch_time = time.time() - self.batch_start
        self.batch_times.append(batch_time)
        
        # 记录GPU和CPU使用情况
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated(0) / 1024**3  # GB
            gpu_memory_reserved = torch.cuda.memory_reserved(0) / 1024**3  # GB
            self.gpu_memory_usage.append(gpu_memory)
            
            # 尝试获取GPU利用率 (需要nvidia-ml-py3)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.gpu_utilization.append(gpu_util.gpu)
            except:
                self.gpu_utilization.append(0)
        else:
            self.gpu_memory_usage.append(0)
            self.gpu_utilization.append(0)
        
        # CPU内存使用率
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        self.cpu_memory_usage.append(memory_percent)
    
    def record_timing(self, data_loading_time, forward_time, backward_time, optimizer_time):
        self.data_loading_times.append(data_loading_time)
        self.forward_times.append(forward_time)
        self.backward_times.append(backward_time)
        self.optimizer_times.append(optimizer_time)
    
    def print_performance_summary(self):
        print("=" * 60)
        print("🚀 训练性能总结")
        print("=" * 60)
        
        total_time = sum(self.epoch_times)
        total_batches = len(self.batch_times)
        
        print(f"⏱️  总训练时间: {total_time:.2f}s")
        print(f"📊 总batch数: {total_batches}")
        print(f"⚡ 平均epoch时间: {np.mean(self.epoch_times):.2f}s")
        print(f"⚡ 平均batch时间: {np.mean(self.batch_times):.4f}s")
        
        # 计算训练速度 (samples/s)
        if total_batches > 0:
            avg_batch_time = np.mean(self.batch_times)
            samples_per_second = 48 / avg_batch_time  # 假设batch_size=64
            print(f"📈 训练速度: {samples_per_second:.2f} samples/s")
        
        print(f"📊 详细时间分布:")
        if self.data_loading_times:
            print(f"   数据加载: {np.mean(self.data_loading_times):.4f}s ({np.mean(self.data_loading_times)/np.mean(self.batch_times)*100:.1f}%)")
        if self.forward_times:
            print(f"   前向传播: {np.mean(self.forward_times):.4f}s ({np.mean(self.forward_times)/np.mean(self.batch_times)*100:.1f}%)")
        if self.backward_times:
            print(f"   反向传播: {np.mean(self.backward_times):.4f}s ({np.mean(self.backward_times)/np.mean(self.batch_times)*100:.1f}%)")
        if self.optimizer_times:
            print(f"   优化器更新: {np.mean(self.optimizer_times):.4f}s ({np.mean(self.optimizer_times)/np.mean(self.batch_times)*100:.1f}%)")
        
        # GPU利用率统计
        if torch.cuda.is_available() and self.gpu_memory_usage:
            avg_gpu_memory = np.mean(self.gpu_memory_usage)
            max_gpu_memory = max(self.gpu_memory_usage)
            gpu_memory_utilization = (avg_gpu_memory / self.gpu_memory_total) * 100
            
            print(f"🔧 GPU资源利用:")
            print(f"   平均内存使用: {avg_gpu_memory:.2f}GB ({gpu_memory_utilization:.1f}%)")
            print(f"   峰值内存使用: {max_gpu_memory:.2f}GB")
            
            if self.gpu_utilization and any(self.gpu_utilization):
                avg_gpu_util = np.mean([u for u in self.gpu_utilization if u > 0])
                print(f"   平均GPU利用率: {avg_gpu_util:.1f}%")
        
        # CPU资源统计
        if self.cpu_memory_usage:
            avg_cpu_memory = np.mean(self.cpu_memory_usage)
            print(f"💻 CPU资源利用:")
            print(f"   平均内存使用率: {avg_cpu_memory:.1f}%")
        
        print("=" * 60)


def create_smolvla_sac_config() -> SmolVLASACConfig:
    """创建Critic warmup的配置"""
    config = SmolVLASACConfig(
        # 输入特征配置
        input_features={
            "observation.image.handeye": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 600, 800)
            ),
            "observation.image.global": PolicyFeature(
                type=FeatureType.VISUAL, 
                shape=(3, 600, 800)
            ),
            "observation.state": PolicyFeature(
                type=FeatureType.STATE,
                shape=(6,)
            )
        },
        # 输出特征配置
        output_features={
            "action": PolicyFeature(
                type=FeatureType.ACTION,
                shape=(6,)
            )
        },
        # 归一化映射
        normalization_mapping={
            "action": "action",
            "observation.image.handeye": "handeye",
            "observation.image.global": "global",
            "observation.state": "state"
        },
        # Critic网络配置
        num_critics=2,
        critic_lr=3e-4,
        actor_lr=0.0,  # 冻结actor
        temperature_lr=0.0,  # 冻结temperature
        # 网络架构
        state_encoder_hidden_dim=256,
        latent_dim=256,
        vision_encoder_name="helper2424/resnet10",
        image_encoder_hidden_dim=256,
        shared_encoder=False,
        freeze_smolvla_for_critic=True,
        # 启用torch.compile大幅提升训练速度
        use_torch_compile=True,
        # SAC相关配置
        discount=0.99,
        temperature_init=1.0,
        critic_target_update_weight=0.005,
        num_steps=10, # diffusion steps
        target_entropy=None,
        use_backup_entropy=True
    )
    return config


def load_labeled_dataset(dataset_repo_id: str) -> Any:
    """加载标注的数据集"""
    print(f"📥 加载标注数据集: {dataset_repo_id}")
    
    try:
        # 首先尝试从HuggingFace Hub加载
        dataset = load_dataset(dataset_repo_id)
        print(f"✅ 从HuggingFace Hub加载数据集成功，包含 {len(dataset)} 个样本")
        return dataset
    except Exception as e:
        print(f"❌ 从HuggingFace Hub加载数据集失败: {e}")
        print("尝试从本地缓存加载...")
        
        # 尝试从本地缓存加载
        cache_path = os.path.expanduser(f"~/.cache/huggingface/lerobot/{dataset_repo_id}")
        if os.path.exists(cache_path):
            print(f"找到本地缓存: {cache_path}")
            # 使用LeRobot的方式加载本地数据集
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            
            # 创建本地数据集
            local_dataset = LeRobotDataset(
                repo_id=dataset_repo_id,
                root=cache_path,
                episodes=None  # 加载所有episodes
            )
            
            print(f"✅ 从本地缓存加载数据集成功，包含 {len(local_dataset)} 个样本")
            return local_dataset
        else:
            print(f"本地缓存不存在: {cache_path}")
            return None


def compute_dataset_stats(dataset: Any) -> Dict[str, Dict[str, torch.Tensor]]:
    """计算数据集的统计信息用于归一化"""
    print("📊 计算数据集统计信息...")
    # 这里简化处理，实际应该遍历整个数据集
    # 对于测试，我们使用默认值
    stats = {
        "observation.state": {
            "mean": torch.zeros(6),
            "std": torch.ones(6),
        },
        "action": {
            "mean": torch.zeros(6),
            "std": torch.ones(6),
        }
    }
    
    print("✅ 数据集统计信息计算完成")
    return stats
        

def prepare_training_data(dataset, config, max_samples=1000):  # 从2000增加到5000
    """准备训练数据"""
    print("🔧 准备训练数据...")
    
    # 限制样本数量
    if len(dataset) > max_samples:
        print(f"⚠️ 样本数量过多({len(dataset)})，限制为{max_samples}个")
        # 使用Subset而不是切片，避免AttributeError
        dataset = Subset(dataset, list(range(max_samples)))
    
    # 计算数据集统计信息
    print("📊 计算数据集统计信息...")
    
    # 检查数据集质量
    print("🔍 检查数据集质量...")
    sample_batch = dataset[0]
    print(f"   Sample keys: {list(sample_batch.keys())}")
    
    # 检查reward分布
    rewards = [dataset[i]['next.reward'] for i in range(min(100, len(dataset)))]  # 只检查前100个样本
    rewards_tensor = torch.stack(rewards)
    print(f"   Reward stats (前100样本):")
    print(f"     Mean: {rewards_tensor.mean():.4f}")
    print(f"     Std: {rewards_tensor.std():.4f}")
    print(f"     Min: {rewards_tensor.min():.4f}")
    print(f"     Max: {rewards_tensor.max():.4f}")
    
    # 划分训练集和验证集
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, total_size))
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    print(f"✅ 数据准备完成: 训练集 {len(train_dataset)} 样本, 验证集 {len(val_dataset)} 样本")
    
    # 创建DataLoader，使用更大的batch_size提高GPU利用率
    batch_size = 48  # 从32增加到64
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # 多进程并行解码
        pin_memory=True,  # GPU内存优化
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,  # 验证时使用较少worker
        pin_memory=True,
        drop_last=False
    )
    
    print(f"🚀 使用DataLoader优化: {train_loader.num_workers}个训练worker, {val_loader.num_workers}个验证worker")
    print(f"📊 Batch size: {batch_size} (增大以提高GPU利用率)")
    return train_loader, val_loader


def train_critic_offline(
    policy: SmolVLASACPolicy,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: SmolVLASACConfig,
    warmup_config: CriticWarmupConfig,
    wandb_logger=None
) -> bool:
    """离线训练Critic模型"""
    print("🚀 开始Critic离线训练...")
    
    # 初始化性能监控器
    monitor = PerformanceMonitor()
    
    # 设置训练模式
    policy.train()
    
    # 创建优化器（只优化critic）
    optimizer = torch.optim.Adam(
        policy.critic.critic_ensemble.parameters(),
        lr=warmup_config.learning_rate,
        weight_decay=warmup_config.weight_decay
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=warmup_config.scheduler_step_size, 
        gamma=warmup_config.scheduler_gamma
    )
    
    # 训练循环
    num_epochs = warmup_config.epochs
    device = get_safe_torch_device("cuda", log=True)
    policy = policy.to(device)
    print(f"🔧 设备: {device}")
    
    # 记录训练过程
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        monitor.start_epoch()
        
        # 训练阶段
        policy.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # 使用DataLoader进行训练
        for batch_idx, batch in enumerate(train_loader):
            monitor.start_batch()
            
            # 记录数据加载时间
            data_loading_start = time.time()
            
            # 将数据移动到GPU
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)
            
            data_loading_time = time.time() - data_loading_start
            
            # 构建标准格式的batch数据
            batch_data = {
                "state": {
                    "observation.image.handeye": batch["observation.image.handeye"],
                    "observation.image.global": batch["observation.image.global"],
                    "observation.state": batch["observation.state"],
                },
                "action": batch["action"],
                "reward": batch["next.reward"],
                "next_state": {
                    "observation.image.handeye": batch["observation.image.handeye"].clone(),
                    "observation.image.global": batch["observation.image.global"].clone(),
                    "observation.state": batch["observation.state"].clone(),
                    "task": batch["task"],  # 添加task字段，SmolVLA actor需要
                    "action": batch["action"].clone(),  # 添加action字段，SmolVLA actor需要
                },
                "done": batch["next.done"]
            }
            
            forward_start = time.time()
            # 使用policy内置的loss计算
            loss_critic = policy.compute_loss_critic(
                observations=batch_data["state"],
                actions=batch_data["action"],
                rewards=batch_data["reward"],
                next_observations=batch_data["next_state"],
                done=batch_data["done"]
            )
            forward_time = time.time() - forward_start
            
            # 记录时间
            monitor.record_timing(data_loading_time, forward_time, 0, 0)
            
            # 记录反向传播时间
            backward_start = time.time()
            
            # 反向传播
            optimizer.zero_grad()
            loss_critic.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                policy.critic.critic_ensemble.parameters(), 
                max_norm=warmup_config.grad_clip_norm
            )
            
            backward_time = time.time() - backward_start
            
            # 记录优化器更新时间
            optimizer_start = time.time()
            optimizer.step()
            optimizer_time = time.time() - optimizer_start
            
            # 记录性能指标
            monitor.record_timing(data_loading_time, forward_time, backward_time, optimizer_time)
            
            epoch_loss += loss_critic.item()
            num_batches += 1
            
            # 定期显示进度和性能指标
            if batch_idx % warmup_config.log_freq == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                current_batch_time = monitor.batch_times[-1] if monitor.batch_times else 0
                print(f"   📊 Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}")
                print(f"      Avg Loss: {avg_loss:.4f}, Batch Time: {current_batch_time:.4f}s")
            
            monitor.end_batch()

        monitor.end_epoch()
        monitor.print_performance_summary()
        
        # 更新学习率
        scheduler.step()
        
        # 验证阶段
        policy.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # 将数据移动到GPU
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device, non_blocking=True)
                
                # 构建验证batch数据
                val_batch_data = {
                    "state": {
                        "observation.image.handeye": batch["observation.image.handeye"],
                        "observation.image.global": batch["observation.image.global"],
                        "observation.state": batch["observation.state"],
                    },
                    "action": batch["action"],
                    "reward": batch["next.reward"],
                    "next_state": {
                        "observation.image.handeye": batch["observation.image.handeye"].clone(),
                        "observation.image.global": batch["observation.image.global"].clone(),
                        "observation.state": batch["observation.state"].clone(),
                        "task": batch["task"],  # 添加task字段，SmolVLA actor需要
                        "action": batch["action"].clone(),  # 添加action字段，SmolVLA actor需要
                    },
                    "done": batch["next.done"]
                }
                
                # 计算验证loss - 让错误直接暴露
                val_loss_critic = policy.compute_loss_critic(
                    observations=val_batch_data["state"],
                    actions=val_batch_data["action"],
                    rewards=val_batch_data["reward"],
                    next_observations=val_batch_data["next_state"],
                    done=val_batch_data["done"]
                )
                val_loss += val_loss_critic.item()
                val_batches += 1
        
        # 计算平均loss
        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"🔄 Epoch {epoch+1}/{num_epochs}")
        print(f"   训练Loss: {avg_train_loss:.4f}, 验证Loss: {avg_val_loss:.4f}")
        print(f"   学习率: {scheduler.get_last_lr()[0]:.6f}")
        
        # WandB 日志记录
        if wandb_logger:
            log_dict = {
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'learning_rate': scheduler.get_last_lr()[0],
                'epoch': epoch + 1
            }
            # 添加性能指标
            if monitor.epoch_times:
                log_dict['epoch_time'] = monitor.epoch_times[-1]
            if monitor.batch_times:
                log_dict['avg_batch_time'] = np.mean(monitor.batch_times[-len(train_loader):])
            if monitor.gpu_memory_usage and torch.cuda.is_available():
                log_dict['gpu_memory_gb'] = np.mean(monitor.gpu_memory_usage[-len(train_loader):])
            
            wandb_logger.log_dict(log_dict, step=epoch + 1)
        
        # 显示epoch性能指标
        epoch_time = monitor.epoch_times[-1] if monitor.epoch_times else 0
        print(f"   ⏱️  Epoch时间: {epoch_time:.2f}s")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(warmup_config.output_dir, "best_critic_checkpoint")
            os.makedirs(save_path, exist_ok=True)
            
            # 保存模型状态
            checkpoint_data = {
                'critic_state_dict': policy.critic.state_dict(),
                'smolvla_sac_config': config,
                'warmup_config': asdict(warmup_config),
                'epoch': epoch + 1,
                'val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }
            torch.save(checkpoint_data, os.path.join(save_path, "best_critic.pth"))
            
            # 保存配置文件
            config_save_path = os.path.join(save_path, "config.json")
            with open(config_save_path, 'w') as f:
                json.dump(asdict(warmup_config), f, indent=2)
            
            print(f"   💾 保存最佳模型，验证Loss: {best_val_loss:.4f}")
        
        # 定期保存检查点
        if (epoch + 1) % warmup_config.save_freq == 0:
            checkpoint_path = os.path.join(warmup_config.output_dir, f"checkpoint_epoch_{epoch+1}")
            os.makedirs(checkpoint_path, exist_ok=True)
            
            checkpoint_data = {
                'critic_state_dict': policy.critic.state_dict(),
                'smolvla_sac_config': config,
                'warmup_config': asdict(warmup_config),
                'epoch': epoch + 1,
                'val_loss': avg_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }
            torch.save(checkpoint_data, os.path.join(checkpoint_path, "checkpoint.pth"))
            print(f"   📁 保存检查点 epoch_{epoch+1}")
        
        monitor.end_epoch()
        monitor.print_performance_summary()
    
    # 保存最终模型
    final_save_path = os.path.join(warmup_config.output_dir, "final_critic_checkpoint")
    os.makedirs(final_save_path, exist_ok=True)
    
    final_checkpoint_data = {
        'critic_state_dict': policy.critic.state_dict(),
        'smolvla_sac_config': config,
        'warmup_config': asdict(warmup_config),
        'epoch': num_epochs,
        'final_val_loss': avg_val_loss,
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    torch.save(final_checkpoint_data, os.path.join(final_save_path, "final_critic.pth"))
    
    # 保存配置文件
    config_save_path = os.path.join(final_save_path, "config.json")
    with open(config_save_path, 'w') as f:
        json.dump(asdict(warmup_config), f, indent=2)
    
    # 上传到 HuggingFace Hub
    if warmup_config.push_to_hub:
        try:
            upload_to_huggingface(final_save_path, warmup_config)
        except Exception as e:
            print(f"⚠️ HuggingFace上传失败: {e}")
    
    print(f"✅ Critic warmup完成！")
    print(f"   最佳验证Loss: {best_val_loss:.4f}")
    print(f"   最终验证Loss: {avg_val_loss:.4f}")
    print(f"   模型保存在: {warmup_config.output_dir}")
    
    return True





def upload_to_huggingface(model_path: str, config: CriticWarmupConfig):
    """上传模型到HuggingFace Hub"""
    try:
        from huggingface_hub import HfApi, create_repo
        
        print(f"🚀 开始上传模型到 HuggingFace Hub: {config.repo_id}")
        
        # 创建仓库（如果不存在）
        api = HfApi()
        try:
            create_repo(repo_id=config.repo_id, repo_type="model", exist_ok=True)
            print(f"✅ 仓库 {config.repo_id} 创建/确认成功")
        except Exception as e:
            print(f"⚠️ 仓库创建警告: {e}")
        
        # 上传所有文件
        model_path = Path(model_path)
        files_to_upload = list(model_path.glob("*"))
        
        for file_path in files_to_upload:
            if file_path.is_file():
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=file_path.name,
                    repo_id=config.repo_id,
                    repo_type="model"
                )
                print(f"   📤 已上传: {file_path.name}")
        
        # 创建 README.md
        readme_content = f"""---
library_name: lerobot
tags:
- robotics
- critic
- reinforcement-learning
- smolvla
---

# Critic Warmup Model for SmolVLA-SAC

这是一个为 SmolVLA-SAC 策略训练的 Critic 模型预热权重。

## 训练配置

- 数据集: {config.dataset_repo_id}
- 训练轮数: {config.epochs}
- 批大小: {config.batch_size}
- 学习率: {config.learning_rate}

## 使用方法

```python
from lerobot.policies.smolvla_sac.modeling_smolvla_sac import SmolVLASACPolicy
import torch

# 加载模型
checkpoint = torch.load("final_critic.pth")
critic_state_dict = checkpoint['critic_state_dict']

# 在你的 SmolVLASACPolicy 中加载权重
policy.critic.load_state_dict(critic_state_dict)
```

训练于: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=config.repo_id,
            repo_type="model"
        )
        
        print(f"✅ 模型成功上传到 HuggingFace Hub!")
        print(f"🔗 查看地址: https://huggingface.co/{config.repo_id}")
        
    except Exception as e:
        print(f"❌ HuggingFace上传失败: {e}")
        raise


def main():
    """主函数"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Critic模型Warmup训练")
    parser.add_argument(
        "--config_path", 
        type=str, 
        default="src/lerobot/configs/critic_warmup_smolvla_ring.json",
        help="配置文件路径"
    )
    args = parser.parse_args()
    
    # 初始化日志
    init_logging()
    
    print("🚀 Critic模型Warmup开始")
    print("=" * 50)
    
    # 1. 加载配置
    print("=== 步骤1: 加载配置 ===")
    try:
        warmup_config = CriticWarmupConfig.from_json(args.config_path)
        print(f"✅ 从 {args.config_path} 加载配置成功")
        print(f"   数据集: {warmup_config.dataset_repo_id}")
        print(f"   输出目录: {warmup_config.output_dir}")
        print(f"   批大小: {warmup_config.batch_size}")
        print(f"   训练轮数: {warmup_config.epochs}")
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return 1
    
    # 设置随机种子
    if warmup_config.seed is not None:
        set_seed(warmup_config.seed)
        print(f"🌱 设置随机种子: {warmup_config.seed}")
    
    # 创建输出目录
    os.makedirs(warmup_config.output_dir, exist_ok=True)
    
    # 2. 初始化WandB
    wandb_logger = None
    if warmup_config.wandb_enable:
        print("\n=== 步骤2: 初始化WandB ===")
        try:
            import wandb
            wandb.init(
                project=warmup_config.wandb_project,
                entity=warmup_config.wandb_entity,
                name=warmup_config.job_name,
                notes=warmup_config.wandb_notes,
                tags=warmup_config.wandb_tags,
                config=asdict(warmup_config),
                dir=warmup_config.output_dir
            )
            print(f"✅ WandB初始化成功")
            print(f"🔗 运行地址: {wandb.run.get_url()}")
            # 创建简化的logger接口
            class SimpleWandBLogger:
                def log_dict(self, d, step):
                    wandb.log(d, step=step)
            wandb_logger = SimpleWandBLogger()
        except Exception as e:
            print(f"⚠️ WandB初始化失败: {e}，继续训练但不记录")
    else:
        print("\n=== WandB已禁用 ===")
    
    # 3. 创建SmolVLA SAC配置
    print("\n=== 步骤3: 创建SmolVLA SAC配置 ===")
    smolvla_config = create_smolvla_sac_config()
    print("✅ SmolVLA SAC配置创建完成")
    
    # 4. 加载数据集
    print("\n=== 步骤4: 加载数据集 ===")
    dataset = load_labeled_dataset(warmup_config.dataset_repo_id)
    if dataset is None:
        print("❌ 数据集加载失败，退出")
        return 1
    
    # 5. 计算数据集统计信息
    print("\n=== 步骤5: 计算数据集统计信息 ===")
    dataset_stats = compute_dataset_stats(dataset)
    
    # 6. 创建SmolVLASACPolicy
    print("\n=== 步骤6: 创建SmolVLASACPolicy ===")
    try:
        policy = SmolVLASACPolicy(
            config=smolvla_config,
            dataset_stats=dataset_stats,
            smolvla_policy=None  # 让SmolVLASACPolicy自己创建
        )
        print("✅ SmolVLASACPolicy创建成功")
    except Exception as e:
        print(f"❌ SmolVLASACPolicy创建失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 7. 准备训练数据
    print("\n=== 步骤7: 准备训练数据 ===")
    train_loader, val_loader = prepare_training_data(
        dataset, smolvla_config, max_samples=warmup_config.max_samples
    )
    if train_loader is None:
        print("❌ 训练数据准备失败，退出")
        return 1
    
    # 8. 离线训练Critic
    print("\n=== 步骤8: 离线训练Critic ===")
    success = train_critic_offline(
        policy, train_loader, val_loader, smolvla_config, warmup_config, wandb_logger
    )
    
    # 完成训练
    if wandb_logger:
        try:
            wandb.finish()
        except:
            pass
    
    if success:
        print("\n🎉 Critic Warmup完成！")
        print(f"📁 模型保存在: {warmup_config.output_dir}")
        if warmup_config.push_to_hub:
            print(f"🔗 HuggingFace: https://huggingface.co/{warmup_config.repo_id}")
        print("\n下一步:")
        print("1. 使用warmup后的critic进行HILSERL训练")
        print("2. 启动learner: python -m lerobot.scripts.rl.learner --config_path src/lerobot/configs/train_config_hilserl_smolvla_ring.json")
        print("3. 启动actor: python -m lerobot.scripts.rl.actor --config_path src/lerobot/configs/train_config_hilserl_smolvla_ring.json")
        return 0
    else:
        print("\n❌ Critic Warmup失败")
        return 1


if __name__ == "__main__":
    exit(main()) 