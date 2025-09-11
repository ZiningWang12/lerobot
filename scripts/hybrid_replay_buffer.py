#!/usr/bin/env python3
"""
混合ReplayBuffer模块 - 精简版
支持离线+在线数据混合，实现渐进式阶段切换
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 复用现有的ReplayBuffer
from lerobot.utils.buffer import ReplayBuffer
from lerobot.datasets.lerobot_dataset import LeRobotDataset


@dataclass
class MixingStage:
    """混合训练阶段配置"""
    name: str
    episodes: List[int]  # [start, end)
    offline_ratio: float
    description: str


class HybridReplayBuffer:
    """支持离线+在线数据混合的ReplayBuffer"""
    
    def __init__(self, offline_capacity: int, online_capacity: int, device: str = "cpu"):
        self.offline_buffer = ReplayBuffer(
            capacity=offline_capacity,
            device=device,
            state_keys=["observation.images.handeye", "observation.images.global", "observation.state"],
            storage_device="cpu",
            optimize_memory=True
        )
        
        self.online_buffer = ReplayBuffer(
            capacity=online_capacity,
            device=device,
            state_keys=["observation.images.handeye", "observation.images.global", "observation.state"],
            storage_device="cpu",
            optimize_memory=True
        )
        
        self.device = device
        self.episode_count = 0
        self.current_stage = "initial"
        self.stage_transitions = []
        
        # 默认阶段配置
        self.default_stages = {
            "initial": MixingStage("initial", [0, 10], 0.9, "初期：主要使用离线数据，在线数据少量引入"),
            "middle": MixingStage("middle", [10, 20], 0.6, "中期：平衡离线在线数据，在线数据比例提升"),
            "late": MixingStage("late", [20, 30], 0.3, "后期：主要使用在线数据，离线数据作为正则化"),
            "final": MixingStage("final", [30, float('inf')], 0.2, "最终：在线数据主导，离线数据保持稳定性")
        }
        
        print(f"🔧 混合ReplayBuffer初始化完成")
        print(f"   离线缓冲区容量: {offline_capacity}")
        print(f"   在线缓冲区容量: {online_capacity}")
        print(f"   设备: {device}")
    
    def set_mixing_stages(self, stages: Dict[str, MixingStage]):
        """设置混合训练阶段"""
        self.default_stages = stages
        print(f"✅ 混合训练阶段配置已更新: {len(stages)} 个阶段")
    
    def add_lerobot_frame(self, frame: dict, next_frame: dict = None, is_online: bool = True):
        """
        添加LeRobotDataset格式的frame数据
        
        Args:
            frame: LeRobotDataset格式的当前frame
            next_frame: LeRobotDataset格式的下一个frame (可选)
            is_online: 是否为在线数据
        """
        # 转换数据格式
        transition = self._frame_to_transition(frame, next_frame)
        
        # 如果转换失败（例如reward=-1），则跳过该帧
        if transition is None:
            return

        # 添加到对应的buffer
        if is_online:
            self.online_buffer.add(**transition)
        else:
            self.offline_buffer.add(**transition)
    
    def _frame_to_transition(self, frame: dict, next_frame: dict = None) -> dict:
        """将LeRobotDataset格式的frame转换为ReplayBuffer格式的transition"""
        # 提取observation数据
        state = {}
        next_state = {}
        
        # 直接使用原始字段名，不做任何映射
        for key, value in frame.items():
            if key.startswith("observation."):
                # 转换为torch.tensor并添加batch维度
                if isinstance(value, np.ndarray):
                    tensor_value = torch.from_numpy(value)
                elif isinstance(value, torch.Tensor):
                    tensor_value = value
                else:
                    tensor_value = torch.tensor(value)
                
                # 统一处理图像数据格式：确保所有图像都是 (C, H, W) 格式且保持uint8
                if key.endswith(".images") or "images" in key:
                    # 🔧 关键修复：强制保持uint8格式，避免float32内存泄漏
                    if tensor_value.dtype == torch.float32:
                        # 如果已经是float32，转换回uint8 (假设范围是[0,1])
                        tensor_value = (tensor_value * 255).clamp(0, 255).to(torch.uint8)
                    elif tensor_value.dtype != torch.uint8:
                        # 确保所有图像数据都是uint8
                        tensor_value = tensor_value.to(torch.uint8)
                    
                    # 检查图像维度并标准化
                    if tensor_value.dim() == 3:
                        if tensor_value.shape[0] == 3:  # 已经是 (C, H, W) 格式
                            pass
                        elif tensor_value.shape[2] == 3:  # 是 (H, W, C) 格式
                            # 转换为 (C, H, W) 格式
                            tensor_value = tensor_value.permute(2, 0, 1)
                        else:
                            # 其他情况，保持原样
                            pass
                
                # 添加batch维度
                tensor_value = tensor_value.unsqueeze(0) if tensor_value.dim() == 3 else tensor_value
                
                # 直接存储原始字段名
                state[key] = tensor_value
            elif key == "action":
                # 动作字段保持不变
                continue
            elif key.startswith("next."):
                # next字段保持不变
                continue
            else:
                # 其他字段保持不变
                continue
        
        # 处理action数据
        action = None
        action_value = frame["action"]
        
        if isinstance(action_value, np.ndarray):
            action = torch.from_numpy(action_value).unsqueeze(0)
        else:
            action = torch.tensor(action_value).unsqueeze(0) if not isinstance(action_value, torch.Tensor) else action_value.unsqueeze(0)

        # 处理reward和done - 这些字段必须存在，否则core dump
        if "next.reward" not in frame:
            raise KeyError(f"数据集缺少必需的字段: next.reward")
        if "next.done" not in frame:
            raise KeyError(f"数据集缺少必需的字段: next.done")
        
        # 获取reward值，过滤掉reward=-1的帧（ignore标签）
        reward_value = frame["next.reward"].item()
        if reward_value == -1:
            # ❌ 重要：reward=-1表示ignore标签，应该被过滤掉
            # 返回None表示这个frame应该被跳过
            return None
        
        reward = float(reward_value)
        done = bool(frame["next.done"].item())
        
        return {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "truncated": False
        }
    
    def load_offline_data(self, dataset, max_episodes: int = None, robot_features: Dict = None) -> int:
        """
        从LeRobotDataset加载离线数据到offline buffer
        
        Args:
            dataset: LeRobotDataset实例
            max_episodes: 最大加载episode数量，None表示加载所有
            robot_features: 机器人特征配置（可选，用于兼容性）
        
        Returns:
            加载的数据帧数量
        """
        print(f"   数据集: {dataset.repo_id}， 总episodes: {dataset.num_episodes}， 总frames: {dataset.num_frames}")
        print(f"   总episodes: {dataset.num_episodes}")
        print(f"   总frames: {dataset.num_frames}")
        
        if max_episodes is None:
            max_episodes = dataset.num_episodes
        
        max_episodes = min(max_episodes, dataset.num_episodes)
        print(f"   计划加载episodes: {max_episodes}")
        
        offline_data_count = 0
        
        # 获取数据集信息
        print(f"   video_keys: {dataset.meta.video_keys}")
        print(f"   camera_keys: {dataset.meta.camera_keys}")
        
        # 遍历每个episode的帧
        for episode_idx in range(max_episodes):
            print(f"   📖 处理Episode {episode_idx + 1}/{max_episodes}")
            
            # 使用episode_data_index获取episode的帧范围
            episode_start = dataset.episode_data_index["from"][episode_idx].item()
            episode_end = dataset.episode_data_index["to"][episode_idx].item()
            episode_frames = episode_end - episode_start
            
            print(f"      Episode {episode_idx} 帧范围: {episode_start}-{episode_end} (共{episode_frames}帧)")
            
            # 遍历episode中的每一帧
            for frame_idx in range(episode_start, episode_end):
                # 直接使用LeRobotDataset的索引机制获取完整frame数据（包括图像）
                frame = dataset[frame_idx]
                
                # 获取next_frame（如果存在）
                next_frame = None
                if frame_idx < episode_end - 1:
                    next_frame_idx = frame_idx + 1
                    next_frame = dataset[next_frame_idx]
                
                # 添加到offline buffer - 如果字段缺失，这里会直接coredump
                self.add_lerobot_frame(
                    frame=frame,
                    next_frame=next_frame,
                    is_online=False
                )
                offline_data_count += 1
        
        print(f"✅ 离线数据加载完成，共加载 {offline_data_count} 帧数据")
        print(f"   离线缓冲区大小: {len(self.offline_buffer)}")
        
        return offline_data_count
    
    def update_stage(self, episode_count: int) -> MixingStage:
        """根据episode数量更新训练阶段"""
        for stage_name, stage in self.default_stages.items():
            if stage.episodes[0] <= episode_count < stage.episodes[1]:
                if self.current_stage != stage_name:
                    old_stage = self.current_stage
                    self.current_stage = stage_name
                    
                    # 记录阶段转换
                    transition = {
                        'episode': episode_count,
                        'from_stage': old_stage,
                        'to_stage': stage_name,
                        'offline_ratio': stage.offline_ratio,
                        'description': stage.description
                    }
                    self.stage_transitions.append(transition)
                    
                    print(f"🔄 训练阶段切换: {old_stage} -> {stage_name}")
                    print(f"   离线数据比例: {stage.offline_ratio:.1%}")
                    print(f"   说明: {stage.description}")
                
                return stage
        
        # 如果没有匹配的阶段，使用final阶段
        return self.default_stages["final"]
    
    def sample_mixed(self, batch_size: int, episode_count: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """混合采样"""
        # 更新训练阶段
        stage_config = self.update_stage(episode_count)
        offline_ratio = stage_config.offline_ratio
        
        # 计算采样数量
        offline_size = int(batch_size * offline_ratio)
        online_size = batch_size - offline_size
        
        # 确保有足够的数据
        available_offline = len(self.offline_buffer)
        available_online = len(self.online_buffer)
        
        # 调整采样数量
        if available_offline < offline_size:
            offline_size = available_offline
            online_size = min(batch_size - offline_size, available_online)
        if available_online < online_size:
            online_size = available_online
            offline_size = min(batch_size - offline_size, available_offline)
        
        # 采样数据
        offline_batch = {}
        online_batch = {}
        
        if offline_size > 0 and available_offline > 0:
            offline_batch = self.offline_buffer.sample(offline_size)
        
        if online_size > 0 and available_online > 0:
            online_batch = self.online_buffer.sample(online_size)
        
        # 合并batch
        mixed_batch = self._combine_batches(offline_batch, online_batch)
        
        # 🔧 关键修复：预处理图像数据，将uint8转换为float32 [0,1]供训练使用
        mixed_batch = self._preprocess_batch_images(mixed_batch)
        
        # 采样信息
        sampling_info = {
            "stage": self.current_stage,
            "offline_ratio": offline_ratio,
            "offline_size": offline_size,
            "online_size": online_size,
            "available_offline": available_offline,
            "available_online": available_online,
            "total_batch_size": len(mixed_batch.get("state", []))
        }
        
        return mixed_batch, sampling_info
    
    def _combine_batches(self, offline_batch: Dict[str, Any], online_batch: Dict[str, Any]) -> Dict[str, Any]:
        """合并离线在线batch数据"""
        if not offline_batch and not online_batch:
            return {}
        
        if not offline_batch:
            return online_batch
        if not online_batch:
            return offline_batch
        # 合并数据
        combined_batch = {}
        
        for key in offline_batch.keys():
            if key in online_batch:
                # 合并张量
                if isinstance(offline_batch[key], torch.Tensor) and isinstance(online_batch[key], torch.Tensor):
                    combined_batch[key] = torch.cat([offline_batch[key], online_batch[key]], dim=0)
                # 合并字典（如state字段）
                elif isinstance(offline_batch[key], dict) and isinstance(online_batch[key], dict):
                    combined_batch[key] = {}
                    for sub_key in offline_batch[key].keys():
                        combined_batch[key][sub_key] = torch.cat([offline_batch[key][sub_key], online_batch[key][sub_key]], dim=0)  
        return combined_batch
    
    def _preprocess_batch_images(self, batch: dict) -> dict:
        """
        预处理批次中的图像数据，将uint8 [0,255] 转换为 float32 [0,1]
        这是训练前必需的步骤，因为SmolVLA期望输入范围是[0,1]
        """
        if not batch:
            return batch
            
        processed_batch = batch.copy()
        
        # 处理state中的图像
        if "state" in batch and isinstance(batch["state"], dict):
            processed_batch["state"] = {}
            for key, value in batch["state"].items():
                if key.endswith(".images") or "images" in key:
                    # 将uint8 [0,255] 转换为 float32 [0,1]
                    if value.dtype == torch.uint8:
                        processed_batch["state"][key] = value.float() / 255.0
                    else:
                        processed_batch["state"][key] = value
                else:
                    processed_batch["state"][key] = value
        
        # 处理next_state中的图像
        if "next_state" in batch and isinstance(batch["next_state"], dict):
            processed_batch["next_state"] = {}
            for key, value in batch["next_state"].items():
                if key.endswith(".images") or "images" in key:
                    # 将uint8 [0,255] 转换为 float32 [0,1]
                    if value.dtype == torch.uint8:
                        processed_batch["next_state"][key] = value.float() / 255.0
                    else:
                        processed_batch["next_state"][key] = value
                else:
                    processed_batch["next_state"][key] = value
        
        return processed_batch
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """获取缓冲区统计信息"""
        return {
            "current_stage": self.current_stage,
            "offline_buffer": {
                "size": len(self.offline_buffer),
                "capacity": self.offline_buffer.capacity
            },
            "online_buffer": {
                "size": len(self.online_buffer),
                "capacity": self.online_buffer.capacity
            },
            "episode_count": self.episode_count
        }
    
    def clear_buffers(self):
        """清空缓冲区"""
        self.offline_buffer.clear()
        self.online_buffer.clear()
        print("🧹 缓冲区已清空")
    
    def save_buffer_state(self, save_path: str):
        """保存缓冲区状态"""
        os.makedirs(save_path, exist_ok=True)
        
        # 保存元数据
        meta_path = os.path.join(save_path, "buffer_meta.json")
        import json
        meta_data = {
            "episode_count": self.episode_count,
            "current_stage": self.current_stage,
            "stage_transitions": self.stage_transitions,
            "offline_buffer_size": len(self.offline_buffer),
            "online_buffer_size": len(self.online_buffer)
        }
        with open(meta_path, 'w') as f:
            json.dump(meta_data, f, indent=2)
        
        print(f"💾 缓冲区状态已保存到: {save_path}")
        print(f"   离线缓冲区大小: {len(self.offline_buffer)}")
        print(f"   在线缓冲区大小: {len(self.online_buffer)}")
    
    def load_buffer_state(self, load_path: str):
        """加载缓冲区状态"""
        # 加载元数据
        meta_path = os.path.join(load_path, "buffer_meta.json")
        if os.path.exists(meta_path):
            import json
            with open(meta_path, 'r') as f:
                meta_data = json.load(f)
            
            self.episode_count = meta_data.get("episode_count", 0)
            self.current_stage = meta_data.get("current_stage", "initial")
            self.stage_transitions = meta_data.get("stage_transitions", [])
            
            print(f"📥 缓冲区状态已从 {load_path} 加载")
            print(f"   离线缓冲区大小: {meta_data.get('offline_buffer_size', 0)}")
            print(f"   在线缓冲区大小: {meta_data.get('online_buffer_size', 0)}")
        else:
            print(f"⚠️ 未找到缓冲区状态文件: {meta_path}")


def create_hybrid_replay_buffer(config_dict: Dict[str, Any]) -> HybridReplayBuffer:
    """创建混合ReplayBuffer - 工厂函数"""
    offline_capacity = config_dict.get("offline_capacity", 100000)
    online_capacity = config_dict.get("online_capacity", 100000)
    device = config_dict.get("device", "cpu")
    
    return HybridReplayBuffer(offline_capacity, online_capacity, device)


def test_hybrid_replay_buffer():
    """测试混合ReplayBuffer的核心功能"""
    print("🧪 测试混合ReplayBuffer...")
    
    # 创建缓冲区
    buffer = create_hybrid_replay_buffer({
        "offline_capacity": 1000,
        "online_capacity": 1000,
        "device": "cpu"
    })
    
    # 测试离线数据加载
    print("\n📚 测试离线数据加载...")
    try:
        dataset = LeRobotDataset("wzn12/teleop_ring")
        loaded_count = buffer.load_offline_data(dataset, max_episodes=2)
        
        if loaded_count > 0:
            print("✅ 离线数据加载成功！")
            
            # 测试采样功能
            print("\n🧪 测试混合采样...")
            batch, sampling_info = buffer.sample_mixed(4, episode_count=0)
            print(f"✅ 采样成功，批次大小: {sampling_info['total_batch_size']}")
            print(f"   采样信息: {sampling_info}")
            
            # 测试统计信息
            print("\n📊 缓冲区统计:")
            stats = buffer.get_buffer_stats()
            for key, value in stats.items():
                if not key.startswith("stage_"):
                    print(f"   {key}: {value}")
            
            print("\n🎉 混合ReplayBuffer测试完成！")
            return True
        else:
            print("❌ 离线数据加载失败")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_hybrid_replay_buffer() 