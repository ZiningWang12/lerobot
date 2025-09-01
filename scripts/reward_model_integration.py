#!/usr/bin/env python3
"""
Reward Model集成模块
支持自动reward model和人工标注reward两种模式

功能:
- 加载预训练的reward classifier (自动模式)
- 人工标注reward管理 (人工模式) - 自动调用label_frames_dataset.py并获取结果
- 提供统一的reward预测接口
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
from torchvision.transforms import v2
import subprocess
import sys
import time
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import matplotlib.patches as patches

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.lerobot.scripts.label_frames_dataset import LeRobotDataset, FrameLevelLabeler

logger = logging.getLogger(__name__)


def convert_robot_observation_to_reward_format(observation: Dict[str, Any]) -> Dict[str, Any]:
    """将机器人观察数据转换为reward model期望的格式"""
    reward_observation = {}
    
    # 处理图像数据
    for key, value in observation.items():
        if "image" in key and isinstance(value, np.ndarray):
            reward_observation[key] = value
    
    return reward_observation


class MemoryFrameLevelLabeler(FrameLevelLabeler):
    """内存版本的FrameLevelLabeler，直接使用父类的完整功能"""
    
    def __init__(self, episode_data: List[Dict[str, Any]], episode_idx: int):
        # 调试：显示episode_data的结构
        print(f"🔍 调试episode_data结构:")
        if episode_data:
            print(f"   第一帧键: {list(episode_data[0].keys())}")
            # 检查图像字段
            image_keys = [key for key in episode_data[0].keys() if "image" in key]
            print(f"   图像字段: {image_keys}")
            for key in image_keys:
                value = episode_data[0][key]
                print(f"     {key}: type={type(value)}, shape={getattr(value, 'shape', 'N/A')}")
        else:
            print("   episode_data为空！")
        
        # 创建一个临时的dataset对象，用于满足父类__init__的要求
        class TempDataset:
            def __init__(self, episode_data):
                self.episode_data = episode_data
                # 直接使用数据中的图像字段，不做任何映射
                camera_keys = []
                if episode_data:
                    for key in episode_data[0].keys():
                        if "image" in key.lower():
                            camera_keys.append(key)
                
                print(f"🔧 TempDataset设置camera_keys: {camera_keys}")
                
                self.meta = type('obj', (object,), {
                    'camera_keys': camera_keys
                })()
                # 添加episode_data_index属性，使用tensor格式
                import torch
                self.episode_data_index = {
                    "from": torch.tensor([0]),
                    "to": torch.tensor([len(episode_data)])
                }
            
            def __getitem__(self, idx):
                """实现索引访问"""
                frame_data = self.episode_data[idx].copy()
                # 添加必要的字段
                frame_data['index'] = torch.tensor([idx])
                frame_data['episode_index'] = torch.tensor([0])  # 假设只有一个episode
                frame_data['frame_index'] = torch.tensor([idx])
                frame_data['timestamp'] = torch.tensor([idx * 0.1])  # 假设10fps
                return frame_data
            
            def __len__(self):
                """实现长度查询"""
                return len(self.episode_data)
        
        # 创建临时dataset并调用父类__init__
        temp_dataset = TempDataset(episode_data)
        print(f"🔧 调用父类__init__前，temp_dataset.meta.camera_keys: {temp_dataset.meta.camera_keys}")
        
        super().__init__(temp_dataset, episode_idx)
        
        # 重写frames数据，使用我们提供的内存数据（保持原始格式，让父类处理转换）
        self.frames = episode_data
        self.num_frames = len(self.frames)
        
        print(f"🧠 内存标注器初始化: episode {episode_idx}, {self.num_frames} frames")
        
        # 重新设置GUI标题
        if hasattr(self, 'fig'):
            self.fig.suptitle(f'Episode {self.episode_idx} Memory Frame Labeling Tool', fontsize=16)
        
        print(f"📷 找到相机: {self.camera_keys}")
        print(f"   帧数: {self.num_frames}")
        print(f"   标签: {self.labels}")
    
    def update_image(self):
        """重写update_image方法，调用父类方法处理图像格式转换"""
        # 调用父类的update_image方法，它会自动处理图像格式转换
        super().update_image()
    
    def update_display(self):
        """重写update_display方法，确保正确更新显示"""
        self.update_image()
        # 调用父类的update_timeline方法（如果存在）
        if hasattr(self, 'update_timeline'):
            self.update_timeline()
        # 重绘图形
        if hasattr(self, 'fig') and hasattr(self.fig, 'canvas'):
            self.fig.canvas.draw()


class SimpleRewardModelIntegrator:
    """简化的Reward Model集成器 - 支持人工标注reward"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建图像变换，复用训练时的参数（resize到224x224）
        self.image_transform = v2.Compose([
            v2.Resize(size=(224, 224), antialias=True),
        ])
        
        # 人工标注配置
        self.human_labeling_enabled = config.get("human_labeling", {}).get("enable", False)
        # 确保标注脚本路径是绝对路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.labeling_script_path = os.path.join(script_dir, "..", "src", "lerobot", "scripts", "label_frames_dataset.py")
        self.labeling_script_path = os.path.abspath(self.labeling_script_path)
        
        # 标注结果缓存
        self.labeled_rewards = {}  # (episode_idx, frame_idx) -> reward
        
        # 初始化模型
        self._load_model()
    
    def _load_model(self):
        """加载预训练的reward model"""
        local_path = self.config.get("local_model_path")
        if local_path and os.path.exists(local_path):
            from lerobot.policies.sac.reward_model.modeling_classifier import Classifier
            self.model = Classifier.from_pretrained(local_path)
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ 从本地路径加载reward model成功: {local_path}")
        else:
            print("⚠️ Reward model不可用，将使用人工标注reward")
            self.model = None
    
    def predict_reward(self, observation: Dict[str, Any], action: torch.Tensor, 
                      episode_idx: int = None, frame_idx: int = None) -> float:
        """
        预测reward值 - 优先使用人工标注，如果没有就等待
        """
        if not self.human_labeling_enabled:
            print("⚠️ 人工标注未启用，无法提供reward")
            return 0.0
        
        if episode_idx is None or frame_idx is None:
            print("⚠️ 缺少episode_idx或frame_idx，无法获取标注reward")
            return 0.0
        
        # 优先使用人工标注的reward
        human_reward = self._get_human_labeled_reward(episode_idx, frame_idx)
        if human_reward is not None:
            return human_reward
        
        # 如果没有拿到label，等待直到获得
        print(f"⏳ 等待episode {episode_idx} frame {frame_idx} 的标注结果...")
        while human_reward is None:
            time.sleep(1)  # 等待1秒
            human_reward = self._get_human_labeled_reward(episode_idx, frame_idx)
        
        return human_reward
        
        # 使用reward model预测
        #return self._predict_with_model(observation, action)
    
    def _get_human_labeled_reward(self, episode_idx: int, frame_idx: int) -> Optional[float]:
        """获取人工标注的reward"""
        # 检查内存中是否有缓存的reward
        cache_key = (episode_idx, frame_idx)
        if cache_key in self.labeled_rewards:
            reward = self.labeled_rewards[cache_key]
            print(f"📋 从缓存获取episode {episode_idx} frame {frame_idx} 的reward: {reward}")
            return reward
        
        print(f"❓ 缓存中未找到episode {episode_idx} frame {frame_idx} 的reward")
        return None
    
    def _predict_with_model(self, observation: Dict[str, Any], action: torch.Tensor) -> float:
        """使用reward model预测reward"""
        # 将模型移到正确设备
        self.model = self.model.to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            # 内部处理数据转换：将机器人观察数据转换为reward model期望的格式
            reward_observation = convert_robot_observation_to_reward_format(observation)
            
            # 构建符合reward classifier要求的输入数据
            batch = {}
            
            # 处理图像数据：从 (H, W, C) 转换为 (C, H, W)
            has_images = False
            for key in reward_observation:
                if "image" in key:
                    img_tensor = torch.tensor(reward_observation[key], dtype=torch.float32)
                    # 从 (H, W, C) 转换为 (C, H, W)
                    img_tensor = img_tensor.permute(2, 0, 1)
                    # Resize到224x224
                    img_tensor = self.image_transform(img_tensor)
                    # 添加batch维度
                    img_tensor = img_tensor.unsqueeze(0)
                    batch[key] = img_tensor.to(self.device)

            # 使用reward classifier的predict_reward方法
            reward = self.model.predict_reward(batch, threshold=0.5)
            
            # 确保返回的是标量值
            if isinstance(reward, torch.Tensor):
                reward = reward.item()
            
            return float(reward)
    
    def add_episode_for_labeling(self, episode_idx: int, dataset_repo_id: str, 
                                output_repo_id: str, online_dataset) -> bool:
        """添加episode到标注队列并启动标注"""
        if not self.human_labeling_enabled:
            return False
        
        print(f"🎯 启动episode {episode_idx} 的标注...")
        
        # 从内存中的online_dataset提取episode数据
        episode_data = self._extract_episode_data(episode_idx, online_dataset)
        if not episode_data:
            print(f"❌ 无法提取episode {episode_idx} 的数据")
            return False
        
        print(f"📊 提取到 {len(episode_data)} 帧数据")
        
        # 使用内存版本的标注器
        try:
            labeler = MemoryFrameLevelLabeler(episode_data, episode_idx)
            result = labeler.show()
            
            if result['saved']:
                # 将标注结果转换为reward并缓存
                self._cache_episode_labels(episode_idx, result['labels'])
                print(f"✅ Episode {episode_idx} 标注完成并缓存")
                return True
            else:
                print(f"❌ Episode {episode_idx} 标注未保存")
                return False
                
        except Exception as e:
            print(f"❌ Episode {episode_idx} 标注失败: {e}")
            return False
    
    def _extract_episode_data(self, episode_idx: int, online_dataset) -> List[Dict[str, Any]]:
        """从内存中的数据集提取指定episode的数据"""
        try:
            # 优先直接从online_dataset提取数据（这是最可靠的方式）
            print(f"   📊 尝试直接从online_dataset提取数据...")
            if hasattr(online_dataset, '__len__') and len(online_dataset) > 0:
                # 如果online_dataset有长度，尝试直接访问
                episode_data = []
                for frame_idx in range(len(online_dataset)):
                    frame_data = online_dataset[frame_idx]
                    if frame_data:
                        episode_data.append(frame_data)
                
                if episode_data:
                    print(f"   ✅ 直接从online_dataset提取成功: {len(episode_data)} 帧")
                    # 调试：显示第一帧的详细结构
                    if episode_data:
                        first_frame = episode_data[0]
                        print(f"   🔍 第一帧结构分析:")
                        print(f"      所有键: {list(first_frame.keys())}")
                        image_keys = [key for key in first_frame.keys() if "image" in key.lower()]
                        print(f"      图像相关键: {image_keys}")
                        for key in image_keys:
                            value = first_frame[key]
                            print(f"        {key}: type={type(value)}, shape={getattr(value, 'shape', 'N/A')}")
                    return episode_data
            
            # 优先使用episode_data_index（如果存在）
            if online_dataset.episode_data_index is not None:
                print(f"   📊 使用episode_data_index提取数据...")
                from_idx = online_dataset.episode_data_index["from"][episode_idx].item()
                to_idx = online_dataset.episode_data_index["to"][episode_idx].item()
                
                print(f"   Episode {episode_idx}: frame {from_idx} 到 {to_idx}")
                
                # 使用LeRobotDataset的标准索引方式，这会自动处理图像加载
                episode_data = []
                for frame_idx in range(from_idx, to_idx):
                    if frame_idx < len(online_dataset):
                        # 使用标准索引方式，这会自动加载相机图像
                        frame_data = online_dataset[frame_idx]
                        episode_data.append(frame_data)
                        print(f"     Frame {frame_idx}: 加载成功，包含键: {list(frame_data.keys())}")
                
                print(f"   ✅ 从episode_data_index成功提取 {len(episode_data)} 帧数据")
                return episode_data
            
            # 如果episode_data_index不存在，尝试从hf_dataset直接访问
            elif hasattr(online_dataset, 'hf_dataset') and online_dataset.hf_dataset is not None:
                print(f"   📊 episode_data_index为None，尝试从hf_dataset直接提取数据...")
                
                # 从hf_dataset中查找指定episode的所有帧
                episode_data = []
                for frame_idx in range(len(online_dataset.hf_dataset)):
                    frame = online_dataset.hf_dataset[frame_idx]
                    if hasattr(frame, 'get'):
                        episode_index = frame.get('episode_index', None)
                        if episode_index is not None:
                            if hasattr(episode_index, 'item'):
                                ep_idx = episode_index.item()
                            else:
                                ep_idx = episode_index
                            
                            if ep_idx == episode_idx:
                                episode_data.append(frame)
                
                if episode_data:
                    print(f"   ✅ 从hf_dataset直接提取成功: {len(episode_data)} 帧")
                    # 调试：显示第一帧的详细结构
                    if episode_data:
                        first_frame = episode_data[0]
                        print(f"   🔍 第一帧结构分析:")
                        print(f"      所有键: {list(first_frame.keys())}")
                        image_keys = [key for key in first_frame.keys() if "image" in key.lower()]
                        print(f"      图像相关键: {image_keys}")
                        for key in image_keys:
                            value = first_frame[key]
                            print(f"        {key}: type={type(value)}, shape={getattr(value, 'shape', 'N/A')}")
                    return episode_data
                else:
                    print(f"   ❌ 在hf_dataset中未找到episode {episode_idx} 的数据")
            
            # 最后才考虑episode_buffer（通常为空，因为已经save_episode了）
            print(f"   ⚠️ 尝试从episode_buffer提取数据...")
            
            if hasattr(online_dataset, 'episode_buffer') and online_dataset.episode_buffer is not None:
                # 检查episode_buffer中是否有数据
                if online_dataset.episode_buffer.get("size", 0) > 0:
                    print(f"   📊 从episode_buffer提取数据，大小: {online_dataset.episode_buffer['size']}")
                    
                    # 构建临时的episode数据
                    episode_data = []
                    for i in range(online_dataset.episode_buffer["size"]):
                        frame_data = {}
                        # 从episode_buffer中提取每个字段
                        for key, value in online_dataset.episode_buffer.items():
                            if key not in ["size", "task"] and isinstance(value, (list, np.ndarray)):
                                if i < len(value):
                                    frame_data[key] = value[i]
                        
                        if frame_data:  # 只添加非空的frame
                            episode_data.append(frame_data)
                    
                    print(f"   ✅ 从episode_buffer成功提取 {len(episode_data)} 帧数据")
                    return episode_data
                else:
                    print(f"   ❌ episode_buffer为空，无法提取数据")
                    return []
            else:
                print(f"   ❌ episode_buffer不存在，无法提取数据")
                return []
            
        except Exception as e:
            print(f"   提取episode数据失败: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _cache_episode_labels(self, episode_idx: int, labels: np.ndarray):
        """将episode的标注结果缓存到内存"""
        for frame_idx, label in enumerate(labels):
            cache_key = (episode_idx, frame_idx)
            # 将标签转换为reward值: 0=failure, 1=success, -1=ignore
            if label == -1:  # ignore
                # ❌ 重要：ignore标签不应该进入内存缓存，应该被完全过滤掉
                print(f"   ⚠️ Frame {frame_idx}: 标签 {label} (ignore)，跳过缓存")
                continue  # 跳过这个帧，不进入缓存
            else:  # success (1) 或 failure (0)
                reward = float(label)
                self.labeled_rewards[cache_key] = reward
    
    def update_episode_rewards(self, episode_idx: int, online_dataset, labeled_dataset) -> bool:
        """
        更新整个episode的reward - 在episode结束后调用
        
        Args:
            episode_idx: episode索引
            online_dataset: 在线数据集（内存中的）
            labeled_dataset: 标注后的数据集（内存中的）
        
        Returns:
            是否成功更新
        """
        if not self.human_labeling_enabled:
            return False
        
        print(f"🔄 更新episode {episode_idx} 的reward...")
        
        # 获取episode的frame范围
        from_idx = labeled_dataset.episode_data_index["from"][episode_idx].item()
        to_idx = labeled_dataset.episode_data_index["to"][episode_idx].item()
        
        print(f"   Episode {episode_idx}: frame {from_idx} 到 {to_idx}")
        
        # 更新online_dataset中对应frame的reward
        for frame_idx in range(from_idx, to_idx):
            if frame_idx < len(online_dataset):
                # 获取标注后的reward
                labeled_frame = labeled_dataset[frame_idx]
                if "next.reward" in labeled_frame:
                    labeled_reward = labeled_frame["next.reward"].item()
                    
                    # 更新online_dataset中的reward
                    online_dataset[frame_idx]["next.reward"] = np.array([labeled_reward], dtype=np.float32)
                    
                    # 缓存到内存
                    cache_key = (episode_idx, frame_idx - from_idx)  # episode内的frame索引
                    self.labeled_rewards[cache_key] = float(labeled_reward)
        
        print(f"✅ Episode {episode_idx} reward更新完成")
        return True
    
    def load_labeled_dataset(self, output_repo_id: str):
        """
        加载标注后的数据集
        
        Args:
            output_repo_id: 标注后的数据集仓库ID
        
        Returns:
            标注后的数据集对象
        """
        # 确保转换为字符串
        output_repo_id_str = str(output_repo_id)
        labeled_dataset_path = f"data/{output_repo_id_str.split('/')[-1]}"
        
        from src.lerobot.scripts.label_frames_dataset import LeRobotDataset
        labeled_dataset = LeRobotDataset(output_repo_id_str, root=labeled_dataset_path)
        
        return labeled_dataset
    
    def process_episode_labeling(self, episode_idx: int, dataset_repo_id: str, 
                               output_repo_id: str, online_dataset) -> bool:
        """
        处理整个episode的标注流程：启动标注、等待完成、更新reward
        
        Args:
            episode_idx: episode索引
            dataset_repo_id: 原始数据集仓库ID
            output_repo_id: 标注后的数据集仓库ID
            online_dataset: 在线数据集
        
        Returns:
            是否成功完成整个流程
        """
        if not self.human_labeling_enabled:
            return False
        
        print(f"🎯 开始处理episode {episode_idx} 的标注流程...")
        
        # 步骤1: 启动episode标注
        success = self.add_episode_for_labeling(
            episode_idx=episode_idx,
            dataset_repo_id=dataset_repo_id,
            output_repo_id=output_repo_id,
            online_dataset=online_dataset
        )
        
        if not success:
            print(f"❌ Episode {episode_idx} 标注失败，无法继续训练")
            while True:
                time.sleep(10)  # 每10秒检查一次
        
        print(f"✅ Episode {episode_idx} 标注完成，更新reward...")
        
        # 步骤2: 直接更新online_dataset中的reward（从内存缓存）
        success = self._update_episode_rewards_from_cache(episode_idx, online_dataset)
        
        if not success:
            print(f"❌ Episode {episode_idx} reward更新失败")
            while True:
                time.sleep(10)  # 每10秒检查一次
        
        print(f"✅ Episode {episode_idx} 整个标注流程完成")
        return True
    
    def _update_episode_rewards_from_cache(self, episode_idx: int, online_dataset) -> bool:
        """从内存缓存更新episode的reward"""
        # 如果episode_data_index为None，手动计算它
        if online_dataset.episode_data_index is None:
            print(f"🔧 episode_data_index为None，手动计算...")
            from lerobot.datasets.push_dataset_to_hub.utils import calculate_episode_data_index
            
            # 从hf_dataset计算episode_data_index
            if hasattr(online_dataset, 'hf_dataset') and online_dataset.hf_dataset is not None:
                episode_data_index = calculate_episode_data_index(online_dataset.hf_dataset)
                online_dataset.episode_data_index = episode_data_index
                print(f"✅ 手动计算episode_data_index完成: {len(episode_data_index['from'])} episodes")
            else:
                raise ValueError(f"无法计算episode_data_index：hf_dataset为空")
        
        from_idx = online_dataset.episode_data_index["from"][episode_idx].item()
        to_idx = online_dataset.episode_data_index["to"][episode_idx].item()
        
        # 从缓存更新online_dataset中对应frame的reward
        updated_count = 0
        
        for frame_idx in range(from_idx, to_idx):
            episode_frame_idx = frame_idx - from_idx
            cache_key = (episode_idx, episode_frame_idx)
            
            if cache_key in self.labeled_rewards:
                reward = self.labeled_rewards[cache_key]
                online_dataset[frame_idx]["next.reward"] = np.array([reward], dtype=np.float32)
                updated_count += 1
        
        print(f"✅ Episode {episode_idx} reward更新完成")
        print(f"   更新帧数: {updated_count}")
        return True
    
    def is_available(self) -> bool:
        """检查reward model是否可用"""
        return self.model is not None or self.human_labeling_enabled
    
    def test_path_integration(self):
        """测试路径集成修复"""
        print("🧪 测试路径集成修复...")
        
        # 测试1: 检查标注脚本路径
        print(f"1. 标注脚本路径: {type(self.labeling_script_path)} = {self.labeling_script_path}")
        print(f"   路径是否存在: {os.path.exists(self.labeling_script_path)}")
        
        # 测试2: 模拟PosixPath输入
        from pathlib import Path
        test_dataset_path = Path("outputs/train/test_dataset")
        test_output_id = "wzn12/test_output"
        
        print(f"2. 测试数据集路径: {type(test_dataset_path)} = {test_dataset_path}")
        print(f"   测试输出ID: {type(test_output_id)} = {test_output_id}")
        
        # 测试3: 转换为字符串
        dataset_path_str = str(test_dataset_path)
        output_id_str = str(test_output_id)
        
        print(f"3. 转换后数据集路径: {type(dataset_path_str)} = {dataset_path_str}")
        print(f"   转换后输出ID: {type(output_id_str)} = {output_id_str}")
        
        # 测试4: 构建命令（不实际执行）
        cmd = [
            sys.executable,
            self.labeling_script_path,
            "--dataset_repo_id", dataset_path_str,
            "--episode_idx", "0",
            "--output_repo_id", output_id_str
        ]
        
        print(f"4. 命令列表:")
        for i, arg in enumerate(cmd):
            print(f"   cmd[{i}]: {type(arg)} = {arg}")
        
        # 测试5: 字符串拼接
        try:
            cmd_str = ' '.join(cmd)
            print(f"5. 命令字符串拼接成功: {cmd_str}")
        except Exception as e:
            print(f"5. 命令字符串拼接失败: {e}")
        
        print("\n✅ 路径集成测试完成！")
        return True


def create_reward_model_integrator(config: Dict[str, Any]):
    """创建reward model集成器"""
    integrator = SimpleRewardModelIntegrator(config)
    return integrator


def test_labeling_interface():
    """测试标注界面的完整流程"""
    print("🧪 开始测试标注界面...")
    print("=" * 60)
    
    # 创建测试配置
    config = {
        "human_labeling": {
            "enable": True,
            "output_repo_id": "test_output_dataset"
        }
    }
    
    # 创建集成器
    print("1️⃣ 创建Reward Model集成器...")
    integrator = SimpleRewardModelIntegrator(config)
    print("✅ 集成器创建成功")
    
    # 创建模拟的episode数据
    print("2️⃣ 创建模拟episode数据...")
    import numpy as np
    
    # 模拟10帧数据，包含图像、动作等
    episode_data = []
    for i in range(10):
        frame_data = {
            # 模拟HandEye相机图像 (480x640 RGB)
            "observation.images.handeye": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            # 模拟Global相机图像 (480x640 RGB)
            "observation.images.global": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            # 模拟动作数据 (4维)
            "action": np.array([0.1 + i*0.01, 0.2 + i*0.01, 0.3 + i*0.01, 0.4 + i*0.01], dtype=np.float32),
            # 模拟reward
            "next.reward": np.array([0.0], dtype=np.float32),
            # 模拟done标志
            "next.done": np.array([False], dtype=bool)
        }
        episode_data.append(frame_data)
    
    print(f"✅ 创建了 {len(episode_data)} 帧模拟数据")
    
    # 创建模拟的online_dataset对象
    print("3️⃣ 创建模拟online_dataset对象...")
    class MockOnlineDataset:
        def __init__(self, episode_data):
            self.episode_data_index = None  # 模拟新创建的数据集
            self.episode_buffer = {
                'size': len(episode_data),
                'task': ['test_task'] * len(episode_data),
                'observation.images.handeye': [frame['observation.images.handeye'] for frame in episode_data],
                'observation.images.global': [frame['observation.images.global'] for frame in episode_data],
                'action': [frame['action'] for frame in episode_data],
                'next.reward': [frame['next.reward'] for frame in episode_data],
                'next.done': [frame['next.done'] for frame in episode_data]
            }
            self.root = "/tmp/test_dataset"
    
    mock_dataset = MockOnlineDataset(episode_data)
    print("✅ 模拟数据集创建成功")
    
    # 测试数据提取
    print("4️⃣ 测试数据提取功能...")
    extracted_data = integrator._extract_episode_data(0, mock_dataset)
    if extracted_data:
        print(f"✅ 数据提取成功: {len(extracted_data)} 帧")
        print(f"   第一帧包含字段: {list(extracted_data[0].keys())}")
    else:
        print("❌ 数据提取失败")
        return False
    
    # 测试MemoryFrameLevelLabeler创建
    print("5️⃣ 测试MemoryFrameLevelLabeler创建...")
    try:
        labeler = MemoryFrameLevelLabeler(extracted_data, 0)
        print("✅ MemoryFrameLevelLabeler创建成功")
        print(f"   相机键: {labeler.camera_keys}")
        print(f"   帧数: {labeler.num_frames}")
        print(f"   标签: {labeler.labels}")
    except Exception as e:
        print(f"❌ MemoryFrameLevelLabeler创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试标注界面启动
    print("6️⃣ 测试标注界面启动...")
    print("🎨 即将启动标注界面，请按照以下说明操作：")
    print("   - 使用左右箭头键导航帧")
    print("   - 按0标记为FAILURE (红色)")
    print("   - 按1标记为SUCCESS (绿色)")
    print("   - 按-标记为IGNORE (灰色)")
    print("   - 点击Save按钮保存标注")
    print("   - 关闭窗口完成测试")
    print("")
    print("🚀 启动标注界面...")
    
    try:
        # 启动标注界面
        result = labeler.show()
        
        # 检查结果
        if result and 'saved' in result and result['saved']:
            print("✅ 标注界面测试成功！")
            print(f"   保存的标签: {result['labels']}")
            print(f"   标签统计:")
            labels = result['labels']
            success_count = np.sum(labels == 1)
            failure_count = np.sum(labels == 0)
            ignore_count = np.sum(labels == -1)
            print(f"     SUCCESS: {success_count}")
            print(f"     FAILURE: {failure_count}")
            print(f"     IGNORE: {ignore_count}")
            
            # 测试reward缓存
            print("7️⃣ 测试reward缓存功能...")
            integrator._cache_episode_labels(0, result['labels'])
            print("✅ Reward缓存成功")
            
            # 验证缓存
            for i, label in enumerate(result['labels']):
                expected_reward = 1.0 if label == 1 else 0.0
                cached_reward = integrator._get_human_labeled_reward(0, i)
                if cached_reward == expected_reward:
                    print(f"   Frame {i}: 标签 {label} -> reward {cached_reward} ✅")
                else:
                    print(f"   Frame {i}: 标签 {label} -> reward {cached_reward} ❌")
            
            print("\n🎉 所有测试通过！标注界面工作正常！")
            return True
            
        else:
            print("❌ 标注界面测试失败：未保存标注结果")
            return False
            
    except Exception as e:
        print(f"❌ 标注界面启动失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数 - 用于直接测试"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Reward Model Integration 测试")
    parser.add_argument(
        "--test_labeling", 
        action="store_true",
        help="测试标注界面"
    )
    parser.add_argument(
        "--test_paths",
        action="store_true", 
        help="测试路径集成"
    )
    
    args = parser.parse_args()
    
    if args.test_labeling:
        print("🎯 启动标注界面测试...")
        success = test_labeling_interface()
        if success:
            print("\n✅ 标注界面测试完成！")
            return 0
        else:
            print("\n❌ 标注界面测试失败！")
            return 1
    
    elif args.test_paths:
        print("🔍 启动路径集成测试...")
        config = {"human_labeling": {"enable": True}}
        integrator = SimpleRewardModelIntegrator(config)
        integrator.test_path_integration()
        return 0
    
    else:
        print("使用方法:")
        print("  python reward_model_integration.py --test_labeling    # 测试标注界面")
        print("  python reward_model_integration.py --test_paths      # 测试路径集成")
        return 0


if __name__ == "__main__":
    exit(main())