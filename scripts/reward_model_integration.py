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
from src.lerobot.scripts.label_frames_dataset import LeRobotDataset, FrameLevelLabeler

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


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
        if episode_data:
            # 检查图像字段
            image_keys = [key for key in episode_data[0].keys() if "image" in key]
        else:
            pass
        
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
        
        super().__init__(temp_dataset, episode_idx)
        
        # 重写frames数据，使用我们提供的内存数据（保持原始格式，让父类处理转换）
        self.frames = episode_data
        self.num_frames = len(self.frames)
        
        # 重新设置GUI标题
        if hasattr(self, 'fig'):
            self.fig.suptitle(f'Episode {self.episode_idx} Memory Frame Labeling Tool', fontsize=16)
    
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
        
        # Reward smoothing配置
        self.reward_smoothing_enabled = config.get("reward_smoothing", {}).get("enable", True)
        self.smoothing_frames = config.get("reward_smoothing", {}).get("frames", 20)
        self.smoothing_type = config.get("reward_smoothing", {}).get("type", "linear")  # linear, exponential
        
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
            return reward
        
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
        
        # 确保online_dataset的episode_data_index已经更新
        if online_dataset.episode_data_index is None or len(online_dataset.episode_data_index['from']) < online_dataset.num_episodes:
            # 重新计算episode_data_index
            from lerobot.datasets.utils import get_episode_data_index
            online_dataset.episode_data_index = get_episode_data_index(
                online_dataset.meta.episodes, 
                online_dataset.episodes
            )
        
        # 直接使用原始FrameLevelLabeler和online_dataset
        # FrameLevelLabeler会通过EpisodeSampler自动提取指定episode的数据
        labeler = FrameLevelLabeler(online_dataset, episode_idx)
        result = labeler.show()
        
        if result['saved']:
            # 将标注结果转换为reward并缓存
            self._cache_episode_labels(episode_idx, result['labels'])
            success = True
        else:
            success = False
        
        # 显式清理labeler对象和相关资源
        self._cleanup_labeler(labeler)
        del labeler
        
        # 强制垃圾回收和内存清理
        import gc
        gc.collect()
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        return success
    
    def _cleanup_labeler(self, labeler):
        """清理FrameLevelLabeler对象的所有资源"""
        # 清理图像数据
        if hasattr(labeler, 'frames'):
            labeler.frames.clear()
            labeler.frames = None
        
        # 清理matplotlib资源
        if hasattr(labeler, 'fig'):
            import matplotlib.pyplot as plt
            plt.close(labeler.fig)
            labeler.fig = None
        
        # 清理其他可能的大型对象
        if hasattr(labeler, 'labels'):
            labeler.labels = None
        if hasattr(labeler, 'camera_keys'):
            labeler.camera_keys = None
        if hasattr(labeler, 'dataset'):
            labeler.dataset = None
                
    
    def _extract_episode_data(self, episode_idx: int, online_dataset) -> List[Dict[str, Any]]:
        """从内存中的数据集提取指定episode的数据 - 正确版本"""
        print(f"   🔍 提取episode {episode_idx} 数据...")
        
        # 使用LeRobotDataset的正确方式获取数据（包括视频帧）
        episode_data = []
        
        for frame_idx in range(len(online_dataset)):
            # 使用LeRobotDataset的__getitem__，会自动加载视频帧
            frame = online_dataset[frame_idx]
            
            # 直接获取episode_index字段
            ep_idx = frame['episode_index'].item()
            
            if ep_idx == episode_idx:
                episode_data.append(frame)
        
        print(f"   ✅ 提取到 {len(episode_data)} 帧数据")
        return episode_data
    
    def _cache_episode_labels(self, episode_idx: int, labels: np.ndarray):
        """将episode的标注结果缓存到内存"""
        for frame_idx, label in enumerate(labels):
            cache_key = (episode_idx, frame_idx)
            # 将标签转换为reward值: 0=failure, 1=success, -1=ignore
            if label == -1:  # ignore
                # 重要：ignore标签不应该进入内存缓存，应该被完全过滤掉
                continue  # 跳过这个帧，不进入缓存
            else:  # success (1) 或 failure (0)
                reward = float(label)
                self.labeled_rewards[cache_key] = reward
        
        # 应用reward smoothing
        if self.reward_smoothing_enabled:
            self._apply_reward_smoothing(episode_idx, labels)
    
    def _apply_reward_smoothing(self, episode_idx: int, labels: np.ndarray):
        """应用reward smoothing，在reward=1的第一帧前添加平滑过渡"""
        if not self.reward_smoothing_enabled:
            return
        
        # 找到第一个success帧的位置
        success_indices = np.where(labels == 1)[0]
        if len(success_indices) == 0:
            return
        
        first_success_idx = success_indices[0]
        
        # 计算smoothing的起始位置
        smoothing_start = max(0, first_success_idx - self.smoothing_frames)
        smoothing_end = first_success_idx
        
        # 生成smoothing reward值
        if self.smoothing_type == "linear":
            smoothing_rewards = self._generate_linear_smoothing(smoothing_start, smoothing_end)
        elif self.smoothing_type == "exponential":
            smoothing_rewards = self._generate_exponential_smoothing(smoothing_start, smoothing_end)
        else:
            smoothing_rewards = self._generate_linear_smoothing(smoothing_start, smoothing_end)
        
        # 应用smoothing reward
        for i, reward in enumerate(smoothing_rewards):
            frame_idx = smoothing_start + i
            cache_key = (episode_idx, frame_idx)
            
            # 只更新非ignore的帧
            if frame_idx < len(labels) and labels[frame_idx] != -1:
                self.labeled_rewards[cache_key] = reward
    
    def _generate_linear_smoothing(self, start_idx: int, end_idx: int) -> List[float]:
        """生成线性smoothing reward值"""
        if start_idx >= end_idx:
            return []
        
        num_frames = end_idx - start_idx
        rewards = []
        
        for i in range(num_frames):
            # 线性插值：从0到1
            progress = i / num_frames
            reward = progress
            rewards.append(reward)
        
        return rewards
    
    def _generate_exponential_smoothing(self, start_idx: int, end_idx: int) -> List[float]:
        """生成指数smoothing reward值"""
        if start_idx >= end_idx:
            return []
        
        num_frames = end_idx - start_idx
        rewards = []
        
        for i in range(num_frames):
            # 指数插值：从0到1，后期增长更快
            progress = i / num_frames
            # 使用sigmoid-like函数，在后期增长更快
            reward = 1.0 / (1.0 + np.exp(-5.0 * (progress - 0.5)))
            rewards.append(reward)
        
        return rewards
    
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
        
        # 获取episode的frame范围
        from_idx = labeled_dataset.episode_data_index["from"][episode_idx].item()
        to_idx = labeled_dataset.episode_data_index["to"][episode_idx].item()
        
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
        
        
        # 步骤1: 启动episode标注
        success = self.add_episode_for_labeling(
            episode_idx=episode_idx,
            dataset_repo_id=dataset_repo_id,
            output_repo_id=output_repo_id,
            online_dataset=online_dataset
        )
        
        if not success:
            logger.error(f"Episode {episode_idx} 标注失败，无法继续训练")
            while True:
                time.sleep(10)  # 每10秒检查一次
        
        # 步骤2: 直接更新online_dataset中的reward（从内存缓存）
        success = self._update_episode_rewards_from_cache(episode_idx, online_dataset)
        
        if not success:
            logger.error(f"Episode {episode_idx} reward更新失败")
            while True:
                time.sleep(10)  # 每10秒检查一次
        
        return True
    
    def _update_episode_rewards_from_cache(self, episode_idx: int, online_dataset) -> bool:
        """从内存缓存更新episode的reward"""
        # 确保 episode_data_index 可用且覆盖到当前 episode
        if (
            online_dataset.episode_data_index is None
            or len(online_dataset.episode_data_index.get('from', [])) < online_dataset.num_episodes
        ):
            from lerobot.datasets.utils import get_episode_data_index
            online_dataset.episode_data_index = get_episode_data_index(
                online_dataset.meta.episodes,
                online_dataset.episodes,
            )
        
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
        return True
    
    def is_available(self) -> bool:
        """检查reward model是否可用"""
        return self.model is not None or self.human_labeling_enabled
    
    def test_path_integration(self):
        """测试路径集成修复"""
        # 测试1: 检查标注脚本路径
        if not os.path.exists(self.labeling_script_path):
            return False
        
        # 测试2: 模拟PosixPath输入
        from pathlib import Path
        test_dataset_path = Path("outputs/train/test_dataset")
        test_output_id = "wzn12/test_output"
        
        # 测试3: 转换为字符串
        dataset_path_str = str(test_dataset_path)
        output_id_str = str(test_output_id)
        
        # 测试4: 构建命令（不实际执行）
        cmd = [
            sys.executable,
            self.labeling_script_path,
            "--dataset_repo_id", dataset_path_str,
            "--episode_idx", "0",
            "--output_repo_id", output_id_str
        ]
        
        # 测试5: 字符串拼接
        cmd_str = ' '.join(cmd)
        
        return True


def create_reward_model_integrator(config: Dict[str, Any]):
    """创建reward model集成器"""
    integrator = SimpleRewardModelIntegrator(config)
    return integrator


def test_labeling_interface():
    """测试标注界面的完整流程"""
    # 创建测试配置
    config = {
        "human_labeling": {
            "enable": True,
            "output_repo_id": "test_output_dataset"
        }
    }
    
    # 创建集成器
    integrator = SimpleRewardModelIntegrator(config)
    
    # 创建模拟的episode数据
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
    
    # 创建模拟的online_dataset对象
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
    
    # 测试数据提取
    extracted_data = integrator._extract_episode_data(0, mock_dataset)
    if not extracted_data:
        return False
    
    # 测试MemoryFrameLevelLabeler创建
    labeler = MemoryFrameLevelLabeler(extracted_data, 0)
    
    # 测试标注界面启动
    # 启动标注界面
    result = labeler.show()
    
    # 检查结果
    if result and 'saved' in result and result['saved']:
        # 测试reward缓存
        integrator._cache_episode_labels(0, result['labels'])
        
        # 验证缓存
        for i, label in enumerate(result['labels']):
            expected_reward = 1.0 if label == 1 else 0.0
            cached_reward = integrator._get_human_labeled_reward(0, i)
            if cached_reward != expected_reward:
                return False
        
        return True
        
    else:
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
        success = test_labeling_interface()
        return 0 if success else 1
    
    elif args.test_paths:
        config = {"human_labeling": {"enable": True}}
        integrator = SimpleRewardModelIntegrator(config)
        integrator.test_path_integration()
        return 0
    
    else:
        return 0


if __name__ == "__main__":
    exit(main())