#!/usr/bin/env python3
"""
在线强化学习工具模块
包含配置创建、数据集管理、数据采集等所有工具函数
"""

import os
import time
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

# 导入LeRobot组件
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.robots import RobotConfig
from lerobot.teleoperators import TeleoperatorConfig
from lerobot.utils.control_utils import predict_action
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import log_rerun_data


def create_online_dataset(config: Dict[str, Any], robot) -> LeRobotDataset:
    """创建用于记录online数据的LeRobotDataset"""
    print("📦 创建online数据集...")
    
    # 构建数据集特征
    action_features = hw_to_dataset_features(robot.action_features, "action", use_video=True)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", use_video=True)
    
    # 添加标准字段
    dataset_features = {
        **action_features,
        **obs_features,
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None}
    }
    
    # 创建唯一的数据集目录，避免与LeRobotDataset.create的exist_ok=False冲突
    timestamp = time.strftime("%Y%m%d-%H%M")  # 年月日-时分格式
    pid = os.getpid()
    unique_dataset_dir = os.path.join(config.output_dir, f"online_dataset_{timestamp}_{pid}")
    
    # 确保目录不存在（防止冲突）
    if os.path.exists(unique_dataset_dir):
        import shutil
        shutil.rmtree(unique_dataset_dir)
    
    # 创建数据集
    dataset = LeRobotDataset.create(
        repo_id=f"{config.job_name}_online_data",
        fps=config.training.fps if hasattr(config.training, 'fps') else 30,
        root=unique_dataset_dir,
        robot_type=robot.robot_type,
        features=dataset_features,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4
    )
    
    print(f"✅ Online数据集创建成功")
    print(f"   数据集路径: {dataset.root}")
    return dataset


def create_robot_config(config: Dict[str, Any]) -> RobotConfig:
    """创建机器人配置 - 直接使用record.py的配置结构"""
    robot_config_dict = config.env.robot
    
    # 根据robot_type创建对应的配置类
    if robot_config_dict.type == "so101_follower":
        from lerobot.robots.so101_follower import SO101FollowerConfig
        robot_config = SO101FollowerConfig(
            port=robot_config_dict.port,
            id=robot_config_dict.id,  # 添加robot.id参数
            cameras=robot_config_dict.cameras
        )
    else:
        raise ValueError(f"不支持的机器人类型: {robot_config_dict['type']}")
    
    return robot_config


def create_teleop_config(config: Dict[str, Any]) -> TeleoperatorConfig:
    """创建遥操作配置 - 直接使用record.py的配置结构"""
    teleop_config_dict = config["teleop"]
    
    if teleop_config_dict["type"] == "so101_leader":
        from lerobot.teleoperators.so101_leader import SO101LeaderConfig
        teleop_config = SO101LeaderConfig(
            port=teleop_config_dict["port"],
            id=teleop_config_dict["id"]  # 添加teleop.id参数
        )
    else:
        raise ValueError(f"不支持的遥操作类型: {teleop_config_dict['type']}")
    
    return teleop_config


class OnlineDataCollector:
    """在线数据采集器 - 负责策略数据收集"""
    
    def __init__(self, config: Dict[str, Any], robot, policy, device, online_dataset, hybrid_buffer):
        self.config = config
        self.robot = robot
        self.policy = policy
        self.device = device
        self.online_dataset = online_dataset
        self.hybrid_buffer = hybrid_buffer
        
        self.obs_features = self.online_dataset.features
        
    def collect_policy_data(self, episode: int, task_description: str) -> Tuple[float, int, Dict, Dict]:
        """
        收集策略数据
        
        Returns:
            episode_reward: 总奖励
            episode_steps: 总步数
            final_observation: 最终观察状态
            last_action: 最后一个策略动作（用于平滑过渡）
        """
        print(f"🎬 开始收集策略数据 - Episode {episode + 1}")
        
        # 初始化episode
        observation = self.robot.get_observation()
        episode_reward = 0.0
        episode_steps = 0
        last_action = None
        
        # 开始记录episode
        self.online_dataset.episode_buffer = self.online_dataset.create_episode_buffer(episode_index=episode)
        
        # 策略数据收集循环
        for step in range(self.config.training.steps_per_episode):
            # 使用策略生成动作
            observation_frame = build_dataset_frame(self.obs_features, observation, prefix="observation")
            action_values = predict_action(
                observation_frame,
                self.policy,
                self.device,
                False,  # use_amp
                task=task_description,
                robot_type=self.robot.robot_type
            )
            
            # 转换为动作字典
            action = {key: action_values[i].item() for i, key in enumerate(self.robot.action_features)}
            sent_action = self.robot.send_action(action)
            next_observation = self.robot.get_observation()
            actual_reward = 0.0  # 默认reward，会在episode结束后更新
            
            # 组织数据并记录
            self._record_frame_data(
                observation=observation,
                action=sent_action,
                reward=actual_reward,
                next_observation=next_observation,
                done=False,
                episode=episode,
                step=step,
                task_description=task_description
            )
            
            # 可视化
            if self.config.visualization.enable:
                log_rerun_data(observation, sent_action)
            
            episode_reward += actual_reward
            episode_steps += 1
            observation = next_observation
            last_action = sent_action  # 保存最后一个动作
        
        print(f"✅ 策略数据收集完成 - {episode_steps} 步")
        return episode_reward, episode_steps, observation, last_action
    
    def _record_frame_data(self, observation: Dict, action: Dict, reward: float, 
                          next_observation: Dict, done: bool, episode: int, 
                          step: int, task_description: str):
        """记录单帧数据到数据集和缓冲区"""
        
        # 组织标准格式的online数据
        frame_data = self._organize_online_data(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=done,
            episode=episode,
            step=step,
            task_description=task_description
        )
        
        # 记录到online数据集
        self.online_dataset.add_frame(frame_data, task=task_description)
        
        # 存储经验到在线缓冲区
        self._store_to_replay_buffer(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=done
        )
    
    def _organize_online_data(self, observation: Dict, action: Dict, reward: float, 
                             next_observation: Dict, done: bool, episode: int, 
                             step: int, task_description: str) -> Dict:
        """按照LeRobot标准格式组织online数据"""
        
        # 构建标准格式的frame数据
        observation_frame = build_dataset_frame(self.online_dataset.features, observation, prefix="observation")
        action_frame = build_dataset_frame(self.online_dataset.features, action, prefix="action")
        
        # 直接添加非前缀字段
        frame = {
            **observation_frame,
            **action_frame,
            "next.reward": np.array([reward], dtype=np.float32),
            "next.done": np.array([done], dtype=bool)
        }
        
        return frame
    
    def _store_to_replay_buffer(self, observation: Dict, action: Dict, reward: float, 
                               next_observation: Dict, done: bool):
        """存储经验到混合ReplayBuffer"""
        
        # 使用build_dataset_frame标准化数据格式
        observation_frame = build_dataset_frame(self.obs_features, observation, prefix="observation")
        next_observation_frame = build_dataset_frame(self.obs_features, next_observation, prefix="observation")
        
        # 构建完整的frame数据
        complete_frame = {
            **observation_frame,
            "action": np.array([action[key] for key in self.robot.action_features], dtype=np.float32),
            "next.reward": np.array([float(reward)], dtype=np.float32),
            "next.done": np.array([done], dtype=bool)
        }
        
        # 使用add_lerobot_frame方法
        self.hybrid_buffer.add_lerobot_frame(
            frame=complete_frame,
            next_frame=next_observation_frame,
            is_online=True
        )


class TeleopDataAppender:
    """遥操作数据追加器 - 负责在策略数据后追加遥操作数据"""
    
    def __init__(self, config: Dict[str, Any], robot, teleop, online_dataset, hybrid_buffer):
        self.config = config
        self.robot = robot
        self.teleop = teleop
        self.online_dataset = online_dataset
        self.hybrid_buffer = hybrid_buffer
        
        self.obs_features = self.online_dataset.features
    
    def append_teleop_data(self, episode: int, initial_observation: Dict, 
                          initial_reward: float, initial_step: int, 
                          last_policy_action: Dict = None) -> int:
        """
        追加遥操作数据到当前episode，包含平滑过渡机制
        
        Args:
            episode: episode编号
            initial_observation: 初始观察状态
            initial_reward: 初始奖励
            initial_step: 初始步数
            last_policy_action: 最后一个策略动作，用于平滑过渡
        
        Returns:
            total_teleop_steps: 遥操作总步数
        """
        if not self.config.training.teleop_append.enable:
            return 0
        
        print(f"🎮 开始遥操作数据收集，时长: {self.config.training.teleop_append.duration_s}秒")
        
        # 获取平滑过渡步数配置
        smooth_steps = getattr(self.config.training.teleop_append, 'smooth_steps', 15)
        
        # 复用record.py的遥操作循环逻辑
        teleop_start_time = time.perf_counter()
        teleop_steps = 0
        observation = initial_observation
        
        # 获取初始遥操作动作
        initial_teleop_action = self.teleop.get_action()
        
        while time.perf_counter() - teleop_start_time < self.config.training.teleop_append.duration_s:
            start_loop_t = time.perf_counter()
            
            # 获取当前遥操作动作
            current_teleop_action = self.teleop.get_action()
            
            # 应用平滑过渡
            if teleop_steps < smooth_steps and last_policy_action is not None:
                # 计算平滑权重：从1.0（完全策略动作）过渡到0.0（完全遥操作动作）
                smooth_weight = 1.0 - (teleop_steps / smooth_steps)
                smoothed_action = self._smooth_action_transition(
                    last_policy_action, current_teleop_action, smooth_weight
                )
                action_to_send = smoothed_action
            else:
                action_to_send = current_teleop_action
            
            # 发送动作到机器人
            sent_teleop_action = self.robot.send_action(action_to_send)
            next_observation = self.robot.get_observation()
            
            # 组织遥操作数据（与策略数据格式一致）
            task_description = "Pick the black ring (teleop)"
            teleop_frame_data = self._organize_teleop_data(
                observation=observation,
                action=sent_teleop_action,
                reward=initial_reward,  # 使用相同的reward
                next_observation=next_observation,
                done=False,
                episode=episode,
                step=initial_step + teleop_steps,  # 继续步数计数
                task_description=task_description
            )
            
            # 记录遥操作数据到同一个episode
            self.online_dataset.add_frame(teleop_frame_data, task=task_description)
            
            # 存储到在线缓冲区
            self._store_teleop_to_buffer(
                observation=observation,
                action=sent_teleop_action,
                reward=initial_reward,
                next_observation=next_observation,
                done=False
            )
            
            # 可视化遥操作数据
            if self.config.visualization.enable:
                log_rerun_data(observation, sent_teleop_action)
            
            observation = next_observation
            teleop_steps += 1
            
            # 控制循环频率
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / self.config.training.fps - dt_s)
        
        print(f"✅ 遥操作数据收集完成，追加了 {teleop_steps} 帧")
        return teleop_steps
    
    def _organize_teleop_data(self, observation: Dict, action: Dict, reward: float, 
                             next_observation: Dict, done: bool, episode: int, 
                             step: int, task_description: str) -> Dict:
        """组织遥操作数据格式"""
        
        # 构建标准格式的frame数据
        observation_frame = build_dataset_frame(self.online_dataset.features, observation, prefix="observation")
        action_frame = build_dataset_frame(self.online_dataset.features, action, prefix="action")
        
        # 直接添加非前缀字段
        frame = {
            **observation_frame,
            **action_frame,
            "next.reward": np.array([reward], dtype=np.float32),
            "next.done": np.array([done], dtype=bool)
        }
        
        return frame
    
    def _store_teleop_to_buffer(self, observation: Dict, action: Dict, reward: float, 
                               next_observation: Dict, done: bool):
        """存储遥操作数据到缓冲区"""
        
        observation_frame = build_dataset_frame(self.obs_features, observation, prefix="observation")
        next_observation_frame = build_dataset_frame(self.obs_features, next_observation, prefix="observation")
        
        complete_teleop_frame = {
            **observation_frame,
            "action": np.array([action[key] for key in self.robot.action_features], dtype=np.float32),
            "next.reward": np.array([float(reward)], dtype=np.float32),
            "next.done": np.array([done], dtype=bool)
        }
        
        self.hybrid_buffer.add_lerobot_frame(
            frame=complete_teleop_frame,
            next_frame=next_observation_frame,
            is_online=True
        )
    
    def _smooth_action_transition(self, policy_action: Dict, teleop_action: Dict, weight: float) -> Dict:
        """
        在策略动作和遥操作动作之间进行平滑过渡
        
        Args:
            policy_action: 策略动作字典
            teleop_action: 遥操作动作字典
            weight: 平滑权重 (1.0=完全策略动作, 0.0=完全遥操作动作)
        
        Returns:
            smoothed_action: 平滑后的动作字典
        """
        smoothed_action = {}
        
        for key in policy_action.keys():
            if key in teleop_action:
                # 线性插值：weight * policy + (1-weight) * teleop
                policy_val = policy_action[key]
                teleop_val = teleop_action[key]
                smoothed_val = weight * policy_val + (1.0 - weight) * teleop_val
                smoothed_action[key] = smoothed_val
            else:
                # 如果遥操作动作中没有对应的键，使用策略动作
                smoothed_action[key] = policy_action[key]
        
        return smoothed_action


class EpisodeManager:
    """Episode管理器 - 负责episode的完整生命周期"""
    
    def __init__(self, config: Dict[str, Any], online_dataset: LeRobotDataset, 
                 data_collector: OnlineDataCollector, teleop_appender: TeleopDataAppender):
        self.config = config
        self.online_dataset = online_dataset
        self.data_collector = data_collector
        self.teleop_appender = teleop_appender
    
    def run_episode(self, episode: int) -> Dict[str, Any]:
        """
        运行完整的episode（策略数据 + 遥操作数据）
        
        Returns:
            episode_stats: episode统计信息
        """
        print(f"\n🎬 Episode {episode + 1}/{self.config.training.total_episodes}")
        
        task_description = "Pick the black ring"
        
        # 1. 收集策略数据
        episode_reward, episode_steps, final_observation, last_policy_action = self.data_collector.collect_policy_data(
            episode=episode,
            task_description=task_description
        )
        
        # 2. 追加遥操作数据（如果启用）
        if self.teleop_appender.teleop is not None:
            teleop_steps = self.teleop_appender.append_teleop_data(
                episode=episode,
                initial_observation=final_observation,
                initial_reward=episode_reward,
                initial_step=episode_steps,
                last_policy_action=last_policy_action
            )
        
        import psutil
        import gc
        mem_before = psutil.Process().memory_info().rss / 1024 / 1024
        print(f"🔍 Episode {episode} 保存前: 内存{mem_before:.1f}MB")
        # 3. 保存episode数据
        self.online_dataset.save_episode()
        import datasets
        for key in self.online_dataset.hf_dataset.features:
            if isinstance(self.online_dataset.hf_dataset.features[key], datasets.Image):
                del self.online_dataset.hf_dataset.data[key]
        
        # 🔍 内存监控：episode数据采集完成后
        mem_after_collect = psutil.Process().memory_info().rss / 1024 / 1024
        mem_increase_collect = mem_after_collect - mem_before
        print(f"🔍 Episode {episode} 数据保存后: 内存{mem_after_collect:.1f}MB(+{mem_increase_collect:.1f}MB)")
        
        # 4. 返回episode统计信息
        return {
            'episode': episode,
            'episode_reward': episode_reward,
            'policy_steps': episode_steps,
            'teleop_steps': teleop_steps,
            'total_steps': episode_steps + teleop_steps,
            'final_observation': final_observation
        }
        
def create_data_collection_components(config: Dict[str, Any], robot, policy, teleop, 
                                    device, online_dataset, hybrid_buffer) -> Tuple[OnlineDataCollector, TeleopDataAppender, EpisodeManager]:
    """创建数据采集相关组件"""
    
    # 创建数据采集器
    data_collector = OnlineDataCollector(config, robot, policy, device, online_dataset, hybrid_buffer)
    
    # 创建遥操作追加器
    teleop_appender = TeleopDataAppender(config, robot, teleop, online_dataset, hybrid_buffer)
    
    # 创建episode管理器
    episode_manager = EpisodeManager(config, online_dataset, data_collector, teleop_appender)
    
    return data_collector, teleop_appender, episode_manager