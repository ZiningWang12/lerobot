#!/usr/bin/env python

"""
可视化模型在数据集上的推理效果

这个脚本加载一个训练好的模型，在数据集上运行推理，
并使用Rerun可视化预测动作与真实动作的对比。

使用示例:
python visualize_policy_inference.py \
    --policy-path outputs/train/2025-08-04/11-43-08_smolvla/checkpoints/008000/pretrained_model \
    --dataset-repo-id wzn12/teleop_ring \
    --episode-index 0 \
    --device cuda
"""

import argparse
import gc
import logging
import time
from collections.abc import Iterator
from pathlib import Path

import torch
import torch.utils.data
import numpy as np
import rerun as rr
import tqdm
from typing import Optional

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device


class EpisodeSampler(torch.utils.data.Sampler):
    """Sampler for loading frames from a specific episode, from visualize_dataset.py"""
    
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self) -> Iterator:
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy array for visualization, from visualize_dataset.py"""
    assert chw_float32_torch.dtype == torch.float32
    assert chw_float32_torch.ndim == 3
    c, h, w = chw_float32_torch.shape
    assert c < h and c < w, f"expect channel first images, but instead {chw_float32_torch.shape}"
    # 先移到 CPU，然后转换为 numpy
    hwc_uint8_numpy = (chw_float32_torch.cpu() * 255).type(torch.uint8).permute(1, 2, 0).numpy()
    return hwc_uint8_numpy


def build_observation_from_frame(frame: dict, dataset: LeRobotDataset) -> dict:
    """Build observation dict from frame data (similar to build_dataset_frame in record.py)"""
    from lerobot.datasets.utils import build_dataset_frame
    
    # 构建观测数据字典，类似于 robot.get_observation() 的返回格式
    observation_data = {}
    for key, value in frame.items():
        if key.startswith('observation.'):
            # 移除 'observation.' 前缀，获取原始特征名
            feature_name = key[len('observation.'):]
            
            if isinstance(value, torch.Tensor):
                # 对于图像数据，需要转换为 [H, W, C] 格式
                if "image" in key:
                    # 从 [C, H, W] 转换为 [H, W, C] 
                    value_np = value.cpu().numpy()
                    value_np = np.transpose(value_np, (1, 2, 0))  # [C, H, W] -> [H, W, C]
                    # 从 [0,1] 转换为 [0,255]
                    value_np = (value_np * 255).astype(np.uint8)
                    observation_data[feature_name] = value_np
                else:
                    observation_data[feature_name] = value.cpu().numpy()
            else:
                observation_data[feature_name] = value
    
    # 使用 build_dataset_frame 构建符合策略期望的格式（与 record.py 相同）
    try:
        observation_frame = build_dataset_frame(dataset.features, observation_data, prefix="observation")
        return observation_frame
    except Exception as e:
        logging.warning(f"Failed to build observation frame: {e}")
        # 如果构建失败，直接返回处理过的观测数据
        result = {}
        for key, value in observation_data.items():
            result[f"observation.{key}"] = value
        return result


def calculate_action_error(real_action: np.ndarray, predicted_action: np.ndarray) -> dict:
    """Calculate error between real and predicted actions"""
    error = real_action - predicted_action
    mse = np.mean(error ** 2)
    mae = np.mean(np.abs(error))
    max_error = np.max(np.abs(error))
    
    return {
        'error': error,
        'mse': mse,
        'mae': mae,
        'max_error': max_error
    }


def calculate_training_loss(real_action: np.ndarray, predicted_action: np.ndarray) -> dict:
    """Calculate the same MSE loss as used in training (without calling policy.forward)"""
    try:
        # 转换为torch tensor
        real_action_tensor = torch.from_numpy(real_action).float()
        predicted_action_tensor = torch.from_numpy(predicted_action).float()
        
        # 计算MSE loss (和SmolVLA训练时相同)
        mse_loss = torch.nn.functional.mse_loss(predicted_action_tensor, real_action_tensor, reduction='mean')
        training_loss = mse_loss.item()
        
        # 也计算其他有用的loss统计
        l1_loss = torch.nn.functional.l1_loss(predicted_action_tensor, real_action_tensor, reduction='mean').item()
        
        return {
            'training_loss': training_loss,  # MSE loss (和训练时相同)
            'l1_loss': l1_loss,
            'loss_dict': {
                'mse_loss': training_loss,
                'l1_loss': l1_loss
            }
        }
        
    except Exception as e:
        logging.warning(f"Training loss calculation failed: {e}")
        return {
            'training_loss': float('nan'),
            'l1_loss': float('nan'),
            'loss_dict': {}
        }


def visualize_policy_inference(
    dataset: LeRobotDataset, 
    policy, 
    episode_index: int, 
    device: torch.device, 
    save_output: bool = False, 
    output_dir: Path | None = None,
    batch_size: int = 1,
    num_workers: int = 0,
    mode: str = "local",
    web_port: int = 9090,
    ws_port: int = 9087,
) -> Optional[Path]:
    """
    可视化策略推理效果
    
    Args:
        dataset: 数据集
        policy: 策略模型
        episode_index: 回放索引
        device: 计算设备
        save_output: 是否保存输出
        output_dir: 输出目录
        batch_size: 批处理大小
        num_workers: 数据加载器的工作进程数
        mode: 可视化模式 ("local" 或 "distant")
        web_port: Web端口
        ws_port: WebSocket端口
    """
    if save_output:
        assert output_dir is not None, (
            "Set an output directory where to write .rrd files with `--output-dir path/to/directory`."
        )

    repo_id = dataset.repo_id

    logging.info("Loading dataloader")
    episode_sampler = EpisodeSampler(dataset, episode_index)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        sampler=episode_sampler,
    )

    logging.info("Starting Rerun")

    if mode not in ["local", "distant"]:
        raise ValueError(mode)

    spawn_local_viewer = mode == "local" and not save_output
    rr.init(f"{repo_id}/episode_{episode_index}_policy_inference", spawn=spawn_local_viewer)

    # Manually call python garbage collector after `rr.init` to avoid hanging in a blocking flush
    # when iterating on a dataloader with `num_workers` > 0
    gc.collect()

    if mode == "distant":
        rr.serve(open_browser=False, web_port=web_port, ws_port=ws_port)

    logging.info("Logging to Rerun")

    # 用于累积误差统计
    episode_errors = []
    prediction_successes = []

    for batch in tqdm.tqdm(dataloader, total=len(dataloader), desc="可视化策略推理"):
        # iterate over the batch
        for i in range(len(batch["index"])):
            rr.set_time_sequence("frame_index", batch["frame_index"][i].item())
            rr.set_time_seconds("timestamp", batch["timestamp"][i].item())

            # 获取当前帧数据
            frame = {key: batch[key][i] for key in batch.keys()}
            
            # 构建观测数据（与 record.py 相同的方式）
            observation_frame = build_observation_from_frame(frame, dataset)
            
            # 使用 predict_action 进行推理（与 record.py 完全相同）
            try:
                predicted_action_tensor = predict_action(
                    observation_frame,
                    policy,
                    device,
                    policy.config.use_amp,
                    task=frame.get("task", ""),
                    robot_type="so101_follower"  # 根据你的机器人类型调整
                )
                predicted_action_np = predicted_action_tensor.cpu().numpy()
                prediction_success = True
                
            except Exception as e:
                logging.warning(f"Policy prediction failed: {e}")
                predicted_action_np = np.zeros_like(frame["action"].cpu().numpy())
                prediction_success = False

            # 显示图像信息（参考visualize_dataset.py）
            for key in dataset.meta.camera_keys:
                if key in frame:
                    rr.log(key, rr.Image(to_hwc_uint8_numpy(frame[key])))

            # 显示真实动作和预测动作（参考visualize_dataset.py）
            if "action" in frame:
                real_action = frame["action"].cpu().numpy()
                for dim_idx, val in enumerate(real_action):
                    rr.log(f"action/real_{dim_idx}", rr.Scalar(val.item()))
                for dim_idx, val in enumerate(predicted_action_np):
                    rr.log(f"action/predicted_{dim_idx}", rr.Scalar(val.item()))
                
                # 计算并显示动作误差
                action_error = calculate_action_error(real_action, predicted_action_np)
                episode_errors.append(action_error['mse'])
                prediction_successes.append(prediction_success)
                
                for dim_idx, val in enumerate(action_error['error']):
                    rr.log(f"action/error_{dim_idx}", rr.Scalar(val.item()))
                
                # 显示误差统计
                rr.log("action_error/mse", rr.Scalar(action_error['mse']))
                rr.log("action_error/mae", rr.Scalar(action_error['mae']))
                rr.log("action_error/max_error", rr.Scalar(action_error['max_error']))
                rr.log("action_error/prediction_success", rr.Scalar(1.0 if prediction_success else 0.0))
                
                # 计算训练时的loss（和train.py相同的MSE loss）
                training_loss_info = calculate_training_loss(real_action, predicted_action_np)
                training_loss = training_loss_info['training_loss']
                episode_training_losses.append(training_loss)
                
                # 显示训练loss
                rr.log("training_loss/loss", rr.Scalar(training_loss))
                rr.log("training_loss/l1_loss", rr.Scalar(training_loss_info['l1_loss']))
                
                # 显示loss_dict中的详细信息
                loss_dict = training_loss_info['loss_dict']
                for loss_key, loss_value in loss_dict.items():
                    if isinstance(loss_value, (int, float)):
                        rr.log(f"training_loss/{loss_key}", rr.Scalar(loss_value))

            # 显示状态信息（参考visualize_dataset.py）
            if "observation.state" in frame:
                for dim_idx, val in enumerate(frame["observation.state"]):
                    rr.log(f"state/{dim_idx}", rr.Scalar(val.item()))

            # 显示其他信息
            if "next.done" in frame:
                rr.log("next.done", rr.Scalar(frame["next.done"].item()))
            if "next.reward" in frame:
                rr.log("next.reward", rr.Scalar(frame["next.reward"].item()))
            if "next.success" in frame:
                rr.log("next.success", rr.Scalar(frame["next.success"].item()))

            # 显示任务信息
            if "task" in frame:
                rr.log("task", rr.TextDocument(frame["task"]))

    # 计算并显示 episode 总体统计
    if episode_errors:
        avg_mse = np.mean(episode_errors)
        success_rate = np.mean(prediction_successes)
        logging.info(f"Episode {episode_index} Average MSE: {avg_mse:.6f}")
        logging.info(f"Episode {episode_index} Success Rate: {success_rate:.2%}")
        logging.info(f"Total frames processed: {len(episode_errors)}")
        
        rr.log("episode_stats/average_mse", rr.Scalar(avg_mse))
        rr.log("episode_stats/success_rate", rr.Scalar(success_rate))
        rr.log("episode_stats/total_frames", rr.Scalar(len(episode_errors)))

    if mode == "local" and save_output:
        # save .rrd locally
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        repo_id_str = repo_id.replace("/", "_")
        rrd_path = output_dir / f"{repo_id_str}_episode_{episode_index}_policy_inference.rrd"
        rr.save(rrd_path)
        return rrd_path

    elif mode == "distant":
        # stop the process from exiting since it is serving the websocket connection
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Ctrl-C received. Exiting.")

    print("策略推理可视化完成！")
    return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="可视化策略推理效果")
    parser.add_argument("--policy-path", type=str, required=True, help="策略模型路径")
    parser.add_argument("--dataset-repo-id", type=str, required=True, help="数据集仓库ID")
    parser.add_argument("--episode-index", type=int, default=0, help="回放索引")
    parser.add_argument("--device", type=str, default="cuda", help="计算设备")
    parser.add_argument("--save", action="store_true", help="是否保存输出")
    parser.add_argument("--output-dir", type=str, default="visualization_outputs", help="输出目录")
    parser.add_argument("--batch-size", type=int, default=1, help="批处理大小")
    parser.add_argument("--num-workers", type=int, default=0, help="数据加载器的工作进程数")
    parser.add_argument("--mode", type=str, default="local", 
                       help="可视化模式 ('local' 或 'distant')")
    parser.add_argument("--web-port", type=int, default=9090, 
                       help="Web端口 (当使用 --mode distant 时)")
    parser.add_argument("--ws-port", type=int, default=9087, 
                       help="WebSocket端口 (当使用 --mode distant 时)")
    parser.add_argument("--root", type=Path, default=None, 
                       help="数据集本地存储根目录")
    parser.add_argument("--tolerance-s", type=float, default=1e-4, 
                       help="时间戳容差（秒）")
    
    args = parser.parse_args()
    
    # 设置设备
    device = get_safe_torch_device(args.device)
    
    # 加载数据集
    logging.info("Loading dataset")
    dataset = LeRobotDataset(
        args.dataset_repo_id, 
        root=args.root, 
        tolerance_s=args.tolerance_s
    )
    
    # 加载策略
    logging.info("Loading policy")
    # 先读取配置文件确定策略类型
    import json
    from lerobot.configs.types import NormalizationMode
    
    config_path = Path(args.policy_path) / "config.json"
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    policy_type = config_dict.pop("type", "smolvla")  # 移除 type 字段
    logging.info(f"Policy type: {policy_type}")
    
    # 修复归一化映射，将字符串转换为 NormalizationMode 枚举
    if "normalization_mapping" in config_dict:
        norm_mapping = config_dict["normalization_mapping"]
        for key, value in norm_mapping.items():
            if isinstance(value, str):
                norm_mapping[key] = NormalizationMode(value)
    
    # 创建策略配置
    from lerobot.policies.factory import make_policy_config
    policy_config = make_policy_config(policy_type, **config_dict)
    policy_config.pretrained_path = args.policy_path
    
    # 加载策略
    policy = make_policy(policy_config, ds_meta=dataset.meta)
    
    # 创建输出目录
    output_dir = None
    if args.save:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
    
    # 执行可视化（包含 loss 计算）
    visualize_policy_inference(
        dataset=dataset, 
        policy=policy, 
        episode_index=args.episode_index, 
        device=device, 
        save_output=args.save, 
        output_dir=output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mode=args.mode,
        web_port=args.web_port,
        ws_port=args.ws_port,
    )


if __name__ == "__main__":
    main()
    