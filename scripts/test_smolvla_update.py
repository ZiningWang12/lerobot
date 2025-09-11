#!/usr/bin/env python3
"""
测试SmolVLA参数是否在SAC训练中被更新
"""

import torch
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lerobot.policies.smolvla_sac.modeling_smolvla_sac import SmolVLASACPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def test_smolvla_parameter_update():
    """测试SmolVLA参数是否在训练中被更新"""
    
    print("🔍 测试SmolVLA参数更新...")
    
    # 加载数据集
    dataset = LeRobotDataset(
        repo_id="converted_teleop_ring_labeled",
        root="outputs/converted_teleop_ring_labeled",
        download_videos=False
    )
    
    # 创建策略
    policy = SmolVLASACPolicy.from_pretrained(
        pretrained_name_or_path="outputs/train/2025-08-04/11-43-08_smolvla/checkpoints/008000/pretrained_model",
        dataset_stats=dataset.meta.stats
    )
    
    # 手动解冻SmolVLA参数
    print("🔓 解冻SmolVLA参数...")
    for param in policy.smolvla.parameters():
        param.requires_grad = True
    
    # 检查解冻后的参数数量
    trainable_params = sum(p.numel() for p in policy.smolvla.parameters() if p.requires_grad)
    print(f"解冻后SmolVLA可训练参数数量: {trainable_params}")
    
    # 移动到GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = policy.to(device)
    
    # 记录初始参数
    initial_params = {}
    for name, param in policy.smolvla.named_parameters():
        if param.requires_grad:
            initial_params[name] = param.data.clone()
    
    print(f"SmolVLA可训练参数数量: {len(initial_params)}")
    
    # 创建优化器
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.0003)
    
    # 模拟训练步骤
    policy.train()
    
    # 创建一些虚拟数据（确保在GPU上）
    batch_size = 1  # 减小batch size避免内存不足
    dummy_observations = {
        'observation.state': torch.randn(batch_size, 6).to(device),
        'observation.images.handeye': torch.randn(batch_size, 3, 600, 800).to(device),
        'observation.images.global': torch.randn(batch_size, 3, 600, 800).to(device),
    }
    dummy_actions = torch.randn(batch_size, 6).to(device)
    dummy_rewards = torch.randn(batch_size).to(device)
    dummy_done = torch.zeros(batch_size, dtype=torch.bool).to(device)
    
    # 执行几个训练步骤
    for step in range(5):
        # 检查SmolVLA是否在训练模式
        print(f"步骤 {step+1}: SmolVLA训练模式: {policy.smolvla.training}")
        
        # 计算损失
        loss_critic = policy.compute_loss_critic(
            observations=dummy_observations,
            actions=dummy_actions,
            rewards=dummy_rewards,
            next_observations=dummy_observations,
            done=dummy_done
        )
        
        loss_actor = policy.compute_loss_actor(
            observations=dummy_observations
        )
        
        loss_temperature = policy.compute_loss_temperature(
            observations=dummy_observations
        )
        
        total_loss = loss_critic + loss_actor + loss_temperature
        
        # 检查梯度
        smolvla_grads = []
        for name, param in policy.smolvla.named_parameters():
            if param.requires_grad and param.grad is not None:
                smolvla_grads.append(torch.norm(param.grad).item())
        
        print(f"   SmolVLA有梯度的参数: {len(smolvla_grads)}")
        if smolvla_grads:
            print(f"   平均梯度范数: {sum(smolvla_grads)/len(smolvla_grads):.8f}")
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 检查参数变化
        param_changes = []
        for name, param in policy.smolvla.named_parameters():
            if param.requires_grad and name in initial_params:
                change = torch.norm(param.data - initial_params[name]).item()
                param_changes.append(change)
        
        if param_changes:
            avg_change = sum(param_changes) / len(param_changes)
            max_change = max(param_changes)
            print(f"步骤 {step+1}: 平均变化 {avg_change:.8f}, 最大变化 {max_change:.8f}")
        else:
            print(f"步骤 {step+1}: 无参数变化")
    
    # 最终检查
    final_changes = []
    for name, param in policy.smolvla.named_parameters():
        if param.requires_grad and name in initial_params:
            change = torch.norm(param.data - initial_params[name]).item()
            final_changes.append((name, change))
    
    if final_changes:
        final_changes.sort(key=lambda x: x[1], reverse=True)
        print(f"\n最终结果:")
        print(f"有变化的参数: {len([c for c in final_changes if c[1] > 1e-8])}/{len(final_changes)}")
        print(f"平均变化: {sum(c[1] for c in final_changes)/len(final_changes):.8f}")
        print(f"最大变化: {final_changes[0][1]:.8f} ({final_changes[0][0]})")
        
        if any(c[1] > 1e-8 for c in final_changes):
            print("✅ SmolVLA参数确实被更新了！")
        else:
            print("❌ SmolVLA参数没有被更新！")
    else:
        print("❌ 没有可训练的SmolVLA参数！")

if __name__ == "__main__":
    test_smolvla_parameter_update()