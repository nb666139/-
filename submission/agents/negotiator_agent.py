"""
GridSynergy — 博弈Agent (NegotiatorAgent)
基于MATD3多智能体深度强化学习协调多VPP经济调度博弈。

支持两种模式：
1. PyTorch MATD3：完整训练循环+经验回放+模型持久化
2. Demo模式：边际成本贪心策略（无需GPU和PyTorch）
"""

from __future__ import annotations

import copy
import math
import os
import pickle
from collections import deque
from typing import Any, Optional

import numpy as np

# 条件导入PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    _TORCH_AVAILABLE: bool = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None      # type: ignore[assignment]
    F = None       # type: ignore[assignment]
    optim = None   # type: ignore[assignment]

from config import get_config


# ============================================================================
# Actor-Critic 网络定义
# ============================================================================

if _TORCH_AVAILABLE:

    class Actor(nn.Module):  # type: ignore[misc]
        """Actor网络：输出VPP的连续动作 [-1, 1]^action_dim"""

        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Tanh(),
            )

        def forward(self, state: torch.Tensor) -> torch.Tensor:
            return self.net(state)


    class Critic(nn.Module):  # type: ignore[misc]
        """中心化Critic网络：Q(s, a_1, ..., a_M) → [Q1, Q2]"""

        def __init__(self, state_dim: int, total_action_dim: int, hidden_dim: int = 256) -> None:
            super().__init__()
            self.q1 = nn.Sequential(
                nn.Linear(state_dim + total_action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.q2 = nn.Sequential(
                nn.Linear(state_dim + total_action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, state: torch.Tensor, all_actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            sa = torch.cat([state, all_actions], dim=-1)
            return self.q1(sa), self.q2(sa)


# ============================================================================
# 经验回放缓冲区
# ============================================================================

if _TORCH_AVAILABLE:

    class ReplayBuffer:
        """经验回放缓冲区（FIFO）。"""

        def __init__(self, capacity: int, state_dim: int, action_dim: int, n_agents: int) -> None:
            self.capacity = capacity
            self.states = np.zeros((capacity, state_dim), dtype=np.float32)
            self.actions = np.zeros((capacity, n_agents, action_dim), dtype=np.float32)
            self.rewards = np.zeros((capacity, n_agents), dtype=np.float32)
            self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
            self.dones = np.zeros(capacity, dtype=np.float32)
            self._ptr = 0
            self._size = 0

        def push(
            self,
            state: np.ndarray,
            actions: np.ndarray,
            rewards: np.ndarray,
            next_state: np.ndarray,
            done: float,
        ) -> None:
            self.states[self._ptr] = state
            self.actions[self._ptr] = actions
            self.rewards[self._ptr] = rewards
            self.next_states[self._ptr] = next_state
            self.dones[self._ptr] = done
            self._ptr = (self._ptr + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

        def sample(self, batch_size: int) -> tuple:
            indices = np.random.choice(self._size, batch_size, replace=False)
            states = torch.FloatTensor(self.states[indices])
            actions = torch.FloatTensor(self.actions[indices])
            rewards = torch.FloatTensor(self.rewards[indices])
            next_states = torch.FloatTensor(self.next_states[indices])
            dones = torch.FloatTensor(self.dones[indices])
            return states, actions, rewards, next_states, dones

        def __len__(self) -> int:
            return self._size


# ============================================================================
# VPP Agent（单个虚拟电厂）
# ============================================================================

class VPPAgent:
    """单个VPP的经济调度智能体。"""

    def __init__(
        self,
        vpp_id: str,
        state_dim: int,
        action_dim: int,
        capacity_mw: float = 100.0,
        marginal_cost: float = 40.0,
    ) -> None:
        self.vpp_id = vpp_id
        self.capacity_mw = capacity_mw
        self.marginal_cost = marginal_cost
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.cumulative_reward = 0.0

        # PyTorch网络
        self._actor: Optional[Actor] = None
        self._actor_target: Optional[Actor] = None
        self._actor_optimizer: Optional[optim.Adam] = None

        if _TORCH_AVAILABLE:
            cfg = get_config()
            self._actor = Actor(state_dim, action_dim)
            self._actor_target = Actor(state_dim, action_dim)
            self._actor_target.load_state_dict(self._actor.state_dict())
            self._actor_optimizer = optim.Adam(self._actor.parameters(), lr=cfg.agent.actor_lr)

            self.gamma = cfg.agent.gamma
            self.tau = cfg.agent.tau
            self.policy_noise = 0.2
            self.noise_clip = 0.5

    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """选择动作。"""
        if self._actor is not None:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            self._actor.eval()
            with torch.no_grad():
                action = self._actor(state_tensor).cpu().numpy().flatten()
            self._actor.train()
            if explore:
                noise = np.random.normal(0, self.policy_noise, size=action.shape)
                action = np.clip(action + noise, -1.0, 1.0)
            return action

        # Demo模式：基于边际成本的规则化策略
        cost_factor = min(1.0, 30.0 / max(self.marginal_cost, 1.0))
        action = np.array([
            cost_factor * 0.8 * 2 - 1,   # 出力比例 → [-1, 1]
            (1 - cost_factor) * 0.5 * 2 - 1,  # 报价系数
            0.3 * 2 - 1,                 # 备用比例
        ], dtype=np.float32)
        if explore:
            noise = np.random.normal(0, 0.05, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)
        return action

    def get_actor_state_dict(self) -> Optional[dict]:
        if self._actor is not None:
            return self._actor.state_dict()
        return None

    def load_actor_state_dict(self, state_dict: dict) -> None:
        if self._actor is not None:
            self._actor.load_state_dict(state_dict)
            self._actor_target.load_state_dict(state_dict)


# ============================================================================
# MATD3 训练器
# ============================================================================

class MATD3Trainer:
    """MATD3多智能体训练器。

    中心化Critic + 去中心化Actor，支持：
    - 完整训练循环 (train_episode / train)
    - 经验回放 (ReplayBuffer)
    - 模型检查点保存/加载
    - 奖励曲线记录
    """

    def __init__(
        self,
        env,
        n_vpp: int = 4,
        state_dim: int = 12,
        action_dim: int = 3,
    ) -> None:
        """
        参数:
            env: PowerGridEnv环境
            n_vpp: VPP数量
            state_dim: 状态维度
            action_dim: 动作维度
        """
        self._cfg = get_config()
        self.env = env
        self.n_vpp = n_vpp
        self.state_dim = state_dim
        self.action_dim = action_dim

        if not _TORCH_AVAILABLE:
            raise RuntimeError("MATD3Trainer需要安装PyTorch: pip install torch>=2.0.0")

        self.device = torch.device(self._cfg.training.device if torch.cuda.is_available() else "cpu")

        # 创建VPP Agent
        self.agents: list[VPPAgent] = []
        vpp_configs = [
            {"id": "VPP_1", "capacity": 80.0, "cost": 5.0},
            {"id": "VPP_2", "capacity": 60.0, "cost": 6.0},
            {"id": "VPP_3", "capacity": 50.0, "cost": 3.0},
            {"id": "VPP_4", "capacity": 100.0, "cost": 35.0},
        ]
        for i in range(min(n_vpp, len(vpp_configs))):
            cfg = vpp_configs[i]
            self.agents.append(VPPAgent(cfg["id"], state_dim, action_dim, cfg["capacity"], cfg["cost"]))

        # 中心化Critic（每个Agent一个Critic对）
        total_action_dim = n_vpp * action_dim
        self.critics: list[Critic] = [
            Critic(state_dim, total_action_dim) for _ in range(n_vpp)
        ]
        self.critic_targets: list[Critic] = [
            Critic(state_dim, total_action_dim) for _ in range(n_vpp)
        ]
        for i in range(n_vpp):
            self.critic_targets[i].load_state_dict(self.critics[i].state_dict())

        self.critic_optimizers = [
            optim.Adam(c.parameters(), lr=self._cfg.agent.critic_lr)
            for c in self.critics
        ]

        # 经验回放
        self.replay_buffer = ReplayBuffer(
            self._cfg.training.replay_buffer_capacity,
            state_dim, action_dim, n_vpp,
        )

        # 训练配置
        self.batch_size = self._cfg.training.batch_size
        self.gamma = self._cfg.agent.gamma
        self.tau = self._cfg.agent.tau
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_delay = 2
        self._update_count = 0

        # 训练历史
        self.rewards_history: list[float] = []

    def train_episode(self, max_steps: int = 100) -> float:
        """运行一个训练episode。

        返回:
            该episode的总奖励（所有Agent之和）
        """
        obs = self.env.reset()
        episode_reward = 0.0

        for step in range(max_steps):
            # 1. 各VPP独立选择动作（去中心化）
            actions = []
            for i, agent in enumerate(self.agents):
                a = agent.select_action(obs[i] if isinstance(obs, (list, np.ndarray)) and len(np.array(obs).shape) > 1 else obs, explore=True)
                actions.append(a)
            actions_arr = np.array(actions)

            # 2. 环境执行
            next_obs, rewards, done, info = self.env.step(actions_arr)
            done_val = 1.0 if done else 0.0

            # 3. 存入回放缓冲
            if isinstance(obs, (list, np.ndarray)):
                obs_arr = np.array(obs)
                if len(obs_arr.shape) == 2:
                    state_vec = obs_arr.flatten()[:self.state_dim]
                else:
                    state_vec = obs_arr.flatten()[:self.state_dim]
            else:
                state_vec = np.array([obs]).flatten()[:self.state_dim]

            if isinstance(next_obs, (list, np.ndarray)):
                next_arr = np.array(next_obs)
                if len(next_arr.shape) == 2:
                    next_vec = next_arr.flatten()[:self.state_dim]
                else:
                    next_vec = next_arr.flatten()[:self.state_dim]
            else:
                next_vec = np.array([next_obs]).flatten()[:self.state_dim]

            rew_arr = np.array(rewards).flatten()[:self.n_vpp]
            state_vec = np.pad(state_vec, (0, max(0, self.state_dim - len(state_vec))))[:self.state_dim]
            next_vec = np.pad(next_vec, (0, max(0, self.state_dim - len(next_vec))))[:self.state_dim]
            actions_padded = np.pad(actions_arr, ((0, max(0, self.n_vpp - actions_arr.shape[0])), (0, 0)))[:self.n_vpp, :self.action_dim]
            rew_padded = np.pad(rew_arr, (0, max(0, self.n_vpp - len(rew_arr))))[:self.n_vpp]

            self.replay_buffer.push(state_vec, actions_padded, rew_padded, next_vec, done_val)

            # 4. 从缓冲采样训练
            if len(self.replay_buffer) > self.batch_size:
                self._update()

            obs = next_obs
            episode_reward += np.sum(rew_padded)
            if done:
                break

        self.rewards_history.append(episode_reward)
        return episode_reward

    def train(self, n_episodes: int = 5000) -> list[float]:
        """完整训练循环。

        参数:
            n_episodes: 训练episode数

        返回:
            奖励历史列表
        """
        print(f"[MATD3] 开始训练 {n_episodes} episodes, 设备={self.device}")
        for ep in range(n_episodes):
            ep_reward = self.train_episode()
            if ep % 100 == 0:
                avg = np.mean(self.rewards_history[-100:]) if self.rewards_history else 0
                print(f"  Episode {ep}: Reward={ep_reward:.1f}, Avg100={avg:.1f}")
            if ep % self._cfg.training.checkpoint_interval == 0 and ep > 0:
                self.save_checkpoint(f"checkpoints/ep_{ep}.pt")
        print(f"[MATD3] 训练完成。最终Avg100={np.mean(self.rewards_history[-100:]):.1f}")
        return self.rewards_history

    def _update(self) -> None:
        """一次梯度更新（对所有Agent）。"""
        self._update_count += 1
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = batch

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # 展平所有Agent的动作为 (batch, total_action_dim)
        all_actions = actions.view(self.batch_size, -1)

        # 对每个Agent更新Critic
        for i in range(self.n_vpp):
            agent = self.agents[i]
            if agent._actor is None:
                continue

            # 目标动作
            with torch.no_grad():
                next_actions = []
                for j, ag in enumerate(self.agents):
                    if ag._actor_target is not None:
                        next_a = ag._actor_target(next_states)
                        noise = (torch.randn_like(next_a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                        next_a = (next_a + noise).clamp(-1, 1)
                        next_actions.append(next_a)
                    else:
                        next_actions.append(torch.zeros(self.batch_size, self.action_dim))
                next_all_actions = torch.cat(next_actions, dim=-1)

                target_q1, target_q2 = self.critic_targets[i](next_states, next_all_actions)
                target_q = torch.min(target_q1, target_q2)
                target_q = rewards[:, i].unsqueeze(1) + self.gamma * (1 - dones.unsqueeze(1)) * target_q

            # Critic更新
            current_q1, current_q2 = self.critics[i](states, all_actions)
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()

            # 延迟Actor更新
            if self._update_count % self.policy_delay == 0:
                agent_actions = []
                for j, ag in enumerate(self.agents):
                    if ag._actor is not None:
                        agent_actions.append(ag._actor(states))
                    else:
                        agent_actions.append(torch.zeros(self.batch_size, self.action_dim))
                actor_all_actions = torch.cat(agent_actions, dim=-1)
                actor_all_actions_var = actor_all_actions.clone()
                # 用当前Actor的action替换以计算梯度
                if agent._actor is not None:
                    detached = [a.detach() if j != i else a for j, a in enumerate(agent_actions)]
                    actor_input = torch.cat(detached, dim=-1)
                    q1, _ = self.critics[i](states, actor_input)
                    actor_loss = -q1.mean()

                    agent._actor_optimizer.zero_grad()
                    actor_loss.backward()
                    agent._actor_optimizer.step()

            # 软更新目标网络
            self._soft_update(self.critics[i], self.critic_targets[i])
            if self._update_count % self.policy_delay == 0 and agent._actor is not None:
                self._soft_update(agent._actor, agent._actor_target)

    def _soft_update(self, source, target) -> None:
        """软更新目标网络: θ' ← τ·θ + (1-τ)·θ'"""
        with torch.no_grad():
            for sp, tp in zip(source.parameters(), target.parameters()):
                tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def save_checkpoint(self, path: str) -> None:
        """保存训练检查点。"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "agents": [a.get_actor_state_dict() for a in self.agents],
            "critics": [c.state_dict() for c in self.critics],
            "rewards_history": self.rewards_history,
            "update_count": self._update_count,
        }
        torch.save(data, path)
        print(f"[MATD3] 检查点保存: {path}")

    def load_checkpoint(self, path: str) -> None:
        """加载训练检查点。"""
        data = torch.load(path, map_location=self.device)
        for i, agent in enumerate(self.agents):
            if data["agents"][i] is not None and agent._actor is not None:
                agent._actor.load_state_dict(data["agents"][i])
                agent._actor_target.load_state_dict(data["agents"][i])
        for i, critic in enumerate(self.critics):
            critic.load_state_dict(data["critics"][i])
            self.critic_targets[i].load_state_dict(data["critics"][i])
        self.rewards_history = data.get("rewards_history", [])
        self._update_count = data.get("update_count", 0)
        print(f"[MATD3] 检查点加载: {path}")


# ============================================================================
# NegotiatorAgent（多VPP博弈协调器）
# ============================================================================

class NegotiatorAgent:
    """博弈Agent：协调多个VPP之间的经济调度博弈。"""

    def __init__(self, num_vpps: int = 5) -> None:
        self._config = get_config()
        self.num_vpps = num_vpps
        self.state_dim = 12
        self.action_dim = 3

        self.vpp_agents: dict[str, VPPAgent] = {}
        vpp_configs = [
            {"id": "VPP_Wind_1",    "capacity": 80.0,  "cost": 5.0},
            {"id": "VPP_Wind_2",    "capacity": 60.0,  "cost": 6.0},
            {"id": "VPP_Solar_1",   "capacity": 50.0,  "cost": 3.0},
            {"id": "VPP_Thermal_1", "capacity": 100.0, "cost": 35.0},
            {"id": "VPP_Thermal_2", "capacity": 80.0,  "cost": 40.0},
        ]
        for cfg in vpp_configs[:num_vpps]:
            self.vpp_agents[cfg["id"]] = VPPAgent(
                cfg["id"], self.state_dim, self.action_dim,
                cfg["capacity"], cfg["cost"],
            )

        self._prev_payoffs: Optional[dict[str, float]] = None
        self._round = 0

    def negotiate(self, multi_vpp_state: dict[str, Any], planner_plan: dict[str, Any] = None) -> dict[str, Any]:
        """执行多VPP博弈协商。

        参数:
            multi_vpp_state: 市场状态（global_load, market_price 等）
            planner_plan: PlannerAgent 生成的调度方案（含 unit_commitment），用于初始化VPP出力
        """
        self._round += 1

        global_load = float(multi_vpp_state.get("global_load", 300.0))
        market_price = float(multi_vpp_state.get("market_price", 50.0))
        renewable_forecast = float(multi_vpp_state.get("renewable_forecast", 90.0))
        vpp_states = multi_vpp_state.get("vpp_states", {})

        global_state = self._build_global_state(global_load, market_price, renewable_forecast, vpp_states)
        
        # ---- 注入 Planner 方案：将 unit_commitment 映射到 VPP ----
        total_planned_gen = 0.0
        if planner_plan:
            uc = planner_plan.get("unit_commitment", {})
            # 将规划的6台机组出力分配给4个 VPP:
            # VPP_Thermal_1 → G1+G2, VPP_Thermal_2 → G3+G4, VPP_Wind → G5, VPP_Solar → G6
            gen_to_vpp = {
                "G1": "VPP_Thermal_1", "G2": "VPP_Thermal_1",
                "G3": "VPP_Thermal_2", "G4": "VPP_Thermal_2",
                "G5": "VPP_Wind_1", "G6": "VPP_Solar_1",
            }
            vpp_planned = {}
            for gen_name, info in uc.items():
                output = info.get("output_mw", 0) if isinstance(info, dict) else float(info)
                vpp_name = gen_to_vpp.get(gen_name, f"VPP_{gen_name}")
                vpp_planned[vpp_name] = vpp_planned.get(vpp_name, 0.0) + float(output)
            total_planned_gen = sum(vpp_planned.values())
            if total_planned_gen > 0:
                multi_vpp_state["vpp_planned"] = vpp_planned
                multi_vpp_state["total_planned"] = total_planned_gen
        
        payoff_matrix: dict[str, float] = {}
        actions: dict[str, np.ndarray] = {}

        for vpp_id, agent in self.vpp_agents.items():
            vpp_specific_state = self._build_vpp_state(global_state, vpp_id, vpp_states)
            actions[vpp_id] = agent.select_action(vpp_specific_state, explore=(self._round < 30))

        for vpp_id, action in actions.items():
            agent = self.vpp_agents[vpp_id]
            output_ratio = float((action[0] + 1.0) / 2.0)
            bid_factor = float((action[1] + 1.0) / 2.0)
            actual_output = output_ratio * agent.capacity_mw
            revenue = actual_output * market_price
            cost = actual_output * agent.marginal_cost
            profit = revenue - cost
            payoff_matrix[vpp_id] = round(profit, 2)

        equilibrium_reached = self._check_nash_convergence(payoff_matrix)

        dispatch_schedule = {}
        for vpp_id, action in actions.items():
            agent = self.vpp_agents[vpp_id]
            output_ratio = float((action[0] + 1.0) / 2.0)
            bid_factor = float((action[1] + 1.0) / 2.0)
            reserve_ratio = float((action[2] + 1.0) / 2.0)
            dispatch_schedule[vpp_id] = {
                "output_mw": round(output_ratio * agent.capacity_mw, 2),
                "bid_price": round(agent.marginal_cost * (0.8 + 0.4 * bid_factor), 2),
                "reserve_mw": round(reserve_ratio * agent.capacity_mw * 0.15, 2),
                "profit": payoff_matrix.get(vpp_id, 0.0),
            }

        self._prev_payoffs = payoff_matrix

        return {
            "round": self._round,
            "equilibrium_reached": equilibrium_reached,
            "payoff_matrix": payoff_matrix,
            "dispatch_schedule": dispatch_schedule,
            "total_profit": round(sum(payoff_matrix.values()), 2),
            "metadata": {
                "global_load": global_load,
                "market_price": market_price,
                "renewable_forecast": renewable_forecast,
            },
        }

    def _build_global_state(self, global_load: float, market_price: float,
                            renewable_forecast: float, vpp_states: dict) -> np.ndarray:
        state = [
            global_load / 500.0, market_price / 100.0,
            renewable_forecast / 200.0, len(self.vpp_agents) / 10.0,
        ]
        for vpp_id in self.vpp_agents:
            vs = vpp_states.get(vpp_id, {})
            state.append(vs.get("current_output", 0.0) / 200.0)
        while len(state) < self.state_dim:
            state.append(0.0)
        return np.array(state[:self.state_dim], dtype=np.float32)

    def _build_vpp_state(self, global_state: np.ndarray, vpp_id: str,
                         vpp_states: dict) -> np.ndarray:
        vs = vpp_states.get(vpp_id, {})
        vpp_features = [
            vs.get("current_output", 0.0) / 200.0,
            vs.get("remaining_capacity", 100.0) / 200.0,
            vs.get("renewable_available", 0.0) / 100.0,
        ]
        full_state = np.concatenate([global_state, np.array(vpp_features, dtype=np.float32)])
        if len(full_state) > self.state_dim:
            full_state = full_state[:self.state_dim]
        elif len(full_state) < self.state_dim:
            full_state = np.pad(full_state, (0, self.state_dim - len(full_state)))
        return full_state.astype(np.float32)

    def _check_nash_convergence(self, current_payoffs: dict[str, float]) -> bool:
        if self._prev_payoffs is None:
            return False
        total_change = sum(
            abs(current_payoffs[k] - self._prev_payoffs.get(k, 0.0)) / max(abs(self._prev_payoffs.get(k, 0.001)), 0.001)
            for k in current_payoffs
        )
        avg_change = total_change / max(len(current_payoffs), 1)
        return avg_change < self._config.agent.nash_convergence_epsilon

    def nash_equilibrium(self, payoff_matrix: np.ndarray) -> np.ndarray:
        """计算纳什均衡（简化离散策略迭代）。"""
        num_players = payoff_matrix.shape[0]
        best_response = np.zeros(num_players, dtype=np.int32)
        for _ in range(self._config.agent.max_negotiation_rounds):
            prev_best = best_response.copy()
            for i in range(num_players):
                other_indices = tuple(
                    best_response[j] if j != i else slice(None)
                    for j in range(num_players)
                )
                payoffs_i = payoff_matrix[other_indices]
                if isinstance(payoffs_i, np.ndarray) and payoffs_i.ndim > 0:
                    best_response[i] = int(np.argmax(payoffs_i))
            if np.array_equal(prev_best, best_response):
                break
        strategy = np.zeros((num_players, payoff_matrix.shape[1]))
        for i in range(num_players):
            strategy[i, best_response[i]] = 1.0
        return strategy
