# panda_base_env.py - Franka Panda 基础环境框架
"""
基础 PyBullet 环境，包含：
- 单个 Franka Panda 机械臂
- 基础状态/动作空间定义
- 简单的到达目标任务
- 视频录制功能
- 模块化设计，方便扩展

用途：作为项目的起点，后续可以：
1. 修改 URDF（添加滑轨、剑身）
2. 添加第二个机械臂（对手）
3. 自定义奖励函数
4. 集成 PPO 训练
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time
import os
from typing import Optional, Tuple, Dict, Any


class PandaBaseEnv(gym.Env):
    """
    Franka Panda 基础环境
    
    任务：控制末端执行器到达目标位置
    （后续可修改为击剑任务）
    """
    
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 30
    }
    
    def __init__(
        self,
        render_mode: Optional[str] = 'human',
        max_steps: int = 200,
        control_mode: str = 'position',  # 'position' 或 'velocity'
    ):
        super().__init__()
        
        # 基础参数
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.control_mode = control_mode
        self.current_step = 0
        
        # PyBullet 相关
        self.client = None
        self.robot_id = None
        self.target_id = None
        
        # 机器人配置
        self.num_joints = 7  # Panda 有 7 个关节
        self.ee_link_index = 11  # 末端执行器链接索引
        
        # 定义观测空间
        # [关节位置(7), 关节速度(7), 末端位置(3), 目标位置(3), 位置误差(3)]
        obs_dim = 7 + 7 + 3 + 3 + 3  # 总共 23 维
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # 定义动作空间
        # 关节位置增量 (7维，归一化到 [-1, 1])
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32
        )
        
        # 动作缩放参数
        self.action_scale = 0.05  # 每步最大移动 0.05 弧度
        
        # 目标位置范围
        self.target_range = {
            'x': [0.3, 0.7],
            'y': [-0.3, 0.3],
            'z': [0.2, 0.6]
        }
        
        # 初始化环境
        self._setup_pybullet()
        
    def _setup_pybullet(self):
        """初始化 PyBullet 仿真"""
        # 连接 PyBullet
        if self.render_mode == 'human':
            self.client = p.connect(p.GUI)
            # 优化 GUI 显示
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0.5, 0, 0.3]
            )
        else:
            self.client = p.connect(p.DIRECT)
        
        # 设置物理参数
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1./240.)
        
        # 加载地面
        self.plane_id = p.loadURDF("plane.urdf")
        
        # 加载 Panda 机械臂
        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0, 0, 0],
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION
        )
        
        # 打印关节信息（调试用）
        print("\n=== Panda 关节信息 ===")
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            joint_name = info[1].decode('utf-8')
            joint_type = info[2]
            print(f"[{i}] {joint_name:20s} - Type: {joint_type}")
        
        # 创建目标可视化（红色半透明球体）
        self.target_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.05,
            rgbaColor=[1, 0, 0, 0.5]
        )
        self.target_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=self.target_visual,
            basePosition=[0.5, 0, 0.3]
        )
        
        print("✓ PyBullet 环境初始化完成\n")
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置环境到初始状态"""
        super().reset(seed=seed)
        
        self.current_step = 0
        
        # 重置机器人到初始姿态
        initial_joint_positions = [0, -0.3, 0, -2.0, 0, 1.6, 0.785]  # 典型初始姿态
        
        for i in range(self.num_joints):
            p.resetJointState(
                self.robot_id,
                i,
                targetValue=initial_joint_positions[i],
                targetVelocity=0
            )
        
        # 随机化目标位置
        self.target_pos = np.array([
            np.random.uniform(*self.target_range['x']),
            np.random.uniform(*self.target_range['y']),
            np.random.uniform(*self.target_range['z'])
        ])
        
        p.resetBasePositionAndOrientation(
            self.target_id,
            self.target_pos,
            [0, 0, 0, 1]
        )
        
        # 稳定仿真
        for _ in range(10):
            p.stepSimulation()
        
        # 获取初始观测
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """执行一步动作"""
        # 限制动作范围
        action = np.clip(action, -1.0, 1.0)
        
        # 缩放动作到实际关节增量
        action_scaled = action * self.action_scale
        
        # 获取当前关节位置
        current_joint_pos = self._get_joint_positions()
        
        # 计算目标关节位置
        target_joint_pos = current_joint_pos + action_scaled
        
        # 应用关节限制
        target_joint_pos = self._clip_joint_positions(target_joint_pos)
        
        # 执行关节控制
        for i in range(self.num_joints):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_joint_pos[i],
                force=500,  # 最大力矩
                maxVelocity=1.0
            )
        
        # 仿真步进
        p.stepSimulation()
        
        # 更新步数
        self.current_step += 1
        
        # 获取新观测
        obs = self._get_observation()
        
        # 计算奖励和终止条件
        reward, terminated, info = self._compute_reward_and_done(obs)
        
        # 截断条件（达到最大步数）
        truncated = self.current_step >= self.max_steps
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """获取当前观测"""
        # 关节位置和速度
        joint_states = [p.getJointState(self.robot_id, i) for i in range(self.num_joints)]
        joint_positions = np.array([state[0] for state in joint_states])
        joint_velocities = np.array([state[1] for state in joint_states])
        
        # 末端执行器位置
        ee_state = p.getLinkState(
            self.robot_id,
            self.ee_link_index,
            computeForwardKinematics=True
        )
        ee_pos = np.array(ee_state[0])
        
        # 目标位置
        target_pos = self.target_pos
        
        # 位置误差
        pos_error = target_pos - ee_pos
        
        # 组合观测
        obs = np.concatenate([
            joint_positions,      # 7
            joint_velocities,     # 7
            ee_pos,               # 3
            target_pos,           # 3
            pos_error             # 3
        ]).astype(np.float32)
        
        return obs
    
    def _get_joint_positions(self) -> np.ndarray:
        """获取当前关节位置"""
        joint_states = [p.getJointState(self.robot_id, i) for i in range(self.num_joints)]
        return np.array([state[0] for state in joint_states])
    
    def _clip_joint_positions(self, positions: np.ndarray) -> np.ndarray:
        """限制关节位置在有效范围内"""
        # Panda 关节限制（弧度）
        joint_limits = np.array([
            [-2.8973, 2.8973],   # joint 0
            [-1.7628, 1.7628],   # joint 1
            [-2.8973, 2.8973],   # joint 2
            [-3.0718, -0.0698],  # joint 3
            [-2.8973, 2.8973],   # joint 4
            [-0.0175, 3.7525],   # joint 5
            [-2.8973, 2.8973]    # joint 6
        ])
        
        clipped = np.clip(
            positions,
            joint_limits[:, 0],
            joint_limits[:, 1]
        )
        
        return clipped
    
    def _compute_reward_and_done(
        self,
        obs: np.ndarray
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """计算奖励和终止条件"""
        # 从观测中提取位置误差
        pos_error = obs[-3:]  # 最后 3 个元素
        distance = np.linalg.norm(pos_error)
        
        # === 奖励设计（可自定义）===
        
        # 1. 距离奖励（密集奖励）
        reward_distance = -distance * 10.0
        
        # 2. 成功奖励（稀疏奖励）
        success_threshold = 0.05  # 5cm
        reward_success = 100.0 if distance < success_threshold else 0.0
        
        # 3. 时间惩罚（鼓励快速完成）
        reward_time = -0.1
        
        # 总奖励
        reward = reward_distance + reward_success + reward_time
        
        # 终止条件
        terminated = distance < success_threshold
        
        # 额外信息
        info = {
            'distance': distance,
            'success': terminated,
            'reward_distance': reward_distance,
            'reward_success': reward_success,
            'reward_time': reward_time
        }
        
        return reward, terminated, info
    
    def _get_info(self) -> Dict[str, Any]:
        """获取环境信息"""
        obs = self._get_observation()
        distance = np.linalg.norm(obs[-3:])
        
        return {
            'distance': distance,
            'success': False,
            'current_step': self.current_step
        }
    
    def render(self):
        """渲染环境"""
        if self.render_mode == 'rgb_array':
            # 获取相机图像
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0.5, 0, 0.3],
                distance=1.5,
                yaw=45,
                pitch=-30,
                roll=0,
                upAxisIndex=2
            )
            
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=1.0,
                nearVal=0.1,
                farVal=100.0
            )
            
            (_, _, px, _, _) = p.getCameraImage(
                width=640,
                height=480,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            
            # 转换为 RGB 数组
            rgb_array = np.array(px, dtype=np.uint8)
            rgb_array = rgb_array[:, :, :3]  # 去掉 alpha 通道
            
            return rgb_array
        
        elif self.render_mode == 'human':
            # GUI 模式下自动渲染
            time.sleep(1./240.)
    
    def close(self):
        """关闭环境"""
        if self.client is not None:
            p.disconnect(self.client)
            self.client = None


# =============================================================================
# 演示代码
# =============================================================================

def demo_random_policy():
    """演示：随机策略"""
    print("\n" + "="*60)
    print("演示 1: 随机策略")
    print("="*60 + "\n")
    
    # 自动检测是否有显示（远程 SSH 时使用无头模式）
    import os
    has_display = 'DISPLAY' in os.environ and os.environ['DISPLAY']
    render_mode = 'human' if has_display else 'rgb_array'
    
    if not has_display:
        print("⚠️  检测到无显示环境（SSH），使用无头模式")
    
    env = PandaBaseEnv(render_mode=render_mode)
    
    n_episodes = 3
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        
        print(f"\n--- Episode {episode + 1} ---")
        print(f"初始距离: {info['distance']:.4f}m")
        
        for step in range(200):
            # 随机动作
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            # 每 50 步打印一次
            if step % 50 == 0:
                print(f"Step {step:3d}: distance={info['distance']:.4f}m, "
                      f"reward={reward:6.2f}")
            
            if terminated or truncated:
                print(f"\nEpisode 结束于 step {step}")
                print(f"  成功: {info['success']}")
                print(f"  最终距离: {info['distance']:.4f}m")
                print(f"  总奖励: {episode_reward:.2f}")
                break
    
    env.close()


def demo_heuristic_policy():
    """演示：启发式策略（简单 IK）"""
    print("\n" + "="*60)
    print("演示 2: 启发式策略")
    print("="*60 + "\n")
    
    env = PandaBaseEnv(render_mode='human')
    
    obs, info = env.reset()
    episode_reward = 0
    
    print("使用 PyBullet IK 到达目标...")
    
    for step in range(200):
        # 获取当前末端位置和目标位置
        ee_pos = obs[14:17]  # 末端位置
        target_pos = obs[17:20]  # 目标位置
        
        # 使用 PyBullet IK 计算关节目标
        joint_targets = p.calculateInverseKinematics(
            env.robot_id,
            env.ee_link_index,
            target_pos,
            maxNumIterations=100
        )
        
        # 将 IK 结果转换为动作（增量）
        current_joint_pos = obs[:7]
        action = (np.array(joint_targets[:7]) - current_joint_pos) / env.action_scale
        action = np.clip(action, -1.0, 1.0)
        
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        
        if step % 20 == 0:
            print(f"Step {step:3d}: distance={info['distance']:.4f}m")
        
        if terminated or truncated:
            print(f"\n成功! 用了 {step} 步")
            print(f"最终距离: {info['distance']:.4f}m")
            print(f"总奖励: {episode_reward:.2f}")
            break
    
    env.close()


def record_video(
    env: PandaBaseEnv,
    policy_fn,
    output_path: str = 'demo_video.mp4',
    n_episodes: int = 3
):
    """录制视频"""
    print(f"\n录制视频到: {output_path}")
    
    import imageio
    
    frames = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        
        # 添加文本说明（可选）
        print(f"录制 Episode {episode + 1}/{n_episodes}...")
        
        for step in range(200):
            # 渲染当前帧
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            
            # 执行动作
            action = policy_fn(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                # 在结束时停留几帧
                for _ in range(30):  # 1 秒（30 fps）
                    if frame is not None:
                        frames.append(frame)
                break
    
    # 保存视频
    if frames:
        imageio.mimsave(output_path, frames, fps=30)
        print(f"✓ 视频已保存: {output_path} ({len(frames)} 帧)")
    else:
        print("⚠️  没有渲染帧，无法保存视频")


def demo_video_recording():
    """演示：录制视频"""
    print("\n" + "="*60)
    print("演示 3: 录制视频")
    print("="*60 + "\n")
    
    # 创建环境（rgb_array 模式用于录制）
    env = PandaBaseEnv(render_mode='rgb_array')
    
    # 定义策略函数
    def random_policy(obs, env):
        return env.action_space.sample()
    
    def ik_policy(obs, env):
        ee_pos = obs[14:17]
        target_pos = obs[17:20]
        joint_targets = p.calculateInverseKinematics(
            env.robot_id, env.ee_link_index, target_pos
        )
        current_joint_pos = obs[:7]
        action = (np.array(joint_targets[:7]) - current_joint_pos) / env.action_scale
        return np.clip(action, -1.0, 1.0)
    
    # 录制随机策略
    record_video(env, random_policy, 'random_policy.mp4', n_episodes=2)
    
    # 录制 IK 策略
    record_video(env, ik_policy, 'ik_policy.mp4', n_episodes=2)
    
    env.close()
    print("\n✓ 视频录制完成！")


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("="*60)
    print("Franka Panda PyBullet 基础环境")
    print("="*60)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        print("\n请选择演示模式:")
        print("  1. 随机策略")
        print("  2. 启发式策略（IK）")
        print("  3. 录制视频")
        choice = input("\n输入选项 (1/2/3): ").strip()
        mode = choice
    
    if mode == '1':
        demo_random_policy()
    elif mode == '2':
        demo_heuristic_policy()
    elif mode == '3':
        demo_video_recording()
    else:
        print("运行所有演示...")
        demo_random_policy()
        demo_heuristic_policy()
        demo_video_recording()