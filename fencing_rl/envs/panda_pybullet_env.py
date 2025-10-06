import os
import math
from typing import Dict, Tuple, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    import gym
    from gym import spaces

import pybullet as p
import pybullet_data


class PandaPyBulletEnv(gym.Env):
    """
    Franka Panda in PyBullet with a minimal RL-friendly interface:
    - Observation: joint positions/velocities, gripper width, end-effector pose, target position, optional GUI slider
    - Action: Cartesian delta (dx, dy, dz, droll, dpitch, dyaw) mapped by IK to 7-DoF joint targets; applied with POSITION_CONTROL
    - Reward: negative distance to target, collision penalty, action magnitude penalty

    URDF source directory should contain `panda.urdf` and `meshes/`.
    """

    metadata = {"render.modes": ["human", "rgb_array", None]}

    def __init__(
        self,
        urdf_dir: str = "/home/student/tqz/project_iss/panda-gitlab-4242a-main/PandaRobot.jl-master/deps/Panda",
        use_gui: bool = False,
        time_step_s: float = 1.0 / 240.0,
        action_scale_pos_m: float = 0.02,
        action_scale_rot_rad: float = 0.05,
        workspace_bounds: Tuple[np.ndarray, np.ndarray] = (np.array([-0.6, -0.6, 0.0]), np.array([0.6, 0.6, 0.8])),
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.urdf_dir = urdf_dir
        self.urdf_path = os.path.join(self.urdf_dir, "panda.urdf")
        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")

        self.use_gui = use_gui
        self.physics_client = p.connect(p.GUI if self.use_gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(time_step_s)
        p.setGravity(0, 0, -9.81)

        # Plane and robot
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(
            self.urdf_path,
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION | p.URDF_MERGE_FIXED_LINKS,
        )

        # Joint and link discovery
        self.arm_joint_indices = self._discover_arm_joint_indices(self.robot_id)
        if len(self.arm_joint_indices) < 7:
            raise RuntimeError("Expected 7 controllable arm joints for Panda.")
        self.ee_link_index = self._discover_ee_link_index(self.robot_id)

        # GUI slider (optional, appears only in GUI mode)
        self.slider_parameter_id = None
        if self.use_gui:
            self.slider_parameter_id = p.addUserDebugParameter("slider", -1.0, 1.0, 0.0)

        # Action scaling and workspace
        self.action_scale_pos_m = float(action_scale_pos_m)
        self.action_scale_rot_rad = float(action_scale_rot_rad)
        self.workspace_low, self.workspace_high = workspace_bounds

        # Target visual marker
        self.target_visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE, radius=0.02, rgbaColor=[1.0, 0.2, 0.2, 0.9]
        )
        self.target_body_id = p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=self.target_visual_shape,
            basePosition=[0.4, 0.0, 0.3],
        )

        # RNG
        self.np_random = np.random.RandomState(seed)

        # Spaces
        obs_high = np.inf * np.ones(7 + 7 + 1 + 3 + 3 + 1, dtype=np.float32)
        obs_low = -obs_high
        self.observation_space = spaces.Dict(
            {
                "joint_pos": spaces.Box(low=-np.pi, high=np.pi, shape=(7,), dtype=np.float32),
                "joint_vel": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
                "gripper_width": spaces.Box(low=np.array([0.0], dtype=np.float32), high=np.array([0.08], dtype=np.float32), dtype=np.float32),
                "ee_pos": spaces.Box(low=self.workspace_low.astype(np.float32), high=self.workspace_high.astype(np.float32), dtype=np.float32),
                "target_pos": spaces.Box(low=self.workspace_low.astype(np.float32), high=self.workspace_high.astype(np.float32), dtype=np.float32),
                "slider_pos": spaces.Box(low=np.array([-1.0], dtype=np.float32), high=np.array([1.0], dtype=np.float32), dtype=np.float32),
            }
        )
        # Action: 6D Cartesian delta [dx, dy, dz, droll, dpitch, dyaw]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # Internal state
        self.current_target = np.array([0.4, 0.0, 0.3], dtype=np.float32)

        # Reset to a nominal configuration
        self.reset()

    # -------------- Discovery helpers --------------
    def _discover_arm_joint_indices(self, robot_id: int) -> list:
        indices = []
        num_joints = p.getNumJoints(robot_id)
        for j in range(num_joints):
            joint_info = p.getJointInfo(robot_id, j)
            joint_type = joint_info[2]
            if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC:
                # Heuristic: Panda arm joints are the first 7 non-fixed joints
                indices.append(j)
        return indices[:7]

    def _discover_ee_link_index(self, robot_id: int) -> int:
        candidate_names = [
            b"panda_hand",
            b"hand",
            b"panda_link8",
            b"link8",
            b"tool0",
        ]
        last_non_fixed = -1
        for i in range(p.getNumJoints(robot_id)):
            ji = p.getJointInfo(robot_id, i)
            last_non_fixed = i
            link_name = ji[12]
            if any(name in link_name for name in candidate_names):
                return i
        return last_non_fixed

    # -------------- Gym API --------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.np_random.seed(seed)

        # Nominal arm configuration
        nominal = [0.0, -0.6, 0.0, -2.0, 0.0, 1.6, 0.8]
        for i, j in enumerate(self.arm_joint_indices):
            p.resetJointState(self.robot_id, j, nominal[i], targetVelocity=0.0)

        # Randomize target within workspace
        self.current_target = self._sample_target()
        p.resetBasePositionAndOrientation(self.target_body_id, self.current_target.tolist(), [0, 0, 0, 1])

        # Step a bit for stability
        for _ in range(10):
            p.stepSimulation()

        observation = self._get_observation()
        return observation, {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Compute desired EE pose by increment
        ee_pos, ee_orn = self._get_ee_pose()
        dx, dy, dz, droll, dpitch, dyaw = action
        desired_pos = ee_pos + self.action_scale_pos_m * np.array([dx, dy, dz], dtype=np.float32)
        desired_pos = np.minimum(np.maximum(desired_pos, self.workspace_low), self.workspace_high)

        current_euler = np.array(p.getEulerFromQuaternion(ee_orn), dtype=np.float32)
        desired_euler = current_euler + self.action_scale_rot_rad * np.array([droll, dpitch, dyaw], dtype=np.float32)
        desired_orn = p.getQuaternionFromEuler(desired_euler.tolist())

        # IK to get joint targets
        joint_targets = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_link_index,
            desired_pos.tolist(),
            desired_orn,
            solver=p.IK_DLS,
            maxNumIterations=100,
            residualThreshold=1e-4,
        )
        joint_targets = np.asarray(joint_targets[: len(self.arm_joint_indices)], dtype=np.float32)

        p.setJointMotorControlArray(
            self.robot_id,
            jointIndices=self.arm_joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=joint_targets.tolist(),
            forces=[87.0] * len(self.arm_joint_indices),
        )

        # Simulate a small control horizon
        for _ in range(8):
            p.stepSimulation()

        observation = self._get_observation()
        reward, info = self._compute_reward_and_info(observation, action)
        terminated = self._is_success(observation)
        truncated = False
        return observation, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        # GUI mode renders automatically; for rgb capture, use getCameraImage externally
        pass

    def close(self):
        try:
            p.removeBody(self.target_body_id)
        except Exception:
            pass
        if p.isConnected():
            p.disconnect()

    # -------------- Observation / Reward --------------
    def _get_observation(self) -> Dict[str, np.ndarray]:
        joint_states = p.getJointStates(self.robot_id, self.arm_joint_indices)
        joint_pos = np.array([s[0] for s in joint_states], dtype=np.float32)
        joint_vel = np.array([s[1] for s in joint_states], dtype=np.float32)

        gripper_width = np.array([self._get_gripper_width()], dtype=np.float32)
        ee_pos, _ = self._get_ee_pose()
        slider_pos = np.array([self._get_slider_pos()], dtype=np.float32)

        obs = {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "gripper_width": gripper_width,
            "ee_pos": ee_pos.astype(np.float32),
            "target_pos": self.current_target.astype(np.float32),
            "slider_pos": slider_pos,
        }
        return obs

    def _compute_reward_and_info(self, obs: Dict[str, np.ndarray], action: np.ndarray) -> Tuple[float, Dict]:
        distance = float(np.linalg.norm(obs["ee_pos"] - obs["target_pos"]))
        collision = self._in_collision()

        reward_distance = -distance
        reward_collision = -1.0 if collision else 0.0
        reward_action_penalty = -0.01 * float(np.linalg.norm(action))
        reward = reward_distance + reward_collision + reward_action_penalty

        info = {
            "distance": distance,
            "collision": collision,
            "reward_distance": reward_distance,
            "reward_collision": reward_collision,
            "reward_action_penalty": reward_action_penalty,
        }
        return reward, info

    def _is_success(self, obs: Dict[str, np.ndarray]) -> bool:
        return bool(np.linalg.norm(obs["ee_pos"] - obs["target_pos"]) < 0.03)

    # -------------- Utilities --------------
    def _get_ee_pose(self) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        link_state = p.getLinkState(self.robot_id, self.ee_link_index, computeForwardKinematics=True)
        ee_pos = np.array(link_state[4], dtype=np.float32)
        ee_orn = link_state[5]
        return ee_pos, ee_orn

    def _get_gripper_width(self) -> float:
        # Heuristic: if gripper joints exist, try to read their positions; otherwise return 0
        width = 0.0
        try:
            num_joints = p.getNumJoints(self.robot_id)
            finger_positions = []
            for j in range(num_joints):
                ji = p.getJointInfo(self.robot_id, j)
                name = ji[1].decode("utf-8", errors="ignore")
                if "finger" in name or "hand" in name:
                    s = p.getJointState(self.robot_id, j)[0]
                    finger_positions.append(abs(float(s)))
            if len(finger_positions) >= 2:
                width = float(sum(finger_positions[:2]))
        except Exception:
            width = 0.0
        return float(np.clip(width, 0.0, 0.08))

    def _get_slider_pos(self) -> float:
        if self.slider_parameter_id is None:
            return 0.0
        try:
            return float(p.readUserDebugParameter(self.slider_parameter_id))
        except Exception:
            return 0.0

    def _in_collision(self) -> bool:
        # Contacts with plane or self are considered collision. Self-collision requires the flag at load time.
        contacts_world = p.getContactPoints(bodyA=self.robot_id)
        return len(contacts_world) > 0

    def _sample_target(self) -> np.ndarray:
        # Sample a reachable target above the table/plane
        low = np.array([max(self.workspace_low[0], 0.2), self.workspace_low[1], 0.15])
        high = np.array([self.workspace_high[0], self.workspace_high[1], 0.5])
        tgt = self.np_random.uniform(low=low, high=high)
        return tgt.astype(np.float32)

    # -------------- Offscreen rendering --------------
    def get_rgb(
        self,
        width: int = 640,
        height: int = 480,
        view_matrix: Optional[np.ndarray] = None,
        proj_matrix: Optional[np.ndarray] = None,
        use_egl: bool = True,
    ) -> np.ndarray:
        """
        Capture an RGB frame using offscreen rendering (DIRECT). If EGL plugin is
        unavailable, falls back to TinyRenderer.

        Returns: HxWx3 uint8 array (RGB)
        """
        if view_matrix is None:
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=[1.0, 0.0, 0.7],
                cameraTargetPosition=[0.0, 0.0, 0.3],
                cameraUpVector=[0.0, 0.0, 1.0],
            )
        if proj_matrix is None:
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=float(width) / float(height), nearVal=0.01, farVal=3.0
            )

        renderer = p.ER_TINY_RENDERER
        if use_egl:
            try:
                # Try load EGL plugin (idempotent if already loaded)
                p.loadPlugin(p.getPluginFilename("eglRendererPlugin"))
                renderer = p.ER_BULLET_HARDWARE_OPENGL
            except Exception:
                renderer = p.ER_TINY_RENDERER

        img = p.getCameraImage(int(width), int(height), view_matrix, proj_matrix, renderer=renderer)[2]
        rgba = np.reshape(img, (int(height), int(width), 4))
        rgb = rgba[:, :, :3].copy()
        return rgb


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Enable GUI (requires X11)")
    parser.add_argument("--urdf_dir", type=str, default="/home/student/tqz/project_iss/panda-gitlab-4242a-main/PandaRobot.jl-master/deps/Panda")
    args = parser.parse_args()

    env = PandaPyBulletEnv(use_gui=bool(args.gui), urdf_dir=args.urdf_dir)
    obs, _ = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()


