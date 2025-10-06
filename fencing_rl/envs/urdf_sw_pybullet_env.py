import os
import re
from typing import Dict, Tuple, Optional, List

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    import gym
    from gym import spaces

import pybullet as p
import pybullet_data

urdf_dir_default = "/home/student/tqz/project_iss/urdf-sw/urdf"

class URDFSwPyBulletEnv(gym.Env):
    """
    Generic PyBullet env for a Panda-like arm defined in `urdf-sw`.

    Features:
    - Observation: joint positions/velocities, end-effector position, target position, GUI slider.
    - Action: 6D Cartesian delta -> IK -> joint POSITION_CONTROL commands.
    - Reward: negative distance to target + collision penalty + action magnitude penalty.

    Notes:
    - Auto-discovers controllable joints (revolute/prismatic) and an end-effector link.
    - Self-collision enabled via URDF flags.
    - Workspace bounds clamp desired EE pose before IK for safety.
    """

    metadata = {"render.modes": ["human", "rgb_array", None]}

    def __init__(
        self,
        urdf_dir: str = urdf_dir_default,
        urdf_filename: str = "urdf.urdf",
        use_gui: bool = False,
        time_step_s: float = 1.0 / 240.0,
        action_scale_pos_m: float = 0.02,
        action_scale_rot_rad: float = 0.05,
        workspace_bounds: Tuple[np.ndarray, np.ndarray] = (
            np.array([-0.8, -0.8, 0.0], dtype=np.float32),
            np.array([0.8, 0.8, 1.0], dtype=np.float32),
        ),
        ee_link_name_hints: Optional[List[bytes]] = None,
        fix_mesh_paths: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.urdf_dir = urdf_dir
        self.urdf_path = os.path.join(self.urdf_dir, urdf_filename)
        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")

        self.use_gui = use_gui
        self.physics_client = p.connect(p.GUI if self.use_gui else p.DIRECT)
        # Search paths for assets: PyBullet data + project root containing meshes/textures
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        project_root = os.path.abspath(os.path.join(self.urdf_dir, os.pardir))
        p.setAdditionalSearchPath(project_root)
        p.setAdditionalSearchPath(self.urdf_dir)
        # Explicitly add common resource folders if present
        meshes_dir = os.path.join(project_root, "meshes")
        textures_dir = os.path.join(project_root, "textures")
        if os.path.isdir(meshes_dir):
            p.setAdditionalSearchPath(meshes_dir)
        if os.path.isdir(textures_dir):
            p.setAdditionalSearchPath(textures_dir)
        p.setTimeStep(time_step_s)
        p.setGravity(0, 0, -9.81)

        # World: load plane using absolute path to avoid search-path issues
        try:
            self.plane_id = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"))
        except Exception:
            # Fallback: try relative search as last resort
            self.plane_id = p.loadURDF("plane.urdf")

        # Robot (optionally rewrite mesh paths)
        load_urdf_path = (
            self._prepare_urdf_with_fixed_mesh_paths(self.urdf_path)
            if fix_mesh_paths
            else self.urdf_path
        )
        self.robot_id = p.loadURDF(
            load_urdf_path,
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION | p.URDF_MERGE_FIXED_LINKS,
        )

        # Introspection
        self.arm_joint_indices = self._discover_controllable_joints(self.robot_id)
        if len(self.arm_joint_indices) == 0:
            raise RuntimeError("No controllable joints discovered in the robot.")

        if ee_link_name_hints is None:
            ee_link_name_hints = [
                b"panda_hand",
                b"hand",
                b"link8",
                b"tool0",
                b"ee",
                b"gripper",
            ]
        self.ee_link_index = self._discover_ee_link_index(self.robot_id, ee_link_name_hints)

        # GUI slider (optional)
        self.slider_parameter_id = None
        if self.use_gui:
            self.slider_parameter_id = p.addUserDebugParameter("slider", -1.0, 1.0, 0.0)

        # Action scaling & workspace
        self.action_scale_pos_m = float(action_scale_pos_m)
        self.action_scale_rot_rad = float(action_scale_rot_rad)
        self.workspace_low, self.workspace_high = workspace_bounds

        # Target marker
        self.target_visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE, radius=0.02, rgbaColor=[0.9, 0.2, 0.2, 0.9]
        )
        self.target_body_id = p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=self.target_visual_shape,
            basePosition=[0.4, 0.0, 0.3],
        )

        # RNG
        self.np_random = np.random.RandomState(seed)

        # Spaces
        num_joints = len(self.arm_joint_indices)
        self.observation_space = spaces.Dict(
            {
                "joint_pos": spaces.Box(low=-np.pi, high=np.pi, shape=(num_joints,), dtype=np.float32),
                "joint_vel": spaces.Box(low=-np.inf, high=np.inf, shape=(num_joints,), dtype=np.float32),
                "ee_pos": spaces.Box(low=self.workspace_low.astype(np.float32), high=self.workspace_high.astype(np.float32), dtype=np.float32),
                "target_pos": spaces.Box(low=self.workspace_low.astype(np.float32), high=self.workspace_high.astype(np.float32), dtype=np.float32),
                "slider_pos": spaces.Box(low=np.array([-1.0], dtype=np.float32), high=np.array([1.0], dtype=np.float32), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # Internal state
        self.current_target = np.array([0.4, 0.0, 0.3], dtype=np.float32)

        # Initialize
        self.reset()

    # ----------------------------- Discovery helpers -----------------------------
    def _discover_controllable_joints(self, robot_id: int) -> List[int]:
        indices: List[int] = []
        for j in range(p.getNumJoints(robot_id)):
            ji = p.getJointInfo(robot_id, j)
            jt = ji[2]
            if jt == p.JOINT_REVOLUTE or jt == p.JOINT_PRISMATIC:
                indices.append(j)
        return indices

    def _discover_ee_link_index(self, robot_id: int, name_hints: List[bytes]) -> int:
        last_non_fixed = -1
        for i in range(p.getNumJoints(robot_id)):
            ji = p.getJointInfo(robot_id, i)
            last_non_fixed = i
            link_name = ji[12]
            if any(h in link_name for h in name_hints):
                return i
        return last_non_fixed if last_non_fixed >= 0 else p.getNumJoints(robot_id) - 1

    # ------------------------------- Gym interface -------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.np_random.seed(seed)

        # Nominal configuration: set all joints to 0, safe default
        for j in self.arm_joint_indices:
            p.resetJointState(self.robot_id, j, 0.0, targetVelocity=0.0)

        # New random target
        self.current_target = self._sample_target()
        p.resetBasePositionAndOrientation(self.target_body_id, self.current_target.tolist(), [0, 0, 0, 1])

        for _ in range(8):
            p.stepSimulation()

        obs = self._get_observation()
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Current EE pose
        ee_pos, ee_orn = self._get_ee_pose()
        dx, dy, dz, droll, dpitch, dyaw = action

        desired_pos = ee_pos + self.action_scale_pos_m * np.array([dx, dy, dz], dtype=np.float32)
        desired_pos = np.minimum(np.maximum(desired_pos, self.workspace_low), self.workspace_high)

        current_euler = np.array(p.getEulerFromQuaternion(ee_orn), dtype=np.float32)
        desired_euler = current_euler + self.action_scale_rot_rad * np.array([droll, dpitch, dyaw], dtype=np.float32)
        desired_orn = p.getQuaternionFromEuler(desired_euler.tolist())

        # IK
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
            forces=[100.0] * len(self.arm_joint_indices),
        )

        for _ in range(8):
            p.stepSimulation()

        obs = self._get_observation()
        reward, info = self._compute_reward_and_info(obs, action)
        terminated = self._is_success(obs)
        truncated = False
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        pass

    def close(self):
        try:
            p.removeBody(self.target_body_id)
        except Exception:
            pass
        if p.isConnected():
            p.disconnect()

    # ------------------------- Observation / Reward utils ------------------------
    def _get_observation(self) -> Dict[str, np.ndarray]:
        joint_states = p.getJointStates(self.robot_id, self.arm_joint_indices)
        joint_pos = np.array([s[0] for s in joint_states], dtype=np.float32)
        joint_vel = np.array([s[1] for s in joint_states], dtype=np.float32)

        ee_pos, _ = self._get_ee_pose()
        slider_pos = np.array([self._get_slider_pos()], dtype=np.float32)

        obs = {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "ee_pos": ee_pos.astype(np.float32),
            "target_pos": self.current_target.astype(np.float32),
            "slider_pos": slider_pos,
        }
        return obs

    def _compute_reward_and_info(self, obs: Dict[str, np.ndarray], action: np.ndarray):
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

    # --------------------------------- Utilities --------------------------------
    def _get_ee_pose(self):
        link_state = p.getLinkState(self.robot_id, self.ee_link_index, computeForwardKinematics=True)
        ee_pos = np.array(link_state[4], dtype=np.float32)
        ee_orn = link_state[5]
        return ee_pos, ee_orn

    def _prepare_urdf_with_fixed_mesh_paths(self, urdf_path: str) -> str:
        """
        Ensure mesh paths referenced in URDF are valid relative to `urdf_dir`.
        Common exporter emits paths like `urdf/meshes/...` while actual layout is
        `<project_root>/meshes`. We rewrite such occurrences to `../meshes/...` since
        `urdf_path` lives under `<project_root>/urdf`.
        """
        try:
            with open(urdf_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            return urdf_path

        original_content = content

        # Rewrite typical patterns in mesh filename attributes
        # 0) package://urdf/meshes/... -> ../meshes/...
        content = re.sub(r"filename=\\\"package://urdf/meshes/", 'filename="../meshes/', content)
        # 1) urdf/meshes/...  ->  ../meshes/...
        content = re.sub(r"filename=\"urdf/meshes/", 'filename="../meshes/', content)
        # 2) ./urdf/meshes/... -> ../meshes/...
        content = re.sub(r"filename=\"\./urdf/meshes/", 'filename="../meshes/', content)
        # 3) meshes/... (from urdf dir) -> ../meshes/...
        content = re.sub(r"filename=\"meshes/", 'filename="../meshes/', content)

        if content == original_content:
            return urdf_path

        # Write a cached URDF next to original
        fixed_path = os.path.join(self.urdf_dir, "_fixed_mesh_paths.urdf")
        try:
            with open(fixed_path, "w", encoding="utf-8") as f:
                f.write(content)
            return fixed_path
        except Exception:
            return urdf_path

    def _get_slider_pos(self) -> float:
        if self.slider_parameter_id is None:
            return 0.0
        try:
            return float(p.readUserDebugParameter(self.slider_parameter_id))
        except Exception:
            return 0.0

    def _in_collision(self) -> bool:
        contacts = p.getContactPoints(bodyA=self.robot_id)
        return len(contacts) > 0

    def _sample_target(self) -> np.ndarray:
        low = np.array([
            max(self.workspace_low[0], 0.1),
            self.workspace_low[1],
            max(self.workspace_low[2], 0.1),
        ])
        high = np.array([
            self.workspace_high[0],
            self.workspace_high[1],
            min(self.workspace_high[2], 0.6),
        ])
        tgt = self.np_random.uniform(low=low, high=high)
        return tgt.astype(np.float32)

    # --------------------------- Offscreen RGB capture ---------------------------
    def get_rgb(
        self,
        width: int = 640,
        height: int = 480,
        view_matrix: Optional[np.ndarray] = None,
        proj_matrix: Optional[np.ndarray] = None,
        use_egl: bool = True,
    ) -> np.ndarray:
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
    parser.add_argument(
        "--urdf_dir",
        type=str,
        default=urdf_dir_default,
        help="Directory containing the URDF and meshes",
    )
    parser.add_argument(
        "--urdf_filename",
        type=str,
        default="urdf.urdf",
        help="URDF filename inside urdf_dir",
    )
    args = parser.parse_args()

    env = URDFSwPyBulletEnv(use_gui=bool(args.gui), urdf_dir=args.urdf_dir, urdf_filename=args.urdf_filename)
    obs, _ = env.reset()
    for _ in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()


