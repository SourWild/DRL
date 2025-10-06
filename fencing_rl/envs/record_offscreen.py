# record_offscreen.py
import os
import numpy as np
import imageio.v2 as imageio

# 方式一：EGL（更快）——需要服务器安装 EGL 驱动库
USE_EGL = True
if USE_EGL:
    os.environ["PYBULLET_EGL"] = "1"

import pybullet as p
import pybullet_data

# 连接 DIRECT（无 GUI）
client = p.connect(p.DIRECT)
if USE_EGL:
    try:
        p.loadPlugin(p.getPluginFilename("eglRendererPlugin"))
    except Exception:
        print("EGL 插件加载失败，改用 TinyRenderer")
        USE_EGL = False

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(1.0 / 240.0)

# 加载平面与 Panda（替换为你的 URDF 路径）
plane_id = p.loadURDF("plane.urdf")
panda_dir = "/home/student/tqz/project_iss/4242a-main/PandaRobot.jl-master/deps/Panda"
robot_id = p.loadURDF(
    os.path.join(panda_dir, "panda.urdf"),
    basePosition=[0, 0, 0],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    useFixedBase=True,
    flags=p.URDF_USE_SELF_COLLISION | p.URDF_MERGE_FIXED_LINKS,
)

# 简单移动关节，制造动画
arm_joints = []
for j in range(p.getNumJoints(robot_id)):
    jt = p.getJointInfo(robot_id, j)[2]
    if jt in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
        arm_joints.append(j)
arm_joints = arm_joints[:7]

# 相机参数
width, height = 640, 480
view_matrix = p.computeViewMatrix(
    cameraEyePosition=[1.0, 0.0, 0.7],
    cameraTargetPosition=[0.0, 0.0, 0.3],
    cameraUpVector=[0.0, 0.0, 1.0],
)
proj_matrix = p.computeProjectionMatrixFOV(
    fov=60, aspect=width / height, nearVal=0.01, farVal=3.0
)
renderer = p.ER_BULLET_HARDWARE_OPENGL if USE_EGL else p.ER_TINY_RENDERER

# 视频写入
out_path = "panda_offscreen.mp4"
writer = imageio.get_writer(out_path, fps=30, codec="libx264", quality=8)

num_frames = 600
for t in range(num_frames):
    # 简单轨迹：首关节小幅往返
    target = 0.5 * np.sin(2 * np.pi * t / 200.0)
    p.setJointMotorControlArray(
        robot_id,
        jointIndices=arm_joints,
        controlMode=p.POSITION_CONTROL,
        targetPositions=[target, -0.6, 0.0, -2.0, 0.0, 1.6, 0.8],
        forces=[87.0] * len(arm_joints),
    )
    for _ in range(4):
        p.stepSimulation()

    # 抓帧
    img_arr = p.getCameraImage(
        width, height, view_matrix, proj_matrix,
        renderer=renderer
    )[2]  # RGB
    frame = np.reshape(img_arr, (height, width, 4))[:, :, :3]  # RGBA->RGB
    writer.append_data(frame)

writer.close()
p.disconnect()
print(f"saved: {os.path.abspath(out_path)}")