## DRL 项目说明

本仓库包含使用 PyBullet 加载基于 Panda 改造的机械臂（urdf-sw）的环境与脚本，支持：
- 离线渲染录制（DIRECT + 可选 EGL）用于远程 SSH 训练可视化
- 本地 GUI 在线演示（X11/Prime Offload）用于快速检查机械臂与任务设定

### 1. 目录结构（关键部分）
```
DRL/
  fencing_rl/
    envs/
      urdf_sw_pybullet_env.py      # urdf-sw 环境类（状态/动作/IK/碰撞/奖励）
    scripts/
      urdf_sw_record_offscreen.py  # 离线录制到 MP4
      urdf_sw_gui_demo.py          # GUI 在线演示
  urdf-sw/
    urdf/urdf.urdf                 # URDF 主文件
    meshes/*.STL                   # 网格
```

### 2. 环境依赖
- Python 3.10.18
- pybullet, numpy, imageio
- 可选：EGL（服务器端更快的离屏渲染）

安装示例：
```bash
pip install -r fencing_rl/requirements.txt
```

如果要使用 EGL：确保系统安装了合适的 EGL 驱动/库（NVIDIA 驱动通常包含）。

### 3. [请忽略此条，仓库中已包含urdf资源] URDF 资源准备（mesh 路径），
urdf-sw 的 `urdf.urdf` 使用 `package://urdf/meshes/...` 路径，有两种处理方式：
- 方案A（推荐简单可靠）：将 `meshes/` 目录移动到 `urdf/` 下，形成 `urdf/meshes/...`。
- 方案B（不移动目录）：使用环境类的 `fix_mesh_paths=True` 自动改写为 `../meshes/...`。

注意：若选择方案A，则无需 `fix_mesh_paths`。

### 4. 路径与导入（很重要）
./fencing_rl/scripts 路径下两脚本（line7&line9）包含：
```python
import sys
sys.path.append("/home/student/tqz/project_iss")  # 改成 fencing_rl 的父目录绝对路径
```
请将该路径改为你本机上 `fencing_rl` 的父目录绝对路径，确保 `from fencing_rl...` 能正确导入。

同时将脚本中的 `URDF_DIR_DEFAULT` 或 `urdf_dir_default` 改为你本机 `urdf-sw/urdf` 的绝对路径。

### 5. 离线录制（SSH/无显示）
将生成 MP4 文件，适合远程训练时的可视化检查：
```bash
python /home/student/tqz/project_iss/fencing_rl/scripts/urdf_sw_record_offscreen.py \
  --urdf_dir /home/student/tqz/project_iss/urdf-sw/urdf \
  --urdf_filename urdf.urdf \
  --steps 600 --fps 30 --width 640 --height 480 \
  --out urdf_sw_offscreen.mp4 --use_egl --deterministic
```
提示：若未安装 EGL 或加载失败，脚本会自动回退到 TinyRenderer（更慢但可用）。

如未将 `meshes` 移到 `urdf/` 下，请加：`--fix_mesh_paths`。

### 6. 本地 GUI 在线演示（交互/可视化）
适合本地 Ubuntu/NVIDIA 环境快速查看：
```bash
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
python /home/student/tqz/project_iss/fencing_rl/scripts/urdf_sw_gui_demo.py \
  --urdf_dir /home/student/tqz/project_iss/urdf-sw/urdf \
  --urdf_filename urdf.urdf \
  --steps 0 --hz 60 --deterministic
```
- `--steps 0` 表示持续运行；>0 则运行指定步数后退出。
- 若未移动 `meshes` 到 `urdf/` 下，请加 `--fix_mesh_paths`。

GUI 一闪而过的常见原因：
- URDF/mesh 未成功加载（请先用第5步离线录制验证 URDF 能加载）。
- X/驱动/Prime Offload 环境异常（可尝试不加 Offload 变量直接运行，或检查本机 NVIDIA 驱动与 OpenGL）。

### 7. 训练中的调用建议
在你的训练代码中直接使用环境类：
```python
from fencing_rl.envs.urdf_sw_pybullet_env import URDFSwPyBulletEnv

env = URDFSwPyBulletEnv(
    use_gui=False,  # 训练建议 False
    urdf_dir="/home/you/path/to/urdf-sw/urdf",
    urdf_filename="urdf.urdf",
    fix_mesh_paths=False,  # 若 meshes 不在 urdf/ 下，设为 True
)
obs, _ = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
env.close()
```

环境提供：
- 状态：关节位置/速度、末端位置、目标位置、GUI滑块值
- 动作：笛卡尔增量（dx dy dz droll dpitch dyaw）经 IK 转关节指令
- 奖励：距离惩罚 + 碰撞惩罚 + 动作幅度惩罚

### 8. 常见问题排查（FAQ）
- 找不到 `plane.urdf`：已用绝对路径加载；仍失败请检查 `pybullet_data` 安装。
- 找不到 mesh：
  - 方案A：把 `meshes/` 放到 `urdf/` 下（建议）。
  - 方案B：实例化时传 `fix_mesh_paths=True`。
- EGL 加载失败：脚本会自动回退；也可移除 `--use_egl` 改用 TinyRenderer。
- GUI 闪退：先用离线录制验证 URDF OK；再检查 X/驱动/Prime Offload；或去掉 Offload 变量直接运行。

### 9. 复现实验的最小命令
```bash
# 离线录制一段视频
python fencing_rl/scripts/urdf_sw_record_offscreen.py --out demo.mp4 --use_egl --deterministic

# 本地 GUI 演示，持续运行
python fencing_rl/scripts/urdf_sw_gui_demo.py --steps 0 --hz 60 --deterministic
```

如需进一步扩展（末端姿态/速度、滑台读数、任务奖励定制），可修改 `fencing_rl/envs/urdf_sw_pybullet_env.py` 中对应接口。


