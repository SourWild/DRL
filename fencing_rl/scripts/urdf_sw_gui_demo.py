import os
import argparse
import numpy as np
import time

from fencing_rl.envs.urdf_sw_pybullet_env import URDFSwPyBulletEnv
urdf_dir_default = "/home/student/tqz/project_iss/urdf-sw/urdf" # 改成*/urdf-sw/urdf路径

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf_dir", type=str, default=urdf_dir_default)
    parser.add_argument("--urdf_filename", type=str, default="urdf.urdf")
    parser.add_argument("--fix_mesh_paths", action="store_true")
    parser.add_argument("--steps", type=int, default=0, help="0 表示持续运行，>0 表示运行指定步数")
    parser.add_argument("--hz", type=float, default=60.0)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    env = URDFSwPyBulletEnv(
        use_gui=True,
        urdf_dir=args.urdf_dir,
        urdf_filename=args.urdf_filename,
        seed=0,
        fix_mesh_paths=bool(args.fix_mesh_paths),
    )
    obs, _ = env.reset()

    dt = 1.0 / max(1.0, float(args.hz))
    step_count = 0

    try:
        while True:
            if args.deterministic:
                dx = 0.2 * np.sin(2 * np.pi * step_count / 200.0)
                action = np.array([dx, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)

            step_count += 1
            time.sleep(dt)

            if args.steps > 0 and step_count >= args.steps:
                break
            if terminated or truncated:
                obs, _ = env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    main()


