import os
import argparse
import numpy as np
import imageio.v2 as imageio

from fencing_rl.envs.urdf_sw_pybullet_env import URDFSwPyBulletEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf_dir", type=str, default="/home/student/tqz/project_iss/urdf-sw/urdf")
    parser.add_argument("--urdf_filename", type=str, default="urdf.urdf")
    parser.add_argument("--fix_mesh_paths", action="store_true")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--out", type=str, default="urdf_sw_offscreen.mp4")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--use_egl", action="store_true")
    args = parser.parse_args()

    # 环境：DIRECT 渲染（无GUI）
    env = URDFSwPyBulletEnv(
        use_gui=False,
        urdf_dir=args.urdf_dir,
        urdf_filename=args.urdf_filename,
        seed=0,
        fix_mesh_paths=bool(args.fix_mesh_paths),
    )
    obs, _ = env.reset()

    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8)

    try:
        for t in range(args.steps):
            # 示例动作：小幅摆动或随机
            if args.deterministic:
                dx = 0.2 * np.sin(2 * np.pi * t / 200.0)
                action = np.array([dx, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)

            frame = env.get_rgb(args.width, args.height, use_egl=args.use_egl)
            writer.append_data(frame)

            if terminated or truncated:
                obs, _ = env.reset()
    finally:
        writer.close()
        env.close()

    print(f"saved: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()


