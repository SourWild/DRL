import os
import argparse
import numpy as np
import imageio.v2 as imageio

from fencing_rl.envs.panda_pybullet_env import PandaPyBulletEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf_dir", type=str, default="/home/student/tqz/project_iss/4242a-main/PandaRobot.jl-master/deps/Panda")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--out", type=str, default="panda_demo.mp4")
    parser.add_argument("--use_egl", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    env = PandaPyBulletEnv(use_gui=False, urdf_dir=args.urdf_dir)
    obs, _ = env.reset()

    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8)

    try:
        for t in range(args.steps):
            # 示例动作：小幅摆动 + 向目标微调
            if args.deterministic:
                dx = 0.2 * np.sin(2 * np.pi * t / 200.0)
                action = np.array([dx, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)

            # 抓帧
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


