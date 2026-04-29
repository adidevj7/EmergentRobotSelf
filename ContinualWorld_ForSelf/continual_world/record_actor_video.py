import continualworld.gym_compat
# save as: record_actor_video.py

import os
import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import tensorflow as tf

from continualworld.envs import get_single_env
from continualworld.sac.models import MlpActor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, required=True)
    p.add_argument("--actor_ckpt", type=str, required=True,
                   help="Checkpoint prefix, e.g. checkpoints/actor  (NOT actor.index)")
    p.add_argument("--out", type=str, required=True,
                   help="Output path, e.g. videos/hammer.mp4 or videos/hammer.gif")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--camera_name", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)

    # match training architecture if you changed these from defaults
    p.add_argument("--hidden_sizes", type=int, nargs="+", default=[256, 256, 256, 256])
    p.add_argument("--activation", type=str, default="lrelu")
    p.add_argument("--use_layer_norm", action="store_true", default=True)
    p.add_argument("--hide_task_id", action="store_true", default=False)

    return p.parse_args()


def get_activation(name: str):
    name = name.lower()
    if name == "relu":
        return tf.nn.relu
    if name == "tanh":
        return tf.nn.tanh
    if name == "elu":
        return tf.nn.elu
    if name == "lrelu":
        return tf.nn.leaky_relu
    raise ValueError(f"Unsupported activation: {name}")


def unwrap_env(env):
    cur = env
    seen = set()
    while True:
        if id(cur) in seen:
            return cur
        seen.add(id(cur))
        if hasattr(cur, "env"):
            cur = cur.env
        elif hasattr(cur, "_env"):
            cur = cur._env
        else:
            return cur


def reset_compat(env, seed=None):
    try:
        out = env.reset(seed=seed)
    except TypeError:
        out = env.reset()

    if isinstance(out, tuple):
        return out[0]
    return out


def step_compat(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, rew, terminated, truncated, info = out
        done = terminated or truncated
        return obs, rew, done, info
    obs, rew, done, info = out
    return obs, rew, done, info


def render_frame(env, width, height, camera_name=None):
    # try old-style gym/metaworld render first
    try:
        frame = env.render(mode="rgb_array", width=width, height=height, camera_name=camera_name)
        if frame is not None:
            return np.asarray(frame)
    except Exception:
        pass

    # fallback to direct sim offscreen render
    base = unwrap_env(env)
    if hasattr(base, "sim"):
        try:
            if camera_name is None:
                frame = base.sim.render(width, height, mode="offscreen")
            else:
                frame = base.sim.render(width, height, mode="offscreen", camera_name=camera_name)
            frame = np.asarray(frame)
            if frame.ndim == 3:
                frame = frame[::-1, :, :]
            return frame
        except Exception as e:
            raise RuntimeError(f"Both render paths failed. Last sim.render error: {e}")

    raise RuntimeError("Could not find a working render path.")


def main():
    args = parse_args()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    env = get_single_env(args.task, randomization="deterministic")

    actor = MlpActor(
        input_dim=env.observation_space.shape[0],
        action_space=env.action_space,
        hidden_sizes=args.hidden_sizes,
        activation=get_activation(args.activation),
        use_layer_norm=args.use_layer_norm,
        num_heads=1,
        hide_task_id=args.hide_task_id,
    )

    # build model once before loading weights
    dummy = tf.convert_to_tensor(np.zeros((1, env.observation_space.shape[0]), dtype=np.float32))
    actor(dummy)

    # IMPORTANT: pass prefix like checkpoints/actor, not checkpoints/actor.index
    actor.load_weights(args.actor_ckpt)

    all_frames = []

    for ep in range(args.episodes):
        obs = reset_compat(env, seed=args.seed + ep)
        done = False
        ep_ret = 0.0
        ep_len = 0

        while not done:
            obs_batch = tf.convert_to_tensor(obs[None, :], dtype=tf.float32)
            mu, _, _, _ = actor(obs_batch)
            action = mu.numpy()[0]

            frame = render_frame(env, args.width, args.height, args.camera_name)
            all_frames.append(frame)

            obs, rew, done, info = step_compat(env, action)
            ep_ret += float(rew)
            ep_len += 1

        print(f"episode={ep} return={ep_ret:.3f} len={ep_len}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".gif":
        imageio.mimsave(out_path, all_frames, fps=args.fps)
    else:
        with imageio.get_writer(out_path, fps=args.fps) as writer:
            for frame in all_frames:
                writer.append_data(frame)

    env.close()
    print(f"saved video to {out_path}")


if __name__ == "__main__":
    main()
