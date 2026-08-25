import argparse
import csv
import json
import os
import sys
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NS3_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../../.."))
GYM_BINDING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../model/gym-interface/py"))
NS3AI_UTILS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../python_utils"))
for path in (GYM_BINDING_DIR, NS3AI_UTILS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import gymnasium as gym
import ns3ai_gym_env
import numpy as np

from creo_agent import CreoAgent, creo_reward, set_seed
from pre_process import daubechies_denoise, pdpa_action_space


def parse_args():
    parser = argparse.ArgumentParser(description="Train CREO+ DRL agent on one ns3-ai TCP flow.")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Number of SAC minibatch updates")
    parser.add_argument("--sim_seed", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="models/creo_single.pt")
    parser.add_argument("--log", type=str, default="creo_single_train.csv")
    parser.add_argument("--target", type=str, default="ns3ai_creo_single")
    parser.add_argument("--show_ns3_log", action="store_true")
    parser.add_argument("--load", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--capacity_window", type=int, default=64)
    parser.add_argument("--history_len", type=int, default=10)
    parser.add_argument("--bw_trace", default="dataset/bw.txt")
    parser.add_argument("--latency_trace", default="dataset/latency.txt")
    parser.add_argument("--pdpa_lag", type=int, default=3)
    parser.add_argument("--pdpa_tolerance", type=float, default=0.025)
    parser.add_argument("--mini_window", type=int, default=15)
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--metadata", default="")
    return parser.parse_args()


def resolve_local_path(path):
    return path if os.path.isabs(path) else os.path.join(SCRIPT_DIR, path)


def resolve_ns3_path(path):
    return path if os.path.isabs(path) else os.path.join(NS3_ROOT, path)


def read_trace_column(path, column=1):
    values = []
    with open(resolve_ns3_path(path), "r", encoding="utf-8") as trace_file:
        for line in trace_file:
            fields = line.split()
            if len(fields) <= column:
                continue
            try:
                value = float(fields[column])
            except ValueError:
                continue
            if np.isfinite(value) and value > 0.0:
                values.append(value)
    if len(values) < 8:
        raise ValueError(f"Capacity trace {path} has fewer than eight valid samples")
    return np.asarray(values, dtype=np.float64)


def main():
    args = parse_args()
    model_path = resolve_local_path(args.model)
    log_path = resolve_local_path(args.log)
    metadata_path = resolve_local_path(args.metadata or f"{args.model}.json")
    load_path = resolve_local_path(args.load) if args.load else ""
    set_seed(args.seed, args.torch_threads)

    capacity_trace = read_trace_column(args.bw_trace)
    denoised_trace = daubechies_denoise(capacity_trace)
    action_space, pdpa_metadata = pdpa_action_space(
        denoised_trace,
        lag=args.pdpa_lag,
        tolerance=args.pdpa_tolerance,
    )

    agent = CreoAgent(
        action_space=action_space,
        batch_size=args.batch_size,
        capacity_window=args.capacity_window,
        history_len=args.history_len,
        mini_window_segments=args.mini_window,
    )
    if load_path:
        agent.load(load_path)

    ns3_settings = {
        "transport_prot": "TcpRlTimeBased",
        "duration": args.duration,
        "simSeed": args.sim_seed,
        "flows": 1,
        "miniWindow": args.mini_window,
        "bwTrace": args.bw_trace,
        "latencyTrace": args.latency_trace,
    }
    env = gym.make(
        "ns3ai_gym_env/Ns3-v0",
        targetName=args.target,
        ns3Path=NS3_ROOT,
        ns3Settings=ns3_settings,
        disable_env_checker=True,
    )

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow([
            "step", "socket", "reward", "action_idx", "action_multiplier",
            "cwnd_bytes", "target_rate_bps", "throughput_bps",
            "sampled_capacity_bps", "rtt_us", "min_rtt_us", "learn_epoch",
        ])
        try:
            obs, _ = env.reset()
            done = False
            step = 0
            state = agent.observe(obs)

            while not done:
                if float(obs[9]) <= 0.0:
                    action_idx, multiplier = agent.neutral_action()
                else:
                    action_idx, multiplier = agent.choose_action(state)
                cwnd, target_rate = agent.action_to_control(obs, multiplier)
                action = np.array([cwnd, target_rate], dtype=np.uint32)
                next_obs, _, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
                reward = creo_reward(next_obs)
                next_state = agent.observe(next_obs, multiplier, target_rate)
                agent.remember(state, action_idx, reward, next_state, done)
                loss = agent.learn() if agent.learn_steps < args.epochs else None

                writer.writerow(
                    [
                        step,
                        int(obs[0]),
                        reward,
                        action_idx,
                        multiplier,
                        cwnd,
                        target_rate,
                        float(next_obs[15]),
                        float(next_obs[17]),
                        float(next_obs[11]),
                        float(next_obs[12]),
                        agent.learn_steps,
                    ]
                )
                if loss and step % 50 == 0:
                    print(f"step={step} reward={reward:.4f} loss={loss}")
                state = next_state
                obs = next_obs
                step += 1
        except Exception as exc:
            print(f"Exception occurred: {exc}")
            traceback.print_exc()
            return 1
        finally:
            env.close()

    if len(agent.replay) < args.batch_size:
        raise RuntimeError(
            f"Only {len(agent.replay)} transitions were collected; increase --duration"
        )
    while agent.learn_steps < args.epochs:
        loss = agent.learn()
        if loss is None:
            raise RuntimeError("Replay buffer cannot supply a training minibatch")
        if agent.learn_steps % 100 == 0:
            print(f"training epoch={agent.learn_steps}/{args.epochs} loss={loss}")

    metadata = {
        "training_epochs": agent.learn_steps,
        "ns3_interaction_steps": step,
        "duration_s": args.duration,
        "sim_seed": args.sim_seed,
        "training_seed": args.seed,
        "bandwidth_trace": args.bw_trace,
        "latency_trace": args.latency_trace,
        "burst_pacing": True,
        "mini_window_segments": args.mini_window,
        "action_space": action_space,
        "pdpa": pdpa_metadata,
    }
    agent.save(model_path, metadata)
    os.makedirs(os.path.dirname(metadata_path) or ".", exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, sort_keys=True)
    print(f"Saved CREO+ model to {model_path}")
    print(f"Saved training metadata to {metadata_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
