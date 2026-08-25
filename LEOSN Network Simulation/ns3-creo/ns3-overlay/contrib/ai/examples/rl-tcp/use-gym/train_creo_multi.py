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
    parser = argparse.ArgumentParser(
        description="Train one shared CREO+ policy for multiple ns3-ai TCP flows."
    )
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Total SAC minibatch updates for the shared model")
    parser.add_argument("--flows", type=int, default=3)
    parser.add_argument("--sim_seed", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_dir", type=str, default="models/creo_multi")
    parser.add_argument("--load", type=str, default="")
    parser.add_argument("--log", type=str, default="creo_multi_train.csv")
    parser.add_argument("--target", type=str, default="ns3ai_creo_multi")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--capacity_window", type=int, default=64)
    parser.add_argument("--history_len", type=int, default=10)
    parser.add_argument("--bw_trace", default="dataset/bw.txt")
    parser.add_argument("--latency_trace", default="dataset/latency.txt")
    parser.add_argument("--pdpa_lag", type=int, default=3)
    parser.add_argument("--pdpa_tolerance", type=float, default=0.025)
    parser.add_argument("--mini_window", type=int, default=15)
    parser.add_argument("--torch_threads", type=int, default=1)
    return parser.parse_args()


def local_path(path):
    return path if os.path.isabs(path) else os.path.join(SCRIPT_DIR, path)


def ns3_path(path):
    return path if os.path.isabs(path) else os.path.join(NS3_ROOT, path)


def main():
    args = parse_args()
    args.model_dir = local_path(args.model_dir)
    log_path = local_path(args.log)
    set_seed(args.seed, args.torch_threads)

    capacity = np.loadtxt(ns3_path(args.bw_trace), usecols=1)
    action_space, pdpa_metadata = pdpa_action_space(
        daubechies_denoise(capacity),
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
    if args.load:
        agent.load(local_path(args.load))

    ns3_settings = {
        "transport_prot": "TcpRlTimeBased",
        "duration": args.duration,
        "simSeed": args.sim_seed,
        "flows": args.flows,
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

    builders = {}
    pending = {}
    flow_ids = set()
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow([
            "step", "socket", "reward", "action_idx", "action_multiplier",
            "cwnd_bytes", "target_rate_bps", "throughput_bps",
            "sampled_capacity_bps", "rtt_us", "learn_epoch",
        ])
        try:
            obs, _ = env.reset()
            done = False
            step = 0
            while not done:
                socket_id = int(obs[0])
                flow_ids.add(socket_id)
                builder = builders.setdefault(socket_id, agent.new_feature_builder())
                previous = pending.get(socket_id)
                if previous is None:
                    state = agent.observe(obs, builder=builder)
                    reward = 0.0
                else:
                    state = agent.observe(obs, previous[2], previous[3], builder)
                    reward = creo_reward(obs)
                    agent.remember(previous[0], previous[1], reward, state, False)
                    if agent.learn_steps < args.epochs:
                        agent.learn()

                if float(obs[9]) <= 0.0:
                    action_idx, multiplier = agent.neutral_action()
                else:
                    action_idx, multiplier = agent.choose_action(state)
                cwnd, target_rate = agent.action_to_control(obs, multiplier, builder)
                action = np.array([cwnd, target_rate], dtype=np.uint32)
                next_obs, _, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
                pending[socket_id] = (state, action_idx, multiplier, target_rate)

                writer.writerow([
                    step,
                    socket_id,
                    reward,
                    action_idx,
                    multiplier,
                    cwnd,
                    target_rate,
                    float(obs[15]),
                    float(obs[17]),
                    float(obs[11]),
                    agent.learn_steps,
                ])
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
            f"Only {len(agent.replay)} shared transitions were collected; increase --duration"
        )
    while agent.learn_steps < args.epochs:
        agent.learn()

    os.makedirs(args.model_dir, exist_ok=True)
    metadata = {
        "training_epochs": agent.learn_steps,
        "ns3_interaction_steps": step,
        "flows": len(flow_ids),
        "flow_ids": sorted(flow_ids),
        "shared_model": True,
        "action_space": action_space,
        "pdpa": pdpa_metadata,
        "bandwidth_trace": args.bw_trace,
        "latency_trace": args.latency_trace,
        "mini_window_segments": args.mini_window,
    }
    model_path = os.path.join(args.model_dir, "shared.pt")
    agent.save(model_path, metadata)
    with open(os.path.join(args.model_dir, "training-metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(f"Saved shared CREO+ model for {len(flow_ids)} flows to {model_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
