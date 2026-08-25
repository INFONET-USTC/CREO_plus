import argparse
import csv
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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained CREO+ model through ns3-ai.")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--flows", type=int, default=1)
    parser.add_argument("--sim_seed", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="models/creo_single.pt")
    parser.add_argument("--model_dir", type=str, default="models/creo_multi")
    parser.add_argument("--log", type=str, default="creo_test.csv")
    parser.add_argument("--target", type=str, default="")
    parser.add_argument("--bw_trace", default="dataset/bw.txt")
    parser.add_argument("--latency_trace", default="dataset/latency.txt")
    parser.add_argument("--mini_window", type=int, default=15)
    parser.add_argument("--torch_threads", type=int, default=1)
    return parser.parse_args()


def local_path(path):
    return path if os.path.isabs(path) else os.path.join(SCRIPT_DIR, path)


def checkpoint_path(args):
    if args.flows == 1:
        return local_path(args.model)
    shared = os.path.join(local_path(args.model_dir), "shared.pt")
    return shared if os.path.exists(shared) else local_path(args.model)


def main():
    args = parse_args()
    log_path = local_path(args.log)
    set_seed(args.seed, args.torch_threads)
    model_path = checkpoint_path(args)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"CREO+ checkpoint not found: {model_path}")

    agent = CreoAgent(eval_mode=True)
    agent.load(model_path)
    target = args.target or ("ns3ai_creo_single" if args.flows == 1 else "ns3ai_creo_multi")
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
        targetName=target,
        ns3Path=NS3_ROOT,
        ns3Settings=ns3_settings,
        disable_env_checker=True,
    )
    builders = {}
    last_controls = {}

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow([
            "step", "socket", "reward", "action_idx", "action_multiplier",
            "cwnd_bytes", "target_rate_bps", "throughput_bps",
            "sampled_capacity_bps", "rtt_us", "min_rtt_us",
        ])
        try:
            obs, _ = env.reset()
            done = False
            step = 0
            while not done:
                socket_id = int(obs[0])
                builder = builders.setdefault(socket_id, agent.new_feature_builder())
                previous = last_controls.get(socket_id)
                if previous is None:
                    state = agent.observe(obs, builder=builder)
                else:
                    state = agent.observe(obs, previous[0], previous[1], builder)

                if float(obs[9]) <= 0.0:
                    action_idx, multiplier = agent.neutral_action()
                else:
                    action_idx, multiplier = agent.choose_action(state, deterministic=True)
                cwnd, target_rate = agent.action_to_control(obs, multiplier, builder)
                action = np.array([cwnd, target_rate], dtype=np.uint32)
                reward = creo_reward(obs)
                next_obs, _, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
                last_controls[socket_id] = (multiplier, target_rate)

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
                    float(obs[12]),
                ])
                obs = next_obs
                step += 1
        except Exception as exc:
            print(f"Exception occurred: {exc}")
            traceback.print_exc()
            return 1
        finally:
            env.close()
    print(f"Saved CREO+ test log to {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
