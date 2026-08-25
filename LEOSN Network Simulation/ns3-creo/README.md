# CREO+ on ns-3.41

`ns3-overlay` preserves the exact paths expected by ns-3.41 and contains the
required ns3-ai message/Gym runtime, CREO+ TCP examples, datasets, and handover
notification example.

## Prerequisites

On Ubuntu, install the ns-3 and ns3-ai build dependencies:

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build git rsync \
  libboost-program-options-dev libprotobuf-dev protobuf-compiler \
  pybind11-dev python3-dev python3-venv
```

Use Python 3.10, 3.11, or 3.12 for ns-3.41 and create a virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy gymnasium protobuf psutil torch matplotlib
```

## Install into ns-3.41

Obtain the official ns-3.41 source and merge the overlay:

```bash
git clone --branch ns-3.41 --depth 1 https://gitlab.com/nsnam/ns-3-dev.git ns-3.41
rsync -a ns3-overlay/ ns-3.41/
cd ns-3.41
./ns3 configure --enable-examples -- \
  -DPython_EXECUTABLE="$(command -v python)"
./ns3 build ns3ai_creo_single ns3ai_creo_multi
./ns3 build creo-udp-handover-notification
```

The overlay modifies `contrib/ai`, `dataset`, and the CREO+ scratch target. It
does not require changes under `src/internet`.

## Train a single-flow policy

Training should traverse all available scenarios so that the DRL policy
observes diverse capacity and RTT dynamics. As many datasets as possible were generated using different communication parameters. Here, we present LEOSN representative datasets. The provided `dataset/bw.txt` and `dataset/latency.txt` files are example
inputs. Reproduction users should generate additional datasets, choose their
own parameter ranges, and update the trace paths as required.

```bash
cd contrib/ai/examples/rl-tcp/use-gym
python train_creo_single.py \
  --duration 30 --epochs 10000 \
  --bw_trace dataset/bw.txt --latency_trace dataset/latency.txt \
  --model models/creo_single.pt \
  --log results/creo_single_train.csv
```

`--epochs` is the number of discrete SAC minibatch updates. The checkpoint
stores the actor/twin critics, target network, optimizer state, DWT/PDPA action
space, architecture dimensions, and training metadata.

## Train multiple flows

```bash
python train_creo_multi.py \
  --duration 30 --epochs 10000 --flows 3 \
  --model_dir models/creo_multi \
  --log results/creo_multi_train.csv
```

All flows share one network and replay buffer while retaining independent
capacity/metric histories and action state. Training writes
`models/creo_multi/shared.pt` and its metadata.

## Evaluate a checkpoint

```bash
python test_creo.py \
  --duration 30 --model models/creo_single.pt \
  --log results/creo_single_test.csv
```

The Python CSV records reward, selected action, target rate, receiver-side
throughput, sampled capacity, RTT, and minimum RTT. The C++ program writes
trace-aligned `throughput.dat`, `realbw.dat`, `prop.dat`, `queueSize.dat`, and
`cwnd.dat` files under the ns-3 `results/` directory.

## Main targets

- `ns3ai_creo_single`: one trace-driven TCP flow and one DRL agent.
- `ns3ai_creo_multi`: multiple independently controlled TCP flows.
- `creo-udp-handover-notification`: notification/ACK/retry handover workflow.
