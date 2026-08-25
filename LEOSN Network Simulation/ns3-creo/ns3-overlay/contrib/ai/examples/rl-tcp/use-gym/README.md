# CREO+ Connected-Phase Programs

## Python programs

- `pre_process.py`: periodic db2 DWT with MAD/universal soft thresholding and
  Pareto-DP construction of three decrease, one neutral, and three increase
  actions.
- `creo_agent.py`: capacity/trend/fluctuation state builder, three LSTM
  branches, metric CNN, twin critics, entropy-tuned discrete SAC, replay.
- `train_creo_single.py`: single-flow ns3-ai interaction and traning.
- `train_creo_multi.py`: shared-model training with per-flow state/action
  histories for competing flows.
- `test_creo.py`: evaluation.

## C++ programs

- `tcp-rl-env.*`: RTT-adaptive statistical periods, ACK-dispersion capacity
  sampling, observations, and `[cwnd, pacing_rate]` action execution.
- `tcp-rl.*`: TCP congestion-control callbacks and ns3-ai environment binding.


Run all Python commands from this directory. Trace arguments are resolved from
the ns-3 root, while model and CSV paths are resolved from this directory.

```bash
python train_creo_single.py --epochs 10000 --model models/creo_single.pt
python train_creo_multi.py --epochs 10000 --flows 3
python test_creo.py --model models/creo_single.pt
```
