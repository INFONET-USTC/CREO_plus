# Linux Evaluation

The evaluators exercise real Linux TCP sockets while replaying time-varying
capacity and propagation delay on one host.

## Closed-loop model evaluation

```bash
python3 evaluate_creo_drl.py --duration 20 \
  --checkpoint /absolute/path/to/creo_single.pt \
  --output ../results/drl-closed-loop
```

The script launches the shared model daemon, creates client/router/server
network namespaces, configures HTB and bidirectional NetEm, runs `iperf3 -C
creo`, and validates flow/sequence/action acknowledgements from the kernel.

## Direct Internet upload

```bash
python3 evaluate_creo_drl_upload.py --bytes 10000000 \
  --output ../results/drl-outbound
```

`cloudflare_upload.py` creates one TLS socket and sets
`TCP_CONGESTION="creo"` before connecting; the host-wide default CCA remains
unchanged.

## Trace replay outputs

`evaluate_creo.py` and `evaluate_creo_drl.py` write Origin-friendly files:

- `realbw.dat`: replayed bottleneck capacity;
- `throughput.dat`: receiver-side TCP throughput;
- `rtt.dat`: TCP RTT over each measurement interval;
- `base-rtt.dat`: idle path RTT measured before the flow;
- `summary.json`: aggregate throughput, RTT, utilization, and action checks;
- `restore-report.json`: namespace, qdisc, module-parameter, and CCA restoration.

The evaluators snapshot host state before setup and restore it after the run.
