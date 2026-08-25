# CREO+ Linux TCP Congestion Control

This directory deploys CREO+ as an out-of-tree loadable Linux congestion
control algorithm named `creo`. The kernel data plane collects per-flow TCP
telemetry, publishes sequenced state records through `/dev/creo_drl`, consumes
matching model actions, and applies the resulting target to
`sk_pacing_rate` and `snd_cwnd`.

## Prerequisites

```bash
sudo apt install -y build-essential linux-headers-$(uname -r) \
  iproute2 iperf3 python3-venv
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Build and load

```bash
make
sudo insmod ./tcp_creo.ko drl_enabled=1 update_interval_us=100000
sysctl net.ipv4.tcp_available_congestion_control
ls -l /dev/creo_drl
```

`update_interval_us` is the `DeltaT0` cap. Each flow evaluates at
`min(0.5*RTT_prev, sRTT_prev, DeltaT0)`. Configure an FQ qdisc on the sender's
egress interface so Linux can honor socket pacing timestamps:

```bash
sudo tc qdisc replace dev <interface> root fq
```

## Start the shared model service

From the repository layout, the daemon automatically locates the bundled
`models/creo_single.pt` checkpoint inside the ns-3 overlay:

```bash
sudo .venv/bin/python deployment/creo_drl_daemon.py \
  --batch-size 64 --state-dir /var/lib/creo-drl
```

An explicit checkpoint can also be selected:

```bash
sudo .venv/bin/python deployment/creo_drl_daemon.py \
  --checkpoint /absolute/path/to/creo_single.pt \
  --capacity-trace /absolute/path/to/realbw.dat \
  --state-dir /var/lib/creo-drl
```

## Select CREO+ for TCP

Select the CCA per socket, leaving other host traffic unchanged:

```bash
iperf3 -C creo -c <server-ip>
```

Applications use the equivalent socket option:

```c
setsockopt(fd, IPPROTO_TCP, TCP_CONGESTION, "creo", sizeof("creo"));
```

To select it for every newly created TCP socket:

```bash
sudo sysctl -w net.ipv4.tcp_congestion_control=creo
```

## Evaluate and unload

The evaluator creates client/router/server namespaces, replays an HTB/NetEm
capacity-delay trace, runs real Linux TCP sockets, and verifies that every
observed model action is acknowledged by the kernel:

```bash
python3 evaluation/evaluate_creo_drl.py --duration 20 \
  --output results/drl-closed-loop
```

Stop the daemon before unloading:

```bash
sudo rmmod tcp_creo
```

See `deployment/README.md` for the ABI/model service and
`evaluation/README.md` for all result files.
