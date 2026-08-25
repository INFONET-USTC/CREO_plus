# CREO+ TCP Examples

The targets in this directory connect ns-3 TCP congestion-control callbacks to
Python through ns3-ai. The C++ side collects connected-phase state and applies
`cwnd`/pacing actions; the Python side performs DWT, PDPA, discrete SAC
training, checkpoint evaluation, and multi-flow control.

Build from the ns-3 root:

```bash
./ns3 build ns3ai_creo_single ns3ai_creo_multi
```

Training and test commands are documented in `use-gym/README.md`.
