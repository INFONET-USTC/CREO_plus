# LEOSN Network Simulation

This directory contains the transport implementations and trace-driven
experiment programs.

- `ns3-creo/`: CREO+ ns3-ai overlay for ns-3.41, including single-flow,
  multi-flow, deterministic test, and UDP handover targets.
- `leocc-ns3/`: LeoCC ns-3.41 overlay and scripts for generated and SIGCOMM
  traces.
- `LICENSE`: GPL-2.0 license used by the ns-3 components.

Install each overlay into a separate ns-3.41 source tree when running CREO+
and LeoCC experiments. This keeps their build targets and result directories
independent.
