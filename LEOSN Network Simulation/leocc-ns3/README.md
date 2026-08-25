# LeoCC Core Reproduction for ns-3.41

Install the files and apply the CMake patch:

```bash
rsync -a ns3-overlay/ /path/to/ns-3.41/
patch -d /path/to/ns-3.41 -p1 < patches/ns3.41-internet-cmake.patch
cd /path/to/ns-3.41
./ns3 configure --enable-examples
./ns3 build leocc-connected-eval
python3 scratch/run_leocc_connected.py
python3 scratch/run_leocc_handover.py
```

The controller includes startup/drain, dual bandwidth estimates, RTT-guided
target selection, dynamic cruise, ProbeRTT, and reconfiguration adaptation.
