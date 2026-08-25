# Shared CREO+ Model Service

`creo_drl_daemon.py` binds the Linux CCA to `creo_model_service.py`. After each
device notification, the daemon drains up to 64 available flow states, performs
one batched forward pass, returns flow/sequence-matched Q10 actions, and records
kernel acknowledgements. Set `--batch-size` to tune the maximum batch.


## Live kernel service

```bash
sudo ../.venv/bin/python creo_drl_daemon.py \
  --checkpoint /absolute/path/to/creo_single.pt \
  --state-dir /var/lib/creo-drl
```

The device is mode `0600` and accepts one model daemon. State and action
messages carry `flow_id`, state sequence, production timestamp, validity, and
the previously applied action, so actions cannot cross between flows.

## systemd

Install the repository at `/opt/creo-plus`, build/load `tcp_creo.ko`, and then:

```bash
sudo cp creo-drl.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now creo-drl.service
sudo systemctl status creo-drl.service
```

Set `CREO_TORCH_THREADS` in the service environment to control PyTorch CPU
threads. The supplied unit uses one inference thread.
