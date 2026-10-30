# Out-of-Fold Calibrated U-Net Ensembles for Sim-to-Real Flow Forecasting with Reliable Prediction Intervals

Team **Aminos** — training and inference code for both tracks of the [NeurIPS 2026 RealPDE Competition](https://realpdecompetition.github.io/).

| Directory | Track |
|---|---|
| [`track1/`](track1/) | Track 1 — Sim2Real Transfer Learning |
| [`track2/`](track2/) | Track 2 — Long-Term Test-Time Adaptation |

Each directory is a self-contained Docker build context with source code and configuration only: no data and no weights. Every checkpoint and calibration array is produced from scratch by `solution.train()` from the released competition data. The method is described in [`track1/METHOD.md`](track1/METHOD.md) and [`track2/METHOD.md`](track2/METHOD.md).

## Requirements

Docker with the NVIDIA Container Toolkit, one NVIDIA GPU with at least 24 GB of memory, 16 CPU cores and 128 GB of RAM for training.

## Data

Download the competition release from the organizers (see the [competition website](https://realpdecompetition.github.io/)); the data are not redistributed here. Arrange it as

```
realpde_data/
├── train_real/          # real PIV trajectories (*.h5)
├── train_sim/           # simulated trajectories (*.h5)
└── mean_std_real.pt     # official normalisation statistics
```

## Docker images

Pull the prebuilt images

```bash
T1=doomduke2/realpde-aminos@sha256:df728647d57591aea9caa455ed299fd400d6c2e1eb31924a87a5d1933b6894e4
T2=doomduke2/realpde-aminos@sha256:918ea789b7326e7c87285ce0deb2b15eece9aac7e46a34f320e776883da84994
docker pull $T1 && docker tag $T1 aminos-t1     # Track 1
docker pull $T2 && docker tag $T2 aminos-t2     # Track 2
```

or build them from this repository:

```bash
docker build -t aminos-t1 track1     # Track 1
docker build -t aminos-t2 track2     # Track 2   (add --platform linux/amd64 on non-amd64 hosts)
```

Both ways give the tags `aminos-t1` and `aminos-t2` used below.

## Training

```bash
docker run --rm --network none --gpus '"device=0"' --shm-size 32g --cpus 16 --memory 128g \
  -v $PWD/realpde_data:/data:ro -v $PWD/output_t1:/output \
  aminos-t1 python -c "import solution; solution.train('/data', '/output', 'cuda')"
```

Use `aminos-t2` and another output directory for Track 2. One call runs the whole pipeline and writes the inference files to `/output`.

| | Track 1 | Track 2 |
|---|---|---|
| Training time, 1 × A100 80 GB, 16 cores | about 20 h | about 22 h |
| Peak GPU memory | about 15 GB | about 17 GB |
| Peak RAM | about 52 GB | about 64 GB |

With one GPU, the three concurrent training groups share it through NVIDIA MPS, which `train()` starts inside the container after checking that it works; otherwise it falls back to ordinary time-slicing, which is slower. With three or more GPUs, each group gets its own GPU. Logs go to `/output/work/logs`.

## Inference

**Track 1** — each 20-frame window is forecast from itself only:

```python
import solution                                   # inside the image, from /workspace
predictor = solution.load_predictor("/output", "cuda")
out = predictor.predict(x)        # x: float32 (N, 20, 32, 64, 3), physical units, channels (u, v, 0)
out["prediction"], out["lower"], out["upper"]     # each (N, 20, 32, 64, 3)
```

**Track 2** — one trajectory at a time, in normalised units; the previous window's target is revealed after each forecast:

```python
import torch, solution
model = solution.get_ttt_model("/output", torch.device("cuda"))
model.reset_ttt_state()                                     # at the start of every trajectory
prev = None
for x_norm, y_norm in windows:                              # CUDA tensors, each (1, 20, 32, 64, 3)
    prediction, info = model.ttt_step(x_norm, prev)         # info["lower"], info["upper"]
    prev = y_norm                                           # never the current or a future target
```

[`tools/predict_trajectory.py`](tools/predict_trajectory.py) runs either track over every window of one trajectory and saves the inputs, targets, predictions and bounds in physical units:

```bash
mkdir -p results
docker run --rm --gpus '"device=0"' -v $PWD/output_t1:/output:ro -v $PWD/realpde_data:/data:ro \
  -v $PWD/tools:/tools:ro -v $PWD/results:/results \
  aminos-t1 python /tools/predict_trajectory.py T1 /data/train_real/10125_10.h5 /results/T1_10125_10.npz
```

## Repository layout

```
├── track1/                  Track 1 package (Docker build context)
│   ├── Dockerfile
│   ├── solution.py          train() / load_predictor()
│   ├── README.md            hardware, environment, data, randomness
│   ├── METHOD.md            method and training schedule
│   ├── source_manifest.json SHA-256 of every file
│   └── ...                  pipeline, networks, recipes and inference runtime
├── track2/                  Track 2 package, same tree; solution.py exposes get_ttt_model()
├── tools/                   prediction script for trained models
└── verify_sources.py        checks both packages against their manifests
```

The two packages share most of their tree; only `Dockerfile`, `solution.py`, `README.md`, `METHOD.md` and `source_manifest.json` differ. `Track1/realpde_t1_starting_kit_*` and `Track2/realpde_t2_starting_kit_*` are the competition's starting kits, which the pipeline imports.

## License

Our code is released under the Apache License 2.0 ([`LICENSE`](LICENSE)). The bundled competition starting kits belong to the competition organizers.
