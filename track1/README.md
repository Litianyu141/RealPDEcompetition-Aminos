# Aminos: Track 1 training package

## Codabench
- Username: `doomduke2`
- Submission id this package reproduces: **948018** (`t1_v185_s2stack.zip`, md5 `b44fda74b13a17ed4b94a8f587904e67`)
- final_score of that submission: **82.0793**

## How to reproduce
- Build command, if different from `docker build -t <name> .`: none. On a non-amd64 host add `--platform linux/amd64`.
- Train command, if different from the default: none. The organizers' command, `python -c "import solution; solution.train('/data', '/output', 'cuda')"`, runs the whole method; `solution.load_predictor('/output', 'cuda')` returns the predictor.

## Hardware your submitted model was trained on
- GPU model and count: the leaderboard card was assembled during the Development phase from many separate training jobs on NVIDIA V100 32 GB and RTX 4090 24 GB servers, so there is no single run or wall-clock time behind it. This package's `train()` rebuilds the same method in one run; we measured that run under your default allocation on **1 × NVIDIA A100-SXM4-80GB** (same GA100 chip as the A800).
- CPU: 16 cores (`--cpus 16`).
- Memory: 128 GB container limit (`--memory 128g`).
- Training time on this hardware: **19.7 h** for the full `train()` (organizer command plus `--cpus 16 --memory 128g`, no network).
- Peak GPU memory on this hardware: **14,798 MiB** (sampled every 15 s inside the container); container RAM peaked at 52 GB.

## Environment
- Base image: `pytorch/pytorch@sha256:923f687790bec78081c357e71dcd5dcef80b0cc00f6c34484902a5e83362c854` (PyTorch 2.2.2, CUDA 12.1 runtime).
- Added packages and versions (pinned the same way in the Dockerfile): `numpy==1.26.4`, `h5py==3.11.0`, `loguru==0.7.3`, `matplotlib==3.10.9`. PyYAML 6.0.1 comes with the base image.

## Data, weights and code
- Training data used: only what `/data` holds: the 81 real files of `train_real/` (no `7575_0.h5`), the 100 files of `train_sim/`, and `mean_std_real.pt`. The simulation mean and std are recomputed inside `train()` from the 100 sim files at spatial stride two.
- Augmentation: standard transforms of released samples, recorded per recipe in the bundled YAML configs. Some recipes apply random reflection with probability 0.3 (`reflect_p`); some draw windows from all four spatial sub-sampling phase offsets of the full-resolution fields (`phases`). Several losses also include a hard-example L1 term (`hard_l1_a`) or distillation from teacher models that the same `train()` run trains first (`distill_weight`, `distill_ckpt`). No synthetic data are generated.
- Official baseline checkpoints used, if any: none. Every checkpoint, table and calibration array is produced by `train()`; the source tree contains no weights and no fitted arrays.
- Third-party code and its license: none bundled beyond the competition's own starting-kit scorer. Libraries: PyTorch (BSD-3-Clause), NumPy (BSD-3-Clause), h5py (BSD-3-Clause), Loguru (MIT), Matplotlib (PSF-based), PyYAML (MIT).

## Randomness
- Seed(s): fixed seed 0 for every training in the package.
- How runs or checkpoints are selected, if at all: no repeated runs and no seed selection. The two shipping members and the out-of-fold models ship their terminal EMA weights. The older head/band chain keeps its Development-phase rule: each of its trainings keeps the EMA checkpoint with the lowest validation loss on held-out released real trajectories. Those are a fixed 9-trajectory split, or the fold's held-out files for that chain's out-of-fold models. This happens inside `train()`.

## Contact
- Name and email, reachable during the Decision Phase: Tianyu Li (team Aminos), lty2287003972@gmail.com

## Notes
- **Schedule.** `train()` runs three independent groups of trainings concurrently. With one visible GPU, all three share it; the package then starts NVIDIA MPS inside the container so their kernels run together instead of time-slicing. MPS starts only after a probe shows it works, and otherwise the run falls back to time-slicing with the same recipes. With three or more GPUs, each group gets its own GPU. Recipes, seeds and data are the same in every case.
- **Outputs.** Intermediates and logs go to `/output/work`; the inference files are installed at the top of `/output` only after every stage has passed. `/output/work/logs` holds one log per stage. `training_summary.json` records time, peak GPU memory and whether MPS was active.
- **Unused paths are skipped.** The Development-phase pipeline also built some models and tables that the submitted card does not use. `train()` skips them: it trains only the ancestors of this track's two members.
- **Reproduction checks.** Three independent from-scratch runs of this pipeline reproduced the submitted card's scores on the 81 training trajectories within 0.06 points on every leg (rel, tke, mvpe, sps). The runs were on GH200 nodes, 3 × RTX 4090 and 1 × A100. These are in-sample checks. The method is unchanged, so we expect the development-set result to differ from row 948018 only through retraining randomness.
- **Training log.** We attach the complete `/output/work/logs` of the single-GPU A100 run above.
