# Aminos: Track 2 training package

## Codabench
- Username: `doomduke2`
- Submission id this package reproduces: **948640** (`t2_v156g_llrfuse.zip`, md5 `a3aa416379b4348cc9b1e309dcbc10cb`)
- final_score of that submission: **82.3929**

## How to reproduce
- Build command, if different from `docker build -t <name> .`: none. On a non-amd64 host add `--platform linux/amd64`.
- Train command, if different from the default: none. The organizers' command, `python -c "import solution; solution.train('/data', '/output', 'cuda')"`, runs the whole method. `solution.get_ttt_model('/output', 'cuda')` returns the adapter with `reset_ttt_state()` and `ttt_step(input_norm, prev_target_norm=None)`.

## Hardware your submitted model was trained on
- GPU model and count: the leaderboard card was assembled during the Development phase from many separate training jobs on NVIDIA V100 32 GB and RTX 4090 24 GB servers, so there is no single run or wall-clock time behind it. This package's `train()` rebuilds the same method in one run; we measured that run under your default allocation on **1 × NVIDIA A100-SXM4-80GB** (same GA100 chip as the A800).
- CPU: 16 cores (`--cpus 16`).
- Memory: 128 GB container limit (`--memory 128g`).
- Training time on this hardware: **21.5 h** for the full `train()` (organizer command plus `--cpus 16 --memory 128g`, no network).
- Peak GPU memory on this hardware: **16,981 MiB** (sampled every 15 s inside the container); container RAM peaked at 64 GB.

## Environment
- Base image: `w3nhao/realpde-track2@sha256:f07d1d68e19cb7ed5f2f405b6775f4fafb985887ac78356a9599f35365ccf972` (PyTorch 2.2.2), the Track 2 starting-kit image.
- Added packages and versions (pinned the same way in the Dockerfile): `loguru==0.7.3`. NumPy 1.26.4, h5py 3.16.0, Matplotlib 3.10.9 and PyYAML 6.0.1 come with the base image.

## Data, weights and code
- Training data used: only what `/data` holds: the 81 real files of `train_real/` (no `7575_0.h5`), the 100 files of `train_sim/`, and `mean_std_real.pt`. The simulation mean and std are recomputed inside `train()` from the 100 sim files at spatial stride two.
- Augmentation: standard transforms of released samples, recorded per recipe in the bundled YAML configs. Some recipes apply random reflection with probability 0.3 (`reflect_p`); some draw windows from all four spatial sub-sampling phase offsets of the full-resolution fields (`phases`). Several losses also include a hard-example L1 term (`hard_l1_a`) or distillation from teacher models that the same `train()` run trains first (`distill_weight`, `distill_ckpt`). No synthetic data are generated.
- Official baseline checkpoints used, if any: none. Every checkpoint, preconditioner, prior table and calibration array is produced by `train()`; the source tree contains no weights and no fitted arrays.
- Third-party code and its license: none bundled beyond the competition's own starting-kit scorer. Libraries: PyTorch (BSD-3-Clause), NumPy (BSD-3-Clause), h5py (BSD-3-Clause), Loguru (MIT), Matplotlib (PSF-based), PyYAML (MIT).

## Randomness
- Seed(s): fixed seed 0 for every training in the package.
- How runs or checkpoints are selected, if at all: no repeated runs and no seed selection. The two shipping members and the out-of-fold models ship their terminal EMA weights. The older head/calibration chains keep their Development-phase rule: each of their trainings keeps the EMA checkpoint with the lowest validation loss on held-out released real trajectories. Those are a fixed 9-trajectory split, or the fold's held-out files for those chains' out-of-fold models. This happens inside `train()`. Where the Track 2 chain needs a model whose recipe is identical to one the Track 1 chain of the same run has already trained (the RLB128 teacher and its six fold models), it reuses that model instead of training the same recipe a second time. The configs are checked to be identical apart from the output path and the run name.

## Track 2 only
- Does the method call the LLM gateway: **no**.

## Contact
- Name and email, reachable during the Decision Phase: Tianyu Li (team Aminos), lty2287003972@gmail.com

## Notes
- **Schedule.** `train()` runs three independent groups of trainings concurrently. With one visible GPU, all three share it; the package then starts NVIDIA MPS inside the container so their kernels run together instead of time-slicing. MPS starts only after a probe shows it works, and otherwise the run falls back to time-slicing with the same recipes. With three or more GPUs, each group gets its own GPU. Recipes, seeds and data are the same in every case.
- **Outputs.** Intermediates and logs go to `/output/work`; the inference files are installed at the top of `/output` only after every stage has passed. `/output/work/logs` holds one log per stage. `training_summary.json` records time, peak GPU memory and whether MPS was active.
- **Unused paths are skipped.** The Development-phase pipeline also built some models and tables that the submitted card does not use. `train()` skips them: it trains only the ancestors of this track's two members.
- **Reproduction checks.** Four independent from-scratch runs of this pipeline scored on the 81 training trajectories at or above the submitted card on every leg: rel +0.11 to +0.12, tke +0.04 to +0.06, mvpe +0.08 to +0.09, sps +0.42 to +0.45. The runs were on GH200 nodes, on 3 × RTX 4090 and twice on 1 × A100. These are in-sample checks. The method is unchanged, so we expect the development-set result to differ from row 948640 only through retraining randomness.
- **Known differences from the leaderboard card's files.** We compared the retrained cards with the card of row 948640 output by output on the 81 training trajectories. The retrained cards differ from it somewhat more than they differ from each other, for two traceable reasons. Neither is a change of method.
  - The first member (`model.pth`) of row 948640 came from one Development-phase run of the same configuration and teachers. Its log matches ours setting for setting: data windows, teachers, augmentation, steps and the first validation reading. It ended at validation rel-L2 0.0622, while two from-scratch runs of this package ended at 0.0594 and 0.0594.
  - The interval head of row 948640 had been fitted, earlier in the Development phase, on residuals of an older group of fold models that our repository can no longer produce. The package applies the same fitting recipe to the fold models that `train()` itself trains. In the Development phase we measured this choice of fold models to change SPS by at most 0.001 on the same forecaster.
- **Training log.** We attach the complete `/output/work/logs` of the single-GPU A100 run above.
