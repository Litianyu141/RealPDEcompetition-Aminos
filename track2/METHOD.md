# Aminos: Track 2 method

## 1. Summary
Two U-Net forecasters, `T2PHK_H` and `T2PTK_SW`, are blended for the point forecast. On each trajectory, causal test-time adaptation updates the first member's output projection with a fixed ridge preconditioner, using only targets already revealed. Kinetic-energy and time-mean corrections choose among fixed options by follow-the-leader on the revealed history, warm-started by priors fitted from out-of-fold training data. A learned interval head and calibration give the bounds.

## 2. Model
- Architecture and number of parameters:
  - `T2PHK_H`: about 7.96 M parameters, a U-Net.
  - `T2PTK_SW`: about 4.24 M parameters, a self-warp U-Net.
  - Fitted arrays:
    - a 129×129 ridge preconditioner for the first member's last layer;
    - 70×32 kinetic-energy and 10×32 time-mean follow-the-leader warm-start tables;
    - the interval head and calibration tables.
- Inputs and outputs, including preprocessing and normalization:
  - Input `(1, 20, 32, 64, 3)` normalized with the official `mean_std_real.pt`, as on Codabench; output the next 20 frames in the same units.
  - Some recipes also normalize each window by its own statistics and undo it on the output.
  - `ttt_step` returns float32 prediction, lower and upper.
- Starting point: random initialization. No official baseline checkpoint and no external weights.

## 3. Data
- Training data used and how the two are combined: every network is pretrained on `train_sim` (sim statistics recomputed from the 100 files at spatial stride two) and then fine-tuned on `train_real`. Teachers trained earlier in the same run supply distillation targets.
- Validation split carved out of the training data, if any: two fixed, deterministic schemes over the 81 real files, both defined in the package.
  - A 72/9 train/validation split, used by the older head/calibration chains to choose checkpoints.
  - Fixed folds: three Reynolds-number folds and two angle-of-attack folds. Models trained on these folds give out-of-fold predictions for the follow-the-leader priors and the interval fits. Their simulation pretraining uses all 100 public sim files.
- Augmentation: random reflection (`reflect_p` 0.3) and all four spatial sub-sampling phase offsets of the full-resolution fields (`phases`), per recipe in the YAML configs. Some losses add a hard-example L1 term.
- Data left out, and why: `7575_0.h5` is not in the release mount. Nothing else is left out.

## 4. Training
`train()` stages the data, then runs three independent groups concurrently (rows 1–3; on one GPU they share it through NVIDIA MPS), then rows 4–7 in order. Network rows follow their YAML configs. Most use the values below; individual configs differ in steps, distillation and loss terms.

| Stage | Data | Loss | Optimizer and learning rate schedule | Steps or epochs | Batch size |
|---|---|---|---|---|---|
| 1. Interval-head chain shared with Track 1 (group A, first part) | sim, then the real 72/9 split, its fold models, and all 81 files | normalized relative L2 on u, v (plus recipe terms); interval head as in Track 1 | AdamW, weight decay 1e-5, cosine with 3% warm-up; sim lr 1e-3, real lr 2e-3 | 6,000 sim + up to 20,000 real per network | 16 |
| 1b. Track 2 gain head and calibration chain (group A, second part) | as row 1; reuses row 1's models where the recipe is identical | as row 1; calibration tables fitted in closed form | as row 1 | as row 1 | 16 |
| 2. Shipping members (group B) | sim, then all 81 real files; teachers first, then distilled students | normalized relative L2 + distillation, reflection, hard-example terms per config | as row 1 | as row 1; terminal EMA shipped | 16 |
| 3. Out-of-fold models (group C) | the fixed Reynolds and angle folds (sim pretraining on all 100 files) | as row 2 | as row 1 | as row 1; terminal EMA | 16 |
| 4. Last-layer Gram | the 81 real files, shipped first member | 129×129 feature Gram and its ridge-1 inverse (the preconditioner) | — | — | — |
| 5. Fold captures | for each of the five folds, the members' fold versions from row 3 with their own fold Gram and preconditioner, run causally over that fold's held-out trajectories | per-block kinetic-energy and time-mean sufficient statistics | — | — | — |
| 6. Follow-the-leader priors | the captures of row 5 | utility tables on a fixed grid (70×32 and 10×32, scales 4 and 3) | — | — | — |
| 7. Assembly | outputs of rows 1–6 | fitted arrays written into the runtime template; `/output` installed and the interface smoke-tested | — | — | — |

- How runs or checkpoints are selected, if at all: no repeated runs and no seed selection. Rows 2–3 ship terminal EMA weights. Rows 1 and 1b keep, for each of their trainings, the EMA checkpoint with the lowest validation loss on the held-out real trajectories. Paths the shipped card does not use are skipped.
- Seeds, and the run-to-run spread you have seen: seed 0 everywhere. Four independent from-scratch runs scored above the submitted card on the 81 training trajectories: rel +0.11 to +0.12, tke +0.04 to +0.06, mvpe +0.08 to +0.09, sps +0.42 to +0.45. In local five-fold evaluation, changing the seed of one recipe moved the Final score by 0.001–0.008.

## 5. Uncertainty bounds
- The base interval comes from the interval head and the 64-bin table of row 1, combined with the Track 2 calibration tables of row 1b. A causal width adjustment uses only revealed history.
- Bounds are returned with the point forecast in normalized units and are ordered (`lower ≤ upper`).

## 6. Inference
- Track 2:
  - `reset_ttt_state()` clears every trajectory-local accumulator, the last-layer update, any pending revealed target and the CUDA buffers.
  - `ttt_step(input_norm, prev_target_norm)` uses the current window and, when given, the target revealed for the previous step; it never reads current or future targets.
  - The adaptation keeps a causal cross-covariance between the first member's last-layer features and its revealed residuals. It applies a ridge-preconditioned correction to that member's output convolution, which is the only parameter set updated.
  - The kinetic-energy and time-mean corrections pick among fixed options by follow-the-leader on the trajectory's revealed windows, starting from the warm-start tables.
  - One adaptation step per window; no gradient steps.

## 7. Speed
- CUDA-graph capture of the per-step forward.
- A batched reset: one fused copy restores the post-reset state.
- Pre-processing and output un-normalization fused into single passes.
- The two members on separate streams, with the longer one on the critical path.
- Row 948640 scored 92.77 on the Time sub-score.

## 8. What mattered
Effect sizes are online Final deltas between our own consecutive Development-phase uploads, rounded. One upload's Time draw moves Final by about ±0.014. Items marked *approximate* changed together with other parts, or their attribution rests on our notes rather than a clean pair.

Overall, Final rose from 79.80 (our first upload under the current scoring, 2026-08-06) to 82.39 (row 948640). By leg, about +2.1 of the +2.6 came from SPS, +0.2 from TKE, +0.2 from rel and +0.1 from MVPE.

1. **Interval construction.** Early band changes gave about +0.42 through SPS (*approximate*; details not recorded). Porting the learned interval head and its table from Track 1 added about +0.10, with the point legs identical.
2. **Revealed-target follow-the-leader corrections.** Kinetic energy and time mean, with per-block width and out-of-fold warm-start priors. About +0.35 (TKE, SPS, MVPE), the largest documented single step (*approximate*; several parts shipped together).
3. **Two-member blend.** About +0.28 (*approximate*). Distilling the two members from teachers added about +0.09.
4. **Retraining the members with the current recipes.** About +0.06 to +0.16 (*approximate*).
5. **Multi-stream runtime.** About +0.08, through Time only, with accuracy bit-exact.
6. **Small causal corrections.** A test-time bias correction and a half-weighted interval-centre shift each changed accuracy by less than +0.05.
7. **The last-layer ridge adaptation in the final card.** No measurable online change in the accuracy legs (−0.001). The +0.02 Final of that upload came from Time.

## 9. Compute
- Development phase: separate jobs on NVIDIA V100 32 GB and RTX 4090 24 GB servers; there is no single wall-clock time.
- This package, one from-scratch `train()` on 1 × A100-SXM4-80GB, 16 CPU cores and 128 GB, with MPS: 21.5 h; peak GPU memory 16,981 MiB; container RAM peak 64 GB.

## 10. References
- O. Ronneberger, P. Fischer, T. Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015.
- G. Hinton, O. Vinyals, J. Dean. Distilling the Knowledge in a Neural Network. arXiv:1503.02531, 2015.
- I. Loshchilov, F. Hutter. Decoupled Weight Decay Regularization (AdamW). ICLR 2019.
- Y. Sun et al. Test-Time Training with Self-Supervision for Generalization under Distribution Shifts. ICML 2020.
- N. Cesa-Bianchi, G. Lugosi. Prediction, Learning, and Games (follow-the-leader). Cambridge University Press 2006.
- R. Koenker, G. Bassett. Regression Quantiles (pinball loss). Econometrica 1978.
- Libraries: PyTorch, NumPy, h5py, Loguru, Matplotlib, PyYAML.
