# Aminos: Track 1 method

## 1. Summary
Two U-Net forecasters map 20 input frames to the next 20 frames, and their outputs are averaged. Point heads correct the forecast: a variance-deficit gain head, a three-regime anchor chosen from the input window, and a KE-preserving temporal gain. The interval is built from a per-lead base band, a learned spatial interval head (SBAND2) with a 64-bin lookup, and a shift of the interval centre fitted from the two members' difference (SPSR). The interval parts determine the Safe Prediction Score; the members and point heads set rel, tke and mvpe.

## 2. Model
- Architecture and number of parameters:
  - `T1GEG_H`: about 8.0 M parameters, a ConvNeXt-style U-Net with base width 128 and a GEGLU gate.
  - `T2PTK_SW`: about 4.24 M parameters, a self-warp U-Net.
  - The two members are blended with fixed equal weights.
  - Heads: a gain head; the SBAND2 interval head (157,352 parameters, 5 dilated convolution layers); fitted lookup tables for the regime anchor, the temporal gain (LK), the 64-bin band edges and the SPSR centre weights.
- Inputs and outputs, including preprocessing and normalization:
  - Input `(N, 20, 32, 64, 3)` in physical units; u and v are measured and the third channel is zero.
  - Networks see inputs normalized with the official `mean_std_real.pt`; outputs are denormalized back.
  - Some recipes also normalize each window by its own statistics (`in_norm: self`) and undo it on the output.
  - `predict` returns float32 `prediction`, `lower` and `upper` of the same shape.
- Starting point: random initialization. No official baseline checkpoint and no external weights.

## 3. Data
- Training data used and how the two are combined: every network is first pretrained on `train_sim` (normalized with sim statistics recomputed from the 100 files at spatial stride two), then fine-tuned on `train_real`. Teachers trained earlier in the same run supply distillation targets for students.
- Validation split carved out of the training data, if any: two fixed, deterministic schemes over the 81 real files, both defined in the package.
  - A 72/9 train/validation split, used by the older head/band chain to choose checkpoints.
  - Fixed folds: three Reynolds-number folds and two angle-of-attack folds. Models trained on these folds give out-of-fold predictions for fitting the heads and tables. Their simulation pretraining uses all 100 public sim files.
- Augmentation: random reflection (`reflect_p` 0.3) and all four spatial sub-sampling phase offsets of the full-resolution fields (`phases`), per recipe in the YAML configs. Some losses add a hard-example L1 term.
- Data left out, and why: `7575_0.h5` is not in the release mount. Nothing else is left out.

## 4. Training
`train()` stages the data, then runs three independent groups concurrently (rows 1–3; on one GPU they share it through NVIDIA MPS), then the fits in rows 4–7 in order. Every network row uses the recipe in its YAML config. Most use the values below; individual configs differ in steps, distillation and loss terms.

| Stage | Data | Loss | Optimizer and learning rate schedule | Steps or epochs | Batch size |
|---|---|---|---|---|---|
| 1. Head/band chain (group A) | sim, then real 72/9 split, its fold models, and all 81 files for the shipped versions | normalized relative L2 on u, v (plus recipe terms) | AdamW, weight decay 1e-5, cosine with 3% warm-up; sim lr 1e-3, real lr 2e-3 | 6,000 sim + up to 20,000 real per network; heads fitted on the chain's out-of-fold predictions | 16 |
| 2. Shipping members (group B) | sim, then all 81 real files; teachers first, then distilled students | normalized relative L2 + distillation, reflection, hard-example terms per config | as row 1 | as row 1; terminal EMA shipped | 16 |
| 3. Out-of-fold models (group C) | the fixed Reynolds and angle folds (sim pretraining on all 100 files) | as row 2 | as row 1 | as row 1; terminal EMA | 16 |
| 4. SBAND2 interval head | out-of-fold predictions from row 3 (13,202 windows at stride 5) | pinball loss, quantile 0.75 | AdamW lr 3e-4, OneCycle | 15 epochs; then the 64-bin edges are recomputed | 16 |
| 5. Regime anchor and LK gain | out-of-fold predictions of the two members' fold versions | closed-form fits: k-means (K=3, fixed seed) on input-window block means; 8×8 block KE and mean maps; the 3×20×4×8×2 temporal gain from sufficient statistics | — | — | — |
| 6. SPSR centre weights | out-of-fold element captures of the two members | grid search of 20×2 weights over [−0.8, 0.8] in steps of 0.05 | — | — | — |
| 7. Assembly | outputs of rows 1–6 | the fitted arrays are written into the runtime template; `/output` is installed and the interface is smoke-tested | — | — | — |

- How runs or checkpoints are selected, if at all: no repeated runs and no seed selection. Rows 2–3 ship terminal EMA weights. Row 1 keeps, for each of its trainings, the EMA checkpoint with the lowest validation loss on its held-out real trajectories. Paths the shipped card does not use are skipped.
- Seeds, and the run-to-run spread you have seen: seed 0 everywhere.
  - Three independent from-scratch runs matched the submitted card on the 81 training trajectories within 0.06 points on each of rel, tke, mvpe and sps.
  - In local five-fold evaluation, changing the seed of one recipe moved the Final score by 0.001–0.008.

## 5. Uncertainty bounds
1. The interval starts from asymmetric per-lead, per-channel base bands.
2. The SBAND2 head predicts a per-element residual scale. A 64-bin lookup maps that scale to lower and upper half-widths. Fixed persistence-aware adjustments are then applied.
3. SPSR moves only the interval centre, by `U[t, c] · (member0 − member1)`. It changes neither the point forecast nor the interval width.

Every array is fitted from out-of-fold predictions of models trained in the same run.

## 6. Inference
- Track 1: `predict` accepts any batch size and uses only the window it is given; there is no state across calls.
  - It runs the two members and the heads on CUDA, with half-precision member weights where supported.
  - It applies the regime anchor (picked from the current window's block-mean flow), the KE-preserving temporal gain, and the interval construction above.
  - It returns physical float32 arrays.
  - CUDA graphs and a side stream are captured on first use.

## 7. Speed
- Member weights in half precision where supported.
- A cached CUDA graph for the fixed-shape forward pass, with a side stream.
- These runtime paths are unchanged from the submitted card. Row 948018 scored 92.93 on the Time sub-score.

## 8. What mattered
Effect sizes are online Final deltas between our own consecutive Development-phase uploads, rounded. One upload's Time draw moves Final by about ±0.014. Items marked *approximate* changed together with other parts, or their attribution rests on our notes rather than a clean pair.

Overall, Final rose from 79.64 (our first upload under the current scoring, 2026-08-05) to 82.08 (row 948018). By leg, about +1.8 of the +2.4 came from SPS, +0.25 from rel, +0.25 from Time and +0.15 from TKE and MVPE.

1. **Learned spatial interval head (sband).** About +0.23, through SPS (*approximate*). As a contrast, widening the band uniformly by ×1.225 cost −0.26.
2. **Two-member blend of the ConvNeXt and self-warp U-Nets.** About +0.16 to +0.19, through rel and SPS (*approximate*). Distilling the two members from teachers added about +0.04 to +0.06.
3. **Input-conditioned anchoring.** Kinetic-energy and mean maps for three input regimes, fitted out of fold, plus band adjustments. About +0.09, mostly TKE (*approximate*; bundled).
4. **Retraining the members with the current recipes.** About +0.08 to +0.11.
5. **Training the shipped members on all 81 files.** About +0.03 in the accuracy legs.
6. **Interval-centre shift from member disagreement (SPSR) and the retrained interval head (SBAND2).** About +0.03, all through SPS. The point forecast was unchanged.
7. **Runtime: half-precision weights and cached CUDA graphs.** About +0.02, through Time only, with the accuracy legs bit-identical. Within about 1.5 Time draws, so uncertain.

## 9. Compute
- Development phase: separate jobs on NVIDIA V100 32 GB and RTX 4090 24 GB servers; there is no single wall-clock time.
- This package, one from-scratch `train()` on 1 × A100-SXM4-80GB, 16 CPU cores and 128 GB, with MPS: 19.7 h; peak GPU memory 14,798 MiB; container RAM peak 52 GB.

## 10. References
- O. Ronneberger, P. Fischer, T. Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015.
- Z. Liu et al. A ConvNet for the 2020s (ConvNeXt). CVPR 2022.
- N. Shazeer. GLU Variants Improve Transformer (GEGLU). arXiv:2002.05202, 2020.
- G. Hinton, O. Vinyals, J. Dean. Distilling the Knowledge in a Neural Network. arXiv:1503.02531, 2015.
- I. Loshchilov, F. Hutter. Decoupled Weight Decay Regularization (AdamW). ICLR 2019.
- L. N. Smith, N. Topin. Super-Convergence (OneCycle). 2019.
- R. Koenker, G. Bassett. Regression Quantiles (pinball loss). Econometrica 1978.
- Libraries: PyTorch, NumPy, h5py, Loguru, Matplotlib, PyYAML.
