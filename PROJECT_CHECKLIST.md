# Project Status Checklist: Hybrid CNN-ViT Bone Fracture Detection

## Core Architectural Principle (Aligned Across Docs)
- [x] Same X-ray input is used for both branches (no separate data requirement).
- [x] CNN branch extracts local fracture cues (edges, micro-discontinuities, texture shifts).
- [x] ViT branch extracts global relational context (alignment, structural consistency).
- [x] Fusion is intended to exploit inductive bias diversity (local + global complementarity).

## Phase 1: Infrastructure & Setup
- [x] Dataset loading pipeline implemented (MURA-compatible path handling + CSV indexing).
- [x] Training/validation/test dataloaders implemented.
- [x] Platform handling for Windows workers integrated (`num_workers=0` fallback).
- [x] Project dependencies captured in `requirements.txt`.

## Phase 2: Model Development
- [x] CNN-only baseline model implemented.
- [x] ViT-only baseline model implemented.
- [x] Hybrid CNN+ViT model implemented with attention and bottleneck projection.
- [x] Micro-hybrid model implemented for low-compute training.
- [x] Backbone freeze and selective fine-tuning controls implemented.

## Phase 3: Preprocessing & Feature Engineering
- [x] Deterministic sanitization + cropping implemented.
- [x] CLAHE local contrast balancing integrated.
- [x] Wavelet high-frequency boosting integrated.
- [x] Frangi ridge enhancement integrated.
- [x] 3-channel structural/frequency/ridge stack integrated into transforms.

## Phase 4: Training & Validation Framework
- [x] BCE-with-logits binary training pipeline implemented.
- [x] Early stopping + checkpoint saving implemented.
- [x] LR scheduler integrated.
- [x] AMP (mixed precision) integrated for CUDA runs.
- [x] Metrics computation/logging integrated.


## Phase 5: Evaluation Artifacts
- [x] ROC generation pipeline implemented.
- [x] Confusion matrix generation pipeline implemented.
- [x] Classification report JSON export implemented.
- [x] Grad-CAM utility implemented for convolutional target layers.
- [ ] ViT-native Grad-CAM/XAI strategy implemented.

## Phase 6: Reporting
- [x] Comparison chart generation implemented.
- [x] Markdown final report generation implemented.
- [x] Pipeline entrypoint supports train -> artifacts -> report flow.

## Phase 7: Run Status (Execution Evidence)
- [x] Feasibility artifacts exist under `outputs/feasibility_test`.
- [x] Test artifacts exist under `test_outputs`.
- [x] Model checkpoints and history files are being produced.
- [ ] Full multi-experiment training run completed (cnn, vit, hybrid, micro) with final comparative metrics.
- [x] Final report regenerated from a fresh full run.
- [x] Test-split evaluation integrated into final reporting workflow.

## Immediate Next Actions
- [x] Run full pipeline on target experiment set and regenerate all artifacts.
- [ ] Validate and document Grad-CAM behavior per model family (CNN/Hybrid/Micro).
- [x] Add test-set metrics into report aggregation for decision-grade comparison.
- [ ] Record final selected model and deployment-ready inference settings.

## Known Limitations / Open Items
- [ ] ViT branch interpretability remains limited with standard conv-layer Grad-CAM hooks.
- [x] Report path supports split-aware metrics aggregation (val/test).
- [ ] Colab full end-to-end execution should be re-verified for latest code state.

## Run Log (Date-Stamped)
Use one line per run to keep experiment history compact and comparable.

Template:

| Date | Mode | Experiments | Epochs | Data Split | Key Artifacts | Best Metric | Notes |
|---|---|---|---:|---|---|---|---|
| YYYY-MM-DD | local/colab | cnn,vit,hybrid,micro | 10/20 | val/test | roc,cm,gradcam,report | acc=0.0000 / f1=0.0000 / auc=0.0000 | short observation |

Latest Entries:

| Date | Mode | Experiments | Epochs | Data Split | Key Artifacts | Best Metric | Notes |
|---|---|---|---:|---|---|---|---|
| 2026-03-13 | local | feasibility | 1 | val | best_model, training_history, sample_heatmap | feasibility run completed | artifacts present in outputs/feasibility_test |
| 2026-03-13 | local | implementation | N/A | val,test | split-aware artifacts + report wiring | code integrated | updated finalize_artifacts/export_report/run_full_pipeline |
| 2026-03-13 | local | cnn,vit,hybrid | 1 (fast plan) | val,test | train+roc+cm+gradcam+report | report generated | outputs/plan_fast_compare/final_report.md |
| 2026-03-13 | local | micro | 1 (fast plan) | val,test | train+roc+cm+gradcam+report | report generated | outputs/micro_pipeline_run/final_report.md |
| 2026-03-13 | local | micro | 10 (full) | val,test | train+roc+cm+gradcam+report | in progress | active run writing to outputs/micro_full_10ep |
| 2026-03-14 | local | cnn,vit,hybrid,micro | 1 (fast) | val,test | roc,cm,gradcam,report | acc=0.5719 (vit) | verified all 4 models in a single combined run |

## Update Rule
- [ ] After each major run, update this file by toggling completed items and appending date-stamped notes if scope changes.
