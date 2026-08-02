## [2026-07-31] — Repo cleanup (logs / dead helpers)

### Done
- Deleted root training/maintenance `*.log` (~133 MB) and `_free_space_and_restore_tb.ps1`
- Cleared `__pycache__/`; `.gitignore` now ignores `*.log`
- Removed unused `format_recon_comparison` and `_purge_scalar_tags` from `train_autoencoder.py`
- Left intact: `checkpoints/`, `runs/`, `data/`, `.venv/`, paper PNGs, `tiny_imagenet_final_results.csv`

### Remaining
- Optional later: prune old checkpoint/run dirs (e.g. `tiny_imagenet_ae_layer_adapt_300`) — ask before wipe

### Notes for next agent
- Core entrypoint: `train_autoencoder.py`; map: `ARCHITECTURE.md`
- Comparison artifacts at repo root: `recon_compare_full.png`, `recon_compare_l1.png`, `arch_ae_vs_adapt.png`
- Canvas: `C:\Users\shing\.cursor\projects\d-PycharmProjects-Peter-training\canvases\ae-recon-compare.canvas.tsx`
- Epoch-500 metrics (from checkpoints):
  | Model | Path | Train | Val MSE | Val acc (%) |
  |-------|------|-------|---------|-------------|
  | ae_recon | full | 0.295188 | 0.292425 | 57.639 |
  | ae_layer_adapt_500 | full | 0.311378 | 0.290138 | 58.341 |
  | ae_layer_adapt_500 | trim2 | 0.311378 | 0.336187 | 49.303 |
  | ae_recon_l1 | trim2 | 0.332493 | 0.346959 | 45.632 |
- `ae_recon_l1` finished through epoch 500 (`checkpoints/tiny_imagenet_ae_recon_l1/epoch_500.pt`)

## [2026-07-18] — TensorBoard restart after metadata repair

### Done
- Killed stale TB listeners (pids 16668 / 39608 / 46264) on port 6006
- Restarted TensorBoard 2.20.0:
```powershell
.\.venv\Scripts\tensorboard.exe --logdir runs --host 127.0.0.1 --port 6006 --reload_interval 5 --reload_multifile true
```
- Listening at http://127.0.0.1:6006/ (python listener pid 27304)

### Remaining
- Hard-refresh browser tab on TB UI to drop cached legacy cards / wrong timestamps
- Optional: prune adapt_300 checkpoints to interval 20

### Notes for next agent
- Prefer `--reload_multifile true` after event-file rewrites so TB picks up rewritten files

## [2026-07-18] — TB metadata.data_class repair (adapt_300)

### Done
- Added `_repair_summary_data_class`, `_expected_summary_data_class`, `repair_tensorboard_metadata()`, and `--mode repair_tb_metadata` in `train_autoencoder.py`
- `rewrite_tensorboard_scalars` now always repairs metadata after rename (or repair-only when rename map empty); docstrings warn not to run while SummaryWriter is open
- Ran on finished `tiny_imagenet_ae_layer_adapt_300`:
```powershell
$env:PYTHONPATH = ".\.venv\Lib\site-packages"; py -3.11 train_autoencoder.py --mode rewrite_tb_scalars --run-name tiny_imagenet_ae_layer_adapt_300
```
- **15** summary values repaired (`data_class` 0 → 1 for scalars, 0 → 3 for recon images); 0 renames (no legacy tags left)
- EventFileLoader audit: **pass** — `1_Loss/train_pixel`, `Schedule/LR`, `Schedule/Alpha` all `data_class=1`; recon tags `data_class=3`; zero `data_class=0` in run; train_pixel count=60 steps 5..300; wall times monotonic
- Log: `repair_adapt300_tb.log`

### Remaining
- (TB restart done — see section above)
- Optional: prune adapt_300 checkpoints to interval 20

### Notes for next agent
- `.venv\Scripts\python.exe` may be AV-blocked on this machine; use `py -3.11` with `PYTHONPATH=.\.venv\Lib\site-packages` or `pythonw.exe`
- Repair alone: `--mode repair_tb_metadata [--run-name NAME]`

## [2026-07-18] — adapt_300 finished; final TB path-stem rewrite

### Done
- `tiny_imagenet_ae_layer_adapt_300` training completed (epoch 300/300); processes exited
- Re-ran scalar rewrite after train end (old in-memory writers had re-added legacy tags at epochs 295/300):
```powershell
.\.venv\Scripts\python.exe train_autoencoder.py --mode rewrite_tb_scalars --run-name tiny_imagenet_ae_layer_adapt_300
```
- Renamed **12** leftover scalar values (`val_recon_full`/`trim*` + `2_Accuracy/val_full`/`val_trim*` → six path stems)
- EventFileLoader audit: **ALL 7 runs CLEAN** — no `val_full` / `val_bookends` / `val_recon_trim*` / `val_trim*` tags remain
- Log: `rewrite_adapt300_scalars.log`

### Remaining
- Hard-refresh TensorBoard UI (or restart TB) so the sidebar drops cached legacy cards
- Optional: prune adapt_300 checkpoints to interval 20; `restore_tb_recon` if image backfill still needed

### Notes for next agent
- Path-stem cards under `1_Loss` / `2_Accuracy`: `val_recon_{1,2,3}layer{,_latent}` and matching `val_acc_*`
- Live logging already writes path stems; rewrite only needed if a long-running process started before the rename

## [2026-07-18] — TB historical scalar tag rewrite

### Done
- Added `_rename_summary_tags`, per-family `TB_SCALAR_RENAME_BY_RUN`, `rewrite_tensorboard_scalars()`, and `--mode rewrite_tb_scalars` in `train_autoencoder.py`
- Updated `_max_epoch_from_run` probe list with path-stem tags (`val_recon_3layer_latent`, etc.)
- Ran full rewrite (not dry-run) for all 7 runs under `runs/`:
```powershell
.\.venv\Scripts\python.exe train_autoencoder.py --mode rewrite_tb_scalars 2>&1 | Tee-Object -FilePath rewrite_tb_scalars.log
```
- **10,262** scalar values renamed across: `tiny_imagenet_ae_recon` (4000), `tiny_imagenet_ae_layer_500` (4000), `tiny_imagenet_ae_recon_l1` (820), `tiny_imagenet_ae_layer_adapt_500` (300), `tiny_imagenet_ae_layer_adapt_300` (342), `flowers102_glue_layer_latent` (400), `flowers102_glue_recon_latent` (400)
- EventFileLoader verification: finished runs have **ZERO** legacy path tags; new path-stem tags present with expected counts
- Log: `rewrite_tb_scalars.log`
- **Follow-up complete:** adapt_300 finished and was re-rewritten (see section above)

### Remaining
- (cleared by follow-up rewrite)

### Notes for next agent
- Re-migrate a single run: `--mode rewrite_tb_scalars --run-name <name>`; preview with `--dry-run`
- `tiny_imagenet_ae_layer_adapt2` has no `runs/` dir yet — skipped automatically

## [2026-07-18] — TB path stems + freeze random recon

### Done
- Renamed TensorBoard **path** scalar tags to six stems under existing `1_Loss` / `2_Accuracy` (internal dict keys / `decode_trim*` unchanged):
  - ConvAE: full→`3layer_latent`, trim1→`3layer`, trim2→`2layer`, trim3→`1layer`
  - LayerAdapt: full→`3layer_latent`, trim1→`2layer_latent`, trim2→`1layer_latent`
  - layer1→`1layer_latent`; glue full→`3layer_latent`, bookends→`1layer`
- Constants: `RECON_RANDOM_SEED=0`, `TB_VAL_RECON_*` / `TB_VAL_ACC_*` in `train_autoencoder.py`
- `4_Reconstruction_Random` now always uses `seed=RECON_RANDOM_SEED` (same images every epoch)
- Updated `ARCHITECTURE.md` mapping tables + metric defs

### Remaining / Blocked
- **Could not run** `restore_tb_recon` or refresh TensorBoard: agent Shell/Task fail with `SQLITE_FULL` (Cursor DB / disk)
- After freeing space, run:
```powershell
cd d:\PycharmProjects\Peter_training
.\.venv\Scripts\python.exe train_autoencoder.py --mode restore_tb_recon --log-images-every 5 2>&1 | Tee-Object -FilePath restore_tb_recon_pathstem.log
# TB: http://127.0.0.1:6006/ — hard-refresh UI
```
- Historical scalar tags (`val_recon_trim*`, `val_bookends`) remain in old event files until wipe/retrain; new logging uses path stems only
- Optional: delete accidental Tiny ImageNet copy if still present: `C:\Users\shing\.cursor\projects\d-PycharmProjects-Peter-training\terminals\data\`

### Notes for next agent
- New trains write path-stem scalars automatically; restore only rewrites recon **images** with fixed stratified seed

## [2026-07-18] — Train tiny_imagenet_ae_layer_adapt_300

### Done
- Registered `LAYER_ADAPT_300_RUN_NAME` / `ae_layer_adapt_300` in `ACTIVE_RUNS` + `RECON_TAG_BY_RUN` (+ restore_tb_recon LayerAdapt branch); ARCHITECTURE.md run table row
- Started fresh train (same recipe as adapt_500, 300 epochs):
  `--mode layer_adapt --run-name tiny_imagenet_ae_layer_adapt_300 --epochs 300 --alpha-schedule increasing --alpha-start 0 --alpha-end 1 --batch-size 128 --lr 0.01 --checkpoint-every 5 --log-images-every 5`
- Confirmed startup: cuda, alpha 0→1 over 300 epochs, dirs `runs/` + `checkpoints/tiny_imagenet_ae_layer_adapt_300`
- Log: `d:\PycharmProjects\Peter_training\layer_adapt_300.log`

### Remaining
- Train still running (~epoch 97+/300 as of last check; log `layer_adapt_300.log`); monitor to 300
- On completion optionally prune checkpoints to interval 20 + `restore_tb_recon`

### Notes for next agent
- `default_run_name('layer_adapt')` still → `tiny_imagenet_ae_layer_adapt_500`; adapt_300 always needs explicit `--run-name`
- Resume: `--resume checkpoints\tiny_imagenet_ae_layer_adapt_300\latest.pt` with `--start-epoch` / alpha continuity matching ckpt epoch (`$alphaStart = <epoch>/299`)
- Shell tool briefly flaky mid-run; use log + terminal `426321` / process cmdline `*adapt_300*` to monitor

## [2026-07-18] — Delete layer_300; rename layer_adapt → layer_adapt_500

### Done
- Deleted `checkpoints/tiny_imagenet_ae_layer_300/` and `runs/tiny_imagenet_ae_layer_300/`
- Renamed `checkpoints/tiny_imagenet_ae_layer_adapt` → `tiny_imagenet_ae_layer_adapt_500` (same for `runs/`)
- Updated `train_autoencoder.py`: removed `LAYER_300_*` constants; `LAYER_ADAPT_RUN_NAME` = `tiny_imagenet_ae_layer_adapt_500`, `LAYER_ADAPT_RECON_TAG` = `ae_layer_adapt_500` (CLI mode `layer_adapt` unchanged)
- Updated `ARCHITECTURE.md` run table; `tiny_imagenet_final_results.csv` rows `ae_layer_adapt` → `ae_layer_adapt_500`
- Deleted artifact `layer_300_resume.log`
- Ran `restore_tb_recon` — see `restore_tb_recon.log`

### Remaining
- Hard-refresh TensorBoard UI to pick up new recon tags (`ae_layer_adapt_500`)
- Historical scalar tags in event files may still say `ae_layer_adapt` — intentional (loss curves preserved)

### Notes for next agent
- `layer_300` run removed entirely; do not resume
- Accidental Tiny ImageNet copy still at `C:\Users\shing\.cursor\projects\d-PycharmProjects-Peter-training\terminals\data\` (~313 MB) — safe to delete

## [2026-07-18] — TensorBoard recon restore

### Done
- Ran `train_autoencoder.py --mode restore_tb_recon --log-images-every 5` (exit 0, ~293s)
- Log: `d:\PycharmProjects\Peter_training\restore_tb_recon.log`
- Backfilled recon images for: recon (20..500), layer_500 (20..500), layer_300 (5..65), recon_l1 (10..300), layer_adapt (20..500), flowers glue layer/recon (20..100)
- Skipped `tiny_imagenet_ae_layer_adapt2` (no checkpoints)
- Verified each active run has `3_Reconstruction` + `4_Reconstruction_Random` tags
- TB still listening on http://127.0.0.1:6006/ (pid 16668)

### Remaining
- Hard-refresh TensorBoard UI to pick up new event files
- `layer_300` train died mid-epoch (exit `4294967295`); `latest.pt` still epoch **65** — resume with command in section below if desired

### Notes for next agent
- Accidental Tiny ImageNet copy still at `C:\Users\shing\.cursor\projects\d-PycharmProjects-Peter-training\terminals\data\` (~313 MB) — safe to delete

## [2026-07-18] — BLOCKED: SQLITE_FULL while resuming layer_300

### Done
- Disk D: had ~75 GB free; `latest.pt` at epoch **65**
- Started resume with corrected alpha continuity: `--start-epoch 65 --alpha-start 65/299≈0.2174 --alpha-end 1` (naive resume would reset alpha→0)
- Confirmed startup: `Resumed ... at epoch 65`, alpha schedule `0.2174 → 1` over epochs 66–300
- Log: `d:\PycharmProjects\Peter_training\layer_300_resume.log`
- Killed first bad resume (alpha reset to 0)

### Remaining / Blocked
- **SQLITE_FULL** broke Cursor shells/tools mid-epoch-1 — cannot monitor or free space from agent
- Accidental Tiny ImageNet copy under Cursor terminals (from first launch with wrong cwd) — **delete this**:
  - `C:\Users\shing\.cursor\projects\d-PycharmProjects-Peter-training\terminals\data\`
- Then verify train still running; if dead, resume with command below

### Resume command (correct alpha)

```powershell
cd d:\PycharmProjects\Peter_training
$alphaStart = 65/299
.\.venv\Scripts\python.exe train_autoencoder.py --mode layer --dataset tiny_imagenet --run-name tiny_imagenet_ae_layer_300 --epochs 300 --alpha-schedule increasing --alpha-start $alphaStart --alpha-end 1 --start-epoch 65 --bottleneck-size 8 --batch-size 128 --lr 0.01 --checkpoint-every 5 --log-images-every 5 --resume checkpoints\tiny_imagenet_ae_layer_300\latest.pt
```

If `latest.pt` advanced past 65, set `--start-epoch` and `$alphaStart = <epoch>/299` to match `latest.pt` epoch.

### Notes for next agent
- Do **not** resume with `--alpha-start 0` after mid-run — resume overwrites `args.start_epoch` from ckpt and restarts the alpha ramp
- First failed launch (exit 5) used cwd=`terminals/` and may have written `terminals\data\tiny-imagenet-200`

## [2026-07-18] — Truncate ae_recon_l1 after epoch 300

### Done
- Deleted `checkpoints/tiny_imagenet_ae_recon_l1/epoch_{310..370}.pt` (7 files)
- Reset `latest.pt` from `epoch_300.pt` (epoch **300**)
- Rewrote TB event file `events.out.tfevents.1784293541.*` — dropped 324 summary events with step > 300; max step now **300**

### Remaining
- Refresh TensorBoard UI if curves still show >300 (reload / hard refresh)

### Notes for next agent
- `tiny_imagenet_ae_recon_l1` is capped at epoch 300 for both ckpts and TB
