# IN-CXR Membership Inference with DenseNet-121 and Offline RMIA

An end-to-end, Windows-oriented PyTorch experiment for studying **membership inference attacks (MIA)** on the IN-CXR chest X-ray dataset. The pipeline prepares a fixed image-level candidate population, trains a DenseNet-121 target classifier and 16 reference classifiers, generates augmented inference logits, and evaluates **offline Robust Membership Inference Attack (RMIA)** scores.

> **Research implementation / reproduction note:** This repository adapts RMIA to a binary IN-CXR classification task. It is not an official release by the RMIA authors, and it should not be described as an independently verified exact reproduction of the paper. Results depend on the cohort, checkpoints, preprocessing, randomness, and scoring semantics.

## Repository files

| File | Purpose |
| --- | --- |
| `run_incxr_end_to_end.py` | Windows entry point and orchestration; skips existing complete artifacts where supported |
| `prepare_incxr.py` | Builds the candidate manifest, binary labels and reference/target membership masks |
| `incxr_densenet121_lossmia.py` | Trains the target DenseNet-121 and separately reports a loss-based MIA diagnostic |
| `incxr_densenet121_rmia_fixed.py` | Trains references, generates logits, and runs the offline RMIA attack |

**Not required by this runner:** the separate online RMIA, WRN, ResNet-50, and alternative scoring scripts from other experiments.

## Requirements

- Windows 10/11 with PowerShell and a supported Python 3 installation
- PyTorch and torchvision installed as a **compatible pair** (CUDA-enabled builds recommended for full training)
- `numpy`, `scipy`, `scikit-learn`, `Pillow`
- The IN-CXR image files organized into `Normal` and `AbNormal` folders
- Adequate GPU memory, system RAM, storage, and compute time for **17 DenseNet-121 training runs** and 18-query inference over all candidates

Install the remaining Python packages after installing a compatible PyTorch/torchvision pair from the [official PyTorch installation guide](https://pytorch.org/get-started/locally/):

```powershell
python -m pip install numpy scipy scikit-learn pillow
python -c "import torch, torchvision; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

CUDA installation instructions vary by GPU, driver, and PyTorch release; do not assume a single CUDA wheel works on every Windows machine.

## Dataset and directory layout

By default, the runner uses `D:\RMIA_IN_CXR` as its project directory:

```text
D:\RMIA_IN_CXR\
|-- run_incxr_end_to_end.py
|-- prepare_incxr.py
|-- incxr_densenet121_lossmia.py
|-- incxr_densenet121_rmia_fixed.py
|-- data\
|   `-- IN-CXR\
|       |-- Normal\
|       `-- AbNormal\
|-- exp\
|   `-- incxr_16\                 # generated split artifacts
|-- results\
|   `-- incxr_densenet121\       # generated checkpoints and attack results
`-- run_incxr_full.log           # generated execution log
```

Images are **not** downloaded automatically. Supply the dataset yourself and comply with its distribution and usage conditions. The original preparation script expects images immediately within the two class folders, accepting `.png`, `.jpg`, and `.jpeg` extensions. Its expected cohort size for this run is **12,600 images**; if the local dataset differs, the master runner's fixed-cohort validation will stop.

A separate image location can be passed with `--data-root`; on Windows, the runner attempts to create a directory junction under `<project>\data\IN-CXR` when necessary. The simplest supported arrangement is to put images at the default path.

## One-command Windows execution

From PowerShell, in the repository directory:

```powershell
cd D:\RMIA_IN_CXR

python -u .\run_incxr_end_to_end.py `
  --project "D:\RMIA_IN_CXR" `
  --device cuda `
  --workers 0 `
  --infer-batch-size 32 `
  --x-chunk 64
```

For a machine without CUDA, use `--device cpu` (full training will be much slower). Use `--no-amp` to disable mixed precision. The runner uses `--target-batch-size 64` by default.

### Useful runner options

| Option | Default | Meaning |
| --- | --- | --- |
| `--project` | `D:\RMIA_IN_CXR` | Working directory holding `data`, `exp`, and `results` |
| `--data-root` | `<project>/data/IN-CXR` | Input directory immediately containing the class folders |
| `--device` | `auto` | `auto`, `cuda`, or `cpu`; used for reference training, inference, and attack |
| `--workers` | `0` | DataLoader workers; `0` is a conservative Windows default |
| `--target-batch-size` | `64` | Physical training batch size for target model |
| `--infer-batch-size` | `32` | Batch size during augmented inference |
| `--x-chunk` | `64` | Candidate chunk size during pairwise attack scoring |
| `--no-amp` | off | Disable automatic mixed precision |
| `--regenerate-splits` | off | Generate a *new* split; refuses if known model/logit locations already exist |

**Important:** `--device cpu` does not itself force the target-training subprocess to use CPU; that script selects CUDA automatically when available. For exact device control in all stages, inspect the underlying scripts before running.

## Individual stages / recovery

The offline RMIA implementation also supports direct stage execution:

```powershell
# Validate dataset artifacts and existing target checkpoint
python -u .\incxr_densenet121_rmia_fixed.py --project "D:\RMIA_IN_CXR" --stage preflight

# Train or resume reference models
python -u .\incxr_densenet121_rmia_fixed.py --project "D:\RMIA_IN_CXR" --stage train-refs --device cuda --workers 0

# Perform inference and attack using existing checkpoints
python -u .\incxr_densenet121_rmia_fixed.py --project "D:\RMIA_IN_CXR" --stage infer-attack --device cuda --attack-device cuda --infer-batch-size 32 --workers 0 --x-chunk 64

# Recompute just the attack from existing logits
python -u .\incxr_densenet121_rmia_fixed.py --project "D:\RMIA_IN_CXR" --stage attack --attack-device cuda --x-chunk 64
```

Additional supported stages include `infer` and `all`. The reference script also supports `--ref-start`, `--ref-end`, `--restart-ref`, and `--overwrite-logits`; consult its `--help` output. Avoid destructive flags on historical experiment directories.

## Experimental configuration

| Parameter | Setting |
| --- | --- |
| Task | Binary classification (`Normal=0`, `AbNormal=1`) |
| Model | torchvision DenseNet-121, `weights=None`, two-output classifier |
| Candidate count | 12,600 |
| Target member/nonmember count | 6,300 / 6,300 |
| Reference models | 16 |
| Reference assignment | 8 IN and 8 OUT models per candidate |
| Training epochs | 100 |
| Target batch size | 64 by default |
| Optimizer | SGD, learning rate 0.1, momentum 0.9 |
| Weight decay | `5e-4`, applied to parameters with more than one dimension |
| Learning-rate schedule | Cosine annealing |
| Training augmentation | Resize to 224×224; reflect pad 28; random crop; random horizontal flip; tensor conversion |
| Normalization | No ImageNet normalization |
| Inference queries | 18 = 3×3 fixed crops × 2 flip states |
| Reference probability estimate | 20% trimmed mean of eligible OUT-reference confidence signals |
| Attack parameters | Temperature 2.0; soft margin 0.6; offline `a=0.3`; `gamma=2.0` |
| Pairwise vote | Strict likelihood ratio `> gamma` and strict majority `>9/18` |
| Comparison population | Target nonmembers, with self-comparison denominator adjustment |

### Taylor approximation: crucial compatibility detail

This version of `incxr_densenet121_rmia_fixed.py` intentionally uses **released-code-style** Taylor semantics:

```python
for i in range(2, n):
    ...
```

With `n=4`, it includes terms through degree 3, **not** the degree-4 term. This differs from the mathematical fourth-order polynomial `sum(i=0..4) x**i / i!`, which some later experiment variants use. Do **not** silently change this loop when comparing results from the released-style configuration. Record the precise scoring semantics alongside any published number.

## Generated artifacts

```text
exp/incxr_16/
|-- candidates.csv
|-- labels.npy
|-- keep.npy                  # (16, 12600) reference membership bank
`-- target_keep.npy           # (12600,) target membership mask

results/incxr_densenet121/
|-- target/
|   |-- target_densenet121.pt
|   `-- training_history.json
|-- loss_mia/                  # separate target loss-MIA diagnostic
`-- rmia/
    |-- models/
    |   `-- ref_00.pt ... ref_15.pt
    |-- training_logs/
    |-- inference/
    |   |-- target_logits.npy
    |   `-- ref_00_logits.npy ... ref_15_logits.npy
    `-- attack/
        |-- rmia_metrics.json
        |-- rmia_score.npy
        |-- fpr.npy
        |-- tpr.npy
        |-- ratio.npy
        `-- pr_out.npy
```

Each model's logits are expected to have shape `(12600, 18, 2)`.


## Reproducibility and research cautions

1. **Preserve the split files.** Existing checkpoints and logits are tied to the exact ordering in `candidates.csv`, `labels.npy`, `keep.npy`, and `target_keep.npy`. Do not regenerate them over a previous run. Prefer a new project directory for a new cohort or experimental seed.
2. **Image-level, not patient-level separation.** The provided preparation script separates images; it does *not* group by patient identifier. If a patient contributes multiple images, patient information can overlap between member and nonmember cohorts. Claims about patient-level privacy require an explicitly patient-disjoint protocol.
3. **Artifact existence is not full provenance validation.** The launcher skips target training when the checkpoint file exists. Ensure any reused checkpoint matches the current cohort, preprocessing and training configuration.
4. **The target is not epoch-resumable through the master runner.** Reference training supports resumption; interrupted target training may need to be repeated.
5. **No automatic download or patient metadata validation.** Dataset acquisition, rights, provenance, and patient identifiers are outside the current implementation.
6. **Separate loss MIA from RMIA.** The target-training script computes loss MIA as an additional diagnostic; it is not substituted for the final RMIA score.
7. **Pin your environment when publishing.** Record Python, torch, torchvision, CUDA and GPU versions, plus seeds and file hashes, to help interpret cross-system differences.

## Troubleshooting

**`CUDA available: False`** — Install a suitable CUDA-enabled PyTorch build and confirm driver/GPU compatibility. Do not start a large training run assuming GPU acceleration is active.

**`Expected Normal and AbNormal directories`** — Check dataset layout or pass `--data-root` pointing directly to the parent of those class folders.

**Interrupted reference training** — Run the launcher again, or call `--stage train-refs`. The reference script resumes or skips saved checkpoints when compatible.

**Unexpected AUC** — Check that the correct cohort/membership files, exact checkpoints and inference logits were used, and confirm preprocessing and Taylor semantics before concluding an attack algorithm changed.

## RMIA reference

The underlying method is described in *Low-Cost High-Power Membership Inference Attacks* (2024). Cite the original paper and clearly identify this repository as an IN-CXR adaptation with its own training, preprocessing, and evaluation protocol.



## License and dataset redistribution

**No license is granted by this README.** Add a `LICENSE` file only after selecting a license appropriate for code you own and checking any upstream code obligations. Do not redistribute IN-CXR images or patient-derived artifacts unless their terms permit it.
