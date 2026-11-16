# Benchmarking Compact Convolutional Transformers with BigEarthNet

A Compact Convolutional Transformer (CCT) trained from scratch for multi-label land use / land cover
classification on BigEarthNet-19, reaching **78.4% test micro-F1 with 3.94M parameters in 38.5 minutes**
on a single A100 GPU.

This repository also documents a correction. The 2024 version of this project (an M.Sc. Major Research
Project at Toronto Metropolitan University) reported a micro-F1 near 18% and concluded that CCTs were not
viable for this task. That number came from an evaluation bug, not from the models. The notebook here
re-runs the benchmark with a corrected pipeline and reproduces the bug deliberately as a control.

## Results

CCT-7, trained from scratch on the RGB rendering of BigEarthNet-S2 v1.0, 30 epochs, test split of
125,866 patches.

| Decision rule | Precision | Recall | Micro-F1 | Accuracy¹ | Macro-F1 | mAP |
|---|---|---|---|---|---|---|
| Threshold 0.5 | 80.8 | 76.1 | **78.4** | 64.5 | 72.4 | 79.5 |
| Per-class thresholds (tuned on validation) | 75.9 | 81.8 | 78.7 | 64.9 | 73.2 | 79.5 |

¹ Micro-averaged Jaccard index, TP / (TP + FP + FN) — the quantity reported in the accuracy column of
Papoutsis et al.

### Against the published benchmark

| Model | Source | Params | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| **CCT-7** | **This work, RGB** | **3.9M** | **64.5** | **80.8** | **76.1** | **78.4** |
| WRN-B5-ECA | Papoutsis et al. | 5.2M | – | 82.0 | 76.1 | 79.0 |
| WRN-B4-ECA | Papoutsis et al. | 1.0M | – | 82.4 | 75.5 | 78.8 |
| VGG19 | Papoutsis et al. | 20M | 63.6 | 81.7 | 74.2 | 77.7 |
| ResNet50 | Papoutsis et al. | 23M | 62.4 | 78.8 | 75.0 | 76.8 |
| ViT/20 | Papoutsis et al. | 7M | 62.1 | 80.5 | 73.1 | 76.6 |
| K-Branch CNN | Papoutsis et al. | 36M | 47.3 | 63.4 | 65.1 | 64.2 |
| CCT-2, 2024 version (mis-scored) | this project, 2024 | 0.4M | – | 25.1 | 13.5 | 17.6 |

**This comparison is indicative, not like-for-like.** The benchmark models were trained on ~10 Sentinel-2
spectral bands and the full ~590k-patch archive; this work uses 3-band RGB and the 519,284 patches that
remain after cloud- and snow-affected patches are removed. Near-infrared and short-wave infrared carry
much of the signal for vegetation and water, so the RGB model works with less information; the easier
patch set works in its favour. Training a convolutional baseline inside this same pipeline is the next
step — the notebook has a `resnet18` preset for exactly that.

Per-class results are in [`results/per_class.csv`](results/per_class.csv). Best and worst classes:

| Class | Test patches | Precision | Recall | F1 |
|---|---|---|---|---|
| Marine waters | 18,023 | 98.4 | 98.0 | 98.2 |
| Coniferous forest | 39,532 | 87.4 | 85.5 | 86.4 |
| Arable land | 47,150 | 86.7 | 85.1 | 85.9 |
| … | | | | |
| Natural grassland / sparse | 2,799 | 72.3 | 43.5 | 54.3 |
| Industrial or commercial | 2,808 | 65.9 | 45.2 | 53.6 |

## What went wrong in 2024

| # | Problem | Effect |
|---|---|---|
| 1 | Test set built with `shuffle=True`, then predictions scored against `ds["test"]["labels"]` in the original order | Every prediction was compared with the labels of a **different image**. Per-class precision equalled class prevalence (r = 0.999) — the signature of chance-level scoring. |
| 2 | "Accuracy" was Keras `CategoricalAccuracy` | Not a multi-label metric, and not the accuracy of the benchmark paper. It also selected the checkpoint. |
| 3 | `RandomCrop(32, 32)` on 120×120 multi-label patches | The model saw 7% of each patch while being asked for the labels of the whole patch. |
| 4 | PNGs decoded in Python every step; 900-token sequences | ~5 h/epoch, which forced the 32px crop and a 4-epoch run. |
| 5 | Stray quote in the checkpoint path | The "best" weights were never reloaded. |
| 6 | Label smoothing on a 19-way multi-label BCE; constant LR | Smaller effects, all in the wrong direction. |

Two sets of 2024 numbers exist: the MRP paper reports micro-F1 of 17.6% (CCT-2) and 18.0% (CCT-4), while
the previous README of this repository reports 24.8% and 26.0% from later runs. Both sets come from the
same mis-scored evaluation, so neither measures the models.

**The control experiment.** Section 9 of the notebook takes the *working* CCT-7, shuffles its test
predictions, and scores them against labels in the original order:

```
aligned  : micro-F1 0.784  precision 0.808  recall 0.761
shuffled : micro-F1 0.245  precision 0.253  recall 0.238
correlation between shuffled per-class precision and class prevalence: 1.000
```

The 2024 full-patch run reported precision 26.0 / recall 22.8 / F1 24.3 — almost exactly what a working
model scores under shuffled labels. The 2024 models were never scored correctly, so their true
performance is unknown.

## What the corrected pipeline does

- **Decodes the dataset once** into a `uint8` array on local disk and holds the training split in GPU
  memory, so a batch is an index operation rather than 256 PNG decodes. ~1.3 min/epoch.
- **Trains on the full 120×120 patch** with a tokenizer sized for that resolution. The CIFAR tokenizer
  from the CCT paper gives 900 tokens at this input size; one extra conv block gives 225, cutting
  attention cost ~16×.
- **Label-preserving augmentation only** — flips and 90° rotations. No random cropping: the labels
  describe the whole 1.2 km × 1.2 km patch.
- **Evaluates in a single ordered pass**, with predictions and labels collected from the same batches,
  and reports micro-F1, macro-F1, micro-Jaccard and mAP.
- **Resumes after a Colab disconnect** from the last finished epoch.
- **Checks itself before spending GPU hours** by memorising a single batch.

## Repository layout

```
CCT_BigEarthNet_v2.ipynb     corrected pipeline, with the run outputs
paper/                       the rewritten paper (LaTeX source + PDF + figures)
results/                     per-class metrics and the run summary as CSV/JSON
original-2024/               the 2024 paper, notebooks and README, kept for reference
```

## Running it

Open `CCT_BigEarthNet_v2.ipynb` in Google Colab, choose a GPU runtime (A100 or L4 preferred; T4 works),
and *Run all*. The configuration cell at the top controls everything:

```python
MODEL      = "cct7"   # "cct2" | "cct4" | "cct7" | "resnet18"
EPOCHS     = 30
BATCH_SIZE = 256
```

Each model writes to its own folder and the final section tabulates every run found there. The first run
decodes the dataset (~22 GB on local disk, a few minutes); later runs reuse it.

A Hugging Face token is needed to download the dataset. Store it as a Colab secret named `HF_TOKEN`
(key icon in the left sidebar) — never paste it into a cell.

## Data

[`danielz01/BigEarthNet-S2-v1.0`](https://huggingface.co/datasets/danielz01/BigEarthNet-S2-v1.0) on the
Hugging Face Hub: 519,284 Sentinel-2 patches of 120×120 pixels at 10 m resolution, RGB, with cloud- and
snow-affected patches removed. The 43 CORINE classes are mapped to the 19-class nomenclature in the
notebook. Patches average 2.9 labels, and the classes are heavily imbalanced — *Arable land* appears in
37.2% of training patches, *Coastal wetlands* in 0.4%.

## Paper

[`paper/cct_bigearthnet_paper.pdf`](paper/cct_bigearthnet_paper.pdf) — the 2024 Major Research Project,
rewritten around these results. LaTeX source and figures are beside it.

## References

1. I. Papoutsis, N. I. Bountos, A. Zavras, D. Michail, C. Tryfonopoulos, "Benchmarking and scaling of deep
   learning models for land cover image classification," *ISPRS Journal of Photogrammetry and Remote
   Sensing*, vol. 195, pp. 250–268, 2023. [doi:10.1016/j.isprsjprs.2022.11.012](https://doi.org/10.1016/j.isprsjprs.2022.11.012)
2. A. Hassani et al., "Escaping the big data paradigm with compact transformers,"
   [arXiv:2104.05704](https://arxiv.org/abs/2104.05704), 2021.
3. G. Sumbul, M. Charfuelan, B. Demir, V. Markl, "BigEarthNet: A large-scale benchmark archive for remote
   sensing image understanding," *IGARSS*, 2019. [doi:10.1109/igarss.2019.8900532](https://doi.org/10.1109/igarss.2019.8900532)

## License

MIT — see [LICENSE](LICENSE).
