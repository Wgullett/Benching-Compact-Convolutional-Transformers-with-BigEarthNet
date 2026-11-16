# Original 2024 work

Kept for reference and for comparison with the corrected pipeline in the root of this repository.

| File | What it is |
|---|---|
| `MRP_2024_original.pdf` | The Major Research Project as submitted in 2024 |
| `CCT_Experiment_1.ipynb` | CCT-2 (2 layers, 2 heads, dim 128), Keras, 32×32 random crops, 30 epochs |
| `CCT_Experiment_2.ipynb` | CCT-6 (6 layers, 4 heads, dim 256), full 120×120 patches, run in 1-epoch chunks |
| `README_2024.md` | The previous repository README, preserved unchanged |

**The results in these files and in `README_2024.md` are not valid.** Both notebooks build the test set
with `shuffle=True` and then score `model.predict()` output against labels in their original order, so
every prediction was compared with the labels of a different image. Both also carry a stray quotation
mark in the checkpoint path (`"'/content/drive/...`), so the best weights were never reloaded before
evaluation.

`README_2024.md` reads the per-label precision list as evidence that class imbalance drove the low
scores. Those precision values are almost exactly each class's frequency in the dataset
(correlation 0.999), which is what chance-level scoring produces regardless of what the model learned.
Class imbalance is real and does affect the rare classes, but it is not what produced those numbers.

Section 9 of `CCT_BigEarthNet_v2.ipynb` reproduces the bug on a working model as a control.
