<div align="center">

# 🔍 A Spatial-Channel Attention Guided Vision Hybrid Approach for Robust Deepfake Forensics

### CNN and Attention-Based Architectures for Deepfake Image Classification

**Exploring convolutional features, channel and spatial attention, and Transformer self-attention for fake-versus-real image classification.**

![Python](https://img.shields.io/badge/Language-Python-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/API-Keras-D00000?logo=keras&logoColor=white)
![Computer Vision](https://img.shields.io/badge/Task-Computer_Vision-2563EB)
![Code Only](https://img.shields.io/badge/Release-Code_Only-64748B)

</div>


---

> ### 💡 TL;DR
>
> TransCNN provides four custom TensorFlow/Keras architectures for binary deepfake image classification: CNN, CNN + CBAM, CNN + Transformer, and CNN + CBAM + Transformer. The repository organizes model definitions, image loading, augmentation, training, evaluation, and visual explanations into reusable Python modules. This is a **code-only release**; datasets, trained weights, historical results, and notebook outputs are not included.

---

## 📖 Overview

<p align="justify">This project explores how convolutional feature extraction can be combined with attention mechanisms to classify fake and real images. A common CNN backbone is used across four model variants, making the placement and role of each attention component easy to inspect.</p>

<p align="justify">CBAM applies channel attention followed by spatial attention to convolutional feature maps. The Transformer variants reshape intermediate feature maps into spatial token sequences and apply multi-head self-attention before returning them to the convolutional pipeline.</p>

The original research was developed and executed in Kaggle notebooks. This repository was organized afterward for accessibility and source-code inspection. **The modular pipeline has not been rerun to reproduce the historical results.**

---

## 🧭 Contents

- [Model variants](#-model-variants)
- [Architecture](#-architecture)
- [Datasets and preprocessing](#-datasets-and-preprocessing)
- [Training configuration](#-training-configuration)
- [Evaluation metrics](#-evaluation-metrics)
- [Visualization and explanations](#-visualization-and-explanations)
- [Repository structure](#-repository-structure)
- [Usage](#-usage)
- [Provenance and validation](#-provenance-and-validation)
- [Status and license](#-status-and-license)
- [Author](#-author)

---

## 🤖 Model Variants

| Model | Implementation | Attention placement |
| --- | --- | --- |
| **CNN** | `models/cnn.py` | Convolutional baseline without attention |
| **CNN + CBAM** | `models/cnn_cbam.py` | CBAM in all four convolutional blocks |
| **CNN + Transformer** | `models/cnn_transformer.py` | Transformer blocks in convolutional blocks 3 and 4 |
| **CNN + CBAM + Transformer** | `models/cnn_cbam_transformer.py` | CBAM in all blocks; Transformer after CBAM in blocks 3 and 4 |

The original model-builder and attention functions are preserved. Each model supports either locally configured dataset; a separate model file is not required for each dataset.

---

## 🧩 Architecture

### Shared CNN backbone

| Stage | Operations |
| --- | --- |
| Block 1 | Conv2D(32, 3×3, ReLU) → BatchNorm → optional attention → MaxPool(2×2) |
| Block 2 | Conv2D(64, 3×3, ReLU) → BatchNorm → optional attention → MaxPool(2×2) |
| Block 3 | Conv2D(128, 3×3, ReLU) → BatchNorm → optional attention → MaxPool(2×2) |
| Block 4 | Conv2D(256, 3×3, ReLU) → BatchNorm → optional attention → MaxPool(2×2) |
| Final convolution | Conv2D(128, 3×3, ReLU), named `gradcam_conv` |
| Classification head | Global average pooling → Dense(128, ReLU) → Dropout(0.4) → Dense(1, sigmoid) |

All backbone convolutions use same padding. Attention is applied before the corresponding max-pooling operation.

### CBAM

The Convolutional Block Attention Module contains:

1. **Channel attention:** global average and max pooling followed by a shared two-layer MLP, addition, and sigmoid gating. The reduction ratio is 8.
2. **Spatial attention:** channel-wise mean and max maps are concatenated and processed by a 7×7 sigmoid convolution.

Each attention map is multiplied into the corresponding features.

### Transformer

Each Transformer block uses four attention heads, pre-layer normalization with `epsilon=1e-6`, residual connections, a feed-forward hidden dimension of 256, and dropout of 0.1. The attention key dimension is `C // 4`, where `C` is the feature-channel count. There is no explicit positional encoding or class token.

With 224×224 inputs, self-attention operates on **56×56** and **28×28** feature maps in blocks 3 and 4. Some original inline comments mention smaller sizes; the implementation itself is preserved.

---

## 📚 Datasets and Preprocessing

The repository uses two local dataset aliases:

- `deepfake_dataset_1`
- `deepfake_dataset_2`

These names identify local folders rather than public dataset titles. Dataset download links and redistribution permissions are not supplied in this release.

Each dataset is expected under `datasets/<dataset_name>/` with the following structure:

| Partition | Fake images | Real images |
| --- | --- | --- |
| Training | `train/fake/` | `train/real/` |
| Validation | `val/fake/` | `val/real/` |
| Test | `test/fake/` | `test/real/` |
| Optional external evaluation | `real_world/fake/` | `real_world/real/` |

**Class mapping:** `fake = 0`, `real = 1`.

The loader consumes existing partitions. It does not create or change train/validation/test splits.

### Image processing

All four models use **224×224 RGB images** and rescaling by `1/255`.

Training augmentation follows the settings present in the original augmented notebook pipeline:

```yaml
augmentation:
  brightness_range: [0.8, 1.2]
  zoom_range: 0.1
  width_shift_range: 0.1
  height_shift_range: 0.1
  horizontal_flip: true
```

Validation, test, and external-evaluation images are resized and rescaled without augmentation. Training and validation generators shuffle; test and external-evaluation generators do not.

---

## 🧪 Training Configuration

The repository configuration is stored in `configs/experiment.yaml`.

| Setting | Value |
| --- | --- |
| Input shape | 224 × 224 × 3 |
| Optimizer | Adam |
| Initial learning rate | 0.0001 |
| Loss | Binary cross-entropy |
| Maximum epochs | 50 |
| Training metric | Accuracy |
| Callback monitor | Validation loss |
| Best-weight restoration | Enabled in early stopping |

| Model | Batch size | Early-stopping patience | LR reduction factor | LR reduction patience |
| --- | ---: | ---: | ---: | ---: |
| CNN | 16 | 45 | 0.3 | 5 |
| CNN + CBAM | 16 | 25 | 0.5 | 18 |
| CNN + Transformer | 8 | 45 | 0.3 | 5 |
| CNN + CBAM + Transformer | 4 | 45 | 0.3 | 5 |

No new random seed or split-generation procedure has been introduced. Unspecified optimizer and callback arguments retain library defaults.

---

## 📏 Evaluation Metrics

The evaluator collects predictions and ground-truth labels from the same batches. Classification uses the original decision rule:

```python
predicted_label = (probability > 0.5).astype(int)
```

Supported calculations and plots include:

- Accuracy.
- Binary precision, recall, and F1 for **Real / class 1**.
- Confusion matrix.
- ROC/AUC for fake and real classes.

The sigmoid output is the real-class probability; the fake-class score is `1 - probability`.

> **Original evaluation behavior:** the curve labelled “Mean ROC” repeats the real-class ROC calculation. The sample-prediction “Confidence” label always shows the real-class probability. These behaviors are retained and documented rather than silently corrected.

Historical numerical results and prediction files are intentionally omitted from this repository.

---

## 🔎 Visualization and Explanations

The code includes:

1. Training and validation accuracy/loss curves in two layouts.
2. Confusion-matrix heatmaps.
3. ROC curves.
4. Random sample-prediction grids.
5. Grad-CAM heatmaps and overlays.
6. The original alternate Grad-CAM++ calculation from the combined-model notebook.

For CNN + Transformer, the explanation launcher selects the fourth backbone convolution, matching the original notebook's `conv2d_7` target. Other variants use `gradcam_conv`.

The original explanation routines differentiate the single sigmoid real-class score. The alternate Grad-CAM++ routine retains its original derivative calculation and limitations; it is not replaced with a different implementation.

Figures are generated only when requested. This release contains no saved research figures.

---

## 📁 Repository Structure

| Directory / file | Contents |
| --- | --- |
| `configs/` | `experiment.yaml` |
| `data/` | `dataset.py` |
| `models/` | `cnn.py`, `cnn_cbam.py`, `cnn_transformer.py`, `cnn_cbam_transformer.py`, `build.py` |
| `training/` | `train.py` |
| `evaluation/` | `evaluate.py`, `metrics.py`, `explain.py` |
| `utils/` | `config.py`, `plotting.py` |
| `scripts/` | Four `train_*.py` launchers, `evaluate.py`, and `common.py` |
| `docs/` | `provenance.md` |
| Project root | `README.md`, `QUICKSTART.md`, `requirements.txt`, `.gitignore` |

Python package directories also contain `__init__.py` files. Local datasets, generated outputs, checkpoints, and notebooks are excluded from version control.

---

## 🚀 Usage

### Clone the repository

```bash
git clone https://github.com/sabbir5622r/TransCNN.git
cd TransCNN
```

### Install dependencies

Installation is needed only to execute the code, not to read or publish it.

```bash
python -m venv .venv
```

On Windows PowerShell, install into that environment without activating it:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The commands below assume the environment is active. On Windows, you can instead replace `python` with `.\.venv\Scripts\python.exe`.

Dependencies are not pinned because exact historical package versions were not recorded. The Kaggle notebook metadata identifies Python 3.11.13 and a Tesla T4; installing current packages does not recreate that original environment.

### Configure data paths

Edit `configs/experiment.yaml`:

```yaml
datasets:
  deepfake_dataset_1: datasets/deepfake_dataset_1
  deepfake_dataset_2: datasets/deepfake_dataset_2
```

Relative paths resolve from the repository root. Absolute paths are also supported.

### Build a model

```python
from models.cnn_transformer import build_cnn_transformer_binary

model = build_cnn_transformer_binary((224, 224, 3))
model.summary()
```

### Train a selected model

**These optional commands start new experiments. They are not required to publish the repository or inspect the completed research code.** Run them from the repository root:

```bash
python -m scripts.train_cnn --dataset deepfake_dataset_1
python -m scripts.train_cnn_cbam --dataset deepfake_dataset_1
python -m scripts.train_cnn_transformer --dataset deepfake_dataset_1
python -m scripts.train_cnn_cbam_transformer --dataset deepfake_dataset_1
```

Replace `deepfake_dataset_1` with `deepfake_dataset_2` to select the second dataset.

| Optional flag | Behavior |
| --- | --- |
| `--real-world` | Evaluate the additional `real_world` partition |
| `--plots` | Save and display learning curves and evaluation plots |
| `--save-weights` | Export final restored model weights |
| `--config path/to/file.yaml` | Load another configuration |

Training prints test metrics afterward. Without optional flags, it does not save plots or weights.

To make weights available for a later evaluation:

```bash
python -m scripts.train_cnn_transformer --dataset deepfake_dataset_1 --save-weights
```

### Evaluate compatible saved weights

The repository does not include trained weights. If you have compatible weights, evaluate them without training:

```bash
python -m scripts.evaluate --model cnn_transformer --dataset deepfake_dataset_1 --weights outputs/deepfake_dataset_1/cnn_transformer/model.weights.h5
```

The path above is the output of the optional weight-saving command. The evaluator expects weights, not a complete serialized model.

Add `--split real_world` for external evaluation, `--plots` for evaluation figures, or `--explain gradcam` for explanation images. The alternate method is available through `--explain gradcampp`.

Requested files are written under `outputs/<dataset>/<model>/`. That directory is ignored by Git. Weight export is a repository convenience, not a claim that the historical experiments used checkpoint saving or resume support.

---

## 🔬 Provenance and Validation

<p align="justify">The model functions were extracted from the original executed notebooks and checked statically against their source definitions. Python syntax, local module paths, launcher help commands, and evaluation calculations on synthetic batches were checked. TensorFlow model construction, GPU execution, weight loading, and complete end-to-end execution have not been verified after refactoring.</p>

The published configuration intentionally standardizes two settings at the author's request:

- **224×224 input for every model.** The original CNN + CBAM + Transformer Dataset 2 notebook used 180×180.
- **Training augmentation for every model.** Most recorded notebook execution paths used rescaling without augmentation.

These adaptations must not be presented as the exact configuration that generated all historical results. No original CNN + Transformer Dataset 2 notebook was supplied; launcher support for that combination does not establish a historical run.


---

## 📝 Status and License

The original Kaggle research experiments are complete. This repository provides their subsequently organized model code and an adapted local pipeline.

**Release scope:** source code and documentation only. Original notebooks are retained privately; historical outputs and trained models are not distributed here.

A software license has not yet been selected. No open-source license is asserted in this release.

---
## 📄 Related Publication and Bibtex

A shorter conference version of this research was published at ICCIT 2025. This repository includes additional model variants beyond that conference version.

If you use this work, please cite:

```bibtex
@inproceedings{Hossen2025TransCNN,
  title={TransCNN: A Hybrid CNN--Transformer Synergy for Reliable Deepfake Forensics},
  author={Hossen, Md. Sabbir and Saiduzzaman, Md.},
  booktitle={2025 28th International Conference on Computer and Information Technology (ICCIT)},
  year={2025},
  address={Cox's Bazar, Bangladesh},
  organization={IEEE}
}
```



---

## 👤 Author

**Md Sabbir Hossen**  
Research Assistant

Research interests include Natural Language Processing, Computer Vision, Large Language Models, Vision-Language Models, and Efficient AI.

**GitHub:** [sabbir5622r](https://github.com/sabbir5622r)  
**Personal Website:** [sabbir-hossen.com](https://sabbir-hossen.com/)

---

<div align="center">

**Research code for convolutional and attention-based deepfake image classification**

</div>
