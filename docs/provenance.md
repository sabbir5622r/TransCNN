# Refactoring notes

## Source mapping

| Model file | Uploaded source |
| --- | --- |
| `models/cnn.py` | `cnn_dataset1.ipynb` |
| `models/cnn_cbam.py` | `cnn-cbam-dataset1.ipynb` |
| `models/cnn_transformer.py` | `cnn-tranformer-dataset1.ipynb` |
| `models/cnn_cbam_transformer.py` | `cnn-cbam-tranformer-dataset1.ipynb` |

Imports and function definitions are extracted directly. Model instantiation, compilation, summary, and fit calls are organized into launchers and shared training code. Shared attention functions remain duplicated in the four model files to preserve direct correspondence to source.

Evaluation, plots, and basic Grad-CAM derive from `cnn_dataset1.ipynb`; equivalent evaluation calculations appear in the other supplied notebooks. The alternate Grad-CAM++ function derives from `cnn-cbam-tranformer-dataset1.ipynb`. No original outputs are included. Keep the original notebooks privately as the historical record.

## Explicitly requested changes

- All inputs are 224 × 224 × 3. The historical combined-model Dataset 2 notebook used 180 × 180.
- Training augmentation is enabled for every model. The recorded augmented path was used for CBAM Dataset 1, while the other recorded paths used rescaling only.
- Dataset paths use the local aliases `deepfake_dataset_1` and `deepfake_dataset_2`.
- This public package contains code only.

The standardized augmented loader follows the original augmented-loader shuffle settings: training and validation shuffle, test and real-world data do not. Most original rescaling-only test generators left shuffle at its default. The evaluation code still pairs labels and predictions from each same batch.

## Organizational changes

- Hardcoded Kaggle output paths become per-dataset, per-model, per-split local paths. Source naming errors are not propagated into these new paths or evaluation titles.
- Plot generation is optional; numerical metrics print to the terminal. Evaluation returns a metric dictionary. It retains the original in-memory image collection.
- Optional final weight export and a weights-only evaluation command were added for reuse. They are not historical checkpointing behavior.
- The incomplete, unexecuted alternate CNN training cell referring to undefined `checkpoint` is not used. No seed, new splitting procedure, or new metric was added.
- Missing directory and class-folder checks were added to fail clearly rather than silently load an unintended label mapping.
- The two explanation functions have distinct names so one does not overwrite the other.

## Original behavior retained

- Strict binary threshold: probability greater than 0.5.
- Binary precision, recall, and F1 use real/class 1 as positive.
- "Mean ROC" repeats the real-class ROC computation, rather than calculating a macro mean.
- Sample "Confidence" is the real-class probability even for fake predictions.
- Grad-CAM differentiates the single sigmoid real-class score, regardless of the predicted class. Images, heatmaps, and 0.6/0.4 overlays follow the original calculations.
- The alternate Grad-CAM++ calculation retains its original nested derivatives and coefficients, including its lack of checks for missing second/third derivatives. No replacement method was introduced.
- CNN + Transformer originally targeted `conv2d_7`, which its saved summary identifies as the fourth backbone convolution. The launcher selects that corresponding convolution by position, since automatically generated names depend on the Keras session. Other models use `gradcam_conv`.
- Original model comments are unchanged. Transformer comments suggesting 28×28 and 14×14 attention inputs are inaccurate for the actual 224×224 forward path: attention occurs at 56×56 and 28×28 before pooling.

## Limits

No training or historical-result reproduction was performed after refactoring. Exact historical dependency versions and seeds are unavailable. Notebook execution counts show interactive execution and do not establish a complete immutable execution history. Identical dataset counts cannot establish identical image contents between differently uploaded datasets. Dataset construction, licenses, and the original split procedure were not supplied.
