# Geometric Inconsistency Analysis

Tampering detection based on **geometric inconsistency**, using the method of
Farid & Kosecka, *Estimating Planar Surface Orientation Using Bispectral
Analysis* (IEEE TIP, 2007). The method estimates the slant of planar surfaces
from a single image; here it is applied to different regions of the image to
flag those whose orientation is inconsistent with the rest of the scene.

## Files

- `bispectral.py` - the paper's method: bicoherence and orientation estimation.
- `tamper_detect.py` - block-wise analysis: local orientation + inconsistency map.
- `app.py` - Gradio interface: analysis of a **single image**.
- `evaluate.py` - evaluation on a `real/` + `fake/` dataset (AUC, accuracy, ROC).
- `reproduce_paper.py` - reproduces the paper's figures (on synthetic data).
- `fetch_synthwildx_subset.py` - downloads 50+50 images for the evaluation.

## Usage

```bash
pip install -r requirements.txt

python app.py                 # Gradio -> http://127.0.0.1:7860
python reproduce_paper.py     # paper figures
python fetch_synthwildx_subset.py --n 50 --out dataset   # download 50 real + 50 fake
python evaluate.py dataset    # evaluation on the images
```

The Gradio tab analyses one image at a time; `evaluate.py` runs the same
analysis automatically over a whole dataset. The paper reproduction is separate
and uses synthetic data (the true slant is known only there, so the downloaded
images are for analysis/evaluation, not for reproducing the error figures).
