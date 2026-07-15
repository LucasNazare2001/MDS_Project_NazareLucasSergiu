"""
app.py - Gradio interface for analysing a single image.
Estimates surface orientation region by region and highlights the regions
that are geometrically inconsistent with the rest of the scene (paper's method).
Run:  python app.py  ->  http://127.0.0.1:7860
"""
from __future__ import annotations
import numpy as np
import gradio as gr
from PIL import Image

import tamper_detect as td


def run_analysis(image, grid, sigma):
    # analyse one image and return the map + report
    if image is None:
        return None, "Upload an image (preferably with textured surfaces)."
    arr = np.array(image.convert("RGB"))
    r = td.analyze(arr, grid=int(grid), flag_sigma=float(sigma))
    verdict = ("no strong geometric inconsistency"
               if r["frac_inconsistent"] == 0 or r["score"] < 31
               else "regions with inconsistent orientation")
    report = (
        f"### Geometric inconsistency analysis\n"
        f"- Reference orientation: v **{r['ref_v']:+.1f}deg**, h **{r['ref_h']:+.1f}deg**\n"
        f"- Inconsistency score: **{r['score']:.1f}deg**\n"
        f"- Largest deviation: {r['max_deviation']:.1f}deg\n"
        f"- Blocks flagged: {r['frac_inconsistent']*100:.0f}% of {r['n_blocks']}\n\n"
        f"**Result:** {verdict}.\n\n"
        f"*Red = regions whose local surface orientation departs from the "
        f"dominant orientation of the scene.*")
    return r["vis"], report


def make_demo(angle_a, angle_b):
    # generate a test image with two regions at different orientations
    img, _ = td.make_test_image(host_angle=float(angle_a),
                                patch_angle=float(angle_b), seed=0)
    return Image.fromarray(img)


with gr.Blocks(title="Geometric Inconsistency Analysis") as demo:
    gr.Markdown("# Geometric inconsistency analysis\n"
                "Upload an image; the local surface orientation is estimated "
                "region by region and the regions inconsistent with the scene "
                "are highlighted. Method of Farid & Kosecka (IEEE TIP, 2007).")

    with gr.Row():
        img = gr.Image(label="Image", type="pil")
        with gr.Column():
            grid = gr.Slider(3, 6, 4, step=1, label="grid (blocks per axis)")
            sigma = gr.Slider(1.0, 3.5, 2.0, step=0.5, label="sensitivity (sigma)")
            btn = gr.Button("Analyse", variant="primary")
    vis = gr.Image(label="Inconsistency map (red = anomalous)")
    rep = gr.Markdown()
    btn.click(run_analysis, [img, grid, sigma], [vis, rep])

    gr.Markdown("---\n#### Generate a test image")
    with gr.Row():
        aa = gr.Slider(-40, 40, 25, step=5, label="background orientation")
        ab = gr.Slider(-40, 40, -30, step=5, label="central region orientation")
        mk = gr.Button("Generate")
    mk.click(make_demo, [aa, ab], img)


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
