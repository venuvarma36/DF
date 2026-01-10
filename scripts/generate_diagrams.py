"""Generate architecture/flow/model/solution diagrams as PNGs.

Run:
    python scripts/generate_diagrams.py --outdir diagrams
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt

DPI = 150


def ensure_outdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def add_box(ax, x, y, w, h, text, face="#e8f1ff"):
    rect = plt.Rectangle((x, y), w, h, edgecolor="black", facecolor=face, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)


def add_arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.3))


def draw_architecture(out_path: Path):
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.04)
    ax.axis("off")
    add_box(ax, 0.5, 1.2, 1.0, 0.5, "Input")
    add_box(ax, 0.0, 0.5, 1.0, 0.5, "Audio Pipeline\nSpecRNet-Lite")
    add_box(ax, 1.0, 0.5, 1.0, 0.5, "Image Pipeline\nSigLIP (HF)")
    add_box(ax, 0.0, -0.1, 1.0, 0.5, "Calibrated Audio Score")
    add_box(ax, 1.0, -0.1, 1.0, 0.5, "Calibrated Image Score")
    add_box(ax, 0.5, -0.8, 1.0, 0.5, "Fusion\nalpha·audio + beta·image")
    add_box(ax, 0.5, -1.5, 1.0, 0.5, "Final Decision")
    add_arrow(ax, 1.0, 1.2, 0.5, 1.0)
    add_arrow(ax, 1.0, 1.2, 1.5, 1.0)
    add_arrow(ax, 0.5, 0.5, 0.5, 0.4)
    add_arrow(ax, 1.5, 0.5, 1.5, 0.4)
    add_arrow(ax, 0.5, -0.1, 1.0, -0.35)
    add_arrow(ax, 1.5, -0.1, 1.0, -0.35)
    add_arrow(ax, 1.0, -0.8, 1.0, -1.0)
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylim(-1.9, 1.9)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def draw_flow(out_path: Path):
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.06)
    ax.axis("off")
    add_box(ax, -0.2, 0.4, 1.0, 0.5, "Audio\n.wav")
    add_box(ax, 1.0, 0.4, 1.0, 0.5, "Audio Model\nSpecRNet-Lite")
    add_box(ax, 2.2, 0.4, 1.0, 0.5, "Audio Score")
    add_box(ax, -0.2, -0.6, 1.0, 0.5, "Image\n.jpg/.png")
    add_box(ax, 1.0, -0.6, 1.0, 0.5, "Image Model\nSigLIP HF")
    add_box(ax, 2.2, -0.6, 1.0, 0.5, "Image Score")
    add_box(ax, 3.4, -0.1, 1.1, 0.5, "Fusion")
    add_box(ax, 4.8, -0.1, 1.2, 0.5, "Decision\nREAL/FAKE")
    add_arrow(ax, 0.8, 0.65, 1.0, 0.65)
    add_arrow(ax, 2.0, 0.65, 2.2, 0.65)
    add_arrow(ax, 0.8, -0.35, 1.0, -0.35)
    add_arrow(ax, 2.0, -0.35, 2.2, -0.35)
    add_arrow(ax, 3.2, 0.65, 3.4, 0.15)
    add_arrow(ax, 3.2, -0.35, 3.4, 0.15)
    add_arrow(ax, 4.5, 0.15, 4.8, 0.15)
    ax.set_xlim(-0.6, 6.2)
    ax.set_ylim(-1.1, 1.3)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def draw_models(out_path: Path):
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.08)
    ax.axis("off")
    add_box(ax, 0.0, 0.0, 1.5, 1.0, "Audio Model\nSpecRNet-Lite\nLearnable FB + STFT")
    add_box(ax, 2.0, 0.0, 1.5, 1.0, "Image Model\nSigLIP HF\n224x224 input")
    add_box(ax, 4.0, 0.0, 1.5, 1.0, "Fusion\nalpha=0.6 beta=0.4")
    add_box(ax, 6.0, 0.0, 1.2, 1.0, "Calibration\nTemp scaling")
    add_arrow(ax, 1.5, 0.5, 2.0, 0.5)
    add_arrow(ax, 3.5, 0.5, 4.0, 0.5)
    add_arrow(ax, 5.5, 0.5, 6.0, 0.5)
    ax.set_xlim(-0.3, 7.7)
    ax.set_ylim(-0.4, 1.4)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def draw_solutions(out_path: Path):
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.96, bottom=0.08)
    ax.axis("off")
    add_box(ax, 0.0, 0.5, 1.8, 0.9, "CLI\npython detect.py\n(audio/image)" )
    add_box(ax, 2.2, 0.5, 1.8, 0.9, "Web App\nrun_webapp.ps1\n/ web UI")
    add_box(ax, 4.4, 0.5, 1.8, 0.9, "Batch Eval\nscripts/eval_samples.py")
    add_box(ax, 6.6, 0.5, 1.8, 0.9, "Reports\nscripts/generate_reports.py")
    add_box(ax, 3.3, -0.6, 2.0, 0.9, "Outputs\nScores + PNGs")
    add_arrow(ax, 1.8, 0.95, 2.2, 0.95)
    add_arrow(ax, 4.0, 0.95, 4.4, 0.95)
    add_arrow(ax, 6.2, 0.95, 6.6, 0.95)
    add_arrow(ax, 3.1, 0.5, 3.3, -0.15)
    add_arrow(ax, 5.3, 0.5, 5.3, -0.15)
    add_arrow(ax, 7.5, 0.5, 5.3, -0.15)
    ax.set_xlim(-0.4, 8.9)
    ax.set_ylim(-1.0, 1.7)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate architecture/flow/model/solution diagrams as PNGs.")
    parser.add_argument("--outdir", default="diagrams", help="Output directory for PNGs")
    args = parser.parse_args()

    outdir = ensure_outdir(Path(args.outdir))
    draw_architecture(outdir / "architecture.png")
    draw_flow(outdir / "flow.png")
    draw_models(outdir / "models.png")
    draw_solutions(outdir / "solutions.png")
    print(f"Saved diagrams to {outdir}")


if __name__ == "__main__":
    main()
