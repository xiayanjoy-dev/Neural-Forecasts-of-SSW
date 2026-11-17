#!/usr/bin/env python3
"""Paper-1 Figure 1 v4: honest retrospective rolling-refit framework."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parent
OUT_PNG = ROOT / "ssw_paper1_fig1_framework_v4.png"
OUT_PDF = ROOT / "ssw_paper1_fig1_framework_v4.pdf"


def box(axis, xy, width, height, text, edge, fill="#F7F9FB", size=7.5):
    patch = FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.02",
                           linewidth=1.1, edgecolor=edge, facecolor=fill)
    axis.add_patch(patch)
    axis.text(xy[0] + width/2, xy[1] + height/2, text, ha="center", va="center",
              fontsize=size, color="#27313A")


def arrow(axis, start, end, color="#6B7785"):
    axis.annotate("", xy=end, xytext=start,
                  arrowprops=dict(arrowstyle="-|>", lw=1.0, color=color,
                                  shrinkA=2, shrinkB=2))


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.35, 5.6))
    blue, green, orange, red, grey = "#4678B8", "#3F8F67", "#D77C35", "#B84B4B", "#697684"
    for axis in axes.flat:
        axis.set_xlim(0, 1); axis.set_ylim(0, 1); axis.axis("off")

    axis = axes[0, 0]
    axis.set_title("Predictor and issue-time contract", loc="left", fontweight="bold", fontsize=9)
    box(axis, (0.03, 0.56), 0.25, 0.22, "ERA5 1°\nU, V, T; 20 levels", blue)
    box(axis, (0.37, 0.56), 0.25, 0.22, "QG EP +\nequatorial wind", green)
    box(axis, (0.71, 0.56), 0.25, 0.22, "Vortex and heat-flux\nsummaries", orange)
    box(axis, (0.22, 0.18), 0.56, 0.22,
        "45-day input: 180 six-hourly states\nlast input 18 UTC; issue 00 UTC", grey)
    for x in (0.155, 0.495, 0.835): arrow(axis, (x, 0.56), (0.5, 0.40))

    axis = axes[0, 1]
    axis.set_title("Physics-guided probability-head model", loc="left", fontweight="bold", fontsize=9)
    box(axis, (0.04, 0.47), 0.26, 0.25, "Spatial + temporal\nrepresentation", blue)
    box(axis, (0.39, 0.47), 0.24, 0.25, "Shared latent\nstate", grey)
    box(axis, (0.72, 0.60), 0.24, 0.17, "5/10/15-day\nprobability heads", red)
    box(axis, (0.72, 0.34), 0.24, 0.17, "Future-vortex\nauxiliary heads", green)
    arrow(axis, (0.30, 0.595), (0.39, 0.595)); arrow(axis, (0.63, 0.595), (0.72, 0.685))
    arrow(axis, (0.63, 0.595), (0.72, 0.425))
    axis.text(0.51, 0.20, "Event-balanced sampling; scalar prior shift\ndoes not prove population calibration",
              ha="center", va="center", fontsize=7.2, color="#5B3740")

    axis = axes[1, 0]
    axis.set_title("Frozen retrospective rolling refits", loc="left", fontweight="bold", fontsize=9)
    y_positions = (0.72, 0.47, 0.22)
    folds = (("A", 0.03, 0.53, 0.56, 0.84, "fit <=1999", "test 2000–04"),
             ("B", 0.03, 0.58, 0.61, 0.89, "fit <=2004", "test 2005–10"),
             ("C", 0.03, 0.68, 0.71, 0.98, "fit <=2010", "test 2011–21"))
    for y, (label, x0, x1, xt0, xt1, fit, test) in zip(y_positions, folds):
        axis.text(0.00, y+0.04, label, fontweight="bold", fontsize=8)
        axis.add_patch(Rectangle((x0, y), x1-x0, 0.10, facecolor="#C9D1D9", edgecolor="none"))
        axis.add_patch(Rectangle((xt0, y), xt1-xt0, 0.10,
                                 facecolor={"A":blue,"B":orange,"C":green}[label], edgecolor="none"))
        axis.text((x0+x1)/2, y+0.05, fit, ha="center", va="center", fontsize=7)
        axis.text((xt0+xt1)/2, y+0.05, test, ha="center", va="center", fontsize=7, color="white")
    axis.text(0.5, 0.05, "1,918 origins; 16 events; within-fold fitting uses earlier winters",
              ha="center", fontsize=7.2)

    axis = axes[1, 1]
    axis.set_title("Evidence and claim boundary", loc="left", fontweight="bold", fontsize=9)
    items = ((blue, "Earlier-winter weights and normalization within fold"),
             (orange, "Architecture and 21 epochs from 2018–21 development"),
             (green, "A/B/C are retrospective, not prospective"),
             (red, "2022–25 forward stress test: three events"))
    y = 0.77
    for color, text in items:
        axis.add_patch(Rectangle((0.04, y-0.01), 0.025, 0.13, facecolor=color, edgecolor="none"))
        box(axis, (0.07, y-0.01), 0.87, 0.13, text, "#D4DBE2", fill="#F0F3F6", size=7.0)
        y -= 0.19
    axis.text(0.5, 0.03, "Report folds, baselines, probability quality, events and seed spread",
              ha="center", fontsize=6.9)

    for letter, axis in zip("abcd", axes.flat):
        axis.text(-0.04, 1.04, letter, transform=axis.transAxes,
                  fontweight="bold", fontsize=10, va="top")
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.07, top=0.95,
                        wspace=0.20, hspace=0.28)
    fig.savefig(OUT_PNG, dpi=300, facecolor="white")
    fig.savefig(OUT_PDF, facecolor="white")
    plt.close(fig)
    print("SSW_PAPER1_FIG1_V4_OK")


if __name__ == "__main__":
    main()
