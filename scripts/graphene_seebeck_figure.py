#!/usr/bin/env python3
"""Publication figure for the clean-graphene CTKG Seebeck benchmark.

Consumes the CSVs written by scripts/graphene_seebeck_figure.jl:

    python3 scripts/graphene_seebeck_figure.py --data DIR --out notebook/figs/NAME.png

Panel (a): twist-averaged transport distribution vs the Dirac |E| line.
Panel (b): KPM Seebeck at two temperatures collapsed onto the universal
finite-temperature clean-Dirac curve, with the Mott asymptote.
"""

import argparse
import csv
import glob
import os

import matplotlib as mpl
import numpy as np
from matplotlib import font_manager

KB_OVER_E_UV_PER_K = 86.17333262  # (k_B/|e|) in uV/K
E_REF = 0.2
KERNEL_CORE = 0.08
DIRAC_WINDOW = 0.35
KBT = (0.04, 0.06)

BLUE = "#0072B2"
VERMILLION = "#D55E00"
GRAY = "#8a8a8a"

for path in glob.glob(os.path.expanduser("~/Library/Fonts/Inter_18pt-*.ttf")):
    font_manager.fontManager.addfont(path)

mpl.rcParams.update({
    "font.family": "Inter 18pt",
    "font.size": 8.5,
    "mathtext.fontset": "custom",
    "mathtext.rm": "STIX Two Math",
    "mathtext.it": "STIX Two Math:italic",
    "mathtext.bf": "STIX Two Math:bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "savefig.dpi": 400,
})

import matplotlib.pyplot as plt  # noqa: E402  (after rcParams)


def read_csv(path):
    with open(path) as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = np.array([[float(x) for x in row] for row in reader])
    return {name: rows[:, i] for i, name in enumerate(header)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sigma = read_csv(os.path.join(args.data, "sigma.csv"))
    seebeck = read_csv(os.path.join(args.data, "seebeck.csv"))
    dense = read_csv(os.path.join(args.data, "reference_dense.csv"))

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.05, 3.0), constrained_layout=True)

    # -- (a) transport distribution ------------------------------------------
    E = sigma["E"]
    ax_a.fill_betweenx([0, 2.1], -KERNEL_CORE, KERNEL_CORE,
                       color="0.92", zorder=0)
    ax_a.plot(E, np.abs(E) / E_REF, color="black", lw=1.0, ls=(0, (5, 3)),
              zorder=2)
    ax_a.plot(E, sigma["sigma_normalized"], color=BLUE, lw=1.4, zorder=3)
    ax_a.set_xlabel(r"$E/t$")
    ax_a.set_ylabel(r"$\Sigma(E)\,/\,\Sigma(0.2\,t)$")
    ax_a.set_xlim(-0.4, 0.4)
    ax_a.set_ylim(0, 2.1)
    ax_a.set_xticks([-0.4, -0.2, 0, 0.2, 0.4])
    ax_a.text(-0.335, 1.92, "KPM, 16 twists", color=BLUE, ha="left")
    ax_a.text(0.315, 1.30, r"$|E|\,/\,0.2\,t$", color="black", ha="left")
    ax_a.text(0.0, 0.52, "kernel-\nrounded", color="0.45", ha="center",
              va="center", fontsize=7.5)

    # -- (b) Seebeck collapse ------------------------------------------------
    eta = seebeck["eta"]
    mott_eta = np.linspace(3.0, 8.0, 60)
    mott = -KB_OVER_E_UV_PER_K * np.pi**2 / 3 / mott_eta
    ax_b.axhline(0.0, color="0.85", lw=0.6, zorder=0)
    ax_b.plot(mott_eta, mott, color=GRAY, lw=1.0, ls=(0, (5, 3)), zorder=1)
    ax_b.plot(-mott_eta, -mott, color=GRAY, lw=1.0, ls=(0, (5, 3)), zorder=1)
    ax_b.plot(dense["eta"], dense["S_ref"], color="black", lw=1.2, zorder=2)

    for kbt, key, acc_key, color, dy in (
            (KBT[0], "S_kT1", "acc_kT1", BLUE, 0),
            (KBT[1], "S_kT2", "acc_kT2", VERMILLION, 1)):
        S = seebeck[key]
        acc = seebeck[acc_key].astype(bool)
        ax_b.plot(eta[acc], S[acc], "o", ms=3.6, mfc=color, mec=color,
                  zorder=4)
        ax_b.plot(eta[~acc], S[~acc], "o", ms=3.6, mfc="white", mec=color,
                  mew=0.9, zorder=4)

    ax_b.set_xlabel(r"$\eta = \mu\,/\,k_{\mathrm{B}}T$")
    ax_b.set_ylabel(r"$S$  ($\mu$V/K)")
    ax_b.set_xlim(-8.4, 8.4)
    ax_b.set_ylim(-100, 100)
    ax_b.set_xticks([-8, -4, 0, 4, 8])
    ax_b.text(-7.9, -40, "Dirac, analytic", color="black", ha="left")
    ax_b.text(-7.9, -54, r"KPM  $k_{\mathrm{B}}T = 0.04\,t$", color=BLUE,
              ha="left")
    ax_b.text(-7.9, -68, r"KPM  $k_{\mathrm{B}}T = 0.06\,t$", color=VERMILLION,
              ha="left")
    ax_b.text(-7.9, -82, r"open: $|\mu| > 0.35\,t$", color="0.45",
              ha="left", fontsize=7.5)
    ax_b.text(3.8, -93, r"Mott $-\pi^2/(3\eta)$", color=GRAY, ha="left")

    for ax, letter in ((ax_a, "a"), (ax_b, "b")):
        ax.text(-0.14, 1.02, letter, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

    fig.savefig(args.out)
    root, _ = os.path.splitext(args.out)
    fig.savefig(root + ".pdf")
    print("wrote", args.out, "and", root + ".pdf")


if __name__ == "__main__":
    main()
