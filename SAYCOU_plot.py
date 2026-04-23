#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector


def read_yield_file(filename):
    """
    Try to read:
      5 columns: E, Y0, Y1, Y2, Y3plus
    and fall back to:
      4 columns: E, Y0, Y1, Y2
    """
    try:
        data = np.loadtxt(filename, comments="#")
    except Exception as exc:
        raise RuntimeError(f"Could not read input file '{filename}': {exc}")

    if data.ndim != 2:
        raise RuntimeError(f"Input file '{filename}' does not look like a 2D table.")

    ncols = data.shape[1]

    if ncols >= 5:
        energy = data[:, 0]
        y0 = data[:, 1]
        y1 = data[:, 2]
        y2 = data[:, 3]
        y3plus = data[:, 4]
        has_y3plus = True
    elif ncols >= 4:
        energy = data[:, 0]
        y0 = data[:, 1]
        y1 = data[:, 2]
        y2 = data[:, 3]
        y3plus = np.zeros_like(energy)
        has_y3plus = False
        print("Warning: 5th column not found, assuming no Y3+ contribution.")
    else:
        raise RuntimeError(
            f"Input file '{filename}' must contain at least 4 columns: "
            "E, Y0, Y1, Y2 [, Y3plus]"
        )

    return energy, y0, y1, y2, y3plus, has_y3plus


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} YIELD_FILE AREAL_DENSITY_AT_PER_BARN")
        print(f"Example: {sys.argv[0]} yields_output.txt 6.e-2")
        sys.exit(1)

    filename = sys.argv[1]

    try:
        n_areal = float(sys.argv[2])
    except ValueError:
        print("Error: areal density must be a floating-point number in at/b.")
        sys.exit(1)

    energy, y0, y1, y2, y3plus, has_y3plus = read_yield_file(filename)

    ysum = y0 + y1 + y2 + y3plus

    print(f"Input file          : {filename}")
    print(f"Areal density       : {n_areal:.6e} at/b")
    print(f"Number of points    : {len(energy)}")
    print()
    print("Press > r < to reset x range after zooming in.")
    print("Press > x < to switch between log and lin scale on X axis.")
    print("Press > y < to switch between log and lin scale on Y axis.")
    print("Press a legend item to highlight a curve.")
    print("Press > a < to de-highlight a curve.")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(energy, ysum, label=r"$\Sigma Y_i$", color="tab:red")
    ax.plot(energy, y0, label="$Y_0$", color="tab:blue")
    ax.plot(energy, y1, label="$Y_1$", color="tab:green")
    ax.plot(energy, y2, label="$Y_2$", color="tab:brown")

    if has_y3plus:
        ax.plot(energy, y3plus, label=r"$Y_{\geq 3}$ approx.", color="tab:orange")

    ax.set_xlabel("Neutron energy [eV]")
    ax.set_ylabel("Capture yield")
    ax.set_title(f"Capture yield for n = {n_areal:.3e} at/b")
    ax.grid(True)

    leg = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0
    )

    lines = ax.get_lines()
    legend_lines = leg.get_lines()
    line_map = {}

    for legline, origline in zip(legend_lines, lines):
        legline.set_picker(True)
        legline.set_pickradius(8)
        line_map[legline] = origline

    def on_pick(event):
        legline = event.artist
        if legline not in line_map:
            return

        selected = line_map[legline]

        for line in lines:
            if line is selected:
                line.set_linewidth(3.0)
                line.set_alpha(1.0)
                line.set_zorder(10)
            else:
                line.set_linewidth(1.2)
                line.set_alpha(0.25)
                line.set_zorder(1)

        selected.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", on_pick)

    ax.set_xscale("log")
    ax.set_ylim(0, 1)

    x_full_min = np.min(energy)
    x_full_max = np.max(energy)

    def onselect(xmin, xmax):
        if xmin == xmax:
            return
        ax.set_xlim(min(xmin, xmax), max(xmin, xmax))
        if ax.get_yscale() == "log":
            ax.set_ylim(1e-6, 1)
        else:
            ax.set_ylim(0, 1)
        span.set_visible(False)
        fig.canvas.draw_idle()

    span = SpanSelector(
        ax,
        onselect,
        "horizontal",
        useblit=True,
        props=dict(alpha=0.15, facecolor="tab:blue"),
        interactive=False,
        drag_from_anywhere=False,
    )

    def on_key(event):
        if event.key == "r":
            ax.set_xscale("log")
            ax.set_xlim(x_full_min, x_full_max)
            if ax.get_yscale() == "log":
                ax.set_ylim(1e-6, 1)
            else:
                ax.set_ylim(0, 1)
            span.set_visible(False)
            fig.canvas.draw_idle()

        elif event.key == "y":
            if ax.get_yscale() == "linear":
                ax.set_yscale("log")
                ax.set_ylim(1e-6, 1)
            else:
                ax.set_yscale("linear")
                ax.set_ylim(0, 1)
            fig.canvas.draw_idle()

        elif event.key == "x":
            current_xlim = ax.get_xlim()

            if ax.get_xscale() == "linear":
                ax.set_xscale("log")
            else:
                ax.set_xscale("linear")

            ax.set_xlim(current_xlim)
            fig.canvas.draw_idle()

        elif event.key == "a":
            for line in lines:
                line.set_linewidth(1.5)
                line.set_alpha(1.0)
                line.set_zorder(2)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.tight_layout()
    fig.subplots_adjust(right=0.78)
    plt.show()


if __name__ == "__main__":
    main()