"""Paper figures (CI panels, paired-difference forest plot, scaling plot).

One panel per metric, each with its own axis (never two scales on one axis).
Arms are told apart by their axis label, so one color is enough. Whiskers =
95% CI. Area ratio uses a log axis (2x too big sits as far from 1 as 2x too
small), with a dashed line at 1 = right size. Every chart is saved as PNG
(300 dpi) and PDF.

Usage:
    charts = Charts(res, DRAWS, ARMS, NAMES, OUT, percent=PERCENT)
    charts.ci_panels("internal", [("finding IoU", "finding", "IoU")], MAIN, ncols=4, fname="fig_x")
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from bootstrap_ci import fmt_p

INK, MUTED, GRID, SERIES = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
plt.rcParams["pdf.fonttype"] = 42  # embed TrueType, not Type 3 (IEEE PDF eXpress flags Type 3)
plt.rcParams["ps.fonttype"] = 42


def style(ax, grid_axis="x"):
    """Frameless panel, light grid, muted ticks."""
    if ax.get_xscale() == "linear":
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))  # few ticks, so neighbor panels never collide
    ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)


class Charts:
    """res: long results table (dataset, group, metric, arm, point, lo, hi).
    draws: {(dataset, group, metric): draws x arms}, columns in `arms` order.
    names: arm -> display name. out: output folder. percent: metrics shown as %.
    lower_better: metrics whose title says "lower is better"."""

    def __init__(self, res, draws, arms, names, out, percent=(), lower_better=("miss",)):
        self.res, self.draws, self.arms, self.names, self.out = res, draws, list(arms), names, out
        self.percent, self.lower_better = set(percent), set(lower_better)

    def _sel(self, dataset, group, metric):
        r = self.res
        return r[(r.dataset == dataset) & (r.group == group) & (r.metric == metric)].set_index("arm")

    def _save(self, fig, fname):
        fig.savefig(self.out / f"{fname}.png", dpi=300)
        fig.savefig(self.out / f"{fname}.pdf")
        plt.show()

    def ci_panels(self, dataset, columns, arms, ncols, fname, refs=None, xlim=None, wspace=0.12, xlabel=None):
        """Dot + 95% CI per arm, one panel per (header, group, metric) column.
        refs: optional list (one per column) of x values drawn as a dashed line.
        xlim: optional (low, high) shared by every panel, e.g. (0, 100) for shares.
        wspace: gap between panels; raise it when dots sit at both ends of the axis.
        xlabel: optional metric name under every panel (when titles name the question type)."""
        nrows = -(-len(columns) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 0.42 * len(arms) * nrows + 1.0 * nrows),
                                 squeeze=False, constrained_layout=True)
        fig.get_layout_engine().set(wspace=wspace, hspace=0.08)  # frameless panels need a clear gap
        y = np.arange(len(arms))[::-1]  # first arm on top
        for k, ax in enumerate(axes.flat):
            if k >= len(columns):
                ax.axis("off")
                continue
            header, group, metric = columns[k]
            sel = self._sel(dataset, group, metric).loc[arms]
            scale = 100 if metric in self.percent else 1
            p, lo, hi = (sel[c].to_numpy() * scale for c in ["point", "lo", "hi"])
            ax.errorbar(p, y, xerr=[p - lo, hi - p], fmt="o", color=SERIES, ms=8, mec="white", mew=2,
                        elinewidth=1.5, capsize=0)
            if metric == "area ratio":
                ax.set_xscale("log")
                ax.axvline(1, color=MUTED, lw=1, ls=(0, (3, 3)))
                left, right = min(lo.min(), 1) / 1.15, max(hi.max(), 1) * 1.15  # always show 1
                ax.set_xlim(left, right)
                cands = [0.5, 1, 1.25, 1.5, 2, 3, 5] if right / left < 6 else [0.5, 1, 2, 5, 10, 20, 40]
                ticks = [t for t in cands if left <= t <= right]
                ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
                ax.minorticks_off()
            if refs is not None and refs[k] is not None:
                ax.axvline(refs[k], color=MUTED, lw=1, ls=(0, (3, 3)))
            note = (" (%, lower is better)" if metric in self.lower_better
                    else " (1 = right size)" if metric == "area ratio"
                    else "" if metric in self.percent and refs is not None
                    else " (%)" if metric in self.percent else "")
            ax.set_title(header + note, fontsize=10, color=INK, loc="left")
            if xlabel:
                ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
            ax.set_yticks(y, [self.names[a] for a in arms] if k % ncols == 0 else [""] * len(arms))
            ax.set_ylim(-0.6, len(arms) - 0.4)
            ax.margins(x=0.08)
            style(ax)
            if xlim is not None:  # same fixed axis on every panel, 4% margin so dots at the ends stay whole
                pad = 0.04 * (xlim[1] - xlim[0])
                ax.set_xlim(xlim[0] - pad, xlim[1] + pad)
                ticks = np.linspace(*xlim, 5)
                ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        self._save(fig, fname)

    def forest(self, pairs, columns, fname, long, title_of=None):
        """Paired differences with 95% CI: one panel per pair, one row per metric.
        pairs: (label, arm, reference); columns: (header, dataset, group, metric).
        long: the diff table for the same pairs (columns comparison, metric_name,
        p_holm); its Holm p is printed right of each CI. Filled dot = Holm p <
        0.05; hollow = not. Area-ratio columns are left out (different units).
        title_of(label, arm, reference): panel title; default "arm − reference"."""
        title_of = title_of or (lambda label, a, b: f"{self.names[a]} − {self.names[b]}")
        p_holm = long.set_index(["comparison", "metric_name"])["p_holm"]
        columns = [c for c in columns if c[3] != "area ratio"]
        fig, axes = plt.subplots(1, len(pairs), figsize=(3.6 * len(pairs), 0.36 * len(columns) + 1.2),
                                 squeeze=False, constrained_layout=True)
        fig.get_layout_engine().set(wspace=0.08)
        y = np.arange(len(columns))[::-1]
        for k, (ax, (label, a, b)) in enumerate(zip(axes.flat, pairs)):
            for yi, (header, dataset, group, metric) in zip(y, columns):
                d = self.draws[(dataset, group, metric)]
                point = self._sel(dataset, group, metric)["point"]
                dd = d[:, self.arms.index(a)] - d[:, self.arms.index(b)]
                lo, hi = np.percentile(dd, [2.5, 97.5])
                pt = point[a] - point[b]
                p = p_holm[(label, header)]
                ax.errorbar(pt, yi, xerr=[[pt - lo], [hi - pt]], fmt="o", ms=7, color=SERIES, elinewidth=1.5,
                            mfc=SERIES if p < 0.05 else "white", mec=SERIES, mew=1.5, capsize=0)
                ax.annotate(fmt_p(p, "p"), (hi, yi), xytext=(7, 0), textcoords="offset points",  # clears the dot
                            ha="left", va="center", fontsize=9, color=MUTED,
                            bbox=dict(fc="white", ec="none", pad=0.5))  # hides the 0 line behind the text
            ax.axvline(0, color=MUTED, lw=1)
            ax.set_title(title_of(label, a, b), fontsize=10, color=INK, loc="left")
            ax.set_yticks(y, [c[0] for c in columns] if k == 0 else [""] * len(columns))
            ax.set_ylim(-0.6, len(columns) - 0.4)
            ax.margins(x=0.1)
            x0, x1 = ax.get_xlim()
            ax.set_xlim(x0, x1 + 0.35 * (x1 - x0))  # room for the p-value text
            style(ax)
        fig.supxlabel("difference (arm minus reference), 95% CI; p = paired bootstrap p-value, Holm-corrected; "
                      "hollow = p ≥ 0.05", fontsize=9, color=MUTED)
        self._save(fig, fname)

    def scaling(self, panels, arms, x, xticklabels, xlabel, fname):
        """Line + 95% CI over a training-data axis, one panel per measure.
        panels: (dataset, group, metric, title); arms: one per x point;
        x / xticklabels: positions and labels of the points."""
        fig, axes = plt.subplots(1, len(panels), figsize=(10 / 3 * len(panels), 3.4), constrained_layout=True)
        for ax, (ds, g, m, title) in zip(np.atleast_1d(axes), panels):
            sel = self._sel(ds, g, m).loc[arms]
            p, lo, hi = sel["point"].to_numpy(), sel["lo"].to_numpy(), sel["hi"].to_numpy()
            ax.errorbar(x, p, yerr=[p - lo, hi - p], color=SERIES, lw=2, marker="o", ms=8,
                        mec="white", mew=2, capsize=0, elinewidth=1.5)
            # label only the end points, beside the point on the side away from the line
            rising = p[-1] > p[0]
            for xi, pi, dx, ha, dy in [(x[0], p[0], 10, "left", -14 if rising else 6),
                                       (x[-1], p[-1], -10, "right", 6 if rising else -14)]:
                ax.annotate(f"{pi:.2f}" if m == "area ratio" else f"{pi:.3f}", (xi, pi), textcoords="offset points",
                            xytext=(dx, dy), ha=ha, fontsize=9, color=INK)
            if m == "area ratio":
                ax.axhline(1, color=MUTED, lw=1, ls=(0, (3, 3)))
            ax.set_title(title, fontsize=10, color=INK, loc="left")
            ax.set_xticks(x, xticklabels, fontsize=9, color=MUTED)
            ax.tick_params(axis="y", labelsize=9, colors=MUTED, length=0)
            ax.tick_params(axis="x", length=0)
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for side in ["top", "right", "left"]:
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            ax.margins(y=0.25)
        fig.supxlabel(xlabel, fontsize=9, color=MUTED)
        self._save(fig, fname)
