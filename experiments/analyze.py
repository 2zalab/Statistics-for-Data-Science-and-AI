"""Build the tables and figures of the paper from ``results/*.csv``.

Usage: python experiments/analyze.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, FIG, TAB = ROOT / "results", ROOT / "paper" / "figures", ROOT / "paper" / "tables"
LEVELS = ["none", "weak", "moderate", "strong"]
ALPHA = 0.05

LABEL = {
    "paired_t": "Paired $t$ (folds)",
    "iid_bootstrap": "i.i.d. bootstrap",
    "nadeau_bengio": "Corrected resampled $t$",
    "block_bootstrap": "Block bootstrap",
    "mixed_model": "Mixed model",
    "dm_hac": "DM--HAC (time only)",
    "da_hac": "DA-HAC (proposed)",
    "da_hac_no_tau": "DA-HAC without $\\hat\\tau^2$ term",
    "da_hac_normal": "DA-HAC, normal reference",
    "hac_block_bw": "HAC, block-size bandwidth",
}
PLAIN = {k: v.replace("$t$", "t").replace("$\\hat\\tau^2$", "τ²") for k, v in LABEL.items()}

# Reference categorical palette (dataviz skill, light mode), fixed slot order.
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STYLE = {  # colour follows the method everywhere; markers give a second encoding
    "da_hac": (SLOT[0], "o"),
    "paired_t": (SLOT[1], "s"),
    "iid_bootstrap": (SLOT[2], "^"),
    "block_bootstrap": (SLOT[3], "D"),
    "mixed_model": (SLOT[4], "v"),
    "dm_hac": (SLOT[5], "P"),
    "nadeau_bengio": (SLOT[6], "X"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 1.6, "lines.markersize": 5,
    "savefig.bbox": "tight", "savefig.dpi": 300,
})


def mc_rate(x):
    x = np.asarray(x, float)
    p = x.mean()
    return p, np.sqrt(p * (1 - p) / len(x))


def fmt_rate(p, se=None, bold=False, low=None, high=None):
    s = f"{100 * p:.1f}"
    if low is not None and (p < low or p > high):
        s = f"\\textbf{{{s}}}"
    return s


def add_flags(df):
    df = df.copy()
    df["reject"] = df.pvalue < ALPHA
    if "truth" in df:
        df["covered"] = (df.ci_low <= df.truth) & (df.truth <= df.ci_high)
    df["width"] = df.ci_high - df.ci_low
    return df


# ---------------------------------------------------------------------------
# Experiment 1
# ---------------------------------------------------------------------------

def exp1_tables(df):
    null = df[df.delta == 0]
    methods = ["paired_t", "iid_bootstrap", "block_bootstrap", "mixed_model", "dm_hac", "da_hac",
               "da_hac_no_tau", "da_hac_normal", "hac_block_bw"]
    n_rep = null.groupby(["level", "method"]).size().min()
    band = 1.96 * np.sqrt(ALPHA * (1 - ALPHA) / n_rep)
    lo, hi = ALPHA - band, ALPHA + band
    lines = []
    for m in methods:
        cells = []
        for lvl in LEVELS:
            sub = null[(null.level == lvl) & (null.method == m)]
            cells.append(fmt_rate(sub.reject.mean(), low=lo, high=hi))
        cov = []
        for lvl in LEVELS:
            sub = df[(df.level == lvl) & (df.method == m)]
            cov.append(f"{100 * sub.covered.mean():.1f}")
        lines.append(f"{LABEL[m]} & " + " & ".join(cells) + " & " + " & ".join(cov) + r" \\")
        if m == "da_hac":
            lines.append(r"\midrule")
    (TAB / "exp1_size.tex").write_text("\n".join(lines) + "\n")

    alt = df[df.delta > 0]
    lines = []
    for m in methods[:6]:
        cells = []
        for lvl in ["none", "moderate", "strong"]:
            # size-adjusted: critical p-value = 5% quantile of the null p-values
            # (undefined when more than 5% of null p-values sit at the smallest
            # attainable bootstrap p-value)
            p0 = null[(null.level == lvl) & (null.method == m)].pvalue
            crit = np.quantile(p0, ALPHA)
            ties = (p0 <= p0.min()).mean() > ALPHA
            for dl in sorted(alt.delta.unique()):
                sub = alt[(alt.level == lvl) & (alt.method == m) & (alt.delta == dl)]
                cells.append("--" if ties else f"{100 * (sub.pvalue <= crit).mean():.0f}")
        lines.append(f"{LABEL[m]} & " + " & ".join(cells) + r" \\")
    (TAB / "exp1_power.tex").write_text("\n".join(lines) + "\n")
    return n_rep, lo, hi


def size_panel(ax, null, methods, title, n_rep):
    band = 1.96 * np.sqrt(ALPHA * (1 - ALPHA) / n_rep)
    ax.axhspan(ALPHA - band, ALPHA + band, color=GRID, alpha=0.9, lw=0, zorder=0)
    ax.axhline(ALPHA, color=MUTED, lw=0.8, ls="--", zorder=1)
    x = np.arange(len(LEVELS))
    for m in methods:
        rates = [null[(null.level == l) & (null.method == m)].reject.mean() for l in LEVELS]
        c, mk = STYLE[m]
        ax.plot(x, rates, color=c, marker=mk, label=PLAIN[m], zorder=3 if m == "da_hac" else 2,
                lw=2.2 if m == "da_hac" else 1.4)
    ax.set_xticks(x, [l.capitalize() for l in LEVELS])
    ax.set_xlabel("Dependence level")
    ax.set_title(title, loc="left", color=INK)


def exp1_figures(df, n_rep):
    null = df[df.delta == 0]
    methods = ["paired_t", "iid_bootstrap", "block_bootstrap", "mixed_model", "dm_hac", "da_hac"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    size_panel(axes[0], null, methods, "(a) Type I error rate", n_rep)
    axes[0].set_ylabel("Rejection rate at $\\alpha=0.05$")
    axes[0].set_yscale("log")
    axes[0].set_yticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5], ["0.01", "0.02", "0.05", "0.10", "0.20", "0.50"])
    axes[0].set_ylim(0.005, 0.8)

    ax = axes[1]
    x = np.arange(len(LEVELS))
    for m in methods:
        widths = [df[(df.level == l) & (df.method == m) & (df.delta == 0)].width.median() for l in LEVELS]
        c, mk = STYLE[m]
        ax.plot(x, widths, color=c, marker=mk, lw=2.2 if m == "da_hac" else 1.4)
    ax.set_yscale("log")
    ax.set_yticks([0.2, 0.3, 0.5, 0.7, 1.0, 1.5], ["0.2", "0.3", "0.5", "0.7", "1.0", "1.5"])
    ax.minorticks_off()
    ax.set_xticks(x, [l.capitalize() for l in LEVELS])
    ax.set_xlabel("Dependence level")
    ax.set_ylabel("Median 95% CI width")
    ax.set_title("(b) Interval width", loc="left", color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, bbox_to_anchor=(0.5, -0.05), fontsize=7.5,
               handlelength=1.8, columnspacing=1.2)
    fig.savefig(FIG / "exp1_size.pdf")
    plt.close(fig)

    alt = df.copy()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True)
    deltas = sorted(alt.delta.unique())
    for ax, lvl in zip(axes, ["none", "moderate", "strong"]):
        for m in methods:
            r = [alt[(alt.level == lvl) & (alt.method == m) & (alt.delta == dl)].reject.mean() for dl in deltas]
            c, mk = STYLE[m]
            ax.plot(deltas, r, color=c, marker=mk, lw=2.2 if m == "da_hac" else 1.4, label=PLAIN[m])
        ax.axhline(ALPHA, color=MUTED, lw=0.8, ls="--")
        ax.set_title("No dependence" if lvl == "none" else f"{lvl.capitalize()} dependence", loc="left", color=INK)
        ax.set_xlabel("True loss gap $\\delta$")
    axes[0].set_ylabel("Rejection rate")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, bbox_to_anchor=(0.5, -0.05), fontsize=7.5)
    fig.savefig(FIG / "exp1_power.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 2
# ---------------------------------------------------------------------------

def exp2_tables_figures(df):
    null = df[df.gamma == 0]
    n_rep = null.groupby(["level", "scheme", "method"]).size().min()
    band = 1.96 * np.sqrt(ALPHA * (1 - ALPHA) / n_rep)
    lo, hi = ALPHA - band, ALPHA + band
    rows = [("random", m) for m in ["paired_t", "iid_bootstrap", "nadeau_bengio"]]
    rows += [("blocked", m) for m in ["paired_t", "iid_bootstrap", "block_bootstrap", "mixed_model",
                                      "dm_hac", "da_hac"]]
    gammas = sorted(df[df.gamma > 0].gamma.unique())
    lines = []
    for i, (sch, m) in enumerate(rows):
        if i == 3:
            lines.append(r"\midrule")
        size = [fmt_rate(null[(null.level == l) & (null.scheme == sch) & (null.method == m)].reject.mean(),
                         low=lo, high=hi) for l in LEVELS]
        pw = [f"{100 * df[(df.level == 'moderate') & (df.scheme == sch) & (df.method == m) & (df.gamma == g)].reject.mean():.1f}"
              for g in gammas]
        lines.append(f"{sch.capitalize()} & {LABEL[m]} & " + " & ".join(size) + " & " + " & ".join(pw) + r" \\")
    (TAB / "exp2_size_power.tex").write_text("\n".join(lines) + "\n")

    # Estimand shift: mean estimate under each scheme vs gamma
    est = df[df.method == "paired_t"].groupby(["level", "gamma", "scheme"]).estimate.agg(["mean", "std", "size"])
    est.to_csv(RES / "exp2_estimand.csv")
    lines = []
    for lvl in LEVELS:
        cells = []
        for g in [0.0] + gammas:
            r = est.loc[(lvl, g, "random")]
            b = est.loc[(lvl, g, "blocked")]
            cells.append(f"{r['mean']:.3f} & {b['mean']:.3f}")
        lines.append(f"{lvl.capitalize()} & " + " & ".join(cells) + r" \\")
    (TAB / "exp2_estimand.tex").write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, sch, methods, title in [
            (axes[0], "random", ["paired_t", "iid_bootstrap", "nadeau_bengio"], "(a) Random 5-fold CV"),
            (axes[1], "blocked", ["paired_t", "iid_bootstrap", "block_bootstrap", "mixed_model", "dm_hac", "da_hac"],
             "(b) Blocked 5-fold CV")]:
        size_panel(ax, null[null.scheme == sch], methods, title, n_rep)
    axes[0].set_ylabel("Rejection rate at $\\alpha=0.05$")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_yticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0], ["0.01", "0.02", "0.05", "0.10", "0.20", "0.50", "1"])
        ax.set_ylim(0.005, 1.0)
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, -0.05), fontsize=7.2,
               columnspacing=1.0, handlelength=1.6)
    fig.savefig(FIG / "exp2_size.pdf")
    plt.close(fig)
    return n_rep, lo, hi


# ---------------------------------------------------------------------------
# Experiment 3
# ---------------------------------------------------------------------------

DATASETS = {"nyc_weather": "NYC weather (rain, log loss)", "pedestrian": "Melbourne pedestrians (log count, MSE)",
            "lasrosas": "Las Rosas maize (yield, MSE)"}
COMPARISONS = {"xgb_vs_rf": "XGBoost vs random forest", "xgb_vs_linear": "XGBoost vs linear"}


def exp3_tables_figures():
    frames = [pd.read_csv(RES / f"exp3_{d}.csv") for d in DATASETS if (RES / f"exp3_{d}.csv").exists()]
    if not frames:
        return
    df = add_flags(pd.concat(frames))
    order = [("random", "paired_t"), ("random", "iid_bootstrap"), ("random", "nadeau_bengio"),
             ("blocked", "paired_t"), ("blocked", "block_bootstrap"), ("blocked", "mixed_model"),
             ("blocked", "dm_hac"), ("blocked", "da_hac")]
    lines = []
    for ds in DATASETS:
        sub_ds = df[df.dataset == ds]
        if sub_ds.empty:
            continue
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{DATASETS[ds]}}}}} \\\\")
        for comp in COMPARISONS:
            for sch, m in order:
                r = sub_ds[(sub_ds.comparison == comp) & (sub_ds.scheme == sch) & (sub_ds.method == m)]
                if r.empty:
                    continue
                r = r.iloc[0]
                p = r.pvalue
                ps = "$<$0.001" if p < 0.001 else f"{p:.3f}"
                lines.append(f"\\quad {COMPARISONS[comp] if (sch, m) == order[0] else ''} & {sch} & {LABEL[m]} & "
                             f"{r.estimate:.4f} [{r.ci_low:.4f}, {r.ci_high:.4f}] & {ps} \\\\")
            lines.append(r"\addlinespace")
    (TAB / "exp3_main.tex").write_text("\n".join(lines) + "\n")

    # Diagnostics of the proposed procedure
    lines = []
    for ds in DATASETS:
        for comp in COMPARISONS:
            r = df[(df.dataset == ds) & (df.comparison == comp) & (df.method == "da_hac")]
            if r.empty:
                continue
            r = r.iloc[0]
            rs = "--" if np.isnan(r.info_range_space) else f"{r.info_range_space:.0f} m"
            rt = "--" if np.isnan(r.info_range_time) else f"{r.info_range_time:.0f} h"
            lines.append(f"{DATASETS[ds].split(' (')[0]} & {COMPARISONS[comp]} & {rs} & {rt} & "
                         f"{r.info_n_eff:.0f} & {r.df:.0f} & {r.info_tau2:.2e} \\\\")
    (TAB / "exp3_diag.tex").write_text("\n".join(lines) + "\n")

    # Forest plot
    plot_order = [("random", "paired_t"), ("random", "nadeau_bengio"), ("blocked", "paired_t"),
                  ("blocked", "block_bootstrap"), ("blocked", "mixed_model"), ("blocked", "da_hac")]
    present = [d for d in DATASETS if not df[df.dataset == d].empty]
    fig, axes = plt.subplots(len(present), 2, figsize=(7.0, 1.55 * len(present) + 0.4), squeeze=False)
    for i, ds in enumerate(present):
        for j, comp in enumerate(COMPARISONS):
            ax = axes[i, j]
            for k, (sch, m) in enumerate(plot_order):
                r = df[(df.dataset == ds) & (df.comparison == comp) & (df.scheme == sch) & (df.method == m)]
                if r.empty:
                    continue
                r = r.iloc[0]
                y = len(plot_order) - k
                c, mk = STYLE[m]
                ax.plot([r.ci_low, r.ci_high], [y, y], color=c, lw=2, solid_capstyle="round")
                ax.plot(r.estimate, y, marker=mk, color=c, ms=5, mec="white", mew=0.8)
            ax.axvline(0, color=MUTED, lw=0.8, ls="--")
            ax.set_yticks(range(len(plot_order), 0, -1),
                          [f"{s[0].upper()}: {PLAIN[m]}" for s, m in plot_order] if j == 0 else [""] * len(plot_order))
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(axis="y", visible=False)
            if i == 0:
                ax.set_title(COMPARISONS[comp], color=INK)
            if j == 1:
                ax.text(1.02, 0.5, DATASETS[ds].split(" (")[0], transform=ax.transAxes, rotation=270,
                        va="center", ha="left", fontsize=7.5, color=MUTED)
    for ax in axes[-1]:
        ax.set_xlabel("Loss differential (A $-$ B); 95% CI")
    fig.tight_layout()
    fig.savefig(FIG / "exp3_forest.pdf")
    plt.close(fig)

    # Seed-null diagnostic
    lines = []
    for ds in DATASETS:
        f = RES / f"exp3_{ds}_seednull.csv"
        if not f.exists():
            continue
        s = pd.read_csv(f)
        s["reject"] = s.pvalue < ALPHA
        cells = []
        for sch, m in order:
            sub = s[(s.scheme == sch) & (s.method == m)]
            cells.append("--" if sub.empty else f"{100 * sub.reject.mean():.0f}")
        lines.append(f"{DATASETS[ds].split(' (')[0]} & " + " & ".join(cells) + r" \\")
    (TAB / "exp3_seednull.tex").write_text("\n".join(lines) + "\n")

    # Variograms
    vfiles = [RES / f"exp3_{d}_variogram.csv" for d in DATASETS if (RES / f"exp3_{d}_variogram.csv").exists()]
    if vfiles:
        v = pd.concat([pd.read_csv(f) for f in vfiles])
        v = v[v.comparison == "xgb_vs_rf"]
        panels = list(v.groupby(["dataset", "direction"], sort=False))
        fig, axes = plt.subplots(1, len(panels), figsize=(7.0, 2.2))
        axes = np.atleast_1d(axes)
        for ax, ((ds, direction), g) in zip(axes, panels):
            ax.plot(g.lag, g.gamma, "o", color=SLOT[0], ms=4)
            h = np.linspace(0, g.lag.max(), 200)
            f0 = g.iloc[0]
            if f0.fit_a > 0:
                ax.plot(h, f0.fit_c0 + f0.fit_c1 * (1 - np.exp(-h / f0.fit_a)), color=SLOT[1], lw=1.6)
            if f0.fit_range > 0:
                ax.axvline(f0.fit_range, color=MUTED, lw=0.8, ls=":")
            unit = "h" if direction == "time" else "m"
            ax.set_xlabel(f"{'Time lag' if direction == 'time' else 'Distance'} ({unit})")
            ax.set_title(DATASETS[ds].split(" (")[0], loc="left", color=INK, fontsize=8)
            ax.set_ylim(bottom=0)
        axes[0].set_ylabel("Semivariance of $\\tilde u$")
        fig.tight_layout()
        fig.savefig(FIG / "exp3_variograms.pdf")
        plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    summary = {}
    if (RES / "exp1.csv").exists():
        df1 = add_flags(pd.read_csv(RES / "exp1.csv"))
        n_rep, lo, hi = exp1_tables(df1)
        exp1_figures(df1, n_rep)
        summary["exp1"] = (n_rep, lo, hi)
    if (RES / "exp2.csv").exists():
        df2 = add_flags(pd.read_csv(RES / "exp2.csv"))
        summary["exp2"] = exp2_tables_figures(df2)
    exp3_tables_figures()
    print(summary)


if __name__ == "__main__":
    main()
