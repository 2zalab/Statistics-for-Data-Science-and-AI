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
    "hering_genton": "Hering--Genton (space--time)",
    "da_hac_rho0": "DA-HAC, $\\rho_0=0$",
    "da_hac_rho0.5": "DA-HAC, $\\rho_0=0.5$",
}
PLAIN = {k: v.replace("$t$", "t").replace("$\\hat\\tau^2$", "τ²").replace("--", "–")
         .replace("$\\rho_0=", "ρ₀=").replace("$", "") for k, v in LABEL.items()}

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
    "hering_genton": (SLOT[7], "*"),
}
RSTAR = [0.25]  # updated by exp1b_tables once the pilot has been analysed
MAIN = ["paired_t", "iid_bootstrap", "block_bootstrap", "mixed_model", "dm_hac", "hering_genton", "da_hac"]
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
    methods = MAIN + ["da_hac_no_tau", "da_hac_normal", "hac_block_bw"]
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
    for m in MAIN:
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
    methods = MAIN
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
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.05), fontsize=7.5,
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
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.05), fontsize=7.5)
    fig.savefig(FIG / "exp1_power.pdf")
    plt.close(fig)

    # Coverage for each true gap and dependence level
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.3), sharey=True)
    for ax, lvl in zip(axes, LEVELS):
        for m in methods:
            cv = [df[(df.level == lvl) & (df.method == m) & (df.delta == dl)].covered.mean() for dl in deltas]
            c, mk = STYLE[m]
            ax.plot(deltas, cv, color=c, marker=mk, lw=2.2 if m == "da_hac" else 1.4, label=PLAIN[m])
        ax.axhline(1 - ALPHA, color=MUTED, lw=0.8, ls="--")
        ax.set_title("No dependence" if lvl == "none" else lvl.capitalize(), loc="left", color=INK)
        ax.set_xlabel("True loss gap $\\delta$")
        ax.set_ylim(0.15, 1.0)
    axes[0].set_ylabel("Coverage of 95% CI")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.05), fontsize=7.5)
    fig.savefig(FIG / "exp1_coverage.pdf")
    plt.close(fig)

    # Boundary case: truncation of the training variance at zero
    lines = []
    for lvl in LEVELS:
        sub = null[null.method == "da_hac"]
        sub = sub[sub.level == lvl]
        if "info_tau2_raw" not in sub:
            break
        trunc = sub.info_tau2_raw <= 0
        r1 = sub[trunc].reject.mean() if trunc.any() else np.nan
        r0 = sub[~trunc].reject.mean() if (~trunc).any() else np.nan
        lines.append(f"{lvl.capitalize()} & {100 * trunc.mean():.1f} & {100 * r1:.1f} & {100 * r0:.1f} & "
                     f"{sub[trunc].df.median():.0f} & {sub[~trunc].df.median():.0f} \\\\")
    (TAB / "exp1_boundary.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Experiment 2
# ---------------------------------------------------------------------------

def exp2_tables_figures(df):
    null = df[df.gamma == 0]
    n_rep = null.groupby(["level", "scheme", "method"]).size().min()
    band = 1.96 * np.sqrt(ALPHA * (1 - ALPHA) / n_rep)
    lo, hi = ALPHA - band, ALPHA + band
    rows = [("random", m) for m in ["paired_t", "iid_bootstrap", "nadeau_bengio"]]
    rows += [("blocked", m) for m in MAIN]
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
            (axes[1], "blocked", MAIN, "(b) Blocked 5-fold CV")]:
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
COMPARISONS = {"xgb_vs_rf": "XGBoost vs random forest", "xgb_vs_linear": "XGBoost vs linear",
               "rf_vs_linear": "Random forest vs linear"}
SCHEME = {"random": "R", "blocked": "B", "forward": "F"}


def holm(pvals):
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def exp3_tables_figures():
    frames = [pd.read_csv(RES / f"exp3_{d}.csv") for d in DATASETS if (RES / f"exp3_{d}.csv").exists()]
    if not frames:
        return
    df = add_flags(pd.concat(frames))
    df["p_holm"] = np.nan
    for _, g in df.groupby(["dataset", "scheme", "method"]):
        if len(g) > 1:
            df.loc[g.index, "p_holm"] = holm(g.pvalue.values)
    order = [("random", "paired_t"), ("random", "iid_bootstrap"), ("random", "nadeau_bengio"),
             ("blocked", "paired_t"), ("blocked", "iid_bootstrap"), ("blocked", "block_bootstrap"),
             ("blocked", "mixed_model"), ("blocked", "dm_hac"), ("blocked", "hering_genton"),
             ("blocked", "da_hac"), ("forward", "paired_t"), ("forward", "block_bootstrap"),
             ("forward", "hering_genton"), ("forward", "da_hac")]
    df = df.reset_index(drop=True)
    lines = []
    for ds in DATASETS:
        sub_ds = df[df.dataset == ds]
        if sub_ds.empty:
            continue
        lines.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textbf{{{DATASETS[ds]}}}}} \\\\")
        for comp in COMPARISONS:
            lines.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{COMPARISONS[comp]}}}}} \\\\")
            for sch, m in order:
                r = sub_ds[(sub_ds.comparison == comp) & (sub_ds.scheme == sch) & (sub_ds.method == m)]
                if r.empty:
                    continue
                r = r.iloc[0]
                p = r.pvalue
                ps = "$<$0.001" if p < 0.001 else f"{p:.3f}"
                digits = 4 if abs(r.estimate) < 0.1 else (2 if abs(r.estimate) < 10 else 1)
                ph = r.p_holm
                phs = "--" if np.isnan(ph) else ("$<$0.001" if ph < 0.001 else f"{ph:.3f}")
                lines.append(f"\\quad {SCHEME[sch]} & {LABEL[m]} & "
                             f"{r.estimate:.{digits}f} [{r.ci_low:.{digits}f}, {r.ci_high:.{digits}f}] & {ps} & {phs} \\\\")
            lines.append(r"\addlinespace")
    (TAB / "exp3_main.tex").write_text("\n".join(lines) + "\n")

    # Diagnostics of the proposed procedure
    lines = []
    for ds in DATASETS:
        for comp in COMPARISONS:
            for sch in ("blocked", "forward"):
                r = df[(df.dataset == ds) & (df.comparison == comp) & (df.method == "da_hac") & (df.scheme == sch)]
                if r.empty:
                    continue
                r = r.iloc[0]
                rs = "--" if np.isnan(r.info_range_space) else f"{r.info_range_space:.0f} m"
                rt = "--" if np.isnan(r.info_range_time) else f"{r.info_range_time:.0f} h"
                rstar = RSTAR[0]
                dec = ("indet." if r.info_range_ratio > rstar else ("reject" if r.pvalue < ALPHA else "not rej."))
                lines.append(f"{DATASETS[ds].split(' (')[0]} & {COMPARISONS[comp]} & {SCHEME[sch]} & {rs} & {rt} & "
                             f"{r.info_n_eff:.0f} & {r.df:.0f} & {r.info_range_ratio:.2f} & {dec} \\\\")
    (TAB / "exp3_diag.tex").write_text("\n".join(lines) + "\n")

    # Forest plot
    plot_order = [("random", "paired_t"), ("random", "nadeau_bengio"), ("blocked", "paired_t"),
                  ("blocked", "block_bootstrap"), ("blocked", "hering_genton"), ("blocked", "da_hac"),
                  ("forward", "da_hac")]
    present = [d for d in DATASETS if not df[df.dataset == d].empty]
    fig, axes = plt.subplots(len(present), 3, figsize=(7.0, 1.7 * len(present) + 0.4), squeeze=False)
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
                          [f"{SCHEME[s]}: {PLAIN[m]}" for s, m in plot_order] if j == 0 else [""] * len(plot_order))
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(axis="y", visible=False)
            if i == 0:
                ax.set_title(COMPARISONS[comp], color=INK, fontsize=8)
            if j == 2:
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
        for sch, m in [("random", "paired_t"), ("random", "iid_bootstrap"), ("blocked", "paired_t"),
                       ("blocked", "iid_bootstrap"), ("blocked", "block_bootstrap"), ("blocked", "mixed_model"),
                       ("blocked", "dm_hac"), ("blocked", "hering_genton"), ("blocked", "da_hac")]:
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


# ---------------------------------------------------------------------------
# Experiment 1b: training correlation, robustness, decision rule
# ---------------------------------------------------------------------------

def choose_threshold(null_dahac):
    """Largest range-ratio threshold whose conditional size stays in the MC band."""
    best = None
    for r in np.round(np.arange(0.10, 0.601, 0.025), 3):
        det = null_dahac[null_dahac.info_range_ratio <= r]
        if len(det) < 100:
            continue
        upper = ALPHA + 1.96 * np.sqrt(ALPHA * (1 - ALPHA) / len(det))
        if det.reject.mean() <= upper:
            best = r
    return best


def rule_rows(null, alt, group, label_fn, rstar):
    """Rejection rates of DA-HAC under the three-way rule, by group."""
    lines = []
    for g in null[group].unique():
        n0 = null[(null[group] == g) & (null.method == "da_hac")]
        det = n0.info_range_ratio <= rstar
        size_all = n0.reject.mean()
        size_det = n0[det].reject.mean() if det.any() else np.nan
        cells = [label_fn(g), f"{n0.info_range_ratio.median():.2f}", f"{100 * (~det).mean():.0f}",
                 f"{100 * size_all:.1f}", "--" if np.isnan(size_det) else f"{100 * size_det:.1f}"]
        if alt is not None:
            a0 = alt[(alt[group] == g) & (alt.method == "da_hac")]
            deta = a0.info_range_ratio <= rstar
            cells.append(f"{100 * (a0.reject & deta).mean():.0f}" if len(a0) else "--")
        lines.append(" & ".join(cells) + r" \\")
    return lines


def exp1b_tables(df1=None, df2=None):
    out = {}
    f = RES / "exp1b_rho.csv"
    if f.exists():
        d = add_flags(pd.read_csv(f))
        d = d[d.delta == 0]
        methods = ["paired_t", "block_bootstrap", "mixed_model", "hering_genton",
                   "da_hac_rho0", "da_hac", "da_hac_rho0.5"]
        lines = []
        for sc in d.scenario.unique():
            lvl, rho = sc.split("_rho")
            cells = [f"{100 * d[(d.scenario == sc) & (d.method == m)].reject.mean():.1f}" for m in methods]
            lines.append(f"{'None' if lvl == 'none' else 'Moderate'} & {rho} & " + " & ".join(cells) + r" \\")
        (TAB / "exp1b_rho.tex").write_text("\n".join(lines) + "\n")

    f = RES / "exp1b_robust.csv"
    if f.exists():
        d = add_flags(pd.read_csv(f))
        names = {"baseline": "Baseline (moderate)", "anisotropic": "Anisotropic (4:1)",
                 "gaussian_cov": "Gaussian covariance", "heteroscedastic": "Heteroscedastic",
                 "nonstationary": "Non-stationary range", "irregular_sites": "Irregular sites"}
        methods = ["paired_t", "block_bootstrap", "mixed_model", "hering_genton", "da_hac"]
        lines = []
        for sc, nm in names.items():
            n0 = d[(d.scenario == sc) & (d.delta == 0)]
            a0 = d[(d.scenario == sc) & (d.delta > 0)]
            cells = [f"{100 * n0[n0.method == m].reject.mean():.1f}" for m in methods]
            da = a0[a0.method == "da_hac"]
            pt = a0[a0.method == "paired_t"]
            cells += [f"{100 * da.covered.mean():.1f}", f"{100 * da.reject.mean():.0f}",
                      f"{100 * pt.reject.mean():.0f}"]
            lines.append(f"{nm} & " + " & ".join(cells) + r" \\")
        (TAB / "exp1b_robust.tex").write_text("\n".join(lines) + "\n")

    fp, fm = RES / "exp1b_range_pilot.csv", RES / "exp1b_range.csv"
    if fp.exists() and fm.exists():
        pilot = add_flags(pd.read_csv(fp))
        rstar = choose_threshold(pilot[(pilot.method == "da_hac")])
        out["rstar"] = rstar
        RSTAR[0] = rstar
        (TAB / "rstar.tex").write_text(f"\\renewcommand{{\\rstar}}{{{rstar:g}}}\n")
        main = add_flags(pd.read_csv(fm))
        lab = lambda g: f"{3 * float(g[1:]):g}"
        lines = rule_rows(main, None, "scenario", lab, rstar)
        (TAB / "exp1b_range.tex").write_text("\n".join(lines) + "\n")
        # pooled validation on the main experiments
        lines = []
        if df1 is not None:
            n1 = df1[df1.delta == 0]
            lines += [r"\multicolumn{6}{@{}l}{\emph{Experiment 1 (alternative: $\delta=0.2$)}}\\"]
            lines += rule_rows(n1, df1[df1.delta == 0.2], "level", str.capitalize, rstar)
        if df2 is not None:
            b = df2[df2.scheme == "blocked"]
            lines += [r"\multicolumn{6}{@{}l}{\emph{Experiment 2 (alternative: $\gamma=0.2$)}}\\"]
            lines += rule_rows(b[b.gamma == 0], b[b.gamma == 0.2], "level", str.capitalize, rstar)
        (TAB / "rule_validation.tex").write_text("\n".join(lines) + "\n")

        # Figure: null rejection rate of DA-HAC and paired t against the range ratio
        pooled = [main.assign(src="1b")]
        if df1 is not None:
            pooled.append(df1[df1.delta == 0].assign(src="1"))
        if df2 is not None:
            pooled.append(df2[(df2.scheme == "blocked") & (df2.gamma == 0)].assign(src="2"))
        allnull = pd.concat(pooled, ignore_index=True)
        for c in ("scenario", "level"):
            if c in allnull:
                allnull[c] = allnull[c].fillna("")
        dah = allnull[allnull.method == "da_hac"].copy()
        keycols = ["src", "scenario", "level", "rep"]
        keycols = [c for c in keycols if c in allnull]
        rr = dah.set_index(keycols).info_range_ratio
        bins = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.7]
        fig, ax = plt.subplots(figsize=(4.6, 2.6))
        for m in ["paired_t", "block_bootstrap", "hering_genton", "da_hac"]:
            sub = allnull[allnull.method == m].set_index(keycols)
            sub = sub.assign(rr=rr.reindex(sub.index).values)
            sub["bin"] = pd.cut(sub.rr, bins)
            g = sub.groupby("bin", observed=True).reject.agg(["mean", "size"])
            g = g[g["size"] >= 50]
            x = [iv.mid for iv in g.index]
            c, mk = STYLE[m]
            ax.plot(x, g["mean"], color=c, marker=mk, lw=2.2 if m == "da_hac" else 1.4, label=PLAIN[m])
        ax.axhline(ALPHA, color=MUTED, lw=0.8, ls="--")
        ax.axvline(rstar, color=INK, lw=0.8, ls=":")
        ax.text(rstar, ax.get_ylim()[1] * 0.97, f" $r^*={rstar:g}$", va="top", fontsize=7.5, color=INK)
        ax.set_xlabel("Estimated range ratio $\\max_q b_q/L_q$ (binned)")
        ax.set_ylabel("Rejection rate of a true null")
        ax.legend(fontsize=7, loc="upper left")
        fig.savefig(FIG / "rule.pdf")
        plt.close(fig)
    return out


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
    exp1b = exp1b_tables(df1 if "exp1" in summary else None, df2 if "exp2" in summary else None)
    summary.update(exp1b)
    exp3_tables_figures()
    print(summary)


if __name__ == "__main__":
    main()
