"""Regenerate the preserved hierarchical-extension tables and figure from runs/.

    cd paiec && python report/make_numbers.py && cd report && pdflatex main && bibtex main && pdflatex main && pdflatex main

Writes report/generated/{numbers.tex, tab_main.tex, tab_ablation.tex, fig_results.pdf}. These outputs support appendix_shared.tex. The selected Rasch-CAIMIRA submission uses the separate
scripts/analyze_feedback.py workflow. CONFIG below selects the historical hierarchical stack.
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "generated"
BUDGETS = ("B0", "B1", "B3", "B7", "B15", "B31")

# ----------------------------------------------------------------------------------------------- CONFIG
# Hold-out runs are runs/v2h__<key>; dev runs are given explicitly (None = no dev run, shown as "--").
# STACK is cumulative: each row adds one decision to the row above it, and its Δ in the ablation table is paired
# against the row above. The LAST row of STACK is the final (submitted) model unless FINAL says otherwise.
STACK = [
    # (hold-out key,       dev run,                    table label)
    ("champion",           "v2d__champion_merged",     "IRT (global evidence)"),
    ("item_A3",            "v2d__item_A3",             "+ online item map"),
    ("item_lr_vi",         "v2d__item_lr_vi_merged",   "+ subject regression, VI"),
    ("gate_best",          "v2d__gate_best_merged",    "+ gate"),
    ("gate_best_r30",      "v2a__gate_best__r30",      "+ acquisition radius 0.30"),
    ("gate_best_llm_r30",  None,                       "+ LLM difficulty prior"),
]
# The selected hierarchical model in the preserved development study. Must be a key in STACK.
FINAL = "gate_best_r30"
FINAL_NAME = "Gate"                # short name used in the text
FINAL_USES_NETWORK = False         # True if it calls an LLM API at test time
# The LLM-prior variant discussed in Sec. 3.5 / 4 (a key in STACK, and the row it is paired against).
LLM_RUN, LLM_CTRL = "gate_best_llm_r30", "gate_best_r30"
LLM_NAME = "Qwen3-32B / Qwen3.8-27B"
BASELINES = [
    ("constant",          "v2d__constant",          "Constant 0.5"),
    ("official_baseline", "v2d__official_baseline", "Official baseline"),
    ("lex_cosine",        "v2d__lex_cosine",        "Lexical $k$NN"),
    ("caimira_ens",       "v2d__caimira_ens",       "CaimiraLite (static)"),
]
REF = "champion"                   # the "IRT" row every model is compared with in Table 2
# ------------------------------------------------------------------------------------------- end CONFIG


def load(run):
    d = ROOT / "runs" / run
    with open(d / "pairs.csv") as f:
        rows = {(r["subject_id"], r["benchmark"], r["seed"]): r for r in csv.DictReader(f)}
    return rows, json.loads((d / "metrics.json").read_text())


def paired(ctrl, run, col="alc"):
    """Paired Δ (run − ctrl): pair mean ± CI, benchmark-clustered mean ± CI, win share, per-budget Δ, per-benchmark Δ."""
    c, _ = load(ctrl)
    r, _ = load(run)
    keys = sorted(set(c) & set(r))
    assert len(keys) == len(c), f"{run} matches only {len(keys)}/{len(c)} pairs of {ctrl}"
    d = np.array([float(r[k][col]) - float(c[k][col]) for k in keys])
    by = {}
    for k, x in zip(keys, d):
        by.setdefault(k[1], []).append(x)
    bm = {b: float(np.mean(v)) for b, v in by.items()}
    bv = np.array(list(bm.values()))
    return dict(
        d=d.mean(), ci=1.96 * d.std(ddof=1) / np.sqrt(len(d)),
        db=bv.mean(), cib=1.96 * bv.std(ddof=1) / np.sqrt(len(bv)),
        win=100 * np.mean(d < 0), n=len(d), nb=len(bv), bench=bm,
        budgets={b: float(np.mean([float(r[k][b]) - float(c[k][b]) for k in keys])) for b in BUDGETS},
    )


def f4(x):
    return f"{x:.4f}"


def f3(x):
    return f"{x:.3f}"


def sg(x, nd=4):
    if round(x, nd) == 0:
        return f"{0:.{nd}f}"
    return f"{'+' if x >= 0 else '$-$'}{abs(x):.{nd}f}"


def ci(x, c, nd=4):
    return f"{sg(x, nd)}\\,{{\\scriptsize$\\pm${c:.{nd}f}}}"


def main():
    OUT.mkdir(exist_ok=True)
    keys = [k for k, _, _ in STACK]
    assert FINAL in keys and LLM_RUN in keys and LLM_CTRL in keys
    hk = lambda k: f"v2h__{k}"  # noqa: E731
    met = {k: load(hk(k))[1] for k, _, _ in STACK + BASELINES}
    dev = {k: (json.loads((ROOT / "runs" / d / "metrics.json").read_text()) if d else None) for k, d, _ in STACK + BASELINES}

    # ---------------------------------------------------------------- Table 2: main comparison (hold-out + dev ALC)
    rows = []
    for i, (k, _, lab) in enumerate(BASELINES + STACK):
        m, dm = met[k], dev[k]
        bb, st = m["by_budget"], m["strata"]
        p = paired(hk(REF), hk(k)) if k != REF else None
        cells = [lab, f4(dm["alc"]) if dm else "--", f4(m["alc"]),
                 *(f3(bb[b]) for b in ("0", "1", "7", "31")),
                 f3(st["seen"]["alc"]), f3(st["unseen"]["alc"]),
                 ci(p["db"], p["cib"]) if p else "0"]
        if k == FINAL:
            cells = [f"\\textbf{{{c}}}" if j in (0, 2) else c for j, c in enumerate(cells)]
        rows.append(" & ".join(cells) + r" \\" + (r" \midrule" if i == len(BASELINES) - 1 else ""))
    (OUT / "tab_main.tex").write_text("\n".join(rows) + "\n")

    # ---------------------------------------------------------------- Table 3: ablation (marginal Δ of each layer)
    abl = []
    for (pk, pdev, _), (k, kdev, lab) in zip(STACK, STACK[1:]):
        h = paired(hk(pk), hk(k))
        dv = paired(pdev, kdev) if pdev and kdev else None
        abl.append(" & ".join([
            lab,
            ci(dv["db"], dv["cib"]) if dv else "--",
            ci(h["db"], h["cib"]), f"{h['win']:.0f}",
            *(sg(h["budgets"][b], 3) for b in ("B0", "B1", "B7", "B31")),
        ]) + r" \\")
    (OUT / "tab_ablation.tex").write_text("\n".join(abl) + "\n")

    # ---------------------------------------------------------------- macros
    fm, fb = met[FINAL], met[FINAL]["by_budget"]
    vs_ref = paired(hk(REF), hk(FINAL))
    vs_base = paired(hk("official_baseline"), hk(FINAL))
    vs_const = paired(hk("constant"), hk(FINAL))
    llm = paired(hk(LLM_CTRL), hk(LLM_RUN))
    ranked = sorted(vs_ref["bench"].items(), key=lambda x: x[1])
    n_better = sum(v < 0 for _, v in ranked)
    first = keys[0]
    t = fm.get("timing", {})
    fit_s = max(met[k].get("timing", {}).get("fit_s", 0) for k in keys)
    mac = {
        "FinalName": FINAL_NAME,
        "FinalNetwork": "calls an LLM API at test time" if FINAL_USES_NETWORK else "makes no network calls",
        "FinalALC": f4(fm["alc"]),
        "FinalDevALC": f4(dev[FINAL]["alc"]) if dev[FINAL] else "--",
        "FinalBzero": f3(fb["0"]), "FinalBone": f3(fb["1"]), "FinalBlast": f3(fb["31"]),
        "FinalECE": f3(fm["ece_alc"]),
        "FinalSeen": f4(fm["strata"]["seen"]["alc"]), "FinalUnseen": f4(fm["strata"]["unseen"]["alc"]),
        "FinalSeenGap": f"{fm['strata']['unseen']['alc'] - fm['strata']['seen']['alc']:.3f}",
        "RefSeenGap": f"{met[REF]['strata']['unseen']['alc'] - met[REF]['strata']['seen']['alc']:.3f}",
        "FinalVsRef": ci(vs_ref["db"], vs_ref["cib"]), "FinalVsRefWin": f"{vs_ref['win']:.0f}",
        "FinalVsRefBetter": f"{n_better}", "NBench": f"{vs_ref['nb']}",
        "BestBench": ranked[0][0].replace("_", r"\_"), "BestBenchD": sg(ranked[0][1], 3),
        "WorstBench": ranked[-1][0].replace("_", r"\_"), "WorstBenchD": sg(ranked[-1][1], 3),
        "FinalVsBase": f"{-vs_base['db']:.4f}", "FinalVsBaseCI": f"{vs_base['cib']:.4f}",
        "FinalVsConst": f"{-vs_const['db']:.4f}",
        "FinalRelBase": f"{100 * (met['official_baseline']['alc'] - fm['alc']) / met['official_baseline']['alc']:.0f}",
        "ConstALC": f4(met["constant"]["alc"]),
        "BaseALC": f4(met["official_baseline"]["alc"]), "BaseBone": f3(met["official_baseline"]["by_budget"]["1"]),
        "BaseDevALC": f4(dev["official_baseline"]["alc"]),
        "RefALC": f4(met[REF]["alc"]), "RefDevALC": f4(dev[REF]["alc"]), "RefBzero": f3(met[REF]["by_budget"]["0"]),
        "LLMName": LLM_NAME, "LLMALC": f4(met[LLM_RUN]["alc"]),
        "LLMGain": ci(llm["db"], llm["cib"]), "LLMWin": f"{llm['win']:.0f}",
        "LLMGainPairs": ci(llm["d"], llm["ci"]),
        "NPairSeeds": f"{vs_ref['n']:,}".replace(",", "{,}"),
        "FitMin": f"{fit_s / 60:.0f}",
        "AcqMs": f"{1000 * t.get('acq_s', 0) / max(t.get('acq_calls', 1), 1):.1f}",
        "MinBzero": f3(min(met[k]["by_budget"]["0"] for k in keys)),
        "MaxBzero": f3(max(met[k]["by_budget"]["0"] for k in keys)),
        "StackFirstALC": f4(met[first]["alc"]),
    }
    (OUT / "numbers.tex").write_text(
        "% Generated by make_numbers.py; regenerate through the archived run inputs.\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))

    # ---------------------------------------------------------------- Figure: (a) Brier vs budget, (b) per-benchmark Δ
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.5, 2.0), gridspec_kw={"width_ratios": [1, 1.25]})
    xs = [0, 1, 3, 7, 15, 31]
    curves = [("constant", "Constant 0.5", "#9a9a9a", ":"), ("official_baseline", "Official baseline", "#c4572e", "--"),
              (REF, "IRT", "#5b7fa6", "-"), (FINAL, FINAL_NAME, "#1d3557", "-")]
    for k, lab, col, ls in curves:
        y = [met[k]["by_budget"][str(x)] for x in xs]
        a.plot(range(len(xs)), y, ls, color=col, lw=1.4, marker="o", ms=2.5, label=lab)
    a.set_xticks(range(len(xs)), [str(x) for x in xs])
    a.set_xlabel("labels revealed per pair (budget)")
    a.set_ylabel("Brier (hold-out)")
    a.set_ylim(0.165, 0.40)
    a.legend(frameon=False, fontsize=6.5, loc="upper right")
    a.set_title("(a)", loc="left", fontsize=8)
    names = [n for n, _ in ranked]
    vals = [v for _, v in ranked]
    b.barh(range(len(names)), vals, color=["#1d3557" if v < 0 else "#c4572e" for v in vals], height=0.7)
    b.axvline(0, color="black", lw=0.6)
    b.set_yticks(range(len(names)), [n.replace("_", " ") for n in names], fontsize=5.5)
    b.invert_yaxis()
    b.set_xlabel(f"$\\Delta$ ALC, {FINAL_NAME} $-$ IRT (hold-out)")
    b.set_title("(b)", loc="left", fontsize=8)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    fig.savefig(OUT / "fig_results.pdf")
    print(f"final {FINAL}: hold-out {mac['FinalALC']}, vs IRT {vs_ref['db']:+.4f} ± {vs_ref['cib']:.4f}; wrote {OUT}")


if __name__ == "__main__":
    main()
