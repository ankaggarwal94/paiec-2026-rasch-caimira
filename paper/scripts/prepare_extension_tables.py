"""Select compact appendix columns from the preserved hierarchical study tables."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def select_columns(source: str, target: str, columns: tuple[int, ...], count: int, header: str) -> None:
    output = []
    for row in (ROOT / "generated" / source).read_text().splitlines():
        cells = row.split(" & ")
        assert len(cells) == count, (source, len(cells), row)
        cells[-1] = cells[-1].removesuffix(r" \\ \midrule").removesuffix(r" \\")
        boundary = r" \midrule" if "CaimiraLite (static)" in row else ""
        output.append(" & ".join(cells[index] for index in columns) + r" \\" + boundary)
    (ROOT / "generated" / target).write_text(header + "\n" + "\n".join(output) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n")


if __name__ == "__main__":
    results_header = r"""\begin{tabular}{@{}lrrrrc@{}}
\toprule
 & \multicolumn{2}{c}{Brier ALC} & \multicolumn{2}{c}{Hold-out Brier} & \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Model & Dev & Hold-out & 0 labels & 31 labels & $\Delta$ vs. IRT \\
\midrule"""
    ablation_header = r"""\begin{tabular}{@{}lccc@{}}
\toprule
Layer added & Dev $\Delta$ ALC & Hold-out $\Delta$ ALC & Win (\%) \\
\midrule"""
    select_columns("tab_main.tex", "tab_extension.tex", (0, 1, 2, 3, 6, 9), 10, results_header)
    select_columns("tab_ablation.tex", "tab_extension_ablation.tex", (0, 1, 2, 3), 8, ablation_header)
    print("Prepared the two hierarchical-extension tables from preserved generated rows.")
