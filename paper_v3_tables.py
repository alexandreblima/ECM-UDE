"""Reformat saved results as the manuscript v3 Tables 5, 7 and 9."""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SETTINGS = {
    "Protocolo original — NCA": "E1, NCA",
    "Protocolo original — NMC Samsung": "E1, NMC",
    "Protocolo original — LG HG2": "E5, HG2",
    "Protocolo original — LG M50T": "E5, M50T (7 cells)",
    "M50T G1 F1 fixed (n=6)": "E5, M50T, next aging block (6 cells)",
    "M50T G2 HPPC all (n=7)": "E5, M50T, HPPC in range (7 cells)",
    "A4b — NCA": "E6 sensitivity, NCA",
    "A4b — NMC Samsung": "E6 sensitivity, NMC",
}
ORDER = tuple(SETTINGS)
VARIANTS = {"output_residual": "Output residual", "ecm_ude": "ECM-UDE"}


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def summary_rows():
    rows = read(ROOT/"results/reviewer_round2/E/synthesis.csv")
    selected = {(r["configuration"], r["variant"]): r for r in rows if r["configuration"] in SETTINGS}
    if len(selected) != len(SETTINGS)*len(VARIANTS):
        raise ValueError("Missing or duplicate Table 5 synthesis rows")
    result = []
    for setting in ORDER:
        for variant, label in VARIANTS.items():
            row = selected[setting, variant]
            def interval(reference):
                return "{:.2f} [{:.2f}, {:.2f}]".format(
                    *[float(row[f"{reference}_{part}_difference_mV"]) for part in ("median", "minimum", "maximum")])
            if setting.startswith("A4b"):
                tost = "--"
            elif "M50T" in setting:
                tost = "Yes" if int(row["vs_pg_equivalent_holm_count"]) == 1 else "No"
            else:
                tost = f'{row["vs_pg_equivalent_holm_count"]}/{row["vs_pg_n_tests"]}'
            result.append([SETTINGS[setting], label, interval("vs_pg"), interval("vs_ecm2"), tost])
    return result


def e1_rows():
    rows = read(ROOT/"results/paper_tables/E1.csv")
    if len(rows) != 18:
        raise ValueError("Expected 18 principal E1 conditions")
    result = []
    for row in rows:
        models = [row[name].replace("EQ/ref", "E").replace("ND/ref", "N")
                  .replace("EQ", "E").replace("ND", "N")
                  for name in ("physics_gradient", "output_residual", "ecm_ude")]
        result.append([{"panasonic": "NCA", "calce": "NMC"}[row["dataset"]],
                       row["condition"], row["ECM2RC_mV"], *models])
    return result


def tost_rows():
    rows = read(ROOT/"results/reviewer_round2/D/tost_pvalues.csv")
    paired = {(r["cell"], r["condition"], r["model"]): r for r in rows
              if r["reference"] == "physics_gradient"}
    result = []
    def p(value):
        if not value:
            return "--"
        number = float(value)
        if number < .001:
            return "<0.001"
        if number == 1:
            return "1"
        return f"{number:.3f}"
    for row in rows:
        if row["reference"] != "ecm_2rc_nls":
            continue
        comparison = paired[row["cell"], row["condition"], row["model"]]
        pg = (["--", "--"] if row["model"] == "physics_gradient" else
              [p(comparison["p_TOST_raw"]), p(comparison["p_TOST_Holm"])])
        result.append([{"panasonic": "NCA", "calce": "NMC"}[row["cell"]],
                       row["condition"], {"physics_gradient": "PG", "output_residual": "OR", "ecm_ude": "UDE"}[row["model"]],
                       f'{float(row["median_difference_mV"]):.3f}',
                       f'{float(row["Hodges_Lehmann_pseudomedian_mV"]):.3f}',
                       p(row["p_TOST_raw"]), p(row["p_TOST_Holm"]), *pg])
    if len(result) != 60:
        raise ValueError("Expected 60 E1 TOST rows")
    return result


def write_table(path, headers, rows):
    with path.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(rows)
    lines = [r"\footnotesize", r"\begin{tabular}{" + "l"*len(headers) + "}", r"\toprule",
             " & ".join(headers) + r" \\", r"\midrule"]
    lines += [" & ".join(r"$<10^{-3}$" if value == "<0.001" else value.replace("_", r"\_")
                         for value in row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.with_suffix(".tex").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="Directory for regenerated tables")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    write_table(args.output/"table5", ["Setting", "Variant", "vs. physics-gradient", "vs. ECM-2RC", "TOST"], summary_rows())
    write_table(args.output/"table7", ["Cell", "Condition", "ECM-2RC", "Physics-gradient", "Output residual", "ECM-UDE"], e1_rows())
    write_table(args.output/"table9", ["Cell", "Condition", "Variant", "Median", "HL", "pT 2RC", "pH 2RC", "pT PG", "pH PG"], tost_rows())
    print(f"Regenerated Tables 5, 7 and 9 in {args.output}")


if __name__ == "__main__":
    main()
