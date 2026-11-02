# ECM-UDE: reproducing the latest reviewer round

## Scope

This standalone package corresponds to commit **91b3a08** on branch
`paper-v2`. The protocol was registered **before fitting**, in commit
**d14935e**. It contains the materials needed for the latest reviewer round:

- **A:** NLS parameter ablations V6-Q/V6-RC, CALCE V6-wide, and A123 V6-OCVT;
- **B:** seed-paired TOST comparisons between variants, with a +/-1 mV margin;
- **C:** an audit of 80 saved training histories and eight IEEE learning curves;
- **D:** the ten-seed Figure 2 and E1, condition, and criterion tables.

Raw data, the manuscript, credentials, the Git database, and new neural-network
training are not included. Earlier results supplied here are **frozen inputs**
to this round; this is not a package for rerunning every paper experiment from
scratch.

Source repository: [alexandreblima/ECM-UDE](https://github.com/alexandreblima/ECM-UDE).

Read [reference/report_reviewer_round.md](reference/report_reviewer_round.md)
for the original scientific report. Reference outputs are available in:

- [parameter ablation](reference/results/param_ablation/);
- [paired variant TOST](reference/results/variant_tost/);
- [training audit](reference/results/training_audit/);
- [publication tables](reference/results/paper_tables/).

## 1. Environment

Use Python **3.10** and a new virtual environment. The exact original Python
version and package versions are recorded in [environment.json](environment.json).
A GPU, Julia, and LaTeX are not required to generate CSV/TeX/PDF outputs.

```bash
cd /path/to/ECM_UDE_reviewer_round
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

On Windows, use `.venv\Scripts\activate` and set the environment variables
using your shell's syntax. PyTorch is required for imports, checkpoints, and
auditing; **none of the package entry-point commands trains neural networks**.

PDFs use Times New Roman when available. Otherwise, install **Liberation Serif**
or **Nimbus Roman**, which are Times-equivalent fonts. No proprietary font is
redistributed. A substitute preserves the required dimensions and minimum
8 pt type size, but may change PDF bytes and small typographic measurements.
To compile the tables, optionally install IEEEtran and booktabs in your TeX
distribution and use `\input{.../results/paper_tables/E1.tex}` in your document.

## 2. Verify and reproduce without raw data

The following commands use only saved results and checkpoints:

```bash
python reproduce.py verify
python reproduce.py validate
python reproduce.py analysis
python reproduce.py figure2
python reproduce.py tables-only
```

- `verify`: checks SHA-256 hashes of all distributed files, without importing
  scientific libraries.
- `validate`: runs scientific checks covering parameters, metrics, explained
  fractions, Wilcoxon zeros/ties, Holm correction, bootstrap intervals, and
  training histories; also validates vector PDFs. It does not require the
  complete historical tree of 1,131 artifacts outside this package's scope.
- `analysis`: rebuilds Parts B/C/D, including paired tests, the training audit,
  eight learning-curve PDFs, and CSV/TeX tables, without new fitting or training.
- `figure2`: regenerates only Figure 2 from the ten saved seeds.
- `tables-only`: reexports the saved ablation table, without simulation or fitting.

Except for `verify`, each command creates a new working copy in
`work/<command>-<id>/code/`. Its location is printed in the terminal.
Generated files are written there, not to `reference/`.
You may also specify `--work /path/to/new-directory`.
Do not reuse an existing directory. Reference inputs remain unchanged.

## 3. Public data for rerunning the Part A NLS fits

Obtain data directly from the original providers, comply with their licenses,
and record the downloaded version. This package **does not download anything
automatically**.

| Dataset | Source | Files used in this round |
|---|---|---|
| A123 A1-007, LFP | [CALCE](https://calce.umd.edu/data), [25 C OCV archive](https://web.calce.umd.edu/batteries/data/A123_OCV25-20120905.zip), [25 C dynamic archive](https://web.calce.umd.edu/batteries/data/A123_DST-US06-FUDS-25.zip) | OCV and DST/US06/FUDS at 0, 10, 25, 40, and 50 C; filenames below |
| Mendeley LFP | [Mendeley V1, DOI 10.17632/29kw38kzwj.1](https://doi.org/10.17632/29kw38kzwj.1) | `PCPD.xls`, `DST FUDS UDDS HPPC WLTS US06.xlsx`, and capacity data in the LFP cell's `25 Degree` folder |
| Panasonic 18650PF | [Mendeley V1, DOI 10.17632/wykht8y7tg.1](https://doi.org/10.17632/wykht8y7tg.1) | UDDS/US06/LA92/HWFET MAT files at 25 C, plus secondary UDDS temperature files |
| CALCE INR18650-20R | [CALCE](https://calce.umd.edu/data) | SP1 low-current OCV at 25 C and four audited SP2 80SOC profiles at 25 C, plus secondary temperature profiles |

Create an external `datasets/` directory with the layout below, or pass an
equivalent directory with `--datasets`. Names, spaces, and capitalization matter:

```text
datasets/
  calce_a123_raw/
    A123_OCV25-20120905.zip
    A123_DST-US06-FUDS-25.zip
    A123_OCV0-20120618.zip
    A123_DST-US06-FUDS-0.zip
    A123_OCV10-20120611.zip
    A123_DST-US06-FUDS-10.zip
    A123_OCV40-20120627.zip
    A123_DST-US06-FUDS-40.zip
    A123_OCV50-20120702.zip
    A123_DST-US06-FUDS-50.zip
    extracted/                     # Original contents of all ten ZIP archives
  mendeley_29kw38kzwj_v1/
    extracted/Battery Test Data/
      LiFePO4 18650 Rechargeable Cell 3.3V 1100 mAh/
        25 Degree/
          PCPD.xls
          DST FUDS UDDS HPPC WLTS US06.xlsx
          1.5C 1C 0.5C.xlsx
  Pan18650/
    25degC_UDDS_Pan18650PF.mat
    25degC_US06_Pan18650PF.mat
    25degC_LA92_Pan18650PF.mat
    25degC_HWFTa_Pan18650PF.mat      # Raw filename; scientific label: HWFET
  calce_20r_raw/
    extracted/
      SP1_25C_LC_OCV_11_5_2015/
        11_5_2015_low current OCV test_SP20-1.xlsx
      SP2_25C_FUDS/11_06_2015_SP20-2_FUDS_80SOC.xls
      SP2_25C_DST/11_05_2015_SP20-2_DST_80SOC.xls
      SP2_25C_US06/11_11_2015_SP20-2_US06_80SOC.xls
      SP2_25C_BJDST/11_12_2015_SP20-2_BJDST_80SOC.xls
```

See [raw_data_manifest.json](raw_data_manifest.json) for referenced paths and
input hashes collected from the original manifests. The loaders define the
required sheet names, test steps, and preparation checks.
Extract ZIP archives while preserving their internal subdirectories; do not
rename workbooks. The 25 C loader also verifies/extracts its archives, but the
thermal manifests expect the 0/10/40/50 C workbooks to be extracted already.

**Important:** the shared replication loader opens secondary targets even
when this ablation scores only the 25 C targets. Also provide UDDS MAT files
at 10, 0, -10, and -20 C and all four CALCE SP2 profiles at 0 and 45 C.
The complete file/hash list is in `raw_data_manifest.json`.

**Do not use** the duplicate Mendeley copy under `Pan18650/external`.
Mendeley Qnom is the measured discharge capacity (~2.50 Ah), not the 1.1 Ah
label. Its nominal temperature is 25 C, with a documented 25/30 C ambiguity.

```bash
# Rerun this round's NLS fits; this may take hours:
python reproduce.py ablation --datasets /absolute/path/to/datasets

# Resimulate saved fits only, without NLS:
python reproduce.py rescore --datasets /absolute/path/to/datasets
```

The `ablation` command reuses earlier V1/V6 fits, as required by the protocol,
and recomputes only this round's new fits. Do not run the legacy
`exp_unified.py` or `exp_paper_v2_seeds.py` drivers: they are included for
imports, dependencies, and auditing, and their `main` functions can train models.

## 4. Expected results and limitations

- CALCE V6-wide: **POSITIVE**, with DST/US06/BJDST reductions of
  **38.94%, 46.03%, and 44.15%** against V1 with audited z0.
- Same-temperature aggregate attribution: **joint**. Thermal-transfer
  attribution is also joint, but only one condition has a positive V6 gain.
- Local thermal recalibration: RC recovers >=70% in **5/8** conditions.
  This is a descriptive count for converged variants, not causal proof.
- V6-OCVT at 10 C: **INCONCLUSIVE_NLS**; none of the three starts converges
  within 500 evaluations. Do not score failed fits or increase the budget.
- Direct TOST: **18/18** equivalent conditions for each pair versus
  physics_gradient.
- 80 histories and eight learning-curve PDFs; Figure 2 uses the ten-seed
  LSTM median[min,max]. Seed ranges are not confidence intervals.

Table metrics are in mV. When rerunning NLS, small library, BLAS, or platform
differences may change optimization paths or convergence. Compare metrics,
tolerances, and recorded attempts rather than requiring identical bytes for
new PDFs, NPZ files, or JSON files containing local paths. Frozen reference
files, however, must match their SHA-256 hashes exactly.

### Package validation

The package was tested using `verify`, the nine `validate` tests, `analysis`,
`figure2`, and `rescore` in relocated working copies outside the repository.
The three publication CSV files (E1, criteria, and conditions) were byte-identical.
All **165 rescored numerical values** agreed with their references within
`1e-10`. External data were used only for this check and are not included.
See [validation_report.json](validation_report.json).

**The full NLS fits were not repeated again while assembling this package**:
the original code and fits are included, and resimulation was tested.
A freshly installed virtual environment on another operating system was not
tested either; dependency versions are pinned to support that step.

## 5. Portability, integrity, and provenance

- `reference/` preserves the original scientific code and copied results;
  their bytes were not edited to replace paths.
- `reproduce.py` translates absolute manifest path prefixes **in memory**
  to the working copy and external dataset directory.
- The original `git show d14935e:reviewer_round_protocol.json` check is replaced,
  only within the wrapper process, by SHA-256 verification of the same
  preregistered bytes. The Git history is not required.
- A Times-equivalent font, if needed, is selected in memory.
- Each execution other than `verify` saves `portable_run.json`.
  `bundle_manifest.json` records the commit, hashes, and scope. The historical
  preservation audit is supplied as evidence; it does not imply that all
  1,131 earlier files are included in this package.
- This README grants no additional code or data licenses. Consult the authors
  and original terms before redistribution. No raw data were copied.
