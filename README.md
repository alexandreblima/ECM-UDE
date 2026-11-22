# ECM-UDE: Neural and Hysteresis Corrections Versus Recalibration of Equivalent-Circuit Battery Models

This repository contains the code, registered protocols, data manifests, and numerical results for a controlled evaluation of voltage prediction with equivalent-circuit models (ECM) of lithium-ion cells. The study asks where the remaining voltage error of a calibrated ECM lies, and whether adding a neural network improves prediction beyond what is obtained by re-estimating, or recalibrating, the circuit itself.

Three kinds of intervention are compared under matched calibration and evaluation rules:

- **Neural corrections:** an output-residual network and a universal differential equation (ECM-UDE) that adds a learned term to the RC-branch dynamics.
- **Hysteresis model:** an explicit one-state hysteresis model for LFP cells.
- **Recalibration:** recalibration of the capacity, initial state of charge, RC parameters, and open-circuit-voltage curve, including temperature interpolation of circuit parameters.

Each neural variant is compared with a physics-gradient control, which is the same circuit trained by the same procedure without a network. This control separates the contribution of the learned term from the effect of re-estimating the circuit.

The evaluation covers public laboratory data from NCA, NMC, and LFP cells, including seven cells of the same model for cell-level inference. Driving profiles, temperatures, and aging states are included. Key features:

- hypotheses and decision criteria registered before model fitting, with amendments recorded separately;
- causal state of charge, computed only from information available before each instant;
- continuous simulation over complete profiles from a single initial state;
- equivalence testing with exact Wilcoxon two one-sided tests and Holm correction, with raw and adjusted p-values;
- ablations that freeze the circuit, set the network to zero after training, and vary initialization and training budget.

The repository documents which analyses can be rerun from the raw datasets and which are regenerated from saved checkpoints and training histories. Raw datasets are not redistributed.

**Manuscript:** A. B. de Lima, "Neural and Hysteresis Corrections Versus Recalibration of Equivalent-Circuit Battery Models: A Controlled Evaluation of Voltage Prediction," IEEE Access, under review.

## Installation

```sh
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest
```

The pinned environment has been tested with Python 3.10. Set `ECM_UDE_DATA`
to the directory containing the raw datasets; the default is `./datasets`.
Raw data are not required for the rendering commands below. Rendering
requires Times New Roman to be installed.

```sh
python figures_ieee.py --figures fig01_dynamic_corrections
python figure_m50t.py
python figure_learning_curves.py
python paper_v3_tables.py --output generated_tables
```

## Contents

- Python files at the root: model implementations, experiments, analysis, and tests.
- `protocols/`: registered protocols and amendments.
- `results/`: derived results, not raw data.
- `REGISTRATION_LOG.md`: original registration sequence and protocol hashes.
- `DATA.md`: dataset sources and verified raw-file hashes.
- `REPRODUCIBILITY.md`: item-by-item scope and remaining limitations.
- `TESTING.md`: test adaptations and limitations.

**Release limitations:** some historical audit scripts still require
excluded private artifacts; see `REPRODUCIBILITY.md`.
