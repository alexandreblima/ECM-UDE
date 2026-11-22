# Tests in the public release

Run `python -m pytest -q`. If the raw data directory does not exist, raw-data
tests are skipped with a message. Set `ECM_UDE_DATA` to the dataset root to run
them. The file `PUBLIC_SHA256.json` checks the distribution's bytes; the
original hashes in historical experiment manifests identify the inputs to
the original runs and do not certify modified release code.

The public runners for CALCE z0, monotone fits, parameter ablation and the
ten-seed extension verify released protocol or source hashes instead of
accessing private Git objects. This establishes release integrity, not byte
identity with historical training code; historical provenance is recorded
separately. The thermal snapshot omits private report files and records
repository-relative paths.

## Adaptations and removed checks

- `test_a123_temperature_interp.py::test_frozen_protocol_threshold_and_coverage_inclusive`:
  replaces private-commit access with the SHA-256 of the published protocol.
- `test_a123_temperature_interp.py::test_calibration_load_reads_no_test_numerics_and_predictions_take_only_three_points`:
  retains calibration and raw-file hash checks, but skips if data are absent.
- `test_e1_error_structure.py::FrozenCheckpointReproductionTests`:
  skips when raw data are absent; checks release hashes independently from
  historical input hashes while retaining checkpoint, raw-input, reproduced
  metric and claim comparisons.
- `test_figures_ieee.py::IEEEFiguresTests`: renders the eight figures in a
  temporary directory, validates PDF properties and published input hashes;
  removes comparisons to unpublished previous PDFs, figure archives, ledger,
  manuscript, and index. Labels and saved-data checks remain.
- `test_figures_ieee.py::PlottedValuesTests`: removes comparisons against
  unpublished previous figure generators; retains numerical checks against
  saved datasets, panel structure, visual encodings and text placement.
- `test_lfp_gonogo.py::RealDataTests` and
  `test_reviewer_round2.py::ReviewerRound2CheckpointTests`: skip only when
  the raw dataset root is absent.
- `test_lfp_gonogo.py::SavedResultTests.test_saved_results_against_independent_scalar_recurrence`,
  `test_lfp_static.py::PersistedTests.test_saved_results_and_previous_files`,
  `test_static_replication.py::PersistedTests.test_independent_scalar_simulation_metrics_and_preservation`,
  `test_a123_temperature.py::PersistedTests.test_scalar_recurrence_all_metrics_bounds_hashes_and_decision`,
  and `test_a123_temperature_interp.py::PersistedTests.test_all_saved_central_sample_voltages_metrics_coverage_and_preservation`:
  replace checks of unpublished trajectory files containing raw measurements
  and private preservation snapshots with public hash, saved metric consistency,
  fit constraints, and independently recomputed decision checks. These tests
  **do not** independently reproduce the excluded voltage trajectories.
- `test_paper_v2.py::PersistedTests.test_saved_monotone_shape_bounds_and_metrics`:
  retains monotonicity, bounds and verdict checks; removes the comparison to
  excluded measured-voltage trajectories.
- `test_paper_v2.py::PersistedTests.test_ten_seed_extension`: checks the
  published protocol hash rather than requiring the private Git commit;
  checkpoint hashes and recomputed statistics remain checked.
- `test_reviewer_round.py::ReviewerRoundTests.test_protocol_and_producer_hashes`
  and `test_principal_tables_and_all_previous_bytes`: replace private-history
  preservation checks with published protocol, source, and result hashes.
- `test_reviewer_round.py::ReviewerRoundTests.test_every_metric_matches_persisted_voltage_and_signed_fraction`:
  removes the excluded voltage-trace comparison, retains saved metric and
  signed-fraction checks.
- `test_reviewer_round.py::ReviewerRoundTests.test_paired_differences_walsh_and_holm_families`:
  reconstructs the omitted comparison summary from published per-seed metrics
  and TOST results; independently checks the seed differences, Walsh
  pseudomedians, exact tails, and Holm correction.
- `test_reviewer_round.py::ReviewerRoundTests.test_saved_histories_selection_curves_and_fonts`:
  resolves history paths relative to the public root; the history hashes,
  selection, curves and PDF checks are retained.
- `test_reviewer_round.py::ReviewerRoundTests.test_existing_lstm_bootstrap_is_paired_seed_percentile`:
  retains the exact paired bootstrap check but removes a metadata check on the
  unpublished comparison summary.
- `test_static_replication.py::ReplicationTests.test_duplicate_low_current_timestamps_preserve_zero_duration`:
  uses a synthetic path in the public dataset layout; the zero-duration
  integration check is unchanged.

`test_public_figures.py` adds checks of the reconstructed paired comparison
and rejection of incomplete saved comparisons.

The report-update tests also check replacement of the legacy section marker
and verification of the translated report against the public integrity manifest.
The A report tests check exact reproduction from reconciled checkpoint losses,
preservation of the corrected summary, and rejection of an unreconciled basis.
