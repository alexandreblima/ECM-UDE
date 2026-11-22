# Derived results

The derived results are released under CC BY 4.0. Raw datasets are not included.

## Corrections to derived files

In the original M50T rows of `reviewer_round2/E/synthesis.csv` and
`synthesis.tex`, the minimum and maximum were incorrectly both equal to the
population median. The public copies use the minimum and maximum of the seven
cell-level seed medians in `reviewer_round2/C/run/cell_differences.csv`, for
both neural variants and both references. Medians and statistical tests were
not changed. The public copy retains unrounded values.

The M50T fixed-aging and restricted-HPPC ranges were checked against their
cell-level seed differences; all 12 variant/reference/cohort ranges matched
and were left unchanged.

The historical `reviewer_round2/D/reproducibility_scope.md` command for
the criteria table was updated to use the released saved-analysis script
instead of the removed private-archive-dependent report script. Its SHA-256
changed from `2862ad954953be1d9b772db83f6e141f2bbb3cf313bbff2b9fd68fd389ed39d9`
to `6d0f14afca96607633c210a823487f18c6c63ef26a42d4c72beb750e4d3506fd`.

| File | SHA-256 before correction | SHA-256 after correction |
|---|---|---|
| `reviewer_round2/E/synthesis.csv` | `c5bc107e2be67ee2f597332a88644de16e772fb483703a1c1fb4ae2033d3853e` | `6b02caf640bb2c4e7e1686fd8f6d02d7edf1e1a50cea24a194a4327e76ffc07d` |
| `reviewer_round2/E/synthesis.tex` | `3f87a170d02674152b9e4804340b878fdb2b8d9abfff2221f32ec110fb090d63` | `cd2560c5436090938a85a2eb74c6662a584b555bed3cf219079ba83c45ca8b3e` |

## Path normalization

Paths in the published copies are relative. Historical hashes in result
manifests describe the original experiment inputs; they are not replaced
with hashes of modified release files. `PUBLIC_SHA256.json` records separate
hashes of the published sources, protocols, and results.

Some published reports were translated into English without changing their
numerical content. Historical manifests retain the hashes of the original
reports; the public integrity manifest records the translated copies.
Decimal commas in prose retain the original notation for traceability.
