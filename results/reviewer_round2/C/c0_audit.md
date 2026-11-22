# C0 — M50T audit and preregistered selection

## Applied rule

For each cell, the order of the C/20 and HPPC diagnostics was compared with the order of UDDS SOC20-80 blocks. The first eligible block is the earliest after both diagnostics. Fragments were ordered by Test_Time and joined only when the boundary interval was increasing, at most 1,5 s, and the net Ah discontinuity was at most 0,01 Ah.

A complete UDDS episode must be bounded by observed transitions, not start/end truncated at an export boundary, have minimum duration 1369 s, median sampling interval between 0,05 and 0,2 s, current amplitude of at least 2 A, and current standard deviation of at least 0,5 A. The source is the first complete episode with a causal SOC reference in the same file; targets are later complete episodes in the same block with an equivalent reference. Data are not transferred between aging blocks.

The nearest C/20 before the block supplies capacity; OCV is the cell's own C/20 discharge branch. The SOC reference point is rest in the same file after the SOC20-80 charge step, identified by recorded current/rest. The reference is SOC=0,8; measured net Ah is integrated only up to the start of the episode. No data after the episode begins contribute to z0.

## Audit and decision

| Cell | First post-diagnostic block | Fragment continuity | Result |
|---|---|---|---|
| G1 | N25 | 10 fragments; monotonically increasing internal intervals; boundaries 0,099-0,101 s; maximum Ah discontinuity 0,0005 | Include |
| V4 | N25 | 10 fragments; monotonically increasing internal intervals; boundaries 0,100 s; no Ah discontinuity | Include |
| V5 | N25 | 10 fragments; monotonically increasing internal intervals; boundaries 0,099-0,101 s; maximum Ah discontinuity 0,0005 | Include |
| W3 | N25 | 10 fragments; monotonically increasing internal intervals; boundaries 0,099-1,000 s; maximum Ah discontinuity 0,0006 | Include |
| W4 | N25 | One fragment; only one complete episode after an SOC reference in the same file | Exclude: no later target |
| W5 | N75 | N25 precedes the first HPPC; N75 is the first block after both diagnostics, but the observed dynamic episode starts mid-step and has no prior reference in the same file | Exclude: no eligible source-target pair |
| W7 | N25 | 10 fragments; monotonically increasing internal intervals; boundaries 0,099-0,999 s; maximum Ah discontinuity 0,0008 | Include |
| W8 | N25 | 10 fragments; monotonically increasing internal intervals; boundaries 0,099-1,001 s; maximum Ah discontinuity 0,0008 | Include |
| W9 | N25 | One fragment starting mid-UDDS step, without prior SOC reference in the same file and without a later episode | Exclude: no eligible source-target pair |
| W10 | N25 | 10 fragments; monotonically increasing internal intervals; boundaries 0,099-1,000 s; maximum Ah discontinuity 0,0004 | Include |

The threshold of six valid cells is met: seven cells remain (G1, V4, V5, W3, W7, W8, W10). W4, W5, and W9 are reported as exclusions, without substitution by a later block that would change the rule.

Raw XLSX files were read only. Cell metrics, fragment names, and exclusions are in `c0_audit.csv`; SHA256 hashes of files actually used will be recorded in the C manifest before fitting.
