# Registration log

The entries below record local protocol-changing commits in chronological order. Automatic session snapshots that did not change a protocol file are omitted. Hashes in this table describe the original committed bytes, not later public adaptations.

| Order | Experiment | Nature | Summary | Protocol | Original SHA-256 | Private commit |
|---:|---|---|---|---|---|---|
| 1 | E2–E3 | Initial registration | Add audited LFP loaders and freeze NLS go-no-go protocol | `protocols/lfp_gonogo_protocol.json` | `0d5bd01a9dccbe7cc414d5c2ee6ee1ea9174270021e087b80c603ab9092f2323` | `da67c21` |
| 2 | E4 | Initial registration | Preregister source-only LFP static transfer experiment | `protocols/lfp_static_protocol.json` | `0c4de6dbb2914e73445cbf6de7b9127ffaa99756e87223cdd7a9becdb8dd44fd` | `50e85e1` |
| 3 | E4 | Initial registration | Preregister NCA NMC static replication and A123 thermal tests | `protocols/static_replication_protocol.json` | `0c2979ea21d574950750d09f7cedb8d4be4b7d62d57ae91a547870a593b7b9c6` | `78fefb3` |
| 4 | E4 | Initial registration | Register local A123 V6 thermal recalibration diagnostic | `protocols/a123_temperature_diagnostic.json` | `09ab71e2b45bd71a4a5649f2913e8f31e876017a2719362f7e299707b237a8fc` | `3d0d047` |
| 5 | E4 | Initial registration | Preregister A123 temperature parameter interpolation | `protocols/a123_temperature_interp_protocol.json` | `f0c6b8718383828856b3578cd96c9a2d25983d5b0cf4b48d83e4a9d25482a3ba` | `b2d945c` |
| 6 | E1 | Initial registration | Preregister ten-seed unified equivalence experiment | `protocols/paper_v2_seeds_protocol.json` | `370af0c15072bd283afe199a01aca1fd0bfc121da6338e538176517f58e4e933` | `d5cdc37` |
| 7 | E4 | Initial registration | Preregister monotonic source-supported V7 robustness | `protocols/paper_v2_monotone_protocol.json` | `b036db296f72f881db489c22d41b29fb86ccdce2207cbfacaa4a74c40fecdfd2` | `8a0e2e2` |
| 8 | E1 | Initial registration | Preregister same-file CALCE initial SOC audit | `protocols/paper_v2_calce_z0_protocol.json` | `1e7e86c04d8ff70c8c8fd92766e2999db0e3ad5f3afaa414ad3bf1da5403f1e7` | `308dc2a` |
| 9 | E4 | Amendment after initial result | Record authorized inconclusive-identification policy before remaining monotone fits | `protocols/paper_v2_monotone_failure_policy.json` | `1cc228f1e288d61eaa4bcc10f3dbff193424184250e0a45c00a61a8e7f2acd69` | `3c149a8` |
| 10 | E1, E4 | Initial registration | Preregister reviewer-round parameter ablation and saved-result audits | `protocols/reviewer_round_protocol.json` | `403de102ce6e0db5b2fcd23ff24313b403ac3a97da04bb43b39dca07bcb2e8e9` | `d14935e` |
| 11 | E5–E6 | Initial registration | Pre-register reviewer round two experiments | `protocols/reviewer_round2_protocol.json` | `1a2749ab993165fd3dc0fdcf95b840ff0e1c56861d4470ba1c17c3809142181b` | `61979b1` |
| 12 | E5–E6 | Amendment before evaluation | Clarify A Holm family across cells | `protocols/reviewer_round2_protocol.json` | `3071817e11e28decec81b8d1c61ee0ad5c17e46c09e9cc84c86d976e93299016` | `488d083` |
| 13 | E5–E6 | Amendment before evaluation | Preserve A3 in registered run order | `protocols/reviewer_round2_protocol.json` | `d8cba296a89f6aa1b92ed03075f289093a4fdff4dfc54d8efe7a6158db5f83f1` | `ca40af4` |
| 14 | E5–E6 | Amendment before evaluation | Register HG2 causal SOC audit rules | `protocols/reviewer_round2_protocol.json` | `d22b8c675335bf8198dabd4d4d27ba3c2b2e44e2312e2a957f42efa07647e250` | `d8f5cf8` |
| 15 | E5–E6 | Amendment after initial result | Register A4 aggregation correction | `protocols/reviewer_round2_protocol.json` | `f3dc341e7654e21d27c79006ac6ed81caa748f40b8c8cb0f186235cc1626a2fb` | `e062b8c` |
| 16 | E5–E6 | Amendment before evaluation | Register HG2 MAT filename resolution | `protocols/reviewer_round2_protocol.json` | `1e3a7c0a1e803495996941cbff887eceac4949251bcf2ffccefd239d4a89336a` | `d29cae9` |
| 17 | E5–E6 | Amendment before evaluation | Finalize M50T episode preregistration | `protocols/reviewer_round2_protocol.json` | `91cc26c2f2f0f15f5acf197ee6ce3afe5736bb3c36ea824a3deddecc872624ac` | `20a64d8` |
| 18 | E5–E6 | Amendment after initial result | Record C smoke serialization correction | `protocols/reviewer_round2_protocol.json` | `435b316537253cbd24943dfb46b8bcb270845f549338c47ec97dc2f40331f08b` | `4970e90` |
| 19 | E5–E6 | Amendment after initial result | Register A4 checkpoint loss reconciliation | `protocols/reviewer_round2_protocol.json` | `04da77a0009b9f9f00d301231066c123f954c2bb1e2cf763872ad1b01ccdfb83` | `bc0c0b7` |
| 20 | E5–E6 | Amendment before evaluation | Amendment 009: register M50T aging and HPPC transfer | `protocols/reviewer_round2_protocol.json` | `3639055feded587428b88f963c004699e52b47e1b66565ec7a119d30485ac83b` | `9e562a8` |
| 21 | E5–E6 | Exploratory analysis | Amendment 010: register exploratory M50T transfer corrections | `protocols/reviewer_round2_protocol.json` | `691c9ba5df415c1b3ef0c83681294d654ab876eac9e64f3e6c9284c197d9a757` | `1aeaf76` |

## Published protocol hashes

These hashes describe the files in this release, independently of the original commits.

| Protocol | SHA-256 |
|---|---|
| `protocols/a123_temperature_diagnostic.json` | `09ab71e2b45bd71a4a5649f2913e8f31e876017a2719362f7e299707b237a8fc` |
| `protocols/a123_temperature_interp_protocol.json` | `f0c6b8718383828856b3578cd96c9a2d25983d5b0cf4b48d83e4a9d25482a3ba` |
| `protocols/lfp_gonogo_protocol.json` | `0d5bd01a9dccbe7cc414d5c2ee6ee1ea9174270021e087b80c603ab9092f2323` |
| `protocols/lfp_static_protocol.json` | `0c4de6dbb2914e73445cbf6de7b9127ffaa99756e87223cdd7a9becdb8dd44fd` |
| `protocols/paper_v2_calce_z0_protocol.json` | `1e7e86c04d8ff70c8c8fd92766e2999db0e3ad5f3afaa414ad3bf1da5403f1e7` |
| `protocols/paper_v2_monotone_failure_policy.json` | `1cc228f1e288d61eaa4bcc10f3dbff193424184250e0a45c00a61a8e7f2acd69` |
| `protocols/paper_v2_monotone_protocol.json` | `b036db296f72f881db489c22d41b29fb86ccdce2207cbfacaa4a74c40fecdfd2` |
| `protocols/paper_v2_seeds_protocol.json` | `370af0c15072bd283afe199a01aca1fd0bfc121da6338e538176517f58e4e933` |
| `protocols/reviewer_round2_protocol.json` | `691c9ba5df415c1b3ef0c83681294d654ab876eac9e64f3e6c9284c197d9a757` |
| `protocols/reviewer_round_protocol.json` | `403de102ce6e0db5b2fcd23ff24313b403ac3a97da04bb43b39dca07bcb2e8e9` |
| `protocols/static_replication_protocol.json` | `0c2979ea21d574950750d09f7cedb8d4be4b7d62d57ae91a547870a593b7b9c6` |

## Path normalization

Absolute local paths in released textual results were normalized. Historical
hashes remain evidence of the original bytes and were not overwritten;
`PUBLIC_SHA256.json` independently records the normalized public source,
protocol, and result bytes, including regenerated manifest files.
