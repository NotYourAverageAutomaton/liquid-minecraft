# Scrub record

Byte copies of the lab originals with these replacements only. Python files still parse.

| file | original md5 | published md5 | replacements |
|---|---|---|---|
| `code/realbody/run_rung3.sh` | `abcdf62c096ff2ef85aa0f8ee159f8e6` | `43c324278cb3e0a3f131a71eb0562be7` | card id (parenthesised) -> removed: 1; seat + post -> (review note): 2 |
| `code/realbody/rung2.js` | `4e2151dbd1185f5c33f03e9cc4568020` | `bfb64b09a267170d065b0d629418eb15` | rcon password literal -> process.env.RCON_PASSWORD: 1; vault page path -> generic: 1; card id (parenthesised) -> removed: 1 |
| `code/realbody/rung3.js` | `87f8490dabda53cb309ca8191013b636` | `0e0ed86aa402d34a5afb0d78a100d2cd` | vault page path -> generic: 1; card id (parenthesised) -> removed: 1; seat + post -> (review note): 2 |
| `code/realbody/score_rung3.py` | `3d57ecd69c0bdf01609535969396dae3` | `fd2b7e5f854548e5b327d3fd35949e12` | vault page path -> generic: 1; card id (parenthesised) -> removed: 1 |
| `code/realbody/trainexport_walkcfc.py` | `6b2629b329f6c43e41dccd22792f86e4` | `b0cea6a91bed37ef4217a0fb681837b2` | agent scratch path -> ./: 2 |
| `code/sim/lnn_budget_test.py` | `581d6eb1ef8d5d3a459e5ccc009c79b8` | `eff5a5b1bf0126b1e90e0e07a4fa28d9` | vault script path -> ./: 1 |
| `code/sim/lnn_hebbian_walk_sim.py` | `584027a1f492efaa2834272f27c1af1d` | `9d870fd5bc8c23979fa3aad2d2fe5e6c` | card id (parenthesised) -> removed: 1; owner handle -> the rule author: 2 |
| `code/sim/lnn_rung1j_initstability.py` | `24b6c5611660e0cc02f3997bd446e543` | `5e9dae1b2c53c46131302cc886f27ccc` | card id (parenthesised) -> removed: 1 |
| `code/sim/lnn_rung1k_readapt_at_s01.py` | `e7fb63fdb1dbe47211fbb912b3f8800c` | `561ec294eb7a33f2791a862e253e3ff9` | card id (parenthesised) -> removed: 1 |

The rcon password literal was removed from code; the live sandbox server password is to be confirmed rotated (the sandbox host is offline as of 2026-11-24).
