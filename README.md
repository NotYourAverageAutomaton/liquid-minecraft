# liquid-minecraft

A small liquid (CfC / LTC-style) motor network that learns to walk and steer a Minecraft body with a local, three-factor, reward-modulated Hebbian rule: no backprop at run time. This repository holds the code and logs behind the claims on the NYAA Minecraft page.

Status: code and simulation logs are public. The real-body run captures are **coming** (the capture host is offline); until they are here, the real-body claims below are marked "captures coming".

## Claim -> evidence

| claim | evidence |
|---|---|
| In simulation, the learning controller reaches 0.8 vs 0.02 frozen and 0.12 with shuffled reward | `code/sim/lnn_hebbian_walk_sim.py`, `logs/sim/lnn_hebbian_walk_sim.stdout.txt` (seeded re-run: 0.79, CI 0.71-0.87; 0.02; 0.12) |
| Simulation cost per tick is small (the page says 110 us; this re-run measured 71 us) | same script and log |
| A weak error signal made it worse (under-shoot drift) | `code/sim/lnn_budget_test.py`, `logs/sim/lnn_budget_test.stdout.txt` |
| A smaller initial scale fixes the fragility: 8/8 learn, 6/8 re-adapt | `code/sim/lnn_rung1j_initstability.py`, `code/sim/lnn_rung1k_readapt_at_s01.py`, their logs |
| Real body, new courses: 96% vs 100%, paths 1.66x, memory reset matched | `code/realbody/rung3.js`, `run_rung3.sh`, `score_rung3.py`, `cfc.js`, `adapter.js`, `walkcfc_weights.json`: **captures coming** |
| Real body, 45-degree turns 4/4; 60-degree 8-21% | `code/realbody/rung2.js`, `score_rung2.py`: **captures coming** |

## Files

- `code/rule/rules.py`: the three-factor learning rule.
- `code/sim/`: pure-Python simulations (seeded, CPU, seconds each). `logs/sim/` holds their re-run output.
- `code/realbody/`: the controller port to a disposable Minecraft body (mineflayer), its runner and scorers, and the trained weights.

To run the real-body scripts you need your own Minecraft server; they read the RCON password from the `RCON_PASSWORD` environment variable and stop if it is missing.

## Scrub

`SCRUB.md` lists every file that differs from the lab original, with both md5s and what was replaced (local paths, internal ids). Numbers were not touched.

## Licence

GPL-3.0 (see `LICENSE`).
