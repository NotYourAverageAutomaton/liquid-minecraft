#!/usr/bin/env python3
"""Score rung 3  against its pre-registration, the pre-registration (lab record)
(+ AMENDMENT 0). Reads one JSONL file per (arm, seed, course) as written by rung3.js. Prints one JSON verdict.

Usage: score_rung3.py <dir>        score the run captures in <dir> (*.jsonl)
       score_rung3.py --selftest   show every bar and every guard can say NO, on synthetic records

Verdict order: STOP (instrument: missing/broken data, wrong artifact) > NOT-EVIDENCE > FAIL > PASS.
"""
import glob, json, math, os, sys

SEEDS = [301, 302, 303, 304, 305]
COURSES = ["0", "1", "2", "3"]
SCORED_ARMS = ["pp", "liq", "liqreset", "zero"]
N_WP = 8
PIN = {"cfc": "b9b16802", "adapter": "7a8435c7", "weights": "e2df06eb"}
BINS = [(0, 45), (45, 90), (90, 135), (135, 180.0001)]


def load(d):
    runs = {}
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        recs = []
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("{"):
                    recs.append(json.loads(line))
        st = next((r for r in recs if r.get("ev") == "start"), None)
        if st is None:
            runs[("?", f, "?")] = {"file": f, "recs": recs, "start": None}
            continue
        runs[(st["arm"], st["seed"], str(st["course"]))] = {"file": f, "recs": recs, "start": st}
    return runs


def p99(xs):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(math.ceil(0.99 * len(xs))) - 1)]


def rate(wps):
    return (sum(1 for w in wps if w["reached"]) / len(wps)) if wps else None


def score(runs):
    stops, ne, md5s = [], [], set()
    # ---- STOP: every scored cell present, complete, clean, pinned
    for arm in SCORED_ARMS:
        for s in SEEDS:
            for c in COURSES:
                r = runs.get((arm, s, c))
                if r is None:
                    stops.append(f"missing {arm} s{s} c{c}"); continue
    for key, r in runs.items():
        if r["start"] is None:
            stops.append(f"no start record in {os.path.basename(r['file'])}"); continue
        evs = [x.get("ev") for x in r["recs"]]
        bad = [e for e in evs if e in ("ALERT", "ERR", "KICK", "END")]
        if bad:
            stops.append(f"{key}: {bad[0]}")
        if "done" not in evs:
            stops.append(f"{key}: no done")
        if sum(1 for e in evs if e == "wp") != N_WP:
            stops.append(f"{key}: {sum(1 for e in evs if e == 'wp')} wp records, want {N_WP}")
        m = r["start"].get("md5", {})
        for k, v in PIN.items():
            if not str(m.get(k, "")).startswith(v):
                stops.append(f"{key}: {k} md5 {str(m.get(k, ''))[:8]} != pin {v}")
        md5s.add(m.get("rung3"))
    if len(md5s) > 1:
        stops.append(f"rung3.js md5 differs across runs: {sorted(x[:8] for x in md5s if x)}")
    # same course for every arm: waypoints must match
    for s in SEEDS:
        for c in COURSES:
            ref = None
            for arm in SCORED_ARMS:
                r = runs.get((arm, s, c))
                if r and r["start"]:
                    w = json.dumps(r["start"]["wps"])
                    if ref is None: ref = w
                    elif w != ref: stops.append(f"s{s} c{c}: waypoints differ across arms")
    if stops:
        return {"verdict": "STOP: " + stops[0], "stops": stops}

    wp = lambda arm, seeds=SEEDS, courses=COURSES: [x for s in seeds for c in courses
                                                  for x in runs[(arm, s, c)]["recs"] if x.get("ev") == "wp"]
    per_seed = {arm: {s: rate(wp(arm, [s])) for s in SEEDS} for arm in SCORED_ARMS}
    pooled = {arm: rate(wp(arm)) for arm in SCORED_ARMS}

    # ---- NOT-EVIDENCE guards
    if pooled["pp"] < 0.90: ne.append(f"P reach {pooled['pp']:.3f} < 0.90 (courses too hard for the reference)")
    if pooled["zero"] > 0.20: ne.append(f"L0 reach {pooled['zero']:.3f} > 0.20 (floor too easy)")
    tick_p99 = {}
    for arm in SCORED_ARMS:
        us = [x["us"] for s in SEEDS for c in COURSES for x in runs[(arm, s, c)]["recs"] if x.get("ev") == "tick"]
        tick_p99[arm] = p99(us)
        if tick_p99[arm] is not None and tick_p99[arm] > 50000:
            ne.append(f"{arm} tick p99 {tick_p99[arm]:.0f} us > 50 ms")
    straight = [x for s in SEEDS for x in runs.get(("liq", s, "straight"), {"recs": []})["recs"] if x.get("ev") == "wp"]
    straight_rate = rate(straight)
    if straight_rate is None: ne.append("control 4 missing: no liq straight-chain runs")
    elif straight_rate < 0.90: ne.append(f"control 4: L on straight chains {straight_rate:.3f} < 0.90 (problem is not the switch)")

    # ---- bars
    b1_seeds = [s for s in SEEDS if per_seed["liq"][s] >= per_seed["pp"][s] - 0.10]
    B1 = len(b1_seeds) >= 4
    B2 = (pooled["liq"] - pooled["zero"]) >= 0.50

    # ---- report-only
    def bins(arm):
        out = {}
        for lo, hi in BINS:
            xs = [x for x in wp(arm) if x["turn"] is not None and lo <= x["turn"] < hi]
            out[f"{lo}-{int(hi)}"] = {"n": len(xs), "reach": rate(xs)}
        return out
    ratios = []
    for s in SEEDS:
        for c in COURSES:
            P = {x["i"]: x for x in runs[("pp", s, c)]["recs"] if x.get("ev") == "wp"}
            for x in runs[("liq", s, c)]["recs"]:
                if x.get("ev") == "wp" and x["reached"] and P[x["i"]]["reached"] and P[x["i"]]["path"] > 0:
                    ratios.append(x["path"] / P[x["i"]]["path"])
    nav = [x for s in SEEDS for c in COURSES for x in runs.get(("nav", s, c), {"recs": []})["recs"] if x.get("ev") == "wp"]

    res = {"per_seed_reach": per_seed, "pooled_reach": pooled,
           "B1_transfer": {"pass": B1, "seeds_within_0.10": b1_seeds},
           "B2_above_floor": {"pass": B2, "liq_minus_zero": pooled["liq"] - pooled["zero"]},
           "not_evidence": ne, "control4_straight_liq": straight_rate, "tick_p99_us": tick_p99,
           "report": {"turn_bins": {a: bins(a) for a in ("pp", "liq", "liqreset")},
                      "path_ratio_liq_over_pp_median": sorted(ratios)[len(ratios) // 2] if ratios else None,
                      "stalls": {a: sum(x["stalls"] for x in wp(a)) for a in SCORED_ARMS},
                      "nav_reach_report_only": rate(nav)},
           "md5": runs[("liq", SEEDS[0], "0")]["start"]["md5"]}
    res["verdict"] = ("NOT-EVIDENCE: " + ne[0]) if ne else ("PASS" if (B1 and B2) else "FAIL")
    return res


# ------------------------------------------------------------------ selftest (synthetic records only)

def synth(arm, s, c, reach_fn, us=200.0, md5=None):
    m = md5 or {"rung3": "r3" * 16, "cfc": PIN["cfc"] + "x" * 24, "adapter": PIN["adapter"] + "x" * 24,
                "weights": PIN["weights"] + "x" * 24}
    turns = [None] + ([0.0] * 7 if c == "straight" else [((s * 37 + int(c) * 11 + i * 53) % 180) * 1.0 for i in range(1, 8)])
    wps = [{"i": i, "x": i, "z": 0, "len": 6.0, "turn": t} for i, t in enumerate(turns)]
    recs = [{"ev": "start", "arm": arm, "seed": s, "course": c, "md5": m, "wps": wps}]
    for w in wps:
        recs.append({"ev": "tick", "us": us, "gap": 1})
        recs.append({"ev": "wp", "arm": arm, "seed": s, "course": c, "i": w["i"], "turn": w["turn"], "len": 6.0,
                     "reached": bool(reach_fn(w["turn"])), "ticks": 50, "path": 6.5, "stalls": 0})
    recs.append({"ev": "done"})
    return {"file": f"{arm}-{s}-{c}.jsonl", "recs": recs, "start": recs[0]}


def world(liq_fn, pp_fn=lambda t: True, zero_fn=lambda t: False, straight_fn=lambda t: True, drop=None, md5_bad=False):
    runs = {}
    for s in SEEDS:
        for c in COURSES:
            for arm, fn in (("pp", pp_fn), ("liq", liq_fn), ("liqreset", liq_fn), ("zero", zero_fn)):
                if drop == (arm, s, c): continue
                m = None
                if md5_bad and arm == "liq" and s == 303:
                    m = {"rung3": "r3" * 16, "cfc": "deadbeef" * 4, "adapter": PIN["adapter"], "weights": PIN["weights"]}
                runs[(arm, s, c)] = synth(arm, s, c, fn, md5=m)
        runs[("liq", s, "straight")] = synth("liq", s, "straight", straight_fn)
    return runs


def selftest():
    ok = True
    def check(name, cond):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + name); ok &= bool(cond)
    v = lambda r: score(r)["verdict"]
    check("all arms good -> PASS", v(world(lambda t: True)) == "PASS")
    check("liq fails sharp turns (>=135) -> FAIL (B1)", v(world(lambda t: t is None or t < 135)) == "FAIL")
    r = score(world(lambda t: t is None or t < 135))
    check("  ...and the sharp-turn bin shows it", r["report"]["turn_bins"]["liq"]["135-180"]["reach"] == 0.0)
    check("liq never reaches -> FAIL (B2 also)", score(world(lambda t: False))["B2_above_floor"]["pass"] is False)
    check("zero arm reaches everything -> NOT-EVIDENCE (floor)", v(world(lambda t: True, zero_fn=lambda t: True)).startswith("NOT-EVIDENCE"))
    check("pp misses half -> NOT-EVIDENCE (reference)", v(world(lambda t: True, pp_fn=lambda t: t is None or t < 90)).startswith("NOT-EVIDENCE"))
    check("straight control fails -> NOT-EVIDENCE (control 4)", v(world(lambda t: True, straight_fn=lambda t: False)).startswith("NOT-EVIDENCE"))
    check("a missing cell -> STOP", v(world(lambda t: True, drop=("liq", 302, "1"))).startswith("STOP"))
    check("wrong cfc md5 -> STOP", v(world(lambda t: True, md5_bad=True)).startswith("STOP"))
    w = world(lambda t: True); w[("liq", 301, "0")]["recs"].insert(3, {"ev": "ALERT", "reason": "NAN"})
    check("an ALERT in a run -> STOP", v(w).startswith("STOP"))
    w = world(lambda t: True); w[("zero", 303, "2")]["recs"].insert(5, {"ev": "END", "r": "socketClosed"})
    check("an END (server dropped) in a run -> STOP, even with done present", v(w).startswith("STOP"))
    w = world(lambda t: True)
    for x in w[("pp", 301, "0")]["recs"]:
        if x.get("ev") == "tick": x["us"] = 90000.0
    w2 = world(lambda t: True)
    for k, rr in w2.items():
        if k[0] == "pp":
            for x in rr["recs"]:
                if x.get("ev") == "tick": x["us"] = 90000.0
    check("slow loop (p99 > 50 ms) -> NOT-EVIDENCE", v(w2).startswith("NOT-EVIDENCE"))
    w = world(lambda t: True); w[("liq", 304, "2")]["start"]["wps"][3]["x"] = 999
    check("waypoints differ across arms -> STOP", v(w).startswith("STOP"))
    # B1 boundary, per seed 32 waypoints, pp 32/32: 3 misses = 0.094 under (within), 4 misses = 0.125 under (out)
    def misses(n_per_seed, seeds_hit):
        w = world(lambda t: True)
        for s in seeds_hit:
            k = 0
            for c in COURSES:
                for x in w[("liq", s, c)]["recs"]:
                    if x.get("ev") == "wp" and k < n_per_seed:
                        x["reached"] = False; k += 1
        return w
    check("B1 boundary: 3 misses/seed on all 5 seeds (0.094 under) -> PASS", v(misses(3, SEEDS)) == "PASS")
    check("B1 boundary: 4 misses/seed on 2 seeds (0.125 under) -> FAIL", v(misses(4, [301, 302])) == "FAIL")
    check("B1 boundary: 4 misses/seed on 1 seed only (4 of 5 within) -> PASS", v(misses(4, [301])) == "PASS")
    print("SELFTEST", "OK" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        sys.exit(0 if selftest() else 1)
    if len(sys.argv) != 2:
        print(__doc__); sys.exit(2)
    print(json.dumps(score(load(sys.argv[1])), indent=1))
