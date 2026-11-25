#!/usr/bin/env bash
# Rung 3 runner . Runs on the sandbox NUC in the bot/ directory (cfc.js, adapter.js, walkcfc_weights.json,
# walkcfc_parity.json, cfc-parity.js, rung3.js, node_modules), with Paper already up on 127.0.0.1 (game 25565, rcon 25575).
# Off-scratch node (/srv/scratch is noexec; memory note 10-20). Writes only into captures/rung3-<stamp>/.
set -euo pipefail
NODE=/usr/bin/node
STAMP=$(date +%Y%m%d-%H%M)
OUT=captures/rung3-$STAMP; mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/runner.log"; }

# control 0 (review note): Paper must be listening on 127.0.0.1 ONLY, for game and rcon, or nothing runs.
for P in 25565 25575; do
  L=$(ss -ltnH "sport = :$P" | awk '{print $4}')
  [ -n "$L" ] || { log "STOP: nothing listening on $P"; exit 3; }
  W=$(printf '%s\n' "$L" | grep -vxF -e "127.0.0.1:$P" -e "[::ffff:127.0.0.1]:$P" || true)   # IPv4 or IPv4-mapped loopback only (review note)
  [ -z "$W" ] || { log "STOP: port $P bound wider than 127.0.0.1: $W"; exit 3; }
done
# rcon password from a 0600 file beside the server config, never from code; fail closed.
PASSF=${RCON_PASS_FILE:-../server/rcon.pass}
[ -f "$PASSF" ] || { log "STOP: rcon pass file $PASSF missing"; exit 3; }
[ "$(stat -c %a "$PASSF")" = 600 ] || { log "STOP: $PASSF must be mode 0600"; exit 3; }
RCON_PASSWORD=$(head -n1 "$PASSF"); [ -n "$RCON_PASSWORD" ] || { log "STOP: $PASSF empty"; exit 3; }
export RCON_PASSWORD

# control 1: parity gate on the md5 that runs (must print PASS, max |JS-PY| < 1e-9)
$NODE cfc-parity.js | tee "$OUT/parity.txt"
grep -q 'PASS' "$OUT/parity.txt" || { log "STOP: parity FAIL"; exit 3; }
# control 2: artifacts that ran
md5sum rung3.js cfc.js adapter.js walkcfc_weights.json walkcfc_parity.json cfc-parity.js run_rung3.sh > "$OUT/artifacts.md5"
sha256sum ../server/paper.jar > "$OUT/jar.sha256" 2>/dev/null || log "WARN: paper.jar not at ../server/paper.jar; record its sha256 by hand"
ss -ltn > "$OUT/ss-ltn.txt"
log "start; artifacts:"; cat "$OUT/artifacts.md5" | tee -a "$OUT/runner.log"

one() {  # arm seed course
  local f="$OUT/$1-s$2-c$3.jsonl"
  log "run $1 s$2 c$3"
  timeout 900 $NODE rung3.js "$1" "$2" "$3" > "$f" 2> "$f.err" || log "rc=$? for $1 s$2 c$3 (scorer will STOP on it)"
  sleep 5                         # let the server drop the last session (fixed username)
}
# scored cells: per course, the four scored arms in a seeded random order
for S in 301 302 303 304 305; do
  for C in 0 1 2 3; do
    ORDER=$(printf 'pp\nliq\nliqreset\nzero\n' | awk -v sd="$S$C" 'BEGIN{srand(sd)} {print rand()"\t"$0}' | sort -n | cut -f2)
    for A in $ORDER; do one "$A" "$S" "$C"; done
  done
  one liq "$S" straight          # control 4
  one pp "$S" straight           # report-only reference for control 4
done
# report-only: pathfinder's own steering on the scored courses
for S in 301 302 303 304 305; do for C in 0 1 2 3; do one nav "$S" "$C"; done; done
( cd "$OUT" && md5sum *.jsonl *.txt *.md5 > MANIFEST.md5 )
log "done; captures in $OUT (MANIFEST.md5). Score off-box: score_rung3.py $OUT"
