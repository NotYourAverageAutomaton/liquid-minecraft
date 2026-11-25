// Rung 3 : the FROZEN liquid unit (LTC-style; cfc.js b9b16802, weights e2df06eb) as the steering
// layer through seeded 8-waypoint courses. Pre-registration: the pre-registration (lab record)
// (+ AMENDMENT 0). Disposable sandbox body only. One invocation = one arm on one course; the runner randomises
// arm order per course.
// Usage: node rung3.js <arm> <seed> <course>
//   arm    in {pp, liq, liqreset, zero, nav}   (nav = mineflayer-pathfinder, report-only)
//   seed   in 301..305 (scored) or any int (selftest/dev)
//   course in 0..3 (scored) or "straight" (control 4: all turns 0 deg)
// Output: one JSON record per line on stdout. Scoring is done by score_rung3.py. No learning: learn()/decay() are
// never called. Nothing here writes outside stdout.
const mineflayer = require('mineflayer')
const { Rcon } = require('rcon-client')
const { observe, actuate, clearControls, headingYaw } = require('./adapter')
const { CfC } = require('./cfc')
const fs = require('fs')
const crypto = require('crypto')

const ARM = process.argv[2], SEED = Number(process.argv[3]), COURSE = process.argv[4]
const ARMS = ['pp', 'liq', 'liqreset', 'zero', 'nav']
if (!ARMS.includes(ARM) || !Number.isInteger(SEED) || !(COURSE === 'straight' || /^[0-3]$/.test(COURSE))) {
  console.error('usage: node rung3.js <pp|liq|liqreset|zero|nav> <seed> <0-3|straight>'); process.exit(2)
}
const WFILE = 'walkcfc_weights.json'
const W = JSON.parse(fs.readFileSync(WFILE))
const out = o => console.log(JSON.stringify(o))
const md5 = f => crypto.createHash('md5').update(fs.readFileSync(f)).digest('hex')

// rcon password: never in code (review note). run_rung3.sh exports it from a 0600 file; fail closed if missing.
const RCON_PASSWORD = process.env.RCON_PASSWORD
if (!RCON_PASSWORD) { console.error('STOP: RCON_PASSWORD not set (run via run_rung3.sh with the 0600 pass file)'); process.exit(2) }

const N_WP = 8, REACH = 1.2, WALK = 4.3, STALL_TICKS = 40, CARROT = 5

// Course: deterministic from (seed, course). Separate LCG per course so arms see identical waypoints.
function makeCourse(seed, course, start) {
  let s = (seed * 1000 + (course === 'straight' ? 99 : Number(course)) * 7919 + 17) % 2147483648
  const r = () => { s = (s * 1103515245 + 12345) % 2147483648; return s / 2147483648 }
  let head = r() * 2 * Math.PI, x = start.x, z = start.z
  const wps = []
  for (let i = 0; i < N_WP; i++) {
    let turnDeg = null
    if (i > 0) {
      const mag = course === 'straight' ? 0 : r() * 180
      const sign = r() < 0.5 ? -1 : 1
      turnDeg = mag
      head += sign * mag * Math.PI / 180
    }
    const len = 4 + r() * 8
    x += len * Math.cos(head); z += len * Math.sin(head)
    wps.push({ i, x, y: start.y, z, len, turn: turnDeg })   // turn = |heading change| INTO this waypoint's segment
  }
  return wps
}
const limitTicks = len => Math.max(60, Math.ceil(3 * len / WALK * 20))

const bot = mineflayer.createBot({ host: '127.0.0.1', port: 25565, username: 'r3bot', auth: 'offline' })
let physTicks = 0
bot.on('physicsTick', () => { physTicks++ })

function alertCheck(ctrl, act) {
  if (ctrl.h.some(v => !Number.isFinite(v)) || act.some(v => !Number.isFinite(v))) return 'NAN'
  if (ctrl.h.some(v => Math.abs(v) > 5)) return 'STATE_EXPLOSION'
  return null
}
function fail(reason, extra) {
  clearControls(bot); out({ ev: 'ALERT', reason, ...extra })
  setTimeout(() => process.exit(3), 300)
}

// One control step toward the current waypoint. Same maths as rung2.js decide() for 'pp' and 'cfc'.
function decide(ctrl, o) {
  const t0 = process.hrtime.bigint()
  let fx, fz
  if (ARM === 'pp') { fx = o.dx / o.dist; fz = o.dz / o.dist }
  else {
    let dx = o.dx, dz = o.dz
    if (o.dist > CARROT) { dx = dx * CARROT / o.dist; dz = dz * CARROT / o.dist }   // goal carrot (rung 2 'cfc')
    const r = ctrl.step([dx / 5, dz / 5, o.vx / 5, o.vz / 5, 0, 0], W.arch.dt)
    fx = r.act[0]; fz = r.act[1]
  }
  if (!Number.isFinite(fx) || !Number.isFinite(fz)) return { fx, fz, us: 0, forward: false }
  const cmd = actuate(bot, fx, fz, { thr: 0.05, sprintThr: 2 })
  return { fx, fz, us: Number(process.hrtime.bigint() - t0) / 1000, forward: cmd.forward }
}

async function steerTo(ctrl, wp, tag) {
  const lim = limitTicks(wp.len)
  let ticks = 0, path = 0, stall = 0, stalls = 0, prev = bot.entity.position.clone(), lastPhys = physTicks
  while (ticks < lim) {
    const o = observe(bot, wp)
    if (o.dist < REACH) return { reached: true, ticks, path, stalls }
    const d = decide(ctrl, o)
    if (ctrl) { const a = alertCheck(ctrl, [d.fx, d.fz]); if (a) { fail(a, { tag }); return null } }
    await bot.waitForTicks(1); ticks++
    const now = bot.entity.position, mv = Math.hypot(now.x - prev.x, now.z - prev.z)
    path += mv
    stall = mv < 0.005 ? stall + 1 : 0
    if (stall === STALL_TICKS) { stalls++; out({ ev: 'stall', tag, ticks }) }
    const gap = physTicks - lastPhys; lastPhys = physTicks
    out({ ev: 'tick', tag, us: +d.us.toFixed(1), gap })
    prev = now.clone()
  }
  return { reached: false, ticks, path, stalls }
}

async function navTo(wp, tag) {          // report-only reference: mineflayer-pathfinder's own loop
  const { goals } = require('mineflayer-pathfinder')
  const lim = limitTicks(wp.len), t0 = physTicks, p0 = bot.entity.position.clone()
  let path = 0, prev = p0.clone()
  bot.pathfinder.setGoal(new goals.GoalNear(wp.x, wp.y, wp.z, 1))
  while (physTicks - t0 < lim) {
    await bot.waitForTicks(1)
    const now = bot.entity.position; path += Math.hypot(now.x - prev.x, now.z - prev.z); prev = now.clone()
    if (Math.hypot(wp.x - now.x, wp.z - now.z) < REACH) { bot.pathfinder.setGoal(null); return { reached: true, ticks: physTicks - t0, path, stalls: 0 } }
  }
  bot.pathfinder.setGoal(null)
  return { reached: false, ticks: physTicks - t0, path, stalls: 0 }
}

bot.once('spawn', async () => {
  const rcon = await Rcon.connect({ host: '127.0.0.1', port: 25575, password: RCON_PASSWORD })
  if (ARM === 'nav') {
    const { pathfinder, Movements } = require('mineflayer-pathfinder')
    bot.loadPlugin(pathfinder)
    const mv = new Movements(bot); mv.canDig = false; mv.allow1by1towers = false; mv.allowSprinting = false
    bot.pathfinder.setMovements(mv)
  }
  await bot.waitForTicks(10)
  await rcon.send('tp r3bot 0.5 ~ 0.5'); await bot.waitForTicks(20)
  const start = bot.entity.position.clone()
  const worldSeed = String(await rcon.send('seed')).trim()
  const wps = makeCourse(SEED, COURSE, start)
  out({ ev: 'start', arm: ARM, seed: SEED, course: COURSE, n_wp: N_WP, reach: REACH,
        md5: { rung3: md5(__filename), cfc: md5('cfc.js'), adapter: md5('adapter.js'), weights: md5(WFILE) },
        world_seed: worldSeed, start: [+start.x.toFixed(2), +start.y.toFixed(2), +start.z.toFixed(2)],
        wps: wps.map(w => ({ i: w.i, x: +w.x.toFixed(2), z: +w.z.toFixed(2), len: +w.len.toFixed(2), turn: w.turn === null ? null : +w.turn.toFixed(1) })) })

  let ctrl = null
  if (ARM !== 'pp' && ARM !== 'nav') {
    ctrl = new CfC(W); ctrl.reset()
    if (ARM === 'zero') ctrl.Wout.forEach(r => r.fill(0))
  }
  for (const wp of wps) {
    if (ARM === 'liqreset' && wp.i > 0) ctrl.reset()       // the only difference between liq and liqreset
    const tag = `${ARM}:s${SEED}:c${COURSE}:w${wp.i}`
    const r = ARM === 'nav' ? await navTo(wp, tag) : await steerTo(ctrl, wp, tag)
    if (!r) return
    const p = bot.entity.position
    out({ ev: 'wp', arm: ARM, seed: SEED, course: COURSE, i: wp.i, turn: wp.turn, len: +wp.len.toFixed(2),
          reached: r.reached, ticks: r.ticks, path: +r.path.toFixed(2), stalls: r.stalls,
          end: [+p.x.toFixed(2), +p.z.toFixed(2)] })
    // a miss moves the target on: the next waypoint starts from wherever the body is (pre-reg Setup)
  }
  clearControls(bot)
  out({ ev: 'done' })
  finished = true                       // a normal finish: a later 'end' is not a failure
  await rcon.end(); setTimeout(() => process.exit(0), 500)
})
// A dead or dropped server must end the run at once, not after the runner's 900 s timeout (review note).
// Each writes its record (the scorer STOPs on ERR/KICK/END or a missing 'done'), then exits non-zero.
let finished = false
function die(rec) {
  if (finished) return
  finished = true
  try { clearControls(bot) } catch (_) {}
  out(rec)
  setTimeout(() => process.exit(4), 300)
}
bot.on('error', e => die({ ev: 'ERR', msg: e.message }))
bot.on('kicked', r => die({ ev: 'KICK', r: String(r) }))
bot.on('end', r => die({ ev: 'END', r: String(r) }))
