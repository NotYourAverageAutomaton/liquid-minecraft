// Rung 2 : sim-trained CfC WALK controller on a disposable sandbox body, scored against the
// pre-registered bars in the pre-registration (lab record) (+ 11-05 amendments).
// Modes:  A <arm>   arm in {cfc, cfc_ood, pp, rand}: 8 bearings x (100 out + back), R=3, T=120 s
//         C <decay> 20k-tick continuous online loop, plasticity on, lr 0.05, random targets within 7
//         E <fault> fault in {nan, stall}: planted fault, the alert must fire
// Every line of output is JSON on stdout (one record per event); scoring is done by score_rung2.py.
const mineflayer = require('mineflayer')
const { Rcon } = require('rcon-client')
// rcon password: never in code; set RCON_PASSWORD (as rung3.js / run_rung3.sh do). Fail closed.
const RCON_PASSWORD = process.env.RCON_PASSWORD
if (!RCON_PASSWORD) { console.error('STOP: RCON_PASSWORD not set'); process.exit(2) }
const { observe, actuate, clearControls, headingYaw } = require('./adapter')
const { CfC } = require('./cfc')
const fs = require('fs')

const MODE = process.argv[2], ARG = process.argv[3]
const W = JSON.parse(fs.readFileSync('walkcfc_weights.json'))
const out = o => console.log(JSON.stringify(o))
let THETA = 0
const fro = M => Math.sqrt(M.reduce((s, r) => s + r.reduce((a, v) => a + v * v, 0), 0))
let seed = 20261105
const rand = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648 }

const bot = mineflayer.createBot({ host: '127.0.0.1', port: 25565, username: 'r2bot', auth: 'offline' })
let physTicks = 0
bot.on('physicsTick', () => { physTicks++ })

function alertCheck(ctrl, act, stall) {
  if (ctrl.h.some(v => !Number.isFinite(v)) || act.some(v => !Number.isFinite(v))) return 'NAN'
  if (ctrl.h.some(v => Math.abs(v) > 5)) return 'STATE_EXPLOSION'
  if (stall >= 100) return 'STALL'
  return null
}
function fail(reason, extra) {
  clearControls(bot); out({ ev: 'ALERT', reason, ...extra })
  setTimeout(() => process.exit(3), 300)
}

// One controlled step toward tgt; returns per-tick record. policy: cfc | cfc_ood | pp | rand
function decide(ctrl, policy, o, randHead, learn) {
  const t0 = process.hrtime.bigint()
  let fx, fz, conf = 1
  if (policy === 'pp') { fx = o.dx / o.dist; fz = o.dz / o.dist }
  else if (policy === 'rand') { fx = Math.cos(randHead); fz = Math.sin(randHead) }
  else {
    let dx = o.dx, dz = o.dz
    if (policy === 'cfc' && o.dist > 5) { dx = dx * 5 / o.dist; dz = dz * 5 / o.dist }  // goal carrot
    const r = ctrl.step([dx / 5, dz / 5, o.vx / 5, o.vz / 5, 0, 0], W.arch.dt)
    fx = r.act[0]; fz = r.act[1]; conf = r.conf
    if (learn) { ctrl.learn(learn.reward, learn.lr, conf); ctrl.decay(learn.decay) }
  }
  if (THETA) { const c = Math.cos(THETA), sn = Math.sin(THETA); [fx, fz] = [c * fx - sn * fz, sn * fx + c * fz] }   // 2b planted heading rotation
  // never send a non-finite command to the server: the alert fires first (Bar E ordering)
  if (!Number.isFinite(fx) || !Number.isFinite(fz)) return { fx, fz, us: 0, intended: NaN, forward: false }
  const cmd = actuate(bot, fx, fz, { thr: 0.05, sprintThr: 2 })
  const us = Number(process.hrtime.bigint() - t0) / 1000
  return { fx, fz, us, intended: headingYaw(fx, fz), forward: cmd.forward, conf }
}

async function leg(ctrl, policy, tgt, budgetTicks, randHead, tag) {
  if (ctrl) ctrl.reset()
  let ticks = 0, prev = bot.entity.position.clone(), lastPhys = physTicks, stall = 0
  while (ticks < budgetTicks) {
    const o = observe(bot, tgt)
    if (o.dist < 3) return { reached: true, ticks }
    const d = decide(ctrl, policy, o, randHead, null)
    if (ctrl) { const a = alertCheck(ctrl, [d.fx, d.fz], 0); if (a) { fail(a, { tag }); return null } }
    if (stall >= 100) { out({ ev: 'stuck', tag }); return { reached: false, ticks, stuck: true } }
    await bot.waitForTicks(1); ticks++
    const now = bot.entity.position, mvx = now.x - prev.x, mvz = now.z - prev.z, mv = Math.hypot(mvx, mvz)
    stall = (d.forward && mv < 0.005) ? stall + 1 : 0
    const gap = physTicks - lastPhys; lastPhys = physTicks
    out({ ev: 'tick', tag, us: +d.us.toFixed(1), gap, intended: +d.intended.toFixed(4),
          actual: mv > 0.02 ? +headingYaw(mvx, mvz).toFixed(4) : null })
    prev = now.clone()
  }
  return { reached: false, ticks }
}

async function modeA(rcon, arm) {
  const sp = { x: 0.5, y: bot.entity.position.y, z: 0.5 }
  for (let b = 0; b < 8; b++) {
    await rcon.send(`tp r2bot ${sp.x} ${sp.y} ${sp.z} 0 0`); await bot.waitForTicks(20)
    const th = b * Math.PI / 4
    const far = { x: sp.x + 100 * Math.cos(th), y: sp.y, z: sp.z + 100 * Math.sin(th) }
    const randHead = rand() * 2 * Math.PI
    const ctrl = (arm === 'cfc' || arm === 'cfc_ood') ? new CfC(W) : null
    const t0 = physTicks
    const r1 = await leg(ctrl, arm, far, 2400, randHead, `A:${arm}:b${b}:out`); if (!r1) return
    let r2 = { reached: false, ticks: 0 }
    if (r1.reached) { r2 = await leg(ctrl, arm, sp, 2400 - r1.ticks, randHead, `A:${arm}:b${b}:back`); if (!r2) return }
    clearControls(bot)
    const p = bot.entity.position
    out({ ev: 'trial', arm, bearing: b * 45, out: r1.reached, back: r2.reached, pass: r1.reached && r2.reached,
          ticks: r1.ticks + r2.ticks, end: [+p.x.toFixed(1), +p.z.toFixed(1)], phys: physTicks - t0 })
  }
}

// 2b step 0 (vault f5b4d67): Wout lesions on the same seeded C course, plasticity off. Separate LCG so targets match frozen.
function lesion(ctrl, arm) {
  const W = ctrl.Wout, rows = W.length, cols = W[0].length
  if (arm === 'zero') W.forEach(r => r.fill(0))
  else if (arm === 'scale0.14' || arm === 'scale0.02') { const f = Number(arm.slice(5)); W.forEach(r => r.forEach((v, j) => { r[j] = v * f })) }
  else if (arm === 'rotate') { const r0 = W[0].slice(), r1 = W[1].slice(); W[0] = r1.map(v => -v); W[1] = r0 }   // act -> 90 deg rotation
  else if (arm === 'shuffle') {
    let s2 = 7; const r2 = () => { s2 = (s2 * 1103515245 + 12345) % 2147483648; return s2 / 2147483648 }
    const flat = W.flat(); for (let i = flat.length - 1; i > 0; i--) { const j = Math.floor(r2() * (i + 1)); [flat[i], flat[j]] = [flat[j], flat[i]] }
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) W[i][j] = flat[i * cols + j]
  } else if (arm !== 'frozen') throw new Error('unknown arm ' + arm)
}

// 2b report-only: effective rotation of output rows 0/1 vs their start (rad); undoing a planted theta would drive it toward -theta
function rotEst(Wc, Wi) { let a = 0, b = 0; for (let j = 0; j < Wc[0].length; j++) { a += Wc[0][j] * Wi[0][j] + Wc[1][j] * Wi[1][j]; b += Wc[1][j] * Wi[0][j] - Wc[0][j] * Wi[1][j] } return Math.atan2(b, a) }

async function modeC(rcon, decay, opt = {}) {
  const K = opt.K ?? 20000, lr = opt.lr ?? (process.argv[4] !== undefined ? Number(process.argv[4]) : 0.05)   // argv[4]: 11-05 post-data frozen control (lr 0)
  await rcon.send('tp r2bot 0.5 ~ 0.5'); await bot.waitForTicks(20)
  const sp = bot.entity.position.clone()
  const ctrl = new CfC(W); ctrl.reset()
  if (opt.lesion) lesion(ctrl, opt.lesion)
  const W0 = ctrl.Wout.map(r => r.slice())
  const rnd = () => ({ x: sp.x + (rand() * 2 - 1) * 7, y: sp.y, z: sp.z + (rand() * 2 - 1) * 7 })
  let tgt = rnd(), lastPhys = physTicks, stall = 0, prev = bot.entity.position.clone()
  let rAbs = 0, rN = 0, rNz = 0   // positive control: reward reaching learn(), reset every 1k ticks
  out({ ev: 'w', k: 0, wf: fro(ctrl.Wout) })
  for (let k = 1; k <= K; k++) {
    const o = observe(bot, tgt)
    if (o.dist < 1.2) { out({ ev: 'reach', k }); tgt = rnd(); clearControls(bot); await bot.waitForTicks(1); lastPhys = physTicks; continue }   // reach tick: no update
    const d = decide(opt.policy === 'pp' ? null : ctrl, opt.policy ?? 'cfc', o, 0, null)
    const a = opt.policy === 'pp' ? null : alertCheck(ctrl, [d.fx, d.fz], stall); if (a) return fail(a, { k })
    await bot.waitForTicks(1)
    // 11-05 fix: reward spans the tick (dist before acting minus dist after); was two reads with no tick between = 0
    // same target both reads (re-targets happen only at the loop top or after this line)
    if (lr > 0 && ctrl && opt.policy !== 'pp') {
      const reward = o.dist - observe(bot, tgt).dist
      ctrl.learn(reward, lr, d.conf); ctrl.decay(decay)
      rAbs += Math.abs(reward); rN++; if (reward !== 0) rNz++
    }
    const now = bot.entity.position, mv = Math.hypot(now.x - prev.x, now.z - prev.z)
    stall = (d.forward && mv < 0.005) ? stall + 1 : 0
    // a stall with a target in reach is a stuck body: re-target instead of dying (logged, counted)
    if (stall >= 100) { out({ ev: 'stuck', k }); tgt = rnd(); stall = 0 }
    const gap = physTicks - lastPhys; lastPhys = physTicks
    out({ ev: 'tick', tag: `C:${decay}`, k, us: +d.us.toFixed(1), gap })
    prev = now.clone()
    if (k % 500 === 0) out({ ev: 'w', k, wf: fro(ctrl.Wout), hmax: Math.max(...ctrl.h.map(Math.abs)), rot: +rotEst(ctrl.Wout, W0).toFixed(4) })
    if (k % 1000 === 0 && lr > 0) { out({ ev: 'r', k, meanAbs: rN ? rAbs / rN : 0, fracNz: rN ? rNz / rN : 0, n: rN }); rAbs = 0; rN = 0; rNz = 0 }
  }
  clearControls(bot)
}

async function modeE(rcon, fault) {
  await rcon.send('tp r2bot 0.5 ~ 0.5'); await bot.waitForTicks(20)
  const sp = bot.entity.position.clone(), tgt = { x: sp.x + 50, y: sp.y, z: sp.z }
  const ctrl = new CfC(W); ctrl.reset()
  if (fault === 'stall') await rcon.send(`fill ${Math.floor(sp.x) + 2} ${Math.floor(sp.y)} ${Math.floor(sp.z) - 2} ${Math.floor(sp.x) + 2} ${Math.floor(sp.y) + 2} ${Math.floor(sp.z) + 2} minecraft:stone`)
  let stall = 0, prev = bot.entity.position.clone()
  for (let k = 0; k < 600; k++) {
    if (fault === 'nan' && k === 100) ctrl.h[0] = NaN
    const o = observe(bot, tgt), d = decide(ctrl, 'cfc', o, 0, null)
    const a = alertCheck(ctrl, [d.fx, d.fz], stall)
    if (a) { if (fault === 'stall') await rcon.send(`fill ${Math.floor(sp.x) + 2} ${Math.floor(sp.y)} ${Math.floor(sp.z) - 2} ${Math.floor(sp.x) + 2} ${Math.floor(sp.y) + 2} ${Math.floor(sp.z) + 2} minecraft:air`); return fail(a, { k, planted: fault }) }
    await bot.waitForTicks(1)
    const now = bot.entity.position
    stall = (d.forward && Math.hypot(now.x - prev.x, now.z - prev.z) < 0.005) ? stall + 1 : 0
    prev = now.clone()
  }
  out({ ev: 'NO_ALERT', planted: fault })   // the planted fault was NOT caught: Bar E fails
}

bot.once('spawn', async () => {
  const rcon = await Rcon.connect({ host: '127.0.0.1', port: 25575, password: RCON_PASSWORD })
  await bot.waitForTicks(10)
  out({ ev: 'start', mode: MODE, arg: ARG, arg2: process.argv[4], weights_seed: W.train.seed, init: W.train.init_scale })
  if (MODE === 'A') await modeA(rcon, ARG)
  else if (MODE === 'C') await modeC(rcon, Number(ARG))
  else if (MODE === 'R') {   // 2b: R <cfc|learn|learn0|pp> <thetaDeg> <K> <rep>   (learn0 = lr 0.05, decay 0: positive control)
    THETA = Number(process.argv[4]) * Math.PI / 180; seed = 20261105 + 1000 * Number(process.argv[6] ?? 0)
    const pol = ARG, K = Number(process.argv[5])
    await modeC(rcon, pol === 'learn' ? 3e-6 : 0, { K, lr: pol.startsWith('learn') ? 0.05 : 0, policy: pol === 'pp' ? 'pp' : 'cfc' })
  }
  else if (MODE === 'S') await modeC(rcon, 0, { K: 5000, lr: 0, lesion: ARG })
  else if (MODE === 'E') { await modeE(rcon, ARG); return }
  out({ ev: 'done' })
  await rcon.end(); setTimeout(() => process.exit(0), 500)
})
bot.on('error', e => out({ ev: 'ERR', msg: e.message }))
bot.on('kicked', r => out({ ev: 'KICK', r: String(r) }))
