// obs/action adapter: sim CfC (continuous 2D force) <-> real mineflayer body (WALK skill; jump gated off).
// MC movement: pressing 'forward' at yaw θ moves the body in world-xz direction (-sin θ, cos θ).
// So to drive toward a force vector (fx,fz), set yaw = atan2(-fx, fz) and press forward.
function headingYaw(fx, fz) { return Math.atan2(-fx, -fz) }

function observe(bot, target) {
  const p = bot.entity.position, v = bot.entity.velocity
  const dx = target.x - p.x, dz = target.z - p.z
  return { dx, dz, vx: v.x, vz: v.z, dist: Math.hypot(dx, dz) }
}
// Quantize continuous force (fx,fz) -> discrete body controls. Returns the command for logging.
function actuate(bot, fx, fz, opts = {}) {
  const thr = opts.thr ?? 0.05, sprintThr = opts.sprintThr ?? 0.6
  const mag = Math.hypot(fx, fz)
  const yaw = headingYaw(fx, fz)
  bot.look(yaw, 0, true).catch(() => {})
  bot.setControlState('forward', mag > thr)
  bot.setControlState('sprint', mag > sprintThr)
  return { yaw, mag, forward: mag > thr, sprint: mag > sprintThr }
}
function clearControls(bot) { ['forward','back','left','right','sprint','jump'].forEach(k => bot.setControlState(k, false)) }
module.exports = { observe, actuate, clearControls, headingYaw }
