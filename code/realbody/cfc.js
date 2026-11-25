// JS port of the validated rung-2a CfC: forward (step) + SIGN-credit Hebbian learn() + optional weight-decay.
// Parity-gated vs Python (forward 1e-15; learn via learnparity trace).
const sig = z => 1 / (1 + Math.exp(-z))
const tanh = Math.tanh
const clip01 = x => Math.max(0, Math.min(1, x))
function matvec(W, x) { const o=new Array(W.length); for(let i=0;i<W.length;i++){let s=0;const Wi=W[i];for(let j=0;j<x.length;j++)s+=Wi[j]*x[j];o[i]=s;} return o }
class CfC {
  constructor(w) { this.Wx=w.Wx; this.Wh=w.Wh; this.Wt=w.Wt; this.Wout=w.Wout.map(r=>r.slice()); this.wconf=w.wconf; this.hid=w.arch.hid; this.reset() }
  reset() { this.h=new Array(this.hid).fill(0) }
  step(obs, dt) {
    const g=matvec(this.Wt,obs).map(sig)
    const wx=matvec(this.Wx,obs), wh=matvec(this.Wh,this.h)
    const cand=wx.map((v,i)=>tanh(v+wh[i]))
    this.h=this.h.map((hi,i)=>(1-g[i]*dt)*hi+(g[i]*dt)*cand[i])
    const act=matvec(this.Wout,this.h).map(tanh)
    let c=0; for(let i=0;i<this.hid;i++) c+=this.wconf[i]*this.h[i]
    this.pre=this.h.slice(); this.post=act.slice()            // stored for learn (= Python)
    return { act, conf: sig(c) }
  }
  learn(reward, lr, conf, active=1.0) {                        // base SIGN-credit (hebbian_dW), Wout only
    const eps=1e-6, gain=reward*clip01(active)*clip01(conf), scale=gain/Math.max(Math.abs(gain),eps)
    for (let i=0;i<this.Wout.length;i++) for (let j=0;j<this.Wout[0].length;j++)
      this.Wout[i][j]+= lr*(this.post[i]*this.pre[j])*scale
  }
  decay(d) { for (let i=0;i<this.Wout.length;i++) for (let j=0;j<this.Wout[0].length;j++) this.Wout[i][j]*=(1-d) }
}
module.exports = { CfC }
