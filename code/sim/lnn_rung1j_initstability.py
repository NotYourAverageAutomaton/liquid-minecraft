"""Pick a  RUNG-1J: is the from-scratch LEARN-RATE fragility (rung-1i: 2/5 seeds never
learn, 0.07) fixable by a practical init/training strategy -- i.e. can we make the no-backprop CfC
controller RELIABLY trainable, which is the prerequisite for any real-body deployment? Pure-python, CPU.

WHY: rung-1i (direction-in-credit, honest negative) RE-LOCATED the lead: the binding barrier for pick-a
is NOT the credit form but the INIT/seed-driven from-scratch learn-rate + re-adapt robustness. A
controller that trains on only 3/5 random inits is not deployable. This rung asks the operational
question directly, with two pre-registered levers, before any further mechanism work.

PRE-REGISTERED (before results). Seeds = outer 1..8. "LEARNED" = eval >= 0.50 (eval n=60). Base
sign-credit rule, lr 0.05, 400-episode single-run budget as the reference.
  LEVER 1 -- INIT-SCALE sweep: the CfC init uses gauss std s for all weight matrices (current 0.3).
    Sweep s in {0.1, 0.2, 0.3, 0.5}, train each of the 8 seeds from scratch, report learned-rate x/8
    and mean score. QUESTION: does any scale give a ROBUST rate (>= 7/8)? (cheap fix, zero extra compute)
  LEVER 2 -- RESTART-AND-SELECT (the standard no-backprop reliability fix): fix s=0.3. For each outer
    seed, spawn K=5 inner inits, PROBE each with P=80 episodes, pick the best by probe-eval (n=30),
    CONTINUE the winner to a 400-episode total, final eval n=60. QUESTION: does restart-and-select
    reach >= 0.50 on >= 7/8 outer seeds? Report its TOTAL episode cost vs a single run honestly
    (it is NOT budget-matched -- 5 probes cost extra; CPU has 404x real-time headroom so the trade is
    compute-for-reliability, free in wall-clock, and that framing is reported, not hidden).
  CONCLUSION LOGIC (pre-registered):
   - an init-scale >= 7/8  -> cheap fix, no compute cost. Adopt that scale.
   - else restart-and-select >= 7/8 -> reliability achievable at ~Nx compute (reported); fine given headroom.
   - else -> from-scratch no-backprop training is fundamentally init-fragile for this task; the real-body
     plan must deploy a KNOWN-GOOD sim-trained init (train in sim, ship the learned weights), NOT train
     from scratch on the body -- which is the realistic deployment path anyway. Negative still decides.
"""
import math, random, time, copy

def gauss_mat(r,n,m,s): return [[r.gauss(0,s) for _ in range(m)] for _ in range(n)]
def matvec(W,x): return [sum(W[i][j]*x[j] for j in range(len(x))) for i in range(len(W))]
def tanh(v): return [math.tanh(z) for z in v]
def sig(z): return 1.0/(1.0+math.exp(-z))
def dot(a,b): return sum(a[i]*b[i] for i in range(len(a)))

def hebbian_dW(pre, post, reward, active=1.0, conf=1.0, eps=1e-6):
    gain=reward*max(0,min(1,active))*max(0,min(1,conf)); denom=max(abs(gain),eps); scale=gain/denom
    return [[ (post[i]*pre[j])*scale for j in range(len(pre))] for i in range(len(post))]

class CfC:
    def __init__(self, obs=4, hid=16, act=2, seed=0, s=0.3):
        r=random.Random(seed)
        self.Wx=gauss_mat(r,hid,obs,s); self.Wh=gauss_mat(r,hid,hid,s)
        self.Wt=gauss_mat(r,hid,obs,s); self.Wout=gauss_mat(r,act,hid,s)
        self.wconf=[r.gauss(0,s) for _ in range(hid)]; self.hid=hid; self.reset()
    def reset(self): self.h=[0.0]*self.hid
    def step(self,obs,dt):
        g=[sig(z) for z in matvec(self.Wt,obs)]
        wx=matvec(self.Wx,obs); wh=matvec(self.Wh,self.h)
        cand=tanh([wx[i]+wh[i] for i in range(self.hid)])
        self.h=[(1-g[i]*dt)*self.h[i]+(g[i]*dt)*cand[i] for i in range(self.hid)]
        act=tanh(matvec(self.Wout,self.h)); conf=sig(dot(self.wconf,self.h))
        self.pre=list(self.h); self.post=list(act); return act,conf
    def learn(self,reward,lr,conf):
        dW=hebbian_dW(self.pre,self.post,reward,1.0,conf)
        for i in range(len(self.Wout)):
            for j in range(len(self.Wout[0])): self.Wout[i][j]+=lr*dW[i][j]

class Walk:
    def __init__(self,seed,gain=1.0): self.r=random.Random(seed); self.gain=gain
    def reset(self):
        self.px,self.pz=self.r.uniform(-5,5),self.r.uniform(-5,5)
        self.vx=self.vz=0.0; self.tx,self.tz=self.r.uniform(-5,5),self.r.uniform(-5,5); self.t=0
        return self._obs()
    def _d(self): return math.hypot(self.tx-self.px,self.tz-self.pz)
    def _obs(self): return [(self.tx-self.px)/5,(self.tz-self.pz)/5,self.vx/5,self.vz/5]
    def step(self,act,dt=0.1):
        d0=self._d()
        ax=max(-1,min(1,act[0])); az=max(-1,min(1,act[1]))
        self.vx=0.85*self.vx+self.gain*ax*2*dt; self.vz=0.85*self.vz+self.gain*az*2*dt
        self.px+=self.vx*dt; self.pz+=self.vz*dt; self.t+=1; d1=self._d()
        return self._obs(),(d0-d1),d1,(d1<0.5 or self.t>=200)

def episode(ctrl,env,learn,lr):
    obs=env.reset(); ctrl.reset()
    for _ in range(200):
        act,conf=ctrl.step(obs,0.1); obs,rew,dist,done=env.step(act)
        if learn: ctrl.learn(rew,lr,conf)
        if dist<0.5: break
        if done: break
    return (dist<0.5),env.t
def ev(ctrl,n,seed0):
    s=0
    for e in range(n):
        a,_=episode(ctrl,Walk(10000+seed0+e),False,0.0); s+=a
    return s/n

SEEDS=list(range(1,9)); N_EVAL=60; LR=0.05
t0=time.time()
print("=== RUNG-1J: init / learning-stability -- is the no-backprop controller RELIABLY trainable? ===")
print(f"seeds={SEEDS} eval_n={N_EVAL}  'learned' = eval>=0.50\n")

# LEVER 1: init-scale sweep
print("LEVER 1 -- INIT-SCALE sweep (from-scratch learned-rate out of 8):")
scale_rows=[]
for s in (0.1,0.2,0.3,0.5):
    scores=[]
    for sd in SEEDS:
        L=CfC(seed=sd,s=s)
        for e in range(400): episode(L,Walk(e,gain=1.0),True,LR)
        scores.append(ev(L,N_EVAL,0))
    nl=sum(1 for x in scores if x>=0.50)
    scale_rows.append((s,nl,sum(scores)/len(scores)))
    print(f"  s={s}: learned {nl}/8  mean {sum(scores)/len(scores):.2f}  scores {[round(x,2) for x in scores]}")

# LEVER 2: restart-and-select at s=0.3
print("\nLEVER 2 -- RESTART-AND-SELECT (s=0.3, K=5 inner probes of 80 ep, continue best to 400 total):")
K=5; PROBE=80
rs_scores=[]; total_ep=0
for sd in SEEDS:
    cands=[]
    for k in range(K):
        L=CfC(seed=sd*100+k, s=0.3)
        for e in range(PROBE): episode(L,Walk(e,gain=1.0),True,LR);
        total_ep+=PROBE
        pscore=ev(L,30,0); total_ep+=0
        cands.append((pscore,k,L))
    cands.sort(key=lambda c:-c[0]); best=copy.deepcopy(cands[0][2]); bk=cands[0][1]
    for e in range(PROBE,400): episode(best,Walk(e,gain=1.0),True,LR);
    total_ep+=(400-PROBE)
    fsc=ev(best,N_EVAL,0); rs_scores.append(fsc)
    print(f"  outer seed {sd}: probes {[round(c[0],2) for c in sorted(cands,key=lambda c:c[1])]} -> picked k{bk} -> final {fsc:.2f}")
rs_learned=sum(1 for x in rs_scores if x>=0.50)
single_cost=400*len(SEEDS)
print(f"\n  restart-and-select learned {rs_learned}/8  scores {[round(x,2) for x in rs_scores]}")
print(f"  episode cost: {total_ep} vs single-run {single_cost} = {total_ep/single_cost:.2f}x (CPU 404x real-time headroom -> free in wall-clock)")

print("\n=== VERDICT (pre-registered conclusion logic) ===")
best_scale=max(scale_rows,key=lambda r:r[1])
if best_scale[1]>=7:
    print(f"  CHEAP FIX: init-scale s={best_scale[0]} gives {best_scale[1]}/8 (>=7) -- adopt it, zero extra compute.")
elif rs_learned>=7:
    print(f"  RELIABILITY via RESTART-AND-SELECT: {rs_learned}/8 (>=7) at {total_ep/single_cost:.1f}x compute (free given headroom).")
    print(f"  Best single init-scale was s={best_scale[0]} at {best_scale[1]}/8 (insufficient alone).")
else:
    print(f"  INIT-FRAGILE CONFIRMED: best init-scale {best_scale[1]}/8, restart-and-select {rs_learned}/8 -- neither robust.")
    print(f"  OPERATIONAL CONSEQUENCE: real-body plan ships a KNOWN-GOOD sim-trained init (train in sim, deploy the")
    print(f"  learned weights), NOT from-scratch training on the body. That is the realistic path regardless.")
print(f"\nwall {time.time()-t0:.1f}s")
