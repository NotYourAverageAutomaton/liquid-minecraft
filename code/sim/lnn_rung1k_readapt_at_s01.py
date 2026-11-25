"""Pick a  RUNG-1K: does the cheap init fix from rung-1j (s=0.1 -> 8/8 from-scratch) ALSO
preserve the re-adaptation robustness (under-shoot drift + sign-reversal re-adapt), or is it a TRADE?
Pure-python, CPU. This closes the rung-1j claim honestly: "fixed from-scratch learning" is only a clean
win if it does NOT break the adaptation that was the other half of the rung-1i barrier.

PRE-REGISTERED (before results): BASE sign-credit rule (always update -- the honest default; no credit
form survived rung-1i). Compare s=0.1 (the rung-1j fix) vs s=0.3 (current). Seeds 1..8, eval n=60.
Only LEARNED seeds (from-scratch >=0.50) enter the adaptation read. Same bar as rung-1h/1i:
  (a) under-shoot g0.5 keep-train >= that seed's no-learn baseline - 0.03  (drift suppressed)
  (b) axis-swap   keep-train >= 0.50                                        (re-adapt preserved)
QUESTION: is s=0.1's both-pass rate >= s=0.3's?
  - s=0.1 both-pass rate >= s=0.3's AND s=0.1 learns more seeds -> CLEAN WIN (adopt s=0.1 outright).
  - s=0.1 learns more but re-adapt both-pass rate DROPS -> TRADE (note it; init-scale buys trainability
    at some adaptation cost; decide by which matters more for the real-body target).
  Honest either way; reports the full per-seed table at both scales.
"""
import math, random, time, copy
def gauss_mat(r,n,m,s): return [[r.gauss(0,s) for _ in range(m)] for _ in range(n)]
def matvec(W,x): return [sum(W[i][j]*x[j] for j in range(len(x))) for i in range(len(W))]
def tanh(v): return [math.tanh(z) for z in v]
def sig(z): return 1.0/(1.0+math.exp(-z))
def dot(a,b): return sum(a[i]*b[i] for i in range(len(a)))
def hebbian_dW(pre,post,reward,active=1.0,conf=1.0,eps=1e-6):
    gain=reward*max(0,min(1,active))*max(0,min(1,conf)); denom=max(abs(gain),eps); scale=gain/denom
    return [[(post[i]*pre[j])*scale for j in range(len(pre))] for i in range(len(post))]
class CfC:
    def __init__(self,obs=4,hid=16,act=2,seed=0,s=0.3):
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
class WalkSwap(Walk):
    def step(self,act,dt=0.1): return super().step([act[1],-act[0]],dt)
def episode(ctrl,env,learn,lr):
    obs=env.reset(); ctrl.reset()
    for _ in range(200):
        act,conf=ctrl.step(obs,0.1); obs,rew,dist,done=env.step(act)
        if learn: ctrl.learn(rew,lr,conf)
        if dist<0.5: break
        if done: break
    return (dist<0.5),env.t
def ev(ctrl,n,seed0,envcls=Walk,gain=1.0):
    s=0
    for e in range(n):
        env=Walk(10000+seed0+e,gain=gain) if envcls is Walk else WalkSwap(10000+seed0+e)
        a,_=episode(ctrl,env,False,0.0); s+=a
    return s/n
SEEDS=list(range(1,9)); N=60; LR=0.05
t0=time.time()
print("=== RUNG-1K: re-adaptation robustness at the rung-1j init fix (s=0.1) vs current (s=0.3) ===")
print("bar: (a) under-shoot adapt >= seed no-learn baseline-0.03 ; (b) axis-swap adapt >= 0.50 ; BASE rule\n")
summary={}
for s in (0.1,0.3):
    rows=[]; learned=0
    print(f"--- s={s} ---")
    for sd in SEEDS:
        L=CfC(seed=sd,s=s)
        for e in range(400): episode(L,Walk(e,1.0),True,LR)
        ls=ev(L,N,0)
        if ls<0.50:
            print(f"  seed {sd}: from-scratch {ls:.2f} NOT LEARNED (excluded)"); continue
        learned+=1
        ub=ev(copy.deepcopy(L),N,0,gain=0.5)
        Aus=copy.deepcopy(L)
        for e in range(200): episode(Aus,Walk(e,0.5),True,LR)
        us=ev(Aus,N,0,gain=0.5)
        Asw=copy.deepcopy(L)
        for e in range(300): episode(Asw,WalkSwap(e),True,LR)
        sw=ev(Asw,N,0,envcls=WalkSwap)
        a=us>=ub-0.03; b=sw>=0.50; both=a and b
        rows.append(both)
        print(f"  seed {sd}: scratch {ls:.2f} | under-shoot {ub:.2f}->{us:.2f} (a:{a}) | axis-swap ->{sw:.2f} (b:{b}) | BOTH:{both}")
    frac=sum(rows)/len(rows) if rows else 0.0
    summary[s]=(learned,sum(rows),len(rows),frac)
    print(f"  => s={s}: learned {learned}/8 ; both-pass {sum(rows)}/{len(rows)} = {frac:.0%}\n")
print("=== VERDICT ===")
l1,p1,n1,f1=summary[0.1]; l3,p3,n3,f3=summary[0.3]
print(f"  s=0.1: learned {l1}/8, re-adapt both-pass {p1}/{n1} ({f1:.0%})")
print(f"  s=0.3: learned {l3}/8, re-adapt both-pass {p3}/{n3} ({f3:.0%})")
if l1>=l3 and f1>=f3:
    print(f"  => CLEAN WIN: s=0.1 learns >= as many seeds AND re-adapts at >= the rate. Adopt s=0.1 outright;")
    print(f"     the init fix solves from-scratch reliability with no adaptation cost.")
elif l1>l3 and f1<f3:
    print(f"  => TRADE: s=0.1 learns more seeds but re-adapts at a LOWER rate. Init-scale buys trainability at")
    print(f"     some adaptation cost; choose by the real-body need (reliable init vs online re-adapt).")
else:
    print(f"  => MIXED/NEGATIVE: see the per-seed table; s=0.1 is not a strict improvement on both axes.")
print(f"\nwall {time.time()-t0:.1f}s")
