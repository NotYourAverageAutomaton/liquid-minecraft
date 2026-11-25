"""Pick a  rung-1 SIM PROOF, pure-Python (no numpy): no-backprop CfC
controller trained by the rule author's 3-factor reward-modulated Hebbian rule
(src/nbp4090/rules.py HebbianUpdateRule) on a CPU-only 2D walk-to-target point-mass.
No MC/GPU/sandbox. Pre-registered H1 (learns vs frozen) + H2 (online re-adapt) +
shuffled-reward control. Honest: failure is a valid outcome."""
import math, random, time, copy

def gauss_mat(r,n,m,s): return [[r.gauss(0,s) for _ in range(m)] for _ in range(n)]
def matvec(W,x): return [sum(W[i][j]*x[j] for j in range(len(x))) for i in range(len(W))]
def tanh(v): return [math.tanh(z) for z in v]
def sig(z): return 1.0/(1.0+math.exp(-z))
def dot(a,b): return sum(a[i]*b[i] for i in range(len(a)))

# ---- the rule author's 3-factor local rule (faithful port): dW ~ reward*active*conf*outer(post,pre)/|gain|
def hebbian_dW(pre, post, reward, active=1.0, conf=1.0, eps=1e-6):
    gain = reward*max(0,min(1,active))*max(0,min(1,conf))   # signed via reward
    denom = max(abs(gain), eps)
    return [[ (post[i]*pre[j])*gain/denom for j in range(len(pre))] for i in range(len(post))]

class CfC:  # minimal closed-form continuous-time cell + readout (the Hebbian target)
    def __init__(self, obs=4, hid=16, act=2, seed=0):
        r=random.Random(seed); s=0.3
        self.Wx=gauss_mat(r,hid,obs,s); self.Wh=gauss_mat(r,hid,hid,s)
        self.Wt=gauss_mat(r,hid,obs,s); self.Wout=gauss_mat(r,act,hid,s)
        self.wconf=[r.gauss(0,s) for _ in range(hid)]; self.hid=hid; self.reset()
    def reset(self): self.h=[0.0]*self.hid
    def step(self,obs,dt):
        g=[sig(z) for z in matvec(self.Wt,obs)]
        wx=matvec(self.Wx,obs); wh=matvec(self.Wh,self.h)
        cand=tanh([wx[i]+wh[i] for i in range(self.hid)])
        self.h=[(1-g[i]*dt)*self.h[i]+(g[i]*dt)*cand[i] for i in range(self.hid)]
        act=tanh(matvec(self.Wout,self.h))
        conf=sig(dot(self.wconf,self.h))
        self.pre=list(self.h); self.post=list(act)
        return act,conf
    def learn(self,reward,lr,conf):
        dW=hebbian_dW(self.pre,self.post,reward,1.0,conf)
        for i in range(len(self.Wout)):
            for j in range(len(self.Wout[0])): self.Wout[i][j]+=lr*dW[i][j]

class Walk:  # 2D point-mass; reward = per-tick decrease in distance to target
    def __init__(self,seed,gain=1.0): self.r=random.Random(seed); self.gain=gain
    def reset(self):
        self.px,self.pz=self.r.uniform(-5,5),self.r.uniform(-5,5)
        self.vx=self.vz=0.0; self.tx,self.tz=self.r.uniform(-5,5),self.r.uniform(-5,5); self.t=0
        return self._obs()
    def _d(self): return math.hypot(self.tx-self.px,self.tz-self.pz)
    def _obs(self): return [(self.tx-self.px)/5,(self.tz-self.pz)/5,self.vx/5,self.vz/5]
    def step(self,act,dt=0.1):
        d0=self._d(); ax=max(-1,min(1,act[0])); az=max(-1,min(1,act[1]))
        self.vx=0.85*self.vx+self.gain*ax*2*dt; self.vz=0.85*self.vz+self.gain*az*2*dt
        self.px+=self.vx*dt; self.pz+=self.vz*dt; self.t+=1; d1=self._d()
        return self._obs(),(d0-d1),d1,(d1<0.5 or self.t>=200)

shuffle_rng=random.Random(99)
def episode(ctrl,env,learn=False,lr=0.0,shuffle=False):
    obs=env.reset(); ctrl.reset(); arrived=False
    for _ in range(200):
        act,conf=ctrl.step(obs,0.1); obs,rew,dist,done=env.step(act)
        if learn: ctrl.learn(shuffle_rng.gauss(0,0.1) if shuffle else rew, lr, conf)
        if dist<0.5: arrived=True; break
        if done: break
    return arrived,env.t
def train(ctrl,n,lr,seed0,shuffle=False):
    for e in range(n): episode(ctrl,Walk(seed0+e),True,lr,shuffle)
def ev(ctrl,n,seed0,gain=1.0):
    s=t=0
    for e in range(n):
        a,tt=episode(ctrl,Walk(10000+seed0+e,gain=gain)); s+=a; t+=tt
    return s/n, t/n
def ci(p,n): se=math.sqrt(max(p*(1-p),1e-9)/n); return (round(p-1.96*se,2),round(p+1.96*se,2))

t0=time.time()
base=CfC(seed=1); fb,_=ev(base,100,0)
L=CfC(seed=1); train(L,400,0.05,0); ls,lt=ev(L,100,0)
S=CfC(seed=1); train(S,400,0.05,0,shuffle=True); ss,_=ev(S,100,0)
pre,_=ev(L,100,0,gain=0.5)
fz=copy.deepcopy(L); fza,_=ev(fz,100,0,gain=0.5)
A=copy.deepcopy(L)
for e in range(200): episode(A,Walk(e,gain=0.5),True,0.05)
pa,_=ev(A,100,0,gain=0.5)
c=CfC(seed=2); o=Walk(0).reset(); n=5000; s=time.time()
for _ in range(n): c.step(o,0.1)
us=(time.time()-s)/n*1e6
print("=== PRE-REGISTERED RESULTS (pick a rung-1 sim, pure-python) ===")
print(f"H1 frozen-baseline : {fb:.2f} CI{ci(fb,100)}")
print(f"H1 Hebbian-learner : {ls:.2f} CI{ci(ls,100)}  avg_ticks {lt:.0f}")
print(f"   shuffled-reward : {ss:.2f}  (control: must stay ~baseline)")
print(f"H2 learned, gain=0.5, no-learn : {pre:.2f}")
print(f"   frozen copy (plasticity OFF): {fza:.2f}")
print(f"   adapt copy  (plasticity ON) : {pa:.2f}")
print(f"CPU {us:.1f} us/tick (MC 50000us @20TPS -> {50000/us:.0f}x headroom)  wall {time.time()-t0:.1f}s")
