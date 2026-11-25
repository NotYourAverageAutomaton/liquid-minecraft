"""Train the s=0.1 WALK CfC (FLAT3D, jump off) reusing the EXACT validated rung-2a classes,
export weights JSON + a parity trace. Fidelity: exec only the DEFS of rung-2a (not its run)."""
import json, random, hashlib, copy
SRC="scripts/lnn_rung2a_mcphysics.py"
src=open(SRC).read()
assert hashlib.md5(src.encode()).hexdigest()=="6d374fb0d5a5dac3534d3865b6a17ab4", "rung-2a source md5 drift"
defs=src.split("t0=time.time()")[0]          # everything before the top-level experiment run
g={}; exec(defs,g)                            # runs imports + def statements only, no side effects
CfC,Walk3D,episode,ev = g["CfC"],g["Walk3D"],g["episode"],g["ev"]

def make_s01(seed):
    c=CfC(seed=seed)                          # init at s=0.3 (hardcoded), then scale draws to s=0.1
    k=0.1/0.3
    for M in (c.Wx,c.Wh,c.Wt,c.Wout):
        for i in range(len(M)):
            for j in range(len(M[i])): M[i][j]*=k
    c.wconf=[w*k for w in c.wconf]
    return c

# train a few seeds at s=0.1, keep the best-learned WALK controller
best=None
for seed in [1,2,3,4,5,6,7,8]:
    L=make_s01(seed)
    for e in range(500): episode(L, Walk3D(e,jumpon=False), True, 0.05)
    r,_=ev(L,100,0,jumpon=False)
    print(f"seed {seed}: reach-rate {r:.2f}")
    if best is None or r>best[0]: best=(r,seed,L)
rate,seed,L=best
print(f"SELECTED seed {seed}, reach-rate {rate:.2f}")

# export weights
W={"arch":{"obs":6,"hid":L.hid,"act":3,"dt":0.1},
   "Wx":L.Wx,"Wh":L.Wh,"Wt":L.Wt,"Wout":L.Wout,"wconf":L.wconf,
   "train":{"src_md5":"6d374fb0d5a5dac3534d3865b6a17ab4","init_scale":0.1,"seed":seed,"episodes":500,"lr":0.05,
            "skill":"FLAT3D-walk-jumpoff","reach_rate":rate}}
json.dump(W, open("./walkcfc_weights.json","w"))

# parity trace: run ONE eval episode, record (obs, act, conf) per step with h reset at start
L2=copy.deepcopy(L); L2.reset()
env=Walk3D(20000,jumpon=False); obs=env.reset()
trace=[]
for _ in range(120):
    act,conf=L2.step(obs,0.1)
    trace.append({"obs":obs,"act":act,"conf":conf})
    obs,rew,arr,done=env.step(act)
    if done: break
json.dump({"dt":0.1,"steps":trace}, open("./walkcfc_parity.json","w"))
print(f"exported weights + parity trace ({len(trace)} steps)")
