"""Is the gain=0.5 drift a 200-tick BUDGET artifact (half speed can't reach in 200)? Re-run
keep-training at gain=0.5 with a 400-tick budget. If pre~post (no drift), the 'drift' was the
timeout, not the rule. Also report arrival-tick + timeout rate at 200 vs 400."""
import math, random, copy, time
B="./lnn_hebbian_walk_sim.py"
exec(open(B).read().split("t0=time.time()")[0])
def roll(ctrl,env,learn,maxt):
    o=env.reset(); ctrl.reset(); arr=False; t=0
    for _ in range(maxt):
        a,c=ctrl.step(o,0.1); o,r,dist,done=env.step(a); t+=1
        if learn: ctrl.learn(r,0.05,c)
        if dist<0.5: arr=True;break
        if dist>=0.5 and done and env.t>=200 and maxt<=200: break  # env's own 200 cap
    return arr,t
def ev(ctrl,n,seed0,gain,maxt):
    # bypass env's internal 200 cap by using a big-cap env for long budgets
    s=0; tt=0
    for e in range(n):
        env=Walk(10000+seed0+e,gain=gain); env.reset()
        o=env._obs(); ctrl.reset(); arr=False; t=0
        for _ in range(maxt):
            a,c=ctrl.step(o,0.1)
            d0=env._d(); ax=max(-1,min(1,a[0])); az=max(-1,min(1,a[1]))
            env.vx=0.85*env.vx+env.gain*ax*0.2; env.vz=0.85*env.vz+env.gain*az*0.2
            env.px+=env.vx*0.1; env.pz+=env.vz*0.1; dist=env._d(); o=env._obs(); t+=1
            if dist<0.5: arr=True;break
        s+=arr; tt+=t
    return s/n, tt/n
def keep(ctrl,n,seed0,gain,maxt):
    for e in range(n):
        env=Walk(seed0+e,gain=gain); env.reset(); o=env._obs(); ctrl.reset()
        for _ in range(maxt):
            a,c=ctrl.step(o,0.1)
            d0=env._d(); ax=max(-1,min(1,a[0])); az=max(-1,min(1,a[1]))
            env.vx=0.85*env.vx+env.gain*ax*0.2; env.vz=0.85*env.vz+env.gain*az*0.2
            env.px+=env.vx*0.1; env.pz+=env.vz*0.1; dist=env._d(); r=d0-dist; o=env._obs()
            ctrl.learn(r,0.05,c)
            if dist<0.5: break
t0=time.time()
L=CfC(seed=1)
for e in range(400): keep(L,1,e,1.0,200)
for maxt in (200,400):
    c=copy.deepcopy(L)
    pre,pt=ev(c,100,0,0.5,maxt); keep(c,200,0,0.5,maxt); post,_=ev(c,100,0,0.5,maxt)
    print(f"gain=0.5, budget {maxt}: pre {pre:.2f} (arr~{pt:.0f}t) -> post {post:.2f}  ({'DRIFT' if post<pre-0.08 else 'NO DRIFT'})")
print(f"wall {time.time()-t0:.1f}s")
