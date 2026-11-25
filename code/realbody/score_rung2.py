# Score rung-2 runs against the pre-registered bars (vault mc-rung2-controller-port-disposable-body.md + 11-05 amendments).
import json, sys, math, glob, os
D = sys.argv[1]
def recs(f): return [json.loads(l) for l in open(f) if l.startswith('{')]
res = {}
# Bar A
A = {}
for arm in ['cfc', 'cfc_ood', 'pp', 'rand']:
    f = f'{D}/A_{arm}.jsonl'
    if os.path.exists(f):
        t = [r for r in recs(f) if r['ev'] == 'trial']
        A[arm] = {'pass': sum(r['pass'] for r in t), 'n': len(t), 'alerts': [r for r in recs(f) if r['ev'] in ('ALERT', 'ERR', 'KICK')],
                  'stuck': sum(r['ev'] == 'stuck' for r in recs(f))}
res['A'] = A
if all(k in A for k in ('cfc', 'pp', 'rand')):
    c, p, r = A['cfc']['pass'], A['pp']['pass'], A['rand']['pass']
    if A['cfc']['n'] != 8 or p < 8: va = 'NO GRADE (feasibility ref pp < 8/8 or incomplete: route mis-design)'
    elif r > 1: va = 'NO GRADE (random >1/8: task too easy, redesign)'
    elif c >= 7: va = 'PASS'
    elif c <= r or c <= 4: va = 'FAIL'
    else: va = 'BETWEEN (5-6/8: report, scope claim)'
    res['A_verdict'] = va
# Bar D from cfc arm ticks
f = f'{D}/A_cfc.jsonl'
if os.path.exists(f):
    tk = [r for r in recs(f) if r['ev'] == 'tick' and r.get('actual') is not None]
    cx = sum(math.cos(r['actual'] - r['intended']) for r in tk) / len(tk)
    sy = sum(math.sin(r['actual'] - r['intended']) for r in tk) / len(tk)
    R = math.hypot(cx, sy)
    res['D'] = {'Rbar': round(R, 3), 'n': len(tk), 'verdict': 'PASS' if R >= 0.7 and len(tk) >= 2000 else ('FAIL' if R < 0.5 else ('CAUTION' if len(tk) >= 2000 else 'TOO FEW TICKS'))}
# Bars B + C
for f in sorted(glob.glob(f'{D}/C_*.jsonl')):
    rr = recs(f); dec = os.path.basename(f)[2:-6]
    tk = [r for r in rr if r['ev'] == 'tick']; us = sorted(r['us'] for r in tk)
    p99 = us[min(len(us) - 1, int(len(us) * 0.99))] if us else None
    miss = sum(r['gap'] > 1 for r in tk) / max(1, len(tk))
    reach = [r['k'] for r in rr if r['ev'] == 'reach']; w = {r['k']: r['wf'] for r in rr if r['ev'] == 'w'}
    first, last = sum(k <= 5000 for k in reach), sum(k > 15000 for k in reach)
    wf15, wf20 = w.get(15000), w.get(20000)
    plateau = (wf15 is not None and wf20 is not None and abs(wf20 - wf15) <= 0.03 * wf15)
    C = {'ticks': len(tk), 'p99_us': p99, 'missed_frac': round(miss, 4), 'reach_first5k': first, 'reach_last5k': last,
         'wf': {k: round(v, 2) for k, v in sorted(w.items()) if k % 2500 == 0}, 'stuck': sum(r['ev'] == 'stuck' for r in rr),
         'alerts': [r for r in rr if r['ev'] in ('ALERT', 'ERR', 'KICK')], 'done': any(r['ev'] == 'done' for r in rr)}
    C['C_verdict'] = ('INCOMPLETE' if not C['done'] else
                      'PASS' if (last >= first - 1 and plateau) else
                      'FAIL (' + ('competence' if last < first - 1 else '') + (' no-plateau' if not plateau else '') + ')')
    C['B_verdict'] = ('PASS' if (p99 is not None and p99 < 25000 and miss <= 0.01 and len(tk) >= 5000) else
                      'FAIL' if (p99 is not None and p99 >= 50000) else 'NOT PASS')
    res[f'C_{dec}'] = C
# Bar E
for fault in ['nan', 'stall']:
    f = f'{D}/E_{fault}.jsonl'
    if os.path.exists(f):
        rr = recs(f); al = [r for r in rr if r['ev'] == 'ALERT']
        want = {'nan': 'NAN', 'stall': 'STALL'}[fault]
        res[f'E_{fault}'] = 'PASS (alert fired: %s at k=%s)' % (al[0]['reason'], al[0].get('k')) if al and al[0]['reason'] == want else 'FAIL ' + json.dumps(al or [r for r in rr if r['ev'] == 'NO_ALERT'])
print(json.dumps(res, indent=1))
