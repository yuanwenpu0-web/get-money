"""对已保存的真实持仓观察点执行逐币正负对照。"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[2]
d=json.loads((ROOT/'docs/probe-historical-profit-exhaustion.json').read_text(encoding='utf-8'))
rng=np.random.default_rng(20261112)
out={}
for split in ('train','validation'):
    groups=[]
    for symbol in d['selected']:
        rows=sorted([r for r in d['records'] if r['symbol']==symbol and r['split']==split],key=lambda r:r['time'])
        if len(rows)<30: continue
        y=np.array([r['y'] for r in rows])
        if np.std(y)==0: continue
        groups.append((rows,y))
    for feature in ('rsi_pullback','exhaustion'):
        real=[]; oracle=[]; weak=[]; nulls=[[] for _ in range(199)]
        for rows,y in groups:
            x=np.array([r[feature] for r in rows]); ry=rankdata(y); rx=rankdata(x)
            if rx.std()==0: continue
            real.append(float(np.corrcoef(rx,ry)[0,1]))
            oracle.append(float(np.corrcoef(ry,ry)[0,1]))
            z=(y-y.mean())/y.std(); noisy=.25*z+np.sqrt(1-.25**2)*rng.normal(size=len(y))
            weak.append(float(np.corrcoef(rankdata(noisy),ry)[0,1]))
            for k in range(199):
                offset=int(rng.integers(1,len(y)))
                nulls[k].append(float(np.corrcoef(rx,np.roll(ry,offset))[0,1]))
        vals=np.array([np.mean(x) for x in nulls])
        observed=float(np.mean(real))
        out[f'{split}/{feature}']={'symbols':len(real),'observed_mean_ic':observed,
            'oracle_mean_ic':float(np.mean(oracle)),'weak_injected_mean_ic':float(np.mean(weak)),
            'negative_mean_ic':float(vals.mean()),'negative_95_percentiles':np.quantile(vals,[.025,.975]).tolist(),
            'negative_replicates':len(vals),'two_sided_shift_p':float((1+np.sum(np.abs(vals)>=abs(observed)))/(len(vals)+1))}
path=ROOT/'docs/probe-historical-profit-exhaustion-calibration.json'
if path.exists(): raise FileExistsError(path)
path.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
