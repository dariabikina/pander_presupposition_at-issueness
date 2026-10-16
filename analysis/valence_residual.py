"""Per-condition deviation from each proposition's own valence-credence line.

One prompt per cell gives one valence value, so there is no per-condition Pander Score. Instead
each proposition's cells are fitted (logit credence on logit valence) and the residuals
show which framings get more credence than their measured valence predicts. Residuals sum to
zero within a fit, so these are positions relative to the other cells. With n=12-16 and skewed
residuals the bootstrap interval and the sign-flip p can disagree; go by the p.

    uv run python analysis/valence_residual.py
"""

import statistics

from scipy.stats import linregress

from pander_score.core.metrics import to_logit
from presup.design import CELL_ORDER as CELLS
from presup.paths import MODELS, pilot_csv, prompt_side_csv, read_csv
from presup.stats import bootstrap as boot
from presup.stats import permutation_p as perm_p

val = {r["sample_id"]: float(r["consensus"])
       for r in read_csv(prompt_side_csv())
       if r["attribute"]=="author_valence" and r["agree"]=="True" and r["consensus"]}

for model in MODELS:
    rows=[r for r in read_csv(pilot_csv(model)) if not r.get("error")]
    print("="*92); print(model)
    for fmt in ("conv","instr"):
        pts={}
        for r in rows:
            if r["format"]!=fmt or r["sample_id"] not in val: continue
            c=r.get("consensus_credence")
            if not c: continue
            pts.setdefault(r["item_id"],{}).setdefault(r["condition"],[]).append(float(c))
        resid={c:[] for c in CELLS}
        nfit=0
        for item, bycell in pts.items():
            xs=[to_logit(val[f"{item}__{fmt}__{c}"]) for c in bycell if f"{item}__{fmt}__{c}" in val]
            cs=[c for c in bycell if f"{item}__{fmt}__{c}" in val]
            ys=[to_logit(statistics.fmean(bycell[c])) for c in cs]
            raw=[val[f"{item}__{fmt}__{c}"] for c in cs]
            if len(xs)<4 or (max(raw)-min(raw))<0.05: continue
            fitres=linregress(xs,ys); nfit+=1
            for c,x,y in zip(cs,xs,ys):
                resid[c].append(y-(fitres.intercept+fitres.slope*x))
        print(f"  {fmt}   {nfit} propositions fitted (>=4 cells with measured valence)")
        print(f"    {'condition':<12}{'mean resid':>12}{'95% CI':>22}{'p':>9}{'n':>5}")
        for c in CELLS:
            v=resid[c]
            if len(v)<2: print(f"    {c:<12}{'-':>12}"); continue
            lo,hi=boot(v); m=statistics.fmean(v); p=perm_p(v)
            flag="  <--" if (lo>0 or hi<0) else ""
            print(f"    {c:<12}{m:+12.3f}{f'[{lo:+.3f},{hi:+.3f}]':>22}{p:>9.4f}{len(v):>5}{flag}")
        print()
