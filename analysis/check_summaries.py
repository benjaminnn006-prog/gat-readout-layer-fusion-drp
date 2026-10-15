from pathlib import Path
import numpy as np,pandas as pd,json
r=Path(__file__).resolve().parents[1];run=pd.read_csv(r/'results/recomputed_runs.csv');diff=pd.read_csv(r/'results/paired_effects_individual.csv');drug=pd.read_csv(r/'results/figure3_drug_source.csv')
assert len(run)==80 and run.id.nunique()==80
for x in diff.itertuples():
 a=run[(run.regime==x.regime)&(run.feature=='compact')&(run['repeat']==x.repeat)].set_index('variant')
 assert abs(float(a.loc[x.variant,x.metric]-a.loc['baseline',x.metric])-x.difference)<1e-12
assert len(diff)==60 and len(drug)==412 and (drug.defined_repeats==5).all()
for rg,g in drug.groupby('regime'):
 assert len(g)==206 and (g.n_pairs_min==g.n_pairs_max).all()
for p in (r/'splits').rglob('test.csv'):
 assert list(pd.read_csv(p,nrows=0).columns)==['Cell','Drug']
print('PASS: 80 summary rows, 60 paired effects, 412 drug-setting counts and split-ID schemas. This checks provided summaries, not a new raw-prediction audit.')
