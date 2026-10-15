"""All 206 drugs in both settings; count paired directly with each mean PCC.
Contract: a mean/sample SD across 5 repeats, b ECDF of mean correlations,
c all 412 drug-setting mean PCC/count pairs. No sampling or inference tests.
Python backend, 6.05-inch width, Arial >=7pt, 600dpi raster and editable vector.
"""
from pathlib import Path
import json
import pandas as pd,numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
(ROOT/'figures').mkdir(exist_ok=True)
DATA=ROOT/'results'
mpl.rcParams.update({'font.family':'Arial','font.size':7,'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'lines.linewidth':.8,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
run=pd.read_csv(DATA/'recomputed_runs.csv');macro=pd.read_csv(DATA/'defined_PCC_by_run.csv');drug=pd.read_csv(DATA/'figure3_drug_source.csv')
fig,axs=plt.subplots(1,3,figsize=(6.05,2.55),layout='constrained')
colors=['#4C78A8','#4B9A91']
for i,regime in enumerate(['warm','cold_cell']):
    r=run[(run.regime==regime)&(run.feature=='compact')&(run.variant=='baseline')]
    m=macro[(macro.regime==regime)&(macro.feature=='compact')&(macro.variant=='baseline')]
    means=[r.PCC.mean(),m.macro_PCC.mean()];sd=[r.PCC.std(ddof=1),m.macro_PCC.std(ddof=1)]
    for j in range(2):
        axs[0].bar(i+[-.16,.16][j],means[j],width=.28,color=colors[j],label=['Overall','Within drug'][j] if i==0 else None)
        axs[0].errorbar(i+[-.16,.16][j],means[j],yerr=sd[j],fmt='none',ecolor='#34414A',capsize=2,elinewidth=.65,capthick=.65)
    d=drug[drug.regime==regime];assert len(d)==206 and (d.defined_repeats==5).all()
    axs[1].plot(np.sort(d.PCC_mean),np.arange(1,207)/206,color=colors[i],ls=['-','--'][i],label=['Pairs','Cells'][i],lw=.8)
    axs[2].scatter(d.n_pairs,d.PCC_mean,s=5,color=colors[i],marker=['o','^'][i],alpha=.55,linewidths=0,label=['Pairs','Cells'][i])
axs[0].set_xticks([0,1],['Pairs','Cells']);axs[0].set_ylim(0,1.18);axs[0].set_yticks([0,.25,.5,.75,1]);axs[0].set_ylabel('PCC');axs[0].legend(loc='upper right',handlelength=1)
axs[1].set_xlim(-.5,1);axs[1].set_xticks([-.5,0,.5,1]);axs[1].set_ylim(0,1.03);axs[1].set_xlabel('Mean within-drug PCC');axs[1].set_ylabel('Fraction of drugs');axs[1].legend(loc='upper left',handlelength=1.6)
axs[2].set_xlim(15,145);axs[2].set_xticks([25,75,125]);axs[2].set_ylim(-.5,1);axs[2].set_yticks([-.5,0,.5,1]);axs[2].set_xlabel('Test observations per drug');axs[2].set_ylabel('Mean within-drug PCC');axs[2].legend(loc='lower right',handlelength=1)
for ax,l,title in zip(axs,['a','b','c'],['Correlation levels','Drug distribution','Correlation and count']):
    ax.set_title(title,pad=9);ax.text(-.14,1.08,l,transform=ax.transAxes,fontweight='bold',fontsize=8)
fig.savefig(ROOT/'figures/figure2_drug_level.png',dpi=600,bbox_inches='tight',pad_inches=.05)
fig.savefig(ROOT/'figures/figure2_drug_level.svg',bbox_inches='tight',pad_inches=.05)
fig.savefig(ROOT/'figures/figure2_drug_level.pdf',bbox_inches='tight',pad_inches=.05)
plt.close(fig)
(ROOT/'metadata/figure_contract.json').write_text(json.dumps({'backend':'Python','n_drug_setting_points':len(drug),'all_drug_repeat_correlations':2060,'panel_a_error':'sample SD, not CI','panel_b':'mean of each drug\'s 5 correlations, not averaged/pooled predictions','panel_c':'each point pairs exact test count and mean correlation for the same drug; all 412 points retained','exclusions':'none for baseline','original_figure1':'unchanged approved user raster'},indent=2),encoding='utf-8')
