from pathlib import Path
import csv,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,'/Users/xiayan/.codex/skills/atmospheric-plotting/scripts')
from climate_plot_style import style_cartesian_axes
rows=list(csv.DictReader((HERE/'analysis/scores_pooled_and_folds.csv').open()))
fig,axs=plt.subplots(2,2,figsize=(7.6,6.35),layout='constrained')
plt.rcParams.update({'pdf.fonttype':42,'font.size':10})
names={'event_balanced':'Neural mean','physical_logistic':'Physical logistic','persistence':'Persistence'}
colors={'event_balanced':'#0077BB','physical_logistic':'#228833','persistence':'#AA3377'}
for ax,metric,label,letter in [(axs[0,0],'pr_auc','PR-AUC minus prevalence','a'),(axs[0,1],'bss','Brier skill score','b')]:
 for model in names:
  vals=[]
  for f in 'ABC':
   for d in [5,10,15]:
    r=next(v for v in rows if v['model']==model and v['fold']==f and int(v['horizon_days'])==d)
    vals.append(float(r[metric])-(int(r['positive_origins'])/int(r['n']) if metric=='pr_auc' else 0))
  # Separate folds rather than connecting across different fitted models.
  for k in range(3):ax.plot(np.arange(3*k,3*k+3),vals[3*k:3*k+3],'-o',ms=4,color=colors[model],label=names[model] if k==0 else None)
 ax.set_xticks(range(9),[f'{f}\n{d}' for f in 'ABC' for d in [5,10,15]])
 ax.set(xlabel='Test fold and forecast window (days)',ylabel=label,title=f'({letter})')
 ax.axhline(0,color='.5',lw=.6);style_cartesian_axes(ax)
j=json.load(open(ROOT/'v55_p0_multiseed_scratch_results.json'))
z=np.load(ROOT/'v55_p0_multiseed_scratch_predictions.npz');sys.path.insert(0,str(ROOT))
from evaluate_v65_natural_hazard import average_precision
ax=axs[1,0]
for key,label,color in [('seed42_probability','Seed42','#0077BB'),('seed73_probability','Seed73','#EE7733'),('seed2026_probability','Seed2026','#AA3377'),('three_seed_equal_mean_probability','Mean','#000000')]:
 ax.plot([5,10,15],[average_precision((z['target_class']<=i).astype(int),z[key][:,i]) for i in range(3)],'-o',label=label,ms=4,color=color)
ax.set(xlabel='Forecast window (days)',ylabel='Pooled PR-AUC',title='(c)',xticks=[5,10,15]);style_cartesian_axes(ax)
fig.legend(*ax.get_legend_handles_labels(),loc='outside lower center',ncols=4,fontsize=8,frameon=False)
ax=axs[1,1];x=np.arange(3)
ax.bar(x-.17,[6,8,10],.34,label='Detected SSW events',color='#228833');ax.bar(x+.17,[20,26,23],.34,label='False-alarm episodes',color='#CC6677')
ax.set(xticks=x,xticklabels=['5 days','10 days','15 days'],ylabel='Count',title='(d)',ylim=(0,35));ax.legend(fontsize=8,frameon=False)
for i,(h,f) in enumerate(zip([6,8,10],[20,26,23])):ax.text(i-.17,h+.5,f'{h}/16',ha='center',fontsize=8);ax.text(i+.17,f+.5,str(f),ha='center',fontsize=8)
style_cartesian_axes(ax)
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='outside upper center',ncols=3,fontsize=9)
fig.savefig(HERE/'analysis/figure2_v17.png',dpi=350);fig.savefig(HERE/'analysis/figure2_v17.pdf');plt.close(fig)

fig,axs=plt.subplots(2,2,figsize=(7.35,5.6),layout='constrained')
for ax in axs.flat:ax.axis('off')
def box(ax,x,y,w,h,text):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.015',facecolor='#F5F7F8',edgecolor='#546E7A',lw=.8));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9)
ax=axs[0,0];ax.set_title('(a) Predictors and timing',loc='left',fontsize=10)
box(ax,.04,.55,.91,.28,'ERA5 temperature and wind\n20 levels · 71 latitudes · 72 longitudes')
box(ax,.04,.08,.91,.28,'45-day history · 180 six-hourly states\nlast observation: 18 UTC before issue day')
ax.annotate('',(.495,.38),(.495,.53),arrowprops={'arrowstyle':'->','color':'black'})
ax=axs[0,1];ax.set_title('(b) Probability model',loc='left',fontsize=10)
box(ax,.03,.58,.94,.29,'Spatial convolutions and attention\nthen four temporal Transformer layers')
box(ax,.03,.08,.94,.29,'Three output probabilities\nonset within 5, 10 and 15 days')
ax.annotate('',(.5,.39),(.5,.56),arrowprops={'arrowstyle':'->','color':'black'})
ax=axs[1,0];ax.set_title('(c) Training and test winters',loc='left',fontsize=10)
box(ax,.02,.51,.96,.38,'A: train 1979–1999 → test 2000–2004\nB: train 1979–2004 → test 2005–2010\nC: train 1979–2010 → test 2011–2021')
ax.text(.5,.27,'Candidate issue dates: 16 Dec–1 Apr\nthen exclude 30 days after onset',ha='center',va='center',fontsize=9)
ax=axs[1,1];ax.set_title('(d) Comparison and verification',loc='left',fontsize=10)
box(ax,.03,.40,.94,.47,'Original event-balanced configuration\nNatural-frequency hazard configuration\nNew fixed-formulation sampling pair\n(the new pair is still running)')
ax.text(.5,.19,'Same eligible origins · three seeds\nRanking · Brier skill · event warnings',ha='center',va='center',fontsize=9)
fig.savefig(HERE/'analysis/figure1_v17_WORKING.png',dpi=350);fig.savefig(HERE/'analysis/figure1_v17_WORKING.pdf');plt.close(fig)
