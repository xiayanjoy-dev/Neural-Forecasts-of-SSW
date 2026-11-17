"""Descriptive figures of frozen paired forecasts; no fitted or selected model."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE=Path(__file__).resolve().parent;OUT=HERE/'figures';OUT.mkdir(exist_ok=True)
scores=list(csv.DictReader((HERE/'analysis/scores_and_warnings.csv').open()))
bins=list(csv.DictReader((HERE/'analysis/reliability_bins.csv').open()))
ci=list(csv.DictReader((HERE/'analysis/paired_bootstrap.csv').open()))
case=list(csv.DictReader((HERE/'analysis/case2021_probabilities.csv').open()))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False})
arms=[('balanced_mean','Balanced','#0077BB'),('uniform_mean','Uniform','#EE7733'),
      ('balanced_prior_mean','Balanced + prior shift','#228833')]
def score(model,fold,d):return next(r for r in scores if r['model']==model and r['fold']==fold and int(r['horizon_days'])==d)
def export(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=350);fig.savefig(OUT/(name+'.pdf'));plt.close(fig)

fig,axs=plt.subplots(2,3,figsize=(8.0,6.3),layout='constrained')
for j,d in enumerate([5,10,15]):
    ax=axs[0,j];ax.plot([0,1],[0,1],color='.6',lw=.8,ls='--')
    for model,label,c in arms:
        rr=[r for r in bins if r['model']==model and r['fold']=='pooled' and int(r['horizon_days'])==d and int(r['count'])>0]
        ax.plot([float(r['mean_probability']) for r in rr],[float(r['observed_frequency']) for r in rr],'-o',ms=4,color=c,label=label)
    ax.set(xlabel='Forecast probability\n(bin mean)',ylabel='Observed frequency',xlim=(0,1),ylim=(0,1))
    ax.text(.02,.96,f'({chr(97+j)}) {d} days',transform=ax.transAxes,va='top')
fig.legend(*axs[0,0].get_legend_handles_labels(),fontsize=9,loc='outside upper center',ncols=3,frameon=False)
ax=axs[1,0]
for model,label,c in arms[:2]:
    for k,f in enumerate('ABC'):
        ax.plot(np.arange(3*k,3*k+3),[float(score(model,f,d)['bss']) for d in [5,10,15]],'-o',ms=4,color=c,label=label if k==0 else None)
ax.axhline(0,color='.5',lw=.8);ax.set(xticks=range(9),xticklabels=[f'{f}\n{d}' for f in 'ABC' for d in [5,10,15]],xlabel='Fold and window (days)',ylabel='Brier skill score')
ax.text(.02,.96,'(d)',transform=ax.transAxes,va='top');ax.legend(fontsize=8,frameon=False)
ax=axs[1,1]
for k,d in enumerate([5,10,15]):
    row=next(r for r in ci if int(r['horizon_days'])==d and r['metric']=='bss')
    delta=float(score('uniform_mean','pooled',d)['bss'])-float(score('balanced_mean','pooled',d)['bss'])
    ax.plot([float(row['lower95']),float(row['upper95'])],[k,k],color='#444444',lw=1.6)
    ax.scatter([delta],[k],color='#EE7733',s=30,zorder=3)
ax.axvline(0,color='.5',lw=.8,ls='--');ax.set(yticks=range(3),yticklabels=['5 days','10 days','15 days'],xlabel='Uniform − balanced Brier skill',ylim=(2.7,-.7));ax.text(.02,.96,'(e)',transform=ax.transAxes,va='top')
ax=axs[1,2]
for model,label,c in arms[:2]:
    ax.plot([int(score(model,'pooled',d)['false_alarm_episodes_total']) for d in [5,10,15]],
            [int(score(model,'pooled',d)['events_hit']) for d in [5,10,15]],'-o',color=c,label=label)
    for d in [5,10,15]:
        offset=(3,4) if model=='balanced_mean' or d==5 else (-19,7) if d==10 else (3,-13)
        ax.annotate(str(d),(int(score(model,'pooled',d)['false_alarm_episodes_total']),int(score(model,'pooled',d)['events_hit'])),xytext=offset,textcoords='offset points',fontsize=8)
ax.set(xlabel='False-alarm episodes',ylabel='SSW events detected (of 16)',ylim=(0,18));ax.text(.02,.96,'(f)',transform=ax.transAxes,va='top');ax.legend(fontsize=8,loc='lower right',frameon=False)
export(fig,'figureS5_controlled_sampling')

fig,axs=plt.subplots(1,3,figsize=(8.0,3.6),layout='constrained')
for ax,d,letter in zip(axs,[5,10,15],'abc'):
    ax.axvspan(.5,d+.1,color='.95',zorder=0);ax.axhline(.2,color='.4',ls='--',lw=.8)
    for model,label,c in arms:
        rr=sorted([r for r in case if r['model']==model and int(r['horizon_days'])==d],key=lambda r:int(r['lead_days']))
        ax.plot([int(r['lead_days']) for r in rr],[float(r['probability']) for r in rr],'-o',ms=3,color=c,label=label)
    ax.set(xlim=(15.5,.5),ylim=(0,1),xticks=[15,10,5,1],xlabel='Days before 5 January 2021',ylabel='Forecast probability')
    ax.text(.02,.96,f'({letter}) {d} days',transform=ax.transAxes,va='top')
fig.legend(*axs[0].get_legend_handles_labels(),fontsize=9,loc='outside upper center',ncols=3,frameon=False)
export(fig,'figureS6_2021_sampling')

# Preserve the existing workflow's four panels, updating only the pending pair.
fig,axs=plt.subplots(2,2,figsize=(7.35,5.6),layout='constrained')
for ax in axs.flat:ax.axis('off')
def box(ax,x,y,w,h,t):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.015',facecolor='#F5F7F8',edgecolor='#546E7A',lw=.8))
    ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=9)
ax=axs[0,0];ax.set_title('(a) Predictors and timing',loc='left',fontsize=10)
box(ax,.04,.55,.91,.28,'ERA5 temperature and wind\n20 levels · 71 latitudes · 72 longitudes');box(ax,.04,.08,.91,.28,'45-day history · 180 six-hourly states\nlast observation: 18 UTC before issue day');ax.annotate('',(.495,.38),(.495,.53),arrowprops={'arrowstyle':'->','color':'black'})
ax=axs[0,1];ax.set_title('(b) Probability model',loc='left',fontsize=10)
box(ax,.03,.58,.94,.29,'Spatial convolutions and attention\nthen four temporal Transformer layers');box(ax,.03,.08,.94,.29,'Three output probabilities\nonset within 5, 10 and 15 days');ax.annotate('',(.5,.39),(.5,.56),arrowprops={'arrowstyle':'->','color':'black'})
ax=axs[1,0];ax.set_title('(c) Training and test winters',loc='left',fontsize=10)
box(ax,.02,.51,.96,.38,'A: train 1979–1999 → test 2000–2004\nB: train 1979–2004 → test 2005–2010\nC: train 1979–2010 → test 2011–2021');ax.text(.5,.27,'Candidate issue dates: 16 Dec–1 Apr\nthen exclude 30 days after onset',ha='center',va='center',fontsize=9)
ax=axs[1,1];ax.set_title('(d) Comparison and verification',loc='left',fontsize=10)
box(ax,.03,.40,.94,.47,'Historical cumulative / hazard models\nControlled balanced / uniform pair\nsame cumulative model and loss\n18 fits: 2 samplers × 3 seeds × 3 folds');ax.text(.5,.19,'Same eligible origins · three seeds\nRanking · Brier skill · event warnings',ha='center',va='center',fontsize=9)
export(fig,'figure1_v19')
print('Exported three PNG/PDF pairs from fixed predictions.')
