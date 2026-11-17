"""Afterok evaluator: fail closed on any missing/misaligned R67 artifact."""
from pathlib import Path
import json,csv
import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score,roc_auc_score
HERE=Path(__file__).resolve().parent;OUT=HERE/'results';SEEDS=[42,73,2026]
def load(path):
 with np.load(path) as z:a={k:z[k] for k in z.files}
 idx=np.argsort(a['forecast_origin'],kind='stable')
 return {k:v[idx] if v.shape[:1]==idx.shape else v for k,v in a.items()}
def metric(p,y,ref):
 return {'pr_auc':float(average_precision_score(y,p)),'roc_auc':float(roc_auc_score(y,p)),'brier':float(np.mean((p-y)**2)),'bss':float(1-np.mean((p-y)**2)/np.mean((ref-y)**2))}
def warnings(p,y,reference,events,days):
    alarm=p>=.20;origin=reference['forecast_origin'];winter=reference['winter']
    tp=int(np.sum(alarm&(y==1)));fp=int(np.sum(alarm&(y==0)));fn=int(np.sum(~alarm&(y==1)))
    event_rows=[]
    for e in events:
        ok=(winter==e['winter'])&(origin<e['onset_step'])&(e['onset_step']<=origin+days*4)&alarm
        event_rows.append({'winter':e['winter'],'onset_step':e['onset_step'],'hit':bool(ok.any()),'first_alarm_lead_days':float(np.max((e['onset_step']-origin[ok])/4)) if ok.any() else None})
    false_by_winter={}
    for w in np.unique(winter):
        dates=np.sort(origin[(winter==w)&alarm&(y==0)])
        false_by_winter[str(int(w))]=int(1+np.sum(np.diff(dates)>20)) if len(dates) else 0
    leads=[r['first_alarm_lead_days'] for r in event_rows if r['hit']]
    return {'CSI':tp/max(1,tp+fp+fn),'POD':tp/max(1,tp+fn),'FAR':fp/max(1,tp+fp),'TP':tp,'FP':fp,'FN':fn,'events_hit':sum(r['hit'] for r in event_rows),'n_events':len(event_rows),'events':event_rows,'mean_first_alarm_lead_days':float(np.mean(leads)) if leads else None,'false_alarm_episodes_total':sum(false_by_winter.values()),'false_alarm_episodes_by_winter':false_by_winter}
def main():
 reference=load(HERE/'historical_reference_predictions.npz');pred={};result={}
 historical=json.loads((HERE/'historical_reference_results.json').read_text())
 for arm in ['balanced','uniform']:
  for seed in SEEDS:
   blocks=[]
   for fold in 'ABC':
    stem=f'r67_cumulative_{arm}_fold{fold}_seed{seed}'
    for suffix in ['_results.json','_fixed_epoch21_predictions.npz']:
     assert (OUT/(stem+suffix)).is_file(),stem+suffix
    exposure=json.loads((OUT/f'exposure_{arm}_{fold}_{seed}.json').read_text());assert len(exposure)==21
    blocks.append(load(OUT/(stem+'_fixed_epoch21_predictions.npz')))
   merged={k:np.concatenate([b[k] for b in blocks]) for k in ['forecast_origin','start','winter','target_class','horizon_probability']}
   idx=np.argsort(merged['forecast_origin']);merged={k:v[idx] for k,v in merged.items()}
   for k in ['forecast_origin','start','winter','target_class']:np.testing.assert_array_equal(merged[k],reference[k])
   p=merged['horizon_probability'];assert p.shape==(1918,3) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
   pred[f'{arm}_seed{seed}']=p
  pred[f'{arm}_mean']=np.mean([pred[f'{arm}_seed{s}'] for s in SEEDS],axis=0)
 for name,p in pred.items():
  result[name]={str(d):{**metric(p[:,j],(reference['target_class']<=j).astype(int),reference['climatology_probability'][:,j]),'fixed_0.20':warnings(p[:,j],(reference['target_class']<=j).astype(int),reference,historical['scratch_three_seed'][str(d)]['event_metrics']['events'],d)} for j,d in enumerate([5,10,15])}
 rng=np.random.default_rng(671112);winters=np.unique(reference['winter']);indices={w:np.where(reference['winter']==w)[0] for w in winters};diff={}
 for j,d in enumerate([5,10,15]):
  values=[];y=(reference['target_class']<=j).astype(int);ref=reference['climatology_probability'][:,j]
  for _ in range(2000):
   ix=np.concatenate([indices[w] for w in rng.choice(winters,len(winters),replace=True)])
   if len(np.unique(y[ix]))<2:continue
   u=metric(pred['uniform_mean'][ix,j],y[ix],ref[ix]);b=metric(pred['balanced_mean'][ix,j],y[ix],ref[ix])
   values.append([u[k]-b[k] for k in ['pr_auc','brier','bss']])
  diff[str(d)]={k:np.quantile(np.array(values)[:,i],[.025,.975]).tolist() for i,k in enumerate(['pr_auc','brier','bss'])}
 np.savez_compressed(OUT/'r67_aligned_predictions.npz',**{k:reference[k] for k in ['forecast_origin','start','winter','target_class']},**pred)
 (OUT/'r67_results.json').write_text(json.dumps({'status':'R67_FIXED_FORMULATION_METRICS_COMPLETE','aligned_origins':1918,'metrics':result,'uniform_minus_balanced_paired_winter_95ci':diff,'note':'No model selection. Event-level/figure/manuscript finalization remains a separate checked step.'},indent=2))
 print('R67_FIXED_FORMULATION_METRICS_COMPLETE',flush=True)
if __name__=='__main__':main()
