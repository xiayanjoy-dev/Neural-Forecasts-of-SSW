"""Frozen-prediction diagnostics for the reviewer-requested sampling contrast.

No model fitting, probability calibration, threshold or seed selection.
"""
from pathlib import Path
from datetime import datetime, timedelta
import csv, hashlib, json, sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INPUT = HERE.parent / '20261117_R67_completed_analysis'
OUT = HERE / 'analysis'
sys.path.insert(0, str(ROOT))
from evaluate_v65_natural_hazard import average_precision, roc_auc

HORIZONS = [5, 10, 15]
SEEDS = [42, 73, 2026]
KEYS = ['forecast_origin', 'start', 'winter', 'target_class']


def load(path):
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    order = np.argsort(a['forecast_origin'], kind='stable')
    return {k: v[order] if v.shape[:1] == order.shape else v for k, v in a.items()}


def dump(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def scores(p, y, climate):
    bs = float(np.mean((p-y)**2)); ref = float(np.mean((climate-y)**2))
    return dict(pr_auc=average_precision(y,p), roc_auc=roc_auc(y,p),
                brier=bs, bss=1-bs/ref, reference_brier=ref,
                mean_probability=float(p.mean()), prevalence=float(y.mean()))


def warnings(p, y, origin, winter, events, days, merge_days=5):
    alarm = p >= .20
    tp = int(np.sum(alarm & (y==1))); fp = int(np.sum(alarm & (y==0)))
    fn = int(np.sum(~alarm & (y==1))); tn = int(np.sum(~alarm & (y==0)))
    erows = []
    for e in events:
        if e['winter'] not in set(winter.tolist()): continue
        eligible = (winter==e['winter']) & (origin<e['onset_step']) & (e['onset_step']<=origin+days*4)
        hit = eligible & alarm
        erows.append(dict(winter=int(e['winter']), onset_step=int(e['onset_step']),
                          hit=bool(hit.any()), first_alarm_lead_days=float(np.max((e['onset_step']-origin[hit])/4)) if hit.any() else None))
    episodes = []
    for wy in np.unique(winter):
        dates = np.sort(origin[(winter==wy) & alarm & (y==0)])
        if not len(dates): continue
        blocks = np.split(dates, np.where(np.diff(dates)>merge_days*4)[0]+1)
        episodes.extend(dict(winter=int(wy), start_step=int(b[0]), end_step=int(b[-1]), alarm_origins=len(b)) for b in blocks)
    lead = [e['first_alarm_lead_days'] for e in erows if e['hit']]
    summary = dict(CSI=tp/max(1,tp+fp+fn), POD=tp/max(1,tp+fn), FAR=fp/max(1,tp+fp),
                   TP=tp, FP=fp, FN=fn, TN=tn, events_hit=sum(e['hit'] for e in erows),
                   n_events=len(erows), mean_first_alarm_lead_days=float(np.mean(lead)) if lead else None,
                   false_alarm_episodes_total=len(episodes))
    return summary, erows, episodes


def reliability(p, y):
    rows=[]; rel=res=0.; mean=float(y.mean()); assigned=0
    for k in range(10):
        mask=(p>=k/10) & ((p<((k+1)/10)) if k<9 else (p<=1))
        n=int(mask.sum()); assigned+=n
        pm=float(p[mask].mean()) if n else None
        ym=float(y[mask].mean()) if n else None
        rows.append(dict(bin=k, lower=k/10, upper=(k+1)/10, count=n,
                         mean_probability=pm, observed_frequency=ym))
        if n: rel+=n/len(y)*(pm-ym)**2; res+=n/len(y)*(ym-mean)**2
    assert assigned==len(y)
    unc=mean*(1-mean)
    return rows, dict(bin_reliability=rel, bin_resolution=res, uncertainty=unc,
                      bin_brier=rel-res+unc, raw_brier=float(np.mean((p-y)**2)))


def self_test():
    p=np.array([.1,.1,.7,1.]); y=np.array([0,1,0,1])
    bins, r=reliability(p,y)
    pp=np.array([b['mean_probability'] for b in bins for _ in range(b['count'])])
    yp=np.array([0,1,0,1])
    assert abs(np.mean((pp-yp)**2)-r['bin_brier'])<1e-12
    origin=np.array([0,4,24,28,80]); winter=np.ones(5,int)
    w,e,ep=warnings(np.array([.3,.3,.3,.3,.1]),np.array([0,0,0,1,0]),origin,winter,[dict(winter=1,onset_step=32)],5)
    assert len(ep)==1 and e[0]['hit'] and e[0]['first_alarm_lead_days']==2
    print('R67_FINAL_DIAGNOSTIC_SELF_TEST_OK')


def main():
    OUT.mkdir(exist_ok=True)
    ref=load(INPUT/'inputs/historical_reference_predictions.npz')
    pred=load(INPUT/'r67_aligned_predictions.npz')
    old=json.loads((INPUT/'r67_results.json').read_text())
    historical=json.loads((INPUT/'inputs/historical_reference_results.json').read_text())
    audit=json.loads((INPUT/'terminal_audit.json').read_text())
    for k in KEYS: np.testing.assert_array_equal(pred[k],ref[k])
    assert len(ref['winter'])==1918 and len(np.unique(ref['winter']))==22
    models={k:pred[k] for k in old['metrics']}
    provenance=[INPUT/'r67_aligned_predictions.npz',INPUT/'r67_results.json',INPUT/'inputs/historical_reference_predictions.npz']
    for seed in SEEDS:
        blocks=[]
        for fold in 'ABC':
            p=INPUT/'inputs'/f'r67_cumulative_balanced_fold{fold}_seed{seed}_prior_diagnostic_fixed_epoch21_predictions.npz'
            provenance.append(p);blocks.append(load(p))
        merged={k:np.concatenate([b[k] for b in blocks]) for k in KEYS+['horizon_probability']}
        order=np.argsort(merged['forecast_origin']);merged={k:v[order] for k,v in merged.items()}
        for k in KEYS: np.testing.assert_array_equal(merged[k],ref[k])
        models[f'balanced_prior_seed{seed}']=merged['horizon_probability']
    models['balanced_prior_mean']=np.mean([models[f'balanced_prior_seed{s}'] for s in SEEDS],axis=0)
    models['historical_corrected_mean']=ref['three_seed_equal_mean_probability']
    assert all(p.shape==(1918,3) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all() for p in models.values())
    dates=[]
    for wy in range(1979,2022):
        start=datetime(wy-1,11,1);stop=datetime(wy,5,1)
        dates.extend(start+timedelta(hours=6*i) for i in range((stop-start).days*4))
    assert len(dates)==31176
    metric_rows=[]; bin_rows=[]; decomposition=[]; event_rows=[]; episode_rows=[]; all_results={}
    for name,p in models.items():
        all_results[name]={}
        for fold in ['pooled','A','B','C']:
            mask=np.ones(1918,bool) if fold=='pooled' else ref['fold']==fold
            all_results[name][fold]={}
            for j,d in enumerate(HORIZONS):
                y=(ref['target_class'][mask]<=j).astype(int)
                m=scores(p[mask,j],y,ref['climatology_probability'][mask,j])
                ev=historical['scratch_three_seed'][str(d)]['event_metrics']['events']
                wm,er,ep=warnings(p[mask,j],y,ref['forecast_origin'][mask],ref['winter'][mask],ev,d)
                head=dict(model=name,fold=fold,horizon_days=d)
                metric_rows.append(dict(**head,n=int(mask.sum()),positive_origins=int(y.sum()),**m,**wm))
                bins,dec=reliability(p[mask,j],y)
                bin_rows.extend(dict(**head,**b) for b in bins);decomposition.append(dict(**head,**dec))
                for e in er:
                    event_rows.append(dict(**head,onset_date=dates[e['onset_step']].isoformat()[:10],**e))
                for e in ep:
                    episode_rows.append(dict(**head,start_date=dates[e['start_step']].isoformat()[:10],end_date=dates[e['end_step']].isoformat()[:10],**e))
                all_results[name][fold][str(d)]=dict(**m,**wm,**dec)
                if fold=='pooled' and name in old['metrics']:
                    previous=old['metrics'][name][str(d)]
                    for k in ['pr_auc','roc_auc','brier','bss']:assert abs(m[k]-previous[k])<1e-12
                    for k in wm:
                        if k not in previous['fixed_0.20']:continue
                        v=previous['fixed_0.20'][k]
                        assert (wm[k] is None and v is None) or (wm[k] is not None and abs(wm[k]-v)<1e-12),(name,d,k)
    macro=[]
    for name in models:
        for d in HORIZONS:
            rows=[r for r in metric_rows if r['model']==name and r['horizon_days']==d and r['fold']!='pooled']
            macro.append(dict(model=name,horizon_days=d,**{k:float(np.mean([r[k] for r in rows])) for k in ['pr_auc','roc_auc','brier','bss']}))
    # Reproduce the registered2000-draw pooled interval exactly; no refitting.
    rng=np.random.default_rng(671112);winters=np.unique(ref['winter']);indices={w:np.where(ref['winter']==w)[0] for w in winters};ci=[]
    for j,d in enumerate(HORIZONS):
        y=(ref['target_class']<=j).astype(int);rows=[]
        for _ in range(2000):
            ix=np.concatenate([indices[w] for w in rng.choice(winters,len(winters),replace=True)])
            if np.unique(y[ix]).size<2:continue
            u=models['uniform_mean'][ix,j];b=models['balanced_mean'][ix,j];yy=y[ix]
            ub=float(np.mean((u-yy)**2));bb=float(np.mean((b-yy)**2));cb=float(np.mean((ref['climatology_probability'][ix,j]-yy)**2))
            rows.append([average_precision(yy,u)-average_precision(yy,b),ub-bb,(bb-ub)/cb])
        for k,key in enumerate(['pr_auc','brier','bss']):
            low,high=np.quantile(np.asarray(rows)[:,k],[.025,.975])
            np.testing.assert_allclose([low,high],old['uniform_minus_balanced_paired_winter_95ci'][str(d)][key],atol=1e-12,rtol=0)
            ci.append(dict(horizon_days=d,metric=key,lower95=float(low),upper95=float(high),valid_draws=len(rows),requested_draws=2000,winter_clusters=22,seed=671112,predictions_fixed=True))
    case=[]
    onset=next(e['onset_step'] for e in historical['scratch_three_seed']['5']['event_metrics']['events'] if e['winter']==2021)
    for name,p in models.items():
        for lead in range(1,16):
            ix=np.where(ref['forecast_origin']==onset-lead*4)[0]
            if not len(ix):continue
            assert len(ix)==1
            for j,d in enumerate(HORIZONS):case.append(dict(model=name,onset_date=dates[onset].isoformat()[:10],lead_days=lead,horizon_days=d,issue_date=dates[int(ref['forecast_origin'][ix[0]])].isoformat()[:10],probability=float(p[ix[0],j]),within_target_window=lead<=d))
    exposure=[]
    for a in audit['artifact_checks']:
        counts=a['class_counts'];retained=sum(counts)
        for j,d in enumerate(HORIZONS):exposure.append(dict(arm=a['arm'],fold=a['fold'],seed=a['seed'],horizon_days=d,epochs=21,retained_total=retained,retained_positive=int(sum(counts[:j+1])),positive_fraction=float(sum(counts[:j+1])/retained),unique_training_examples_not_inferred=True))
    dump('scores_and_warnings.csv',metric_rows);dump('equal_fold_sensitivity.csv',macro)
    dump('reliability_bins.csv',bin_rows);dump('brier_decomposition.csv',decomposition)
    dump('event_inventory.csv',event_rows);dump('false_alarm_episodes.csv',episode_rows)
    dump('paired_bootstrap.csv',ci);dump('case2021_probabilities.csv',case);dump('retained_exposure.csv',exposure)
    report=dict(status='R67_FINAL_FIXED_PREDICTION_DIAGNOSTICS_OK',aligned_origins=1918,winters=22,events=16,
                scores=all_results,paired_intervals=ci,counts=dict(score_rows=len(metric_rows),equal_fold_rows=len(macro),reliability_rows=len(bin_rows),event_rows=len(event_rows),false_episode_rows=len(episode_rows),case2021_rows=len(case),exposure_rows=len(exposure)),
                bootstrap='2000 paired winter resamples with fixed predictions; RNG671112; conditional on two-class draws; no training refits',
                selection='No seed, threshold, checkpoint, feature, calibration or model selection. Prior correction is a training-only diagnostic, not fitted calibration.',
                positive_exposure_matched=False,prior_diagnostic='Stored inference using the training-pool log-odds shift, not a transformation fitted to the test period.',
                source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in provenance})
    (OUT/'diagnostics.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({'status':report['status'],'counts':report['counts'],'means':{name:all_results[name]['pooled'] for name in ['balanced_mean','uniform_mean','balanced_prior_mean']},'case2021':[r for r in event_rows if r['model'] in ['balanced_mean','uniform_mean','balanced_prior_mean'] and r['fold']=='pooled' and r['onset_date'].startswith('2021')]}))


if __name__=='__main__':
    if '--self-test' in sys.argv:self_test()
    else:main()
