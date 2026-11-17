"""Fixed-prediction reviewer diagnostics; no forecast fitting or selection."""
from pathlib import Path
import sys,json,csv,re,hashlib,calendar
from datetime import datetime,timedelta
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_v65_natural_hazard import average_precision,roc_auc,decision_metrics,event_metrics

OUT=HERE/'analysis';OUT.mkdir(exist_ok=True)
def load(name):
    with np.load(ROOT/name) as z: a={k:z[k] for k in z.files}
    idx=np.lexsort((a['target_class'],a['winter'],a['start'],a['forecast_origin']))
    return {k:v[idx] if v.shape[:1]==idx.shape else v for k,v in a.items()}
def write(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def score(p,y,ref):
    bs=np.mean((p-y)**2);br=np.mean((ref-y)**2)
    return {'pr_auc':average_precision(y,p),'roc_auc':roc_auc(y,p),'brier':float(bs),'bss':float(1-bs/br),'reference_brier':float(br)}

def main():
    a=load('v55_p0_multiseed_scratch_predictions.npz');b=load('v52_p3_physical_baselines_predictions.npz');c=load('v65_natural_hazard_predictions.npz')
    for key in ['forecast_origin','start','winter','target_class','fold']:
        np.testing.assert_array_equal(a[key],b[key]);np.testing.assert_array_equal(a[key],c[key])
    models={'event_balanced':a['three_seed_equal_mean_probability'],'natural_hazard':c['three_seed_equal_mean_probability'],'physical_logistic':b['physical_logistic_probability'],'persistence':b['persistence_probability']}
    for x in models.values():assert x.shape==(1918,3) and np.isfinite(x).all() and ((x>=0)&(x<=1)).all()
    v65=json.loads((ROOT/'v65_natural_hazard_results.json').read_text())
    rows=[];warnings=[];cis=[]
    for f in ['pooled','A','B','C']:
        mask=np.ones(1918,bool) if f=='pooled' else a['fold']==f
        for j,d in enumerate([5,10,15]):
            y=(a['target_class'][mask]<=j).astype(int);ref=b['climatology_probability'][mask,j]
            for name,x in models.items():
                rows.append({'model':name,'fold':f,'horizon_days':d,'n':int(mask.sum()),'positive_origins':int(y.sum()),**score(x[mask,j],y,ref)})
    write('scores_pooled_and_folds.csv',rows)
    macro=[]
    for name in models:
        for d in [5,10,15]:
            r=[v for v in rows if v['model']==name and v['horizon_days']==d and v['fold']!='pooled']
            macro.append({'model':name,'horizon_days':d,**{k:float(np.mean([v[k] for v in r])) for k in ['pr_auc','brier','bss']}})
    write('equal_fold_sensitivity.csv',macro)
    for j,d in enumerate([5,10,15]):
        y=(a['target_class']<=j).astype(int)
        events=v65['scratch_three_seed'][str(d)]['event_metrics']['events']
        events=[{**e,'horizon':d} for e in events]
        for name,x in models.items():
            for t in [.05,.10,.15,.20,.25,.30,.40,.50]:
                em=event_metrics(x[:,j]>=t,y,a['forecast_origin'],a['winter'],events)
                warnings.append({'model':name,'horizon_days':d,'threshold':t,**decision_metrics(x[:,j]>=t,y),'events_hit':em['events_hit'],'n_events':em['n_events'],'false_episodes':em['false_alarm_episodes_total'],'mean_lead':em['mean_first_alarm_lead_days']})
    write('threshold_event_tradeoffs.csv',warnings)
    rng=np.random.default_rng(671111);winters=np.unique(a['winter']);indices={w:np.where(a['winter']==w)[0] for w in winters}
    for j,d in enumerate([5,10,15]):
        target=(a['target_class']<=j).astype(int)
        for name in ['natural_hazard','physical_logistic']:
            deltas=[]
            for _ in range(2000):
                idx=np.concatenate([indices[w] for w in rng.choice(winters,len(winters),replace=True)])
                if np.unique(target[idx]).size<2:continue
                x=score(models[name][idx,j],target[idx],b['climatology_probability'][idx,j]);z=score(models['event_balanced'][idx,j],target[idx],b['climatology_probability'][idx,j])
                deltas.append([x[k]-z[k] for k in ['pr_auc','brier','bss']])
            for k,metric in enumerate(['pr_auc','brier','bss']):
                v=np.asarray(deltas)[:,k];cis.append({'model_minus_event_balanced':name,'horizon_days':d,'metric':metric,'lower95':float(np.quantile(v,.025)),'upper95':float(np.quantile(v,.975)),'valid_draws':len(v),'requested_draws':2000})
    write('paired_winter_bootstrap.csv',cis)
    training=[];exposure=[]
    for fold,i in zip('ABC',range(3)):
        text=(ROOT/f'v65_nat_hazard_b_454395_{i}.out').read_text()
        m=re.search(r'train: (\d+) windows classes=\[([^]]+)\]',text);n=int(m[1]);counts=np.array([int(v) for v in m[2].split(',')])
        q=re.search(r'eligible=(\d+) matched_samples=(\d+) updates=(\d+) train_events=(\d+)',text)
        for j,d in enumerate([5,10,15]):training.append({'fold':fold,'horizon_days':d,'eligible_origins':n,'positive_origins':int(counts[:j+1].sum()),'prevalence':float(counts[:j+1].sum()/n),'eligible_primary_events':int(q[4]),'nominal_draws_per_epoch':int(q[2]),'updates_per_epoch':int(q[3]),'retained_origins_per_epoch':int(q[3])*16})
        for seed,k in zip([42,73,2026],[i,i+3,i+6]):
            for line in (ROOT/f'v65_nat_hazard_b_454395_{k}.out').read_text().splitlines():
                if line.startswith('{"epoch"'):
                    r=json.loads(line);exposure.append({'fold':fold,'seed':seed,'epoch':r['epoch']+1,'nominal_5d_positive_draws':r['sampled_time_class_counts'][0],'nominal_draws':sum(r['sampled_time_class_counts']),'note':'sampler indices BEFORE drop_last, not actual retained-batch exposure'})
    write('training_inventory_from_logs.csv',training);write('historical_nominal_exposure.csv',exposure)
    prior_rows=[];raw_probability=np.empty((1918,3));max_ap_change=0.
    for fold,i in zip('ABC',range(3)):
        text=(ROOT/f'v65_nat_hazard_b_454395_{i}.out').read_text()
        m=re.search(r'train: (\d+) windows classes=\[([^]]+)\]',text)
        cnt=np.array([int(v) for v in m[2].split(',')]);tp=np.cumsum(cnt)[:3]/cnt.sum()
        sampled=.2+.4*np.r_[0.,np.cumsum(cnt[1:3])/cnt[1:4].sum()]
        shift=np.log(tp/(1-tp))-np.log(sampled/(1-sampled))
        mask=a['fold']==fold
        pc=models['event_balanced'][mask]
        # Algebraic inverse of the documented shift, not checkpoint inference.
        raw_probability[mask]=pc/(pc+(1-pc)*np.exp(shift))
        for j,d in enumerate([5,10,15]):
            y=(a['target_class'][mask]<=j).astype(int)
            ap1=average_precision(y,pc[:,j]);ap0=average_precision(y,raw_probability[mask,j]);max_ap_change=max(max_ap_change,abs(ap1-ap0))
            prior_rows.append({'fold':fold,'horizon_days':d,'training_prevalence':float(tp[j]),'sampled_prevalence':float(sampled[j]),'logit_shift':float(shift[j]),'pr_auc_corrected':ap1,'pr_auc_algebraically_unshifted':ap0,'brier_corrected':float(np.mean((pc[:,j]-y)**2)),'brier_algebraically_unshifted':float(np.mean((raw_probability[mask,j]-y)**2))})
    assert max_ap_change<1e-12
    write('prior_shift_diagnostic.csv',prior_rows)
    np.savez_compressed(OUT/'algebraically_unshifted_probabilities.npz',forecast_origin=a['forecast_origin'],fold=a['fold'],probability=raw_probability)
    # Map the compressed winter axis to real calendar dates; verify total exactly.
    times=[]
    for wy in range(1979,2022):
        start=datetime(wy-1,11,1);stop=datetime(wy,5,1)
        times.extend(start+timedelta(hours=6*i) for i in range((stop-start).days*4))
    assert len(times)==31176
    label_source=json.loads((HERE/'labels_source_snapshot.json').read_text())
    onsets=[datetime.fromisoformat(v['date']) for v in label_source['events'] if v['date']<'2022']
    time_index={v:i for i,v in enumerate(times)}
    meta_rows=[]
    for fold,end in [('A',1999),('B',2004),('C',2010)]:
        kept=[];primary_events=set();recovery_excluded=0
        for wy in range(1979,end+1):
            first=datetime(wy-1,11,1)+timedelta(days=45)
            last=datetime(wy,5,1)-timedelta(days=30)
            day=first
            while day<=last:
                if any(t<=day<t+timedelta(days=30) for t in onsets):recovery_excluded+=1
                else:
                    future=[t for t in onsets if day<t<=day+timedelta(days=30)]
                    lag=(min(future)-day).days if future else 999
                    cl=next((i for i,v in enumerate([5,10,15,30]) if lag<=v),4)
                    kept.append((day,cl,wy))
                    if cl==0:primary_events.add(min(future).isoformat()[:10])
                day+=timedelta(days=1)
        c0=[v for v in training if v['fold']==fold and v['horizon_days']==5][0]
        assert len(kept)==c0['eligible_origins'],(fold,len(kept),c0)
        assert len(primary_events)==c0['eligible_primary_events']
        meta_rows.append({'fold':fold,'first_training_winter':1979,'last_training_winter':end,'training_winters':end-1978,'catalogue_events':sum(datetime(1978,11,1)<=t<datetime(end,5,1) for t in onsets),'eligible_primary_events':len(primary_events),'eligible_origins':len(kept),'first_training_origin':min(v[0] for v in kept).isoformat(),'last_training_origin':max(v[0] for v in kept).isoformat(),'recovery_excluded':recovery_excluded,'eligible_event_dates':';'.join(sorted(primary_events))})
    write('training_calendar_reconstructed_verified.csv',meta_rows)
    calendar_rows=[]
    for w in winters:
        ix=np.where(a['winter']==w)[0];dates=[times[int(v)] for v in a['forecast_origin'][ix]]
        calendar_rows.append({'winter':int(w),'first_origin':min(dates).isoformat(),'last_origin':max(dates).isoformat(),'origins':len(ix)})
    write('outer_calendar_support.csv',calendar_rows)
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,3,figsize=(9,3.2),layout='constrained')
    for ax,j in zip(axs,range(3)):
        for name,color in [('event_balanced','#0077BB'),('natural_hazard','#EE7733')]:
            r=[v for v in warnings if v['model']==name and v['horizon_days']==[5,10,15][j]]
            ax.plot([v['false_episodes'] for v in r],[v['events_hit'] for v in r],'-o',ms=3,label=name.replace('_',' '),color=color)
            v=next(v for v in r if v['threshold']==.2);ax.scatter(v['false_episodes'],v['events_hit'],s=65,marker='s',facecolors='none',edgecolors=color)
        ax.set(xlabel='False-alarm episodes',ylabel='SSW events detected',ylim=(-.5,16.5),title=f'({chr(97+j)}) {[5,10,15][j]} days')
    fig.legend(*axs[0].get_legend_handles_labels(),loc='outside upper center',ncols=2)
    fig.savefig(OUT/'threshold_event_tradeoffs.png',dpi=300);fig.savefig(OUT/'threshold_event_tradeoffs.pdf');plt.close(fig)
    report={'status':'CPU_DIAGNOSTICS_EXECUTED','aligned_origins':1918,'events':16,'bootstrap':'2000 paired winter-cluster draws; predictions held fixed; RNG671111; no refitting','new_training_completed':False,'prior_shift_check':{'foldwise_ap_max_difference':max_ap_change,'method':'algebraic inversion of documented correction, not checkpoint inference'},'training_exposure_caveat':'Historical logs record full sampler draws; actual retained exposure unavailable. R67 records retained draws explicitly.','tables':{'pooled_and_fold_rows':len(rows),'macro_rows':len(macro),'threshold_rows':len(warnings),'bootstrap_rows':len(cis),'training_inventory_rows':len(training),'nominal_exposure_rows':len(exposure),'calendar_rows':len(calendar_rows),'prior_shift_rows':len(prior_rows)},'source_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['v55_p0_multiseed_scratch_predictions.npz','v65_natural_hazard_predictions.npz','v52_p3_physical_baselines_predictions.npz']}}
    (OUT/'results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
