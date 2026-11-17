"""Reviewer-prompted cumulative-output sampling pair, not model selection.

Both arms use the frozen cumulative model and raw-probability monotonicity
penalty, identical losses, initialization and update budget. Only the origin
sampling policy changes. Old baseline/V65 remain separately labelled anchors.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.environ.get('SSW_FROZEN_CODE', '/share/home/xiayan/era5_nudging'))
import numpy as np
import torch
from torch.utils.data import DataLoader as TorchLoader, Sampler, Dataset
import ssw_train_v50_1deg_rolling as base

HERE = Path(__file__).resolve().parent


class OriginSampler(Sampler):
    def __init__(self, dataset, arm, samples, seed):
        self.dataset, self.arm, self.samples = dataset, arm, samples
        self.rng = np.random.default_rng(seed + 67111)
        self.epoch = 0
        self.positive_by_event = {}
        for index in np.flatnonzero(dataset.labels == 0):
            origin = dataset.starts[index] + base.CFG['input_steps']
            event = dataset.ssw_idx[(dataset.ssw_idx > origin) &
                                   (dataset.ssw_idx <= origin + 20)]
            assert len(event) == 1
            self.positive_by_event.setdefault(int(event[0]), []).append(int(index))
        self.hard = np.flatnonzero((dataset.labels >= 1) & (dataset.labels <= 3))
        self.background = np.flatnonzero(dataset.labels == 4)
        self.onsets = sorted(self.positive_by_event)
        assert samples <= len(dataset) and samples % 5 == 0

    def __len__(self):
        return self.samples

    def __iter__(self):
        if self.arm == 'uniform':
            indices = self.rng.choice(len(self.dataset), self.samples, replace=False)
        else:
            indices = []
            for event in self.onsets:
                for _ in range(12):
                    indices.extend([self.rng.choice(self.positive_by_event[event]),
                                    *self.rng.choice(self.hard, 2, replace=True),
                                    *self.rng.choice(self.background, 2, replace=True)])
            indices = np.asarray(indices)
            self.rng.shuffle(indices)
        used = indices[:(len(indices) // base.CFG['batch_size']) * base.CFG['batch_size']]
        self.rows.append({'epoch': self.epoch + 1, 'draws': len(indices),
                          'used': len(used), 'unique_used': len(np.unique(used)),
                          'used_class_counts': np.bincount(self.dataset.labels[used], minlength=5).tolist()})
        self.epoch += 1
        return iter(indices.tolist())


def self_test():
    class Toy:
        labels = np.r_[np.zeros(5, int), np.ones(20, int), np.full(275, 4, int)]
        starts = np.arange(300) * 4
        ssw_idx = np.array([200])
        def __len__(self): return len(self.labels)
    toy = Toy()
    # Make five preceding starts map to one event under the frozen origin rule.
    toy.starts[:5] = np.arange(0, 20, 4)
    s = OriginSampler(toy, 'uniform', 60, 42); s.rows = []
    a = list(s); assert len(a) == len(set(a)) == 60
    s = OriginSampler(toy, 'balanced', 60, 42); s.rows = []
    a = list(s); assert sum(toy.labels[a] == 0) == 12
    assert sum(toy.labels[a] == 1) == 24
    assert sum(toy.labels[a] == 4) == 24
    assert s.rows[0]['used'] == 48
    assert hashlib.sha256(Path(base.pilot.base.__file__).read_bytes()).hexdigest() == base.pilot.EXPECTED_CORE_SHA256
    print('R67_SAMPLING_SELF_TEST_OK', flush=True)


def configure(arm, fold):
    item = base.FOLDS[fold]
    base.CFG.update(version=f'r67_cumulative_{arm}_fold{fold}',
                    train_end_year=item['train_end_year'], val_end_year=item['test_end_year'],
                    epochs=45, out_dir=str(HERE/'results'), fixed_train_epochs=21,
                    fixed_alarm_threshold=0.20)
    return item


def data_check():
    base.pilot.verify_core_source()
    files, steps, cum, mask = base.pilot._build_all_file_list()
    labels = base._labels_in_winter_space(cum, steps, mask)
    season, _, _ = base.pilot.base.assign_step_seasons(steps, files)
    result = {'files': len(files), 'steps': int(steps.sum()), 'folds': {}}
    for f, spec in base.FOLDS.items():
        n = sum((season[labels] > spec['train_end_year']) & (season[labels] <= spec['test_end_year']))
        assert n == spec['expected_test_events']
        result['folds'][f] = int(n)
    assert len(files) == 258 and steps.sum() == 31176
    (HERE/'data_check.json').write_text(json.dumps(result, indent=2))
    print('R67_DATA_CHECK_OK', flush=True)


def run(arm, fold, seed):
    configure(arm, fold)
    assert 'H100' in torch.cuda.get_device_name(0)
    free, total = torch.cuda.mem_get_info()
    assert free >= 70*1024**3, f'Allocated GPU is occupied: {free} free'
    print('R67_GPU_PREFLIGHT',os.environ.get('SLURM_JOB_NODELIST'),
          os.environ.get('CUDA_VISIBLE_DEVICES'), torch.cuda.get_device_name(0),free,flush=True)
    rows = []
    train_prior = None

    def loader(dataset, *args, **kwargs):
        if isinstance(dataset, base.PrimaryEventIndex):
            sampler = OriginSampler(dataset.dataset, arm, len(dataset), seed)
            sampler.rows = rows
            kwargs.pop('shuffle', None)
            return TorchLoader(dataset.dataset, *args, sampler=sampler, **kwargs)
        return TorchLoader(dataset, *args, **kwargs)

    def no_shift(dataset, device):
        nonlocal train_prior
        c = np.bincount(dataset.labels, minlength=5)
        train_prior = np.cumsum(c)[:3]/c.sum()
        return train_prior, train_prior.copy(), np.zeros(3), torch.zeros(3,device=device)

    original_evaluate = base._evaluate_outer_test
    def evaluate(model, loader_, shift, onsets, season, fold_, stem, device, amp):
        result = original_evaluate(model,loader_,shift,onsets,season,fold_,stem,device,amp)
        if arm == 'balanced':
            # Training-only diagnostic, not a fitted calibration or primary result.
            ds = loader_.dataset
            # Train pool class proportions recorded before the test is instantiated.
            tp = train_prior
            sampled = np.array([0.2, 0.2 + 0.4*class_fraction[1],
                                0.2 + 0.4*(class_fraction[1]+class_fraction[2])])
            logit = lambda p: np.log(np.clip(p,1e-7,1-1e-7)/(1-np.clip(p,1e-7,1-1e-7)))
            correction=torch.tensor(logit(tp)-logit(sampled),dtype=torch.float32,device=device)
            original_evaluate(model,loader_,correction,onsets,season,fold_,stem+'_prior_diagnostic',device,amp)
        return result

    # Capture only training hard-pool composition; never infer it from outer labels.
    class_fraction = None
    def remember_shift(dataset, device):
        nonlocal class_fraction
        c = np.bincount(dataset.labels,minlength=5).astype(float)
        class_fraction = c/max(1,c[1:4].sum())
        return no_shift(dataset,device)

    base.DataLoader = loader
    base._prior_shift = remember_shift
    base._evaluate_outer_test = evaluate
    base.configure_fold = lambda requested: configure(arm,requested)
    base.FIXED_TRAIN_EPOCHS = 21
    base.FIXED_ALARM_THRESHOLD = 0.20
    base.PROTOCOL = {'name':'reviewer_sampling_pair', 'arm':arm,'epochs':21,
                     'sampling_only_contrast':True,'prior_shift_during_training':'zero in both arms',
                     'monotonicity':'same penalty on raw cumulative probabilities',
                     'outer_test_selection':False,'seeds':[42,73,2026]}
    base.run_fold(fold,seed)
    (HERE/'results'/f'exposure_{arm}_{fold}_{seed}.json').write_text(json.dumps(rows,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.add_argument('--data-check',action='store_true')
    p.add_argument('--arm',choices=['balanced','uniform']);p.add_argument('--fold',choices=list(base.FOLDS));p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if a.self_test:self_test()
    elif a.data_check:data_check()
    else:run(a.arm,a.fold,a.seed)
