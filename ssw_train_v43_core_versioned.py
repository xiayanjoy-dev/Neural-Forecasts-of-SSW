"""Direct-precursor primary 5-day SSW forecast with 10/15-day auxiliaries.

The first head predicts whether a major SSW begins in the next five days.
Independent 10/15-day cumulative heads provide auxiliary supervision.
Training samples 5-day positives by event at 1:4 against a 50/50 mixture of
near-event and background negatives. A level-aware precursor branch explicitly
preserves U(60N, 10 hPa), polar-cap temperature, 100-hPa eddy heat flux, and
upper/lower-stratospheric EP-divergence time series. A continuous auxiliary
head predicts the minimum U(60N, 10 hPa) in the next five days so that every
sample supplies vortex-evolution supervision despite the small event count.
The precursor trend summary also reaches the forecast heads directly, and a
second continuous head predicts the five-day vortex-wind change.
"""
import copy
import numpy as np
from netCDF4 import Dataset as NcFile
import os, sys, glob, time, warnings, json
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

CFG = {
    'version': 'v43',
    'data_dir': '/share/home/xiayan/era5_nudging/merged_full/',
    'ep_dir': '/share/home/xiayan/era5_nudging/ep_flux_6h/',
    'out_dir': '/share/home/xiayan/ssw_phase1',

    # Spatial domain
    'lat_range': (20, 90),
    'n_lev': 20,

    # Temporal windows
    'input_days': 45,
    'lead_days': 5,
    'stride': 4,  # one forecast initialization per day

    # Data selection
    'winter_months': [11, 12, 1, 2, 3, 4],
    'ssw_window_days': 60,
    'max_winter_samples': 20000,

    # EP flux
    'ep_input': True,
    'ep_lat_range': (40, 75),
    'ep_lev_range': (500, 10),

    # Model
    'd_model': 64,
    'nhead': 8,
    'n_layers': 4,
    'dim_feedforward': 256,
    'dropout': 0.20,
    'drop_path': 0.05,
    'lead_times': [5, 10, 15],
    'n_time_classes': 5,
    'ep_enc_dim': 16,
    'precursor_dim': 9,
    'precursor_summary_dim': 18,
    'precursor_summary_enc_dim': 24,
    'precursor_summary_lag_days': [5],
    'u_channel': 1,
    't_channel': 0,
    'v_channel': 2,
    'idx_10hpa': 14,
    'idx_30hpa': 12,
    'idx_50hpa': 11,
    'idx_100hpa': 9,
    'idx_60n_downsampled': 8,
    'polar_cap_last_index': 8,
    'heat_flux_lat_slice': [4, 13],
    'ep_upper_level_start': 6,

    # Training
    'epochs': 45,
    'batch_size': 16,
    'lr': 1.5e-4,
    'weight_decay': 0.005,
    'negative_per_positive': 4,
    'hard_negatives_per_positive': 2,
    'positive_draws_per_event': 12,
    'deterministic_positive_leads': False,
    'horizon_loss_weights': [1.0, 0.3, 0.2],
    'monotonic_loss_weight': 0.05,
    'vortex_level_aux_loss_weight': 0.10,
    'vortex_change_aux_loss_weight': 0.10,
    'selection_bss_floor': -0.01,
    'ema_decay': 0.995,
    'grad_clip_norm': 1.0,
    'use_bf16': True,
    'noise_std': 0.02,
    'jitter_max': 0,

    # Label definition
    'label_window': 5,
    'max_horizon_days': 30,
    'post_onset_exclusion_days': 30,
    'alarm_merge_days': 5,

    # Chronological split (years)
    'train_end_year': 2017,
    'val_end_year': 2025,

    'n_spatial_queries': 4,
    'seed': 42,
    'early_stop_patience': 10,
    'early_stop_min_epochs': 25,

    # One warmup+cosine schedule, with no restarts or plateau scheduler.
    'warmup_epochs': 5,
    'lr_eta_min': 1e-6,
}
CFG['input_steps'] = CFG['input_days'] * 4
CFG['lead_steps'] = CFG['lead_days'] * 4
CFG['window_len'] = CFG['input_steps']
CFG['label_steps'] = CFG['label_window'] * 4
CFG['max_horizon_steps'] = CFG['max_horizon_days'] * 4
CFG['num_workers'] = 0
CFG['n_input_channels'] = 4     # v36_clean: t, u, v, u_dz

torch.manual_seed(CFG['seed'])
np.random.seed(CFG['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def set_seed(seed):
    CFG['seed'] = int(seed)
    torch.manual_seed(CFG['seed'])
    np.random.seed(CFG['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CFG['seed'])


# ====================================================================
# Data loading helpers
# ====================================================================

def _get_month_from_file(fname):
    base = os.path.basename(fname)
    parts = base.replace('.nc', '').split('_')
    for p in parts:
        if len(p) == 6 and p.isdigit():
            return int(p[4:6])
    for i, p in enumerate(parts):
        if p == 'ep' and i >= 2:
            return int(parts[i-1])
    return None

def _is_winter_month(fname):
    m = _get_month_from_file(fname)
    return m in CFG['winter_months'] if m else True


def _year_month_from_file(fname):
    base = os.path.basename(fname).replace('.nc', '')
    parts = base.split('_')
    for part in parts:
        if len(part) == 6 and part.isdigit():
            return int(part[:4]), int(part[4:6])
    for i, part in enumerate(parts):
        if (part == 'ep' and i >= 2 and
                len(parts[i-2]) == 4 and len(parts[i-1]) == 2 and
                parts[i-2].isdigit() and parts[i-1].isdigit()):
            return int(parts[i-2]), int(parts[i-1])
    raise ValueError('Cannot parse YYYYMM from %s' % fname)


def _winter_year_from_file(fname):
    year, month = _year_month_from_file(fname)
    return year + 1 if month in (11, 12) else year


def _training_stop_step(steps_pf, files, train_end_year):
    is_train = np.array(
        [_winter_year_from_file(f) <= train_end_year for f in files],
        dtype=bool)
    if np.any(is_train[1:] > is_train[:-1]):
        raise ValueError('Training files are not a contiguous chronological prefix')
    return int(np.asarray(steps_pf)[is_train].sum())


def _build_all_file_list():
    all_files = sorted(glob.glob(CFG['data_dir'] + 'era5_uvt_full_*.nc'))
    all_steps, valid_all = [], []
    for f in all_files:
        try:
            with NcFile(f, 'r') as ds:
                all_steps.append(ds.dimensions['valid_time'].size)
                valid_all.append(f)
        except Exception:
            print('  SKIP corrupted: %s' % os.path.basename(f), flush=True)
    all_cum = np.cumsum([0] + all_steps)
    winter_mask = np.array([_is_winter_month(f) for f in valid_all], dtype=bool)
    return valid_all, np.array(all_steps), all_cum, winter_mask

def _full2winter_step(full_step, all_cum, all_steps, winter_mask):
    fi = np.searchsorted(all_cum, full_step, side='right') - 1
    if fi < 0 or fi >= len(winter_mask) or not winter_mask[fi]:
        return -1
    offset_in_file = full_step - all_cum[fi]
    winter_before = int((all_steps * winter_mask)[:fi].sum())
    return int(winter_before + offset_in_file)


# ====================================================================
# Data loading (v36_clean: adds U_dz as 4th channel)
# ====================================================================

def preload_uvt():
    """Load winter-only UVT data + compute U_dz (vertical wind shear).
    Returns data (4-channel: t, u, v, u_dz), qbo, steps_pf, files, norm_stats.
    """
    files = sorted(glob.glob(CFG['data_dir'] + 'era5_uvt_full_*.nc'))
    files = [f for f in files if _is_winter_month(f)]
    print('Winter files: %d' % len(files), flush=True)

    with NcFile(files[0], 'r') as ds:
        lat = ds.variables['latitude'][:]
        plev = ds.variables['pressure_level'][:]
    lat_idx = np.where((lat >= CFG['lat_range'][0]) & (lat <= CFG['lat_range'][1]))[0]
    eq_idx = np.where(np.abs(lat) <= 5)[0]
    n_lev = CFG['n_lev']
    lat_small = lat_idx[::2]       # downsample lat x2
    n_lat_small = len(lat_small)
    n_lon_half = 72                # downsample lon x2

    # Find level indices for U_dz = U50 - U200
    idx_200 = int(np.abs(plev - 200).argmin())
    idx_50  = int(np.abs(plev - 50).argmin())
    print('  U_dz: U(%d hPa) - U(%d hPa), idx %d - %d' % (plev[idx_50], plev[idx_200], idx_50, idx_200), flush=True)
    CFG['u_dz_lev_hi'] = idx_50
    CFG['u_dz_lev_lo'] = idx_200

    steps_pf, valid_files = [], []
    for _f in files:
        try:
            with NcFile(_f, 'r') as _ds:
                steps_pf.append(_ds.dimensions['valid_time'].size)
                valid_files.append(_f)
        except Exception:
            print('  SKIP corrupted: %s' % os.path.basename(_f), flush=True)
    files = valid_files
    steps_pf = np.array(steps_pf, dtype=np.int32)
    cum = np.cumsum(np.concatenate([[0], steps_pf]))
    total = int(cum[-1])

    # v36_clean: 4 channels (t, u, v, u_dz)
    n_channels = CFG['n_input_channels']
    data = np.empty((total, n_channels, n_lev, n_lat_small, n_lon_half), dtype=np.float32)
    qbo_u = np.empty((total, n_lev), dtype=np.float32)
    t0, off = time.time(), 0
    for fi, f in enumerate(files):
        with NcFile(f, 'r') as ds:
            ts = steps_pf[fi]
            # t, u, v
            data[off:off+ts, 0] = ds.variables['t'][:, :n_lev, lat_small, ::2]
            u_3d = ds.variables['u'][:, :n_lev, lat_small, ::2]
            data[off:off+ts, 1] = u_3d
            data[off:off+ts, 2] = ds.variables['v'][:, :n_lev, lat_small, ::2]
            # U_dz: U50 - U200, then repeat across all levels as 4th channel
            u_50 = ds.variables['u'][:, idx_50, lat_small, ::2]   # (T, lat, lon)
            u_200 = ds.variables['u'][:, idx_200, lat_small, ::2]  # (T, lat, lon)
            u_dz_2d = u_50 - u_200                                 # (T, lat, lon)
            data[off:off+ts, 3] = u_dz_2d[:, np.newaxis, :, :]     # broadcast across levels
            # QBO from equatorial mean
            u_full = ds.variables['u'][:, :n_lev, eq_idx, :]
            qbo_u[off:off+ts] = u_full.mean(axis=(2, 3))
        off += ts
        if (fi+1) % 50 == 0:
            print('  %d/%d (%d steps, %ds)' % (fi+1, len(files), off,
                  time.time()-t0), flush=True)

    train_stop = _training_stop_step(steps_pf, files, CFG['train_end_year'])
    print('  Normalization fit: first %d/%d winter steps (winter <= %d)' % (
        train_stop, total, CFG['train_end_year']), flush=True)

    # Fit normalization on training winters only. Missing values become zero
    # only after normalization, so zero represents the training mean.
    norm_stats = {}
    for vi, nm in enumerate(['t', 'u', 'v', 'u_dz']):
        v = data[:, vi]
        n_missing = int(np.isnan(v).sum())
        mu = float(np.nanmean(v[:train_stop]))
        sd = float(np.nanstd(v[:train_stop]))
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError('Invalid %s standard deviation: %s' % (nm, sd))
        norm_stats[nm + '_mean'] = mu
        norm_stats[nm + '_std'] = sd
        print('  %s norm: mean=%.4f std=%.4f missing=%d' % (
            nm, mu, sd, n_missing), flush=True)
    qbo_mean = np.nanmean(qbo_u[:train_stop], axis=0)
    qbo_std = np.nanstd(qbo_u[:train_stop], axis=0)
    norm_stats['qbo_mean'] = qbo_mean.tolist()
    norm_stats['qbo_std'] = qbo_std.tolist()

    # Normalize in-place and mean-fill missing values in normalized space.
    for vi in range(n_channels):
        data[:, vi] = (data[:, vi] - norm_stats[['t_mean','u_mean','v_mean','u_dz_mean'][vi]]) \
                      / norm_stats[['t_std','u_std','v_std','u_dz_std'][vi]]
    qbo_u[:] = (qbo_u - qbo_mean) / (qbo_std + 1e-8)
    np.nan_to_num(data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(qbo_u, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    return data, qbo_u, steps_pf, files, norm_stats


def preload_ep_aligned(steps_pf, files):
    """Load EP divergence, align by YYYYMM, and normalize on training winters."""
    ep_files = sorted(glob.glob(CFG['ep_dir'] + '*_ep_flux_6h.nc'))
    ep_files = [f for f in ep_files if _is_winter_month(f)]
    print('EP winter files: %d' % len(ep_files), flush=True)
    if not ep_files:
        raise FileNotFoundError('No EP files matching *_ep_flux_6h.nc')

    with NcFile(ep_files[0], 'r') as ds:
        ep_lat = ds.variables['lat'][:]
        ep_level = ds.variables['level'][:]

    ep_lat_idx = np.where(
        (ep_lat >= CFG['ep_lat_range'][0]) & (ep_lat <= CFG['ep_lat_range'][1])
    )[0]
    n_ep_lat = len(ep_lat_idx)
    ep_lev_idx = np.where(
        (ep_level >= CFG['ep_lev_range'][1]) & (ep_level <= CFG['ep_lev_range'][0])
    )[0]
    n_ep_lev = len(ep_lev_idx)
    CFG['n_ep_lat'] = n_ep_lat
    CFG['n_ep_lev'] = n_ep_lev
    print('  EP lat: %d, lev: %d (%.0f-%.0f hPa)' % (
        n_ep_lat, n_ep_lev, ep_level[ep_lev_idx[-1]], ep_level[ep_lev_idx[0]]), flush=True)

    ep_dict = {}
    for f in ep_files:
        base = os.path.basename(f)
        parts = base.replace('.nc', '').split('_')
        yyyymm = None
        if 'epdiv' in base:
            yyyymm = parts[-1] if len(parts[-1]) == 6 and parts[-1].isdigit() else None
        else:
            for i, p in enumerate(parts):
                if p == 'ep' and i >= 2:
                    yyyymm = parts[i-2] + parts[i-1]
                    break
        if yyyymm and len(yyyymm) == 6:
            ep_dict[yyyymm] = f

    total_uvt = int(np.sum(steps_pf))
    ep_data = np.full(
        (total_uvt, n_ep_lat, n_ep_lev), np.nan, dtype=np.float32)
    off, matched = 0, 0
    t0 = time.time()
    for fi, f in enumerate(files):
        year, month = _year_month_from_file(f)
        yyyymm = '%04d%02d' % (year, month)
        n_steps = int(steps_pf[fi])
        if yyyymm not in ep_dict:
            raise FileNotFoundError('Missing EP month %s for %s' % (yyyymm, f))
        with NcFile(ep_dict[yyyymm], 'r') as ds:
            epdiv = ds.variables['epdiv'][:]
            if epdiv.shape[0] != n_steps:
                raise ValueError(
                    'EP/UVT time mismatch for %s: %d != %d' %
                    (yyyymm, epdiv.shape[0], n_steps))
            ep_data[off:off+n_steps] = (
                epdiv[:, ep_lev_idx, :][:, :, ep_lat_idx].transpose(0, 2, 1))
        matched += 1
        off += n_steps
        if (fi + 1) % 100 == 0:
            print('  EP aligned %d/%d (%ds)' % (
                fi + 1, len(files), time.time() - t0), flush=True)
    if matched != len(files):
        raise ValueError('EP alignment incomplete: %d/%d' % (matched, len(files)))

    train_stop = _training_stop_step(
        steps_pf, files, CFG['train_end_year'])
    ep_mu = np.nanmean(ep_data[:train_stop], axis=(0, 1), keepdims=True)
    ep_sd = np.nanstd(ep_data[:train_stop], axis=(0, 1), keepdims=True)
    if not np.all(np.isfinite(ep_sd)) or np.any(ep_sd <= 0):
        raise ValueError('Invalid EP normalization standard deviation')
    ep_data = (ep_data - ep_mu) / ep_sd
    np.nan_to_num(ep_data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    print('  EP norm fitted on training winters: mean [%.2f, %.2f], '
          'std [%.2f, %.2f]' % (
              ep_mu.min(), ep_mu.max(), ep_sd.min(), ep_sd.max()), flush=True)

    ep_cum = np.cumsum(np.concatenate([[0], steps_pf]))
    return (ep_data, steps_pf, ep_cum,
            ep_mu.reshape(-1).tolist(), ep_sd.reshape(-1).tolist())


def load_ssw_labels():
    path = os.path.join(CFG['out_dir'], 'ssw_onsets_final.npy')
    ssw = np.load(path)
    print('SSW onsets: %d' % len(ssw), flush=True)
    return ssw


# ====================================================================
# Winter season grouping
# ====================================================================

def assign_step_seasons(steps_pf, files):
    cum = np.concatenate([[0], np.asarray(steps_pf, dtype=np.int32)]).cumsum()
    total = cum[-1]
    season_of_step = np.full(total, -1, dtype=np.int32)
    year_of_step = np.full(total, -1, dtype=np.int32)
    for fi, f in enumerate(files):
        base = os.path.basename(f)
        parts = base.replace('.nc', '').split('_')
        yyyymm = None
        for p in parts:
            if len(p) == 6 and p.isdigit():
                yyyymm = p; break
        year, month = int(yyyymm[:4]), int(yyyymm[4:6])
        if month in [11, 12]:
            winter_year = year + 1
        elif month in [1, 2, 3, 4]:
            winter_year = year
        else:
            winter_year = -1
        season_of_step[cum[fi]:cum[fi+1]] = winter_year
        year_of_step[cum[fi]:cum[fi+1]] = year
    return season_of_step, year_of_step, cum


# ====================================================================
# Dataset (v36_clean — 4-channel input with U_dz)
# ====================================================================

def time_class_for_start(gs, ssw_idx):
    """Classify the first strictly future onset into four bins or no onset."""
    forecast_origin = gs + CFG['input_steps']
    future = ssw_idx[
        (ssw_idx > forecast_origin) &
        (ssw_idx <= forecast_origin + CFG['max_horizon_steps'])]
    if len(future) == 0:
        return 4
    delta = int(future.min() - forecast_origin)
    if delta <= 5 * 4:
        return 0
    if delta <= 10 * 4:
        return 1
    if delta <= 15 * 4:
        return 2
    return 3


class SSWDatasetV43(Dataset):
    """At-risk inputs with a five-class time-to-first-onset target."""
    def __init__(self, data, qbo, steps_pf, files,
                 ep, ep_steps_pf, ep_cum,
                 ssw_idx, season_of_step, year_of_step, cum_uvt,
                 split='train', train_end_year=2017, val_end_year=2025):
        self.data = data
        self.qbo = qbo
        self.ep = ep
        self.steps_pf = steps_pf
        self.cum_uvt = cum_uvt
        self.ep_cum = ep_cum
        self.season_of_step = season_of_step
        self.year_of_step = year_of_step
        self.ssw_idx = ssw_idx
        self.training = split == 'train'

        # Input and verification interval must remain within one winter.
        all_starts = []
        required_span = CFG['input_steps'] + CFG['max_horizon_steps']
        total = int(cum_uvt[-1])
        for gs in range(0, total - required_span + 1, CFG['stride']):
            seasons = season_of_step[gs:gs + required_span]
            if seasons[0] < 0 or not np.all(seasons == seasons[0]):
                continue
            all_starts.append(gs)
        all_starts = np.asarray(all_starts, dtype=np.int64)
        print('  Total winter windows: %d' % len(all_starts), flush=True)

        # Split by winter year: Nov/Dec stay with Jan-Apr of the same winter.
        def _year_of_start(gs):
            return int(season_of_step[gs])

        if split == 'train':
            valid = [s for s in all_starts if _year_of_start(s) <= train_end_year]
        elif split == 'val':
            valid = [s for s in all_starts
                     if train_end_year < _year_of_start(s) <= val_end_year]
        else:
            valid = [s for s in all_starts if _year_of_start(s) > val_end_year]

        class_starts = [[] for _ in range(CFG['n_time_classes'])]
        excluded_recovery = 0
        recovery_steps = CFG['post_onset_exclusion_days'] * 4
        for s in valid:
            forecast_origin = s + CFG['input_steps']
            in_recovery = np.any(
                (ssw_idx <= forecast_origin) &
                (forecast_origin < ssw_idx + recovery_steps))
            if in_recovery:
                excluded_recovery += 1
                continue
            target_class = time_class_for_start(s, ssw_idx)
            class_starts[target_class].append(s)

        background_class = CFG['n_time_classes'] - 1
        max_background = min(
            CFG['max_winter_samples'],
            len(class_starts[background_class]))
        np.random.shuffle(class_starts[background_class])
        class_starts[background_class] = (
            class_starts[background_class][:max_background])

        self.starts = []
        labels = []
        for class_id, starts in enumerate(class_starts):
            self.starts.extend(starts)
            labels.extend([class_id] * len(starts))
        self.labels = np.asarray(labels, dtype=np.int64)
        idx = np.arange(len(self.starts))
        np.random.shuffle(idx)
        self.starts = [self.starts[i] for i in idx]
        self.labels = self.labels[idx]
        self.winters = np.asarray(
            [season_of_step[s] for s in self.starts], dtype=np.int32)

        counts = np.bincount(
            self.labels, minlength=CFG['n_time_classes'])
        print('  %s: %d windows classes=%s, recovery excluded=%d' % (
            split, len(self.starts), counts.tolist(),
            excluded_recovery), flush=True)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        gs = self.starts[idx]
        ge = gs + CFG['input_steps']

        x = np.array(self.data[gs:ge])
        q = np.array(self.qbo[gs:ge])
        ep_in = np.array(self.ep[gs:ge])
        # Strictly future daily means: (origin, origin + 5 days], matching
        # the inclusive classification horizon at exactly day 5.
        future_u60_10 = self.data[
            ge + 1:ge + CFG['lead_steps'] + 1,
            CFG['u_channel'], CFG['idx_10hpa'],
            CFG['idx_60n_downsampled'], :].mean(axis=-1)
        future_daily_u60_10 = future_u60_10.reshape(
            CFG['lead_days'], 4).mean(axis=1)
        vortex_target = float(np.min(future_daily_u60_10))
        current_daily_u60_10 = float(self.data[
            ge - 4:ge, CFG['u_channel'], CFG['idx_10hpa'],
            CFG['idx_60n_downsampled'], :].mean())
        vortex_change_target = vortex_target - current_daily_u60_10

        # Augmentation is training-only; validation must be deterministic.
        if self.training and np.random.random() < 0.5:
            x += np.random.randn(*x.shape).astype(np.float32) * CFG['noise_std']
        if (self.training and CFG['jitter_max'] > 0 and
                np.random.random() < 0.3):
            j = np.random.randint(-CFG['jitter_max'], CFG['jitter_max'] + 1)
            if j != 0:
                for arr in [x, q, ep_in]:
                    arr[...] = np.roll(arr, j, axis=0)
                    if j > 0: arr[:j] = 0
                    else: arr[j:] = 0

        return {
            'x': torch.from_numpy(np.ascontiguousarray(x)).float(),
            'q': torch.from_numpy(np.ascontiguousarray(q)).float(),
            'ep_in': torch.from_numpy(np.ascontiguousarray(ep_in)).float(),
            'y': torch.tensor(self.labels[idx]).long(),
            'vortex_target': torch.tensor(vortex_target).float(),
            'vortex_change_target': torch.tensor(
                vortex_change_target).float(),
            'start': torch.tensor(gs).long(),
            'winter': torch.tensor(self.season_of_step[gs]).long(),
        }


# ====================================================================
# Model (v36_clean — 4-channel input)
# ====================================================================

class SSWDropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep_prob)
        x = x / keep_prob
        return x * mask


class Conv3DEncoderV36(nn.Module):
    """v36_clean: 4 input channels (t, u, v, u_dz)."""
    def __init__(self, d_model=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(4, 16, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.GroupNorm(4, 16), nn.ReLU(),
            nn.Conv3d(16, 32, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.GroupNorm(8, 32), nn.ReLU(),
            nn.Conv3d(32, 48, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.GroupNorm(12, 48), nn.ReLU(),
            nn.Conv3d(48, 64, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.GroupNorm(16, 64), nn.ReLU(),
        )
        self.proj = nn.Linear(64, d_model)

    def forward(self, x):
        B, T, C, Z, H, W = x.shape
        f = self.conv(x.reshape(B * T, C, Z, H, W))
        f = f.mean(dim=2)
        return f


class SpatialCrossAttention(nn.Module):
    def __init__(self, d=64, nhead=4, n_queries=4):
        super().__init__()
        self.n_queries = n_queries
        self.query = nn.Parameter(torch.randn(1, n_queries, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, nhead, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, spatial_feat):
        B, d, H, W = spatial_feat.shape
        tokens = spatial_feat.reshape(B, d, H * W).transpose(1, 2)
        q = self.query.expand(B, -1, -1)
        out = self.attn(q, self.norm(tokens), self.norm(tokens))[0]
        return out.mean(dim=1)


class EPEncoderV36(nn.Module):
    """2D EP flux encoder."""
    def __init__(self, n_ep_lat, n_ep_lev, d_ep=16, d_model=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3), padding=(1, 1)),
            nn.GroupNorm(2, 8), nn.GELU(),
            nn.Conv2d(8, d_ep, (3, 3), padding=(1, 1)),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(d_ep, d_model)

    def forward(self, ep_in):
        B, T, H, W = ep_in.shape
        ep = ep_in.reshape(B * T, 1, H, W)
        f = self.net(ep)
        f = self.pool(f).reshape(B, T, CFG['ep_enc_dim'])
        return self.proj(f)


class PosEncV36(nn.Module):
    def __init__(self, d, drop=0.1, maxlen=250):
        super().__init__()
        self.drop = nn.Dropout(p=drop)
        pe = torch.zeros(maxlen, d)
        pos = torch.arange(0, maxlen, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class TransformerLayerV36(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout, drop_path):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model), nn.Dropout(dropout))
        self.norm2 = nn.LayerNorm(d_model)
        self.drop_path = SSWDropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path(self.self_attn(self.norm1(x), self.norm1(x),
                                               self.norm1(x))[0])
        x = x + self.drop_path(self.ff(self.norm2(x)))
        return x


class SSW3DNetV43(nn.Module):
    def __init__(self):
        super().__init__()
        d = CFG['d_model']
        self.enc3d = Conv3DEncoderV36(d)
        self.spatial_attn = SpatialCrossAttention(
            d, nhead=4, n_queries=CFG.get('n_spatial_queries', 4))
        ep_lat = CFG.get('n_ep_lat', 19)
        ep_lev = CFG.get('n_ep_lev', 11)
        self.enc_ep = EPEncoderV36(ep_lat, ep_lev, CFG['ep_enc_dim'], d)
        self.precursor_enc = nn.Sequential(
            nn.LayerNorm(CFG['precursor_dim']),
            nn.Linear(CFG['precursor_dim'], 32), nn.GELU(),
            nn.Linear(32, d),
        )
        self.f3d_norm = nn.LayerNorm(d)
        self.ep_norm = nn.LayerNorm(d)
        self.precursor_norm = nn.LayerNorm(d)
        summary_dim = CFG['precursor_dim'] * (
            1 + len(CFG['precursor_summary_lag_days']))
        if summary_dim != CFG['precursor_summary_dim']:
            raise ValueError(
                'precursor_summary_dim does not match configured lags')
        self.precursor_summary_enc = nn.Sequential(
            nn.LayerNorm(summary_dim),
            nn.Linear(
                CFG['precursor_summary_dim'],
                CFG['precursor_summary_enc_dim']),
            nn.GELU(),
        )
        self.pe = PosEncV36(d, CFG['dropout'], CFG['window_len'] + 1)
        t_layers = []
        for i in range(CFG['n_layers']):
            dp = CFG['drop_path'] * (i / max(1, CFG['n_layers'] - 1))
            t_layers.append(TransformerLayerV36(
                d, CFG['nhead'], CFG['dim_feedforward'], CFG['dropout'], dp))
        self.tf = nn.Sequential(*t_layers)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.qbo_enc = nn.Sequential(
            nn.LayerNorm(CFG['n_lev']),
            nn.Linear(CFG['n_lev'], 16), nn.GELU(),
            nn.Linear(16, 12),
        )
        latent_dim = d + 12 + CFG['precursor_summary_enc_dim']
        self.ln = nn.LayerNorm(latent_dim)
        self.shared_fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.GELU(), nn.Dropout(CFG['dropout']),
        )
        self.time_head = nn.Sequential(
            nn.Linear(latent_dim, 24), nn.GELU(),
            nn.Dropout(CFG['dropout']),
            nn.Linear(24, len(CFG['lead_times'])),
        )
        self.vortex_head = nn.Sequential(
            nn.Linear(latent_dim, 24), nn.GELU(),
            nn.Linear(24, 1),
        )
        self.vortex_change_head = nn.Sequential(
            nn.Linear(latent_dim, 24), nn.GELU(),
            nn.Linear(24, 1),
        )

    def _precursor_series(self, x, ep_in):
        u10_60 = x[
            :, :, CFG['u_channel'], CFG['idx_10hpa'],
            CFG['idx_60n_downsampled'], :].mean(dim=-1)
        polar_end = CFG['polar_cap_last_index'] + 1
        u10_cap = x[
            :, :, CFG['u_channel'], CFG['idx_10hpa'],
            :polar_end, :].mean(dim=(-1, -2))
        t10_cap = x[
            :, :, CFG['t_channel'], CFG['idx_10hpa'],
            :polar_end, :].mean(dim=(-1, -2))
        t30_cap = x[
            :, :, CFG['t_channel'], CFG['idx_30hpa'],
            :polar_end, :].mean(dim=(-1, -2))
        t50_cap = x[
            :, :, CFG['t_channel'], CFG['idx_50hpa'],
            :polar_end, :].mean(dim=(-1, -2))

        lat_start, lat_stop = CFG['heat_flux_lat_slice']
        t100 = x[
            :, :, CFG['t_channel'], CFG['idx_100hpa'],
            lat_start:lat_stop, :]
        v100 = x[
            :, :, CFG['v_channel'], CFG['idx_100hpa'],
            lat_start:lat_stop, :]
        t_eddy = t100 - t100.mean(dim=-1, keepdim=True)
        v_eddy = v100 - v100.mean(dim=-1, keepdim=True)
        heat_flux100 = (t_eddy * v_eddy).mean(dim=(-1, -2))

        ep_upper_start = CFG['ep_upper_level_start']
        ep_lower = ep_in[:, :, :, :ep_upper_start].mean(dim=(-1, -2))
        ep_upper = ep_in[:, :, :, ep_upper_start:].mean(dim=(-1, -2))
        return torch.stack([
            u10_60, u10_cap, t10_cap, t30_cap, t50_cap, heat_flux100,
            ep_lower, ep_upper,
            u10_60 - u10_60.mean(dim=1, keepdim=True),
        ], dim=-1)

    def forward(self, x, q, ep_in=None, return_feats=False,
                return_aux=False):
        B, T = x.shape[0], x.shape[1]
        spatial = self.enc3d(x)
        f3d = self.spatial_attn(spatial).reshape(B, T, -1)
        f_ep = self.enc_ep(ep_in) if ep_in is not None else 0
        precursor_series = self._precursor_series(x, ep_in)
        precursor = self.precursor_enc(precursor_series)
        f_fused = (
            self.f3d_norm(f3d) +
            self.ep_norm(f_ep) +
            self.precursor_norm(precursor)) / np.sqrt(3.0)
        cls = self.cls.expand(B, -1, -1)
        ft = self.pe(torch.cat([cls, f_fused], dim=1))
        ft = self.tf(ft)
        latent = ft[:, 0]
        q_profile = q[:, :CFG['input_steps']].mean(dim=1)
        q_feat = self.qbo_enc(q_profile)
        recent_precursor = precursor_series[:, -4:].mean(dim=1)
        precursor_summary_parts = [recent_precursor]
        for lag_days in CFG['precursor_summary_lag_days']:
            lag_start = -(lag_days + 1) * 4
            lag_stop = -lag_days * 4
            lagged_precursor = precursor_series[
                :, lag_start:lag_stop].mean(dim=1)
            precursor_summary_parts.append(
                recent_precursor - lagged_precursor)
        precursor_summary = torch.cat(precursor_summary_parts, dim=1)
        precursor_summary_feat = self.precursor_summary_enc(
            precursor_summary)
        feat = torch.cat([
            latent, q_feat, precursor_summary_feat], dim=1)
        feat = self.shared_fc(self.ln(feat))
        logits = self.time_head(feat)
        vortex_prediction = self.vortex_head(feat).squeeze(-1)
        vortex_change_prediction = self.vortex_change_head(
            feat).squeeze(-1)
        if return_aux:
            return logits, vortex_prediction, vortex_change_prediction
        if return_feats:
            return logits, latent, q_feat
        return logits


def probability_metrics(prob, targets, threshold):
    prob = np.asarray(prob)
    targets = np.asarray(targets)
    p = prob >= threshold
    tp = int(((p == 1) & (targets == 1)).sum())
    fp = int(((p == 1) & (targets == 0)).sum())
    fn = int(((p == 0) & (targets == 1)).sum())
    tn = int(((p == 0) & (targets == 0)).sum())
    pod = tp / (tp + fn + 1e-8)
    far = fp / (tp + fp + 1e-8)
    csi = tp / (tp + fp + fn + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    return {
        'POD': pod, 'FAR': far, 'CSI': csi, 'ACC': acc,
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        'threshold': float(threshold),
    }


def best_probability_metrics(prob, targets):
    """Exact development-set CSI scan, including all-positive/all-negative."""
    prob = np.asarray(prob, dtype=np.float64)
    candidates = np.unique(prob)
    candidates = np.concatenate((
        [np.nextafter(candidates.min(), -np.inf)],
        candidates,
        [np.nextafter(candidates.max(), np.inf)]))
    best = probability_metrics(prob, targets, candidates[0])
    for threshold in candidates[1:]:
        current = probability_metrics(prob, targets, threshold)
        if current['CSI'] > best['CSI']:
            best = current
    return best


def cross_validated_threshold_predictions(prob, targets, winters):
    """Choose a CSI threshold on other winters and apply it to the held winter."""
    prob = np.asarray(prob, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    winters = np.asarray(winters, dtype=np.int32)
    predicted = np.zeros(len(targets), dtype=bool)
    thresholds = {}
    unique_winters = np.unique(winters)
    for winter in unique_winters:
        held = winters == winter
        development = ~held
        if (development.sum() == 0 or
                len(np.unique(targets[development])) < 2):
            threshold = 0.5
        else:
            threshold = best_probability_metrics(
                prob[development], targets[development])['threshold']
        predicted[held] = prob[held] >= threshold
        thresholds[str(int(winter))] = float(threshold)
    return predicted, thresholds


def event_diagnostics(predicted, targets, starts, winters, ssw_idx,
                      season_of_step, horizon_days, thresholds):
    """Event diagnostics using leave-one-winter-out threshold decisions."""
    predicted = np.asarray(predicted, dtype=bool)
    targets = np.asarray(targets, dtype=np.int64)
    starts = np.asarray(starts, dtype=np.int64)
    winters = np.asarray(winters, dtype=np.int32)
    origins = starts + CFG['input_steps']
    horizon_steps = int(horizon_days * 4)

    events = []
    for onset in ssw_idx:
        winter = int(season_of_step[onset])
        if not (CFG['train_end_year'] < winter <= CFG['val_end_year']):
            continue
        eligible = (
            (origins < onset) &
            (onset <= origins + horizon_steps))
        alarms = eligible & predicted
        lead_days = (
            float(np.max((onset - origins[alarms]) / 4.0))
            if np.any(alarms) else None)
        events.append({
            'onset_step': int(onset),
            'winter': winter,
            'hit': bool(np.any(alarms)),
            'first_alarm_lead_days': lead_days,
            'n_eligible_initializations': int(eligible.sum()),
        })

    false_by_winter = {}
    merge_steps = CFG['alarm_merge_days'] * 4
    for winter in sorted(np.unique(winters).tolist()):
        false_origins = np.sort(origins[
            (winters == winter) & (targets == 0) & predicted])
        episodes = 0
        previous = None
        for origin in false_origins:
            if previous is None or origin - previous > merge_steps:
                episodes += 1
            previous = origin
        false_by_winter[str(int(winter))] = int(episodes)

    hits = sum(event['hit'] for event in events)
    leads = [
        event['first_alarm_lead_days'] for event in events
        if event['first_alarm_lead_days'] is not None]
    return {
        'horizon_days': int(horizon_days),
        'thresholds_by_held_winter': thresholds,
        'n_events': len(events),
        'events_hit': int(hits),
        'event_pod': float(hits / max(1, len(events))),
        'mean_first_alarm_lead_days': (
            float(np.mean(leads)) if leads else None),
        'false_alarm_episodes_by_winter': false_by_winter,
        'false_alarm_episodes_total': int(sum(false_by_winter.values())),
        'events': events,
        'note': (
            'Each winter uses a threshold selected on all other validation '
            'winters. Neural-model checkpoint selection still uses the full '
            '2018-2025 validation block.'),
    }


# ====================================================================
# Training (v43 — direct precursor trend and vortex-change supervision)
# ====================================================================

def horizon_probability_metrics(horizon_probability, target_class):
    """Return threshold-free and best-CSI diagnostics for each horizon."""
    results = {}
    for class_id, horizon in enumerate(CFG['lead_times']):
        prob = horizon_probability[:, class_id]
        target = (target_class <= class_id).astype(np.int64)
        prevalence = float(target.mean())
        try:
            auc_roc = float(roc_auc_score(target, prob))
            auc_pr = float(average_precision_score(target, prob))
        except ValueError:
            auc_roc, auc_pr = 0.5, prevalence
        brier = float(brier_score_loss(target, prob))
        climatology_brier = prevalence * (1.0 - prevalence)
        results[str(horizon)] = {
            'auc_pr': auc_pr,
            'auc_roc': auc_roc,
            'brier': brier,
            'brier_skill_score': float(
                1.0 - brier / max(climatology_brier, 1e-8)),
            'prevalence': prevalence,
            'best_threshold_metrics': best_probability_metrics(prob, target),
        }
    return results


def _update_ema(ema_model, model):
    with torch.no_grad():
        decay = CFG['ema_decay']
        for ema_parameter, parameter in zip(
                ema_model.parameters(), model.parameters()):
            ema_parameter.mul_(decay).add_(parameter, alpha=1.0 - decay)
        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


def train_split(split_name, train_end_year, val_end_year,
                data, qbo, steps_pf, files, ep, ep_steps_pf, ep_cum,
                ssw_idx, season_of_step, year_of_step, cum_uvt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\n===== Split: %s (train<=%d, val<=%d) =====' %
          (split_name, train_end_year, val_end_year), flush=True)
    print('Device: %s' % device, flush=True)
    if device.type == 'cuda':
        print('GPU: %s' % torch.cuda.get_device_name(0), flush=True)

    train_ds = SSWDatasetV43(
        data, qbo, steps_pf, files, ep, ep_steps_pf, ep_cum,
        ssw_idx, season_of_step, year_of_step, cum_uvt,
        split='train', train_end_year=train_end_year, val_end_year=val_end_year)
    val_ds = SSWDatasetV43(
        data, qbo, steps_pf, files, ep, ep_steps_pf, ep_cum,
        ssw_idx, season_of_step, year_of_step, cum_uvt,
        split='val', train_end_year=train_end_year, val_end_year=val_end_year)

    class PrimaryEventIdx(Dataset):
        """One 5-day positive to four negatives, balanced across SSW events."""
        def __init__(self, ds):
            self.ds = ds
            positive_indices = np.where(ds.labels == 0)[0]
            self.hard_indices = np.where(
                (ds.labels >= 1) &
                (ds.labels < CFG['n_time_classes'] - 1))[0]
            self.background_indices = np.where(
                ds.labels == CFG['n_time_classes'] - 1)[0]
            self.positive_by_onset = {}
            for index in positive_indices:
                origin = ds.starts[index] + CFG['input_steps']
                future = ds.ssw_idx[
                    (ds.ssw_idx > origin) &
                    (ds.ssw_idx <= origin + CFG['lead_steps'])]
                if len(future) == 0:
                    raise ValueError(
                        'Each 5-day positive must map to a future onset')
                self.positive_by_onset.setdefault(
                    int(future.min()), []).append(index)
            self.onsets = sorted(self.positive_by_onset)
            if (not self.onsets or len(self.hard_indices) == 0 or
                    len(self.background_indices) == 0):
                raise ValueError(
                    'Positive, hard-negative, and background pools are required')
            self.blocks = (
                len(self.onsets) * CFG['positive_draws_per_event'])
        def __len__(self):
            return self.blocks * (CFG['negative_per_positive'] + 1)
        def __getitem__(self, idx):
            block, offset = divmod(
                idx, CFG['negative_per_positive'] + 1)
            if offset == 0:
                onset = self.onsets[block % len(self.onsets)]
                onset_indices = self.positive_by_onset[onset]
                if CFG['deterministic_positive_leads']:
                    draw = block // len(self.onsets)
                    src_idx = onset_indices[draw % len(onset_indices)]
                else:
                    src_idx = np.random.choice(onset_indices)
            elif offset <= CFG['hard_negatives_per_positive']:
                src_idx = np.random.choice(self.hard_indices)
            else:
                src_idx = np.random.choice(self.background_indices)
            return self.ds[src_idx]

    train_index = PrimaryEventIdx(train_ds)
    train_loader = DataLoader(
        train_index, CFG['batch_size'], shuffle=True,
        num_workers=CFG['num_workers'], drop_last=True)
    val_loader = DataLoader(
        val_ds, CFG['batch_size'], shuffle=False,
        num_workers=CFG['num_workers'])

    model = SSW3DNetV43().to(device)
    ema_model = copy.deepcopy(model).to(device).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    print('Params: %d' % sum(p.numel() for p in model.parameters()), flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    warmup_epochs = CFG['warmup_epochs']
    eta_ratio = CFG['lr_eta_min'] / CFG['lr']

    def lr_factor(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        denominator = max(1, CFG['epochs'] - warmup_epochs - 1)
        progress = min(1.0, (epoch - warmup_epochs) / denominator)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return eta_ratio + (1.0 - eta_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_factor)

    class_counts = np.bincount(
        train_ds.labels, minlength=CFG['n_time_classes']).astype(np.float64)
    train_horizon_prior = (
        np.cumsum(class_counts)[:len(CFG['lead_times'])] /
        class_counts.sum())
    hard_slice = slice(1, CFG['n_time_classes'] - 1)
    hard_count = class_counts[hard_slice].sum()
    sample_class_prior = np.zeros(CFG['n_time_classes'])
    block_size = CFG['negative_per_positive'] + 1
    sample_class_prior[0] = 1.0 / block_size
    hard_fraction = CFG['hard_negatives_per_positive'] / block_size
    sample_class_prior[hard_slice] = (
        hard_fraction * class_counts[hard_slice] / hard_count)
    sample_class_prior[-1] = (
        (CFG['negative_per_positive'] -
         CFG['hard_negatives_per_positive']) / block_size)
    sampled_horizon_prior = np.cumsum(
        sample_class_prior)[:len(CFG['lead_times'])]

    def _logit(probability):
        probability = np.clip(probability, 1e-7, 1.0 - 1e-7)
        return np.log(probability / (1.0 - probability))

    prior_logit_shift = (
        _logit(train_horizon_prior) -
        _logit(sampled_horizon_prior))
    prior_shift_tensor = torch.tensor(
        prior_logit_shift, dtype=torch.float32, device=device)
    amp_enabled = (
        device.type == 'cuda' and CFG['use_bf16'] and
        torch.cuda.is_bf16_supported())
    print('Train horizon prior=%s sampled horizon prior=%s' % (
        np.round(train_horizon_prior, 6).tolist(),
        np.round(sampled_horizon_prior, 6).tolist()), flush=True)
    print('Primary events=%d, positive draws/epoch=%d, '
          'hard/background negatives=%d/%d per positive' % (
              len(train_index.onsets), train_index.blocks,
              CFG['hard_negatives_per_positive'],
              CFG['negative_per_positive'] -
              CFG['hard_negatives_per_positive']), flush=True)
    print('Prior logit shift=%s' %
          np.round(prior_logit_shift, 4).tolist(), flush=True)
    print('BF16 autocast: %s' % amp_enabled, flush=True)

    best_pr5, best_pr5_epoch = -1.0, -1
    best_brier5, best_brier5_epoch = float('inf'), -1
    no_improve, start_time = 0, time.time()
    stem = '%s_seed%d_%s' % (
        CFG['version'], CFG['seed'], split_name)
    log_path = os.path.join(CFG['out_dir'], 'train_%s_log.txt' % stem)
    best_pr5_path = os.path.join(
        CFG['out_dir'], 'ssw_%s_best_pr5.pt' % stem)
    best_brier5_path = os.path.join(
        CFG['out_dir'], 'ssw_%s_best_brier5.pt' % stem)
    last_path = os.path.join(CFG['out_dir'], 'ssw_%s_last.pt' % stem)
    log_file = open(log_path, 'w')
    horizon_weights = torch.tensor(
        CFG['horizon_loss_weights'], dtype=torch.float32, device=device)

    for epoch in range(CFG['epochs']):
        model.train()
        train_loss = 0.0
        for batch_index, batch in enumerate(train_loader):
            x = batch['x'].to(device, non_blocking=True)
            q = batch['q'].to(device, non_blocking=True)
            ep_input = (
                batch['ep_in'].to(device, non_blocking=True)
                if CFG['ep_input'] else None)
            target_class = batch['y'].to(device, non_blocking=True)
            targets = torch.stack([
                (target_class <= class_id).float()
                for class_id in range(len(CFG['lead_times']))], dim=1)
            optimizer.zero_grad()
            with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16,
                    enabled=amp_enabled):
                (logits, vortex_prediction,
                 vortex_change_prediction) = model(
                    x, q, ep_input, return_aux=True)
                loss_by_horizon = F.binary_cross_entropy_with_logits(
                    logits, targets, reduction='none')
                forecast_loss = (
                    loss_by_horizon * horizon_weights).sum(dim=1).mean()
                forecast_loss /= horizon_weights.sum()
                corrected_probability = torch.sigmoid(
                    logits.float() + prior_shift_tensor)
                monotonic_loss = F.relu(
                    corrected_probability[:, :-1] -
                    corrected_probability[:, 1:]).mean()
                vortex_loss = F.smooth_l1_loss(
                    vortex_prediction.float(),
                    batch['vortex_target'].to(device))
                vortex_change_loss = F.smooth_l1_loss(
                    vortex_change_prediction.float(),
                    batch['vortex_change_target'].to(device))
                loss = (
                    forecast_loss +
                    CFG['monotonic_loss_weight'] * monotonic_loss +
                    CFG['vortex_level_aux_loss_weight'] * vortex_loss +
                    CFG['vortex_change_aux_loss_weight'] *
                    vortex_change_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), CFG['grad_clip_norm'])
            optimizer.step()
            _update_ema(ema_model, model)
            train_loss += loss.item()
            if batch_index > 0 and batch_index % 50 == 0:
                print('  epoch %d batch %d/%d' % (
                    epoch, batch_index, len(train_loader)), flush=True)

        ema_model.eval()
        all_logits, all_targets = [], []
        all_vortex_prediction, all_vortex_target = [], []
        all_vortex_change_prediction, all_vortex_change_target = [], []
        validation_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                q = batch['q'].to(device)
                ep_input = (
                    batch['ep_in'].to(device) if CFG['ep_input'] else None)
                target_class_batch = batch['y'].to(device)
                targets_batch = torch.stack([
                    (target_class_batch <= class_id).float()
                    for class_id in range(len(CFG['lead_times']))], dim=1)
                with torch.autocast(
                        device_type=device.type, dtype=torch.bfloat16,
                        enabled=amp_enabled):
                    (raw_logits, vortex_prediction,
                     vortex_change_prediction) = ema_model(
                         x, q, ep_input, return_aux=True)
                corrected_logits = raw_logits.float() + prior_shift_tensor
                validation_loss += (
                    F.binary_cross_entropy_with_logits(
                        corrected_logits, targets_batch, reduction='none') *
                    horizon_weights).sum(dim=1).mean().item()
                all_logits.append(corrected_logits.cpu())
                all_targets.append(target_class_batch.cpu())
                all_vortex_prediction.append(vortex_prediction.float().cpu())
                all_vortex_target.append(batch['vortex_target'])
                all_vortex_change_prediction.append(
                    vortex_change_prediction.float().cpu())
                all_vortex_change_target.append(
                    batch['vortex_change_target'])

        corrected_logits = torch.cat(all_logits)
        target_class = torch.cat(all_targets).numpy()
        horizon_probability = torch.sigmoid(
            corrected_logits).numpy()
        horizon_metrics = horizon_probability_metrics(
            horizon_probability, target_class)
        macro_pr = float(np.mean([
            horizon_metrics[str(horizon)]['auc_pr']
            for horizon in CFG['lead_times']]))
        validation_loss /= len(val_loader) * horizon_weights.sum().item()
        monotonic_violation = float(np.mean(
            horizon_probability[:, :-1] >
            horizon_probability[:, 1:]))
        primary = horizon_metrics['5']
        vortex_prediction = torch.cat(
            all_vortex_prediction).numpy()
        vortex_target = torch.cat(all_vortex_target).numpy()
        vortex_rmse = float(np.sqrt(np.mean(
            (vortex_prediction - vortex_target) ** 2)))
        vortex_correlation = float(np.corrcoef(
            vortex_prediction, vortex_target)[0, 1])
        vortex_change_prediction = torch.cat(
            all_vortex_change_prediction).numpy()
        vortex_change_target = torch.cat(
            all_vortex_change_target).numpy()
        vortex_change_rmse = float(np.sqrt(np.mean(
            (vortex_change_prediction - vortex_change_target) ** 2)))
        vortex_change_correlation = float(np.corrcoef(
            vortex_change_prediction, vortex_change_target)[0, 1])
        lr_now = optimizer.param_groups[0]['lr']
        line = (
            'E%3d lr=%.1e loss=%.4f val=%.4f PR5=%.4f '
            'PR10/15=%s ROC5=%.4f BSS5=%+.3f '
            'U5rmse=%.3f U5r=%.3f dU5rmse=%.3f dU5r=%.3f '
            'mono=%.3f [%dm]' % (
                epoch, lr_now, train_loss / len(train_loader),
                validation_loss, primary['auc_pr'],
                '/'.join('%.4f' % horizon_metrics[str(h)]['auc_pr']
                         for h in CFG['lead_times'][1:]),
                primary['auc_roc'], primary['brier_skill_score'],
                vortex_rmse, vortex_correlation,
                vortex_change_rmse, vortex_change_correlation,
                monotonic_violation,
                int((time.time() - start_time) / 60)))
        print(line, flush=True)
        log_file.write(line + '\n')
        log_file.flush()

        state = {
            'model_state': ema_model.state_dict(),
            'epoch': epoch,
            'primary_auc_pr': primary['auc_pr'],
            'primary_auc_roc': primary['auc_roc'],
            'primary_brier': primary['brier'],
            'primary_brier_skill_score': primary['brier_skill_score'],
            'macro_auc_pr': macro_pr,
            'validation_loss': validation_loss,
            'monotonic_violation_fraction': monotonic_violation,
            'vortex_aux_rmse': vortex_rmse,
            'vortex_aux_correlation': vortex_correlation,
            'vortex_change_aux_rmse': vortex_change_rmse,
            'vortex_change_aux_correlation': vortex_change_correlation,
            'horizon_metrics': horizon_metrics,
            'train_horizon_prior': train_horizon_prior.tolist(),
            'sampled_horizon_prior': sampled_horizon_prior.tolist(),
            'prior_logit_shift': prior_logit_shift.tolist(),
            'cfg': CFG,
        }
        torch.save(state, last_path)
        selection_eligible = (
            primary['brier_skill_score'] >= CFG['selection_bss_floor'])
        if selection_eligible and primary['auc_pr'] > best_pr5:
            best_pr5 = primary['auc_pr']
            best_pr5_epoch = epoch
            no_improve = 0
            torch.save(state, best_pr5_path)
            print('  [%d] New best PR5=%.4f ROC5=%.4f BSS5=%+.3f' % (
                epoch, primary['auc_pr'], primary['auc_roc'],
                primary['brier_skill_score']), flush=True)
        else:
            no_improve += 1
        if primary['brier'] < best_brier5:
            best_brier5 = primary['brier']
            best_brier5_epoch = epoch
            torch.save(state, best_brier5_path)

        scheduler.step()
        if (epoch + 1 >= CFG['early_stop_min_epochs'] and
                best_pr5_epoch >= 0 and
                no_improve >= CFG['early_stop_patience']):
            message = 'Early stopping at epoch %d' % epoch
            print('  ' + message, flush=True)
            log_file.write(message + '\n')
            break

    if best_pr5_epoch < 0:
        fallback = torch.load(
            best_brier5_path, map_location='cpu', weights_only=False)
        torch.save(fallback, best_pr5_path)
        best_pr5 = float(fallback['primary_auc_pr'])
        best_pr5_epoch = int(fallback['epoch'])
        message = (
            'No checkpoint met BSS floor; using best-Brier checkpoint '
            'as the PR fallback')
        print('  ' + message, flush=True)
        log_file.write(message + '\n')

    def evaluate_checkpoint(checkpoint_path, checkpoint_label):
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False)
        ema_model.load_state_dict(checkpoint['model_state'])
        ema_model.eval()
        logits, targets, starts, winters = [], [], [], []
        vortex_predictions, vortex_targets = [], []
        vortex_change_predictions, vortex_change_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                q = batch['q'].to(device)
                ep_input = (
                    batch['ep_in'].to(device) if CFG['ep_input'] else None)
                with torch.autocast(
                        device_type=device.type, dtype=torch.bfloat16,
                        enabled=amp_enabled):
                    (raw_logits, vortex_prediction,
                     vortex_change_prediction) = ema_model(
                         x, q, ep_input, return_aux=True)
                logits.append(raw_logits.float().cpu())
                targets.append(batch['y'])
                starts.append(batch['start'])
                winters.append(batch['winter'])
                vortex_predictions.append(vortex_prediction.float().cpu())
                vortex_targets.append(batch['vortex_target'])
                vortex_change_predictions.append(
                    vortex_change_prediction.float().cpu())
                vortex_change_targets.append(
                    batch['vortex_change_target'])
        prior_shift = torch.tensor(
            checkpoint['prior_logit_shift'], dtype=torch.float32)
        horizon_probability = torch.sigmoid(
            torch.cat(logits) + prior_shift).numpy()
        target_class = torch.cat(targets).numpy()
        starts_array = torch.cat(starts).numpy()
        winters_array = torch.cat(winters).numpy()
        vortex_prediction_array = torch.cat(
            vortex_predictions).numpy()
        vortex_target_array = torch.cat(vortex_targets).numpy()
        vortex_change_prediction_array = torch.cat(
            vortex_change_predictions).numpy()
        vortex_change_target_array = torch.cat(
            vortex_change_targets).numpy()
        metrics = horizon_probability_metrics(
            horizon_probability, target_class)

        for class_id, horizon in enumerate(CFG['lead_times']):
            binary_target = (target_class <= class_id).astype(np.int64)
            predicted, thresholds = cross_validated_threshold_predictions(
                horizon_probability[:, class_id],
                binary_target, winters_array)
            decision_metrics = probability_metrics(
                predicted.astype(np.float32), binary_target, 0.5)
            decision_metrics.pop('threshold')
            metrics[str(horizon)]['loow_threshold_metrics'] = decision_metrics
            metrics[str(horizon)]['event_metrics'] = event_diagnostics(
                predicted, binary_target, starts_array, winters_array,
                ssw_idx, season_of_step, horizon, thresholds)

        prediction_path = os.path.join(
            CFG['out_dir'], '%s_%s_predictions.npz' %
            (stem, checkpoint_label))
        np.savez_compressed(
            prediction_path,
            horizon_probability=horizon_probability,
            target_class=target_class,
            start=starts_array,
            forecast_origin=starts_array + CFG['input_steps'],
            winter=winters_array,
            vortex_prediction=vortex_prediction_array,
            vortex_target=vortex_target_array,
            vortex_change_prediction=vortex_change_prediction_array,
            vortex_change_target=vortex_change_target_array,
            horizons_days=np.asarray(CFG['lead_times']))
        return {
            'checkpoint': checkpoint_path,
            'prediction_file': prediction_path,
            'epoch': int(checkpoint['epoch']),
            'primary_auc_pr': metrics['5']['auc_pr'],
            'macro_auc_pr': float(np.mean([
                metrics[str(h)]['auc_pr'] for h in CFG['lead_times']])),
            'validation_loss': float(checkpoint['validation_loss']),
            'vortex_aux_rmse': float(np.sqrt(np.mean(
                (vortex_prediction_array - vortex_target_array) ** 2))),
            'vortex_aux_correlation': float(np.corrcoef(
                vortex_prediction_array, vortex_target_array)[0, 1]),
            'vortex_change_aux_rmse': float(np.sqrt(np.mean(
                (vortex_change_prediction_array -
                 vortex_change_target_array) ** 2))),
            'vortex_change_aux_correlation': float(np.corrcoef(
                vortex_change_prediction_array,
                vortex_change_target_array)[0, 1]),
            'horizons': metrics,
        }

    best_pr5_evaluation = evaluate_checkpoint(
        best_pr5_path, 'best_pr5')
    best_brier5_evaluation = evaluate_checkpoint(
        best_brier5_path, 'best_brier5')
    five_day = best_pr5_evaluation['horizons']['5']
    five_day_events = five_day['event_metrics']
    log_file.write(
        '\nBest PR5: epoch %d, PR5=%.4f, BSS5=%+.4f\n' %
        (best_pr5_epoch, best_pr5,
         five_day['brier_skill_score']))
    log_file.write(
        'Best Brier5: epoch %d, Brier5=%.5f\n' %
        (best_brier5_epoch, best_brier5))
    log_file.write(
        'LOOW 5d events: %d/%d hit; false episodes=%d\n' % (
            five_day_events['events_hit'], five_day_events['n_events'],
            five_day_events['false_alarm_episodes_total']))
    log_file.close()
    print(
        'Split %s done. Best PR5=%.4f, BSS5=%+.3f, '
        'LOOW 5d events=%d/%d, false episodes=%d' % (
            split_name, best_pr5, five_day['brier_skill_score'],
            five_day_events['events_hit'], five_day_events['n_events'],
            five_day_events['false_alarm_episodes_total']), flush=True)
    return {
        'best_primary_auc_pr': best_pr5,
        'best_primary_pr_epoch': best_pr5_epoch,
        'best_primary_brier': best_brier5,
        'best_primary_brier_epoch': best_brier5_epoch,
        'best_pr5_evaluation': best_pr5_evaluation,
        'best_brier5_evaluation': best_brier5_evaluation,
        'best_pr5_checkpoint': best_pr5_path,
        'best_brier5_checkpoint': best_brier5_path,
        'last_checkpoint': last_path,
    }


# ====================================================================
# Main
# ====================================================================

def run_self_test():
    """Fast synthetic checks for leakage boundaries and model tensor shapes."""
    origin = CFG['input_steps']
    background_class = CFG['n_time_classes'] - 1
    assert time_class_for_start(
        0, np.array([origin])) == background_class
    previous_horizon = 0
    for class_id, horizon_days in enumerate(CFG['lead_times']):
        assert time_class_for_start(
            0, np.array([
                origin + previous_horizon * 4 + 1])) == class_id
        assert time_class_for_start(
            0, np.array([origin + horizon_days * 4])) == class_id
        previous_horizon = horizon_days
    extended_class = len(CFG['lead_times'])
    assert time_class_for_start(
        0, np.array([
            origin + CFG['lead_times'][-1] * 4 + 1])) == extended_class
    assert time_class_for_start(
        0, np.array([
            origin + CFG['max_horizon_steps']])) == extended_class
    assert time_class_for_start(
        0, np.array([
            origin + CFG['max_horizon_steps'] + 1])) == background_class

    ssw = np.array([181], dtype=np.int64)
    total = 320
    time_values = np.arange(total, dtype=np.float32)
    data = np.broadcast_to(
        time_values[:, None, None, None, None],
        (total, 4, 20, 19, 1)).copy()
    qbo = np.zeros((total, CFG['n_lev']), dtype=np.float32)
    ep = np.zeros((total, 2, 2), dtype=np.float32)
    steps = np.array([total], dtype=np.int32)
    cum = np.array([0, total], dtype=np.int64)
    season = np.full(total, 2018, dtype=np.int32)
    year = np.full(total, 2018, dtype=np.int32)
    ds = SSWDatasetV43(
        data, qbo, steps, ['synthetic_201801.nc'],
        ep, steps, cum, ssw, season, year, cum,
        split='val', train_end_year=2017, val_end_year=2025)
    idx = ds.starts.index(0)
    item1, item2 = ds[idx], ds[idx]
    assert item1['x'].shape[0] == CFG['input_steps']
    assert item1['x'][-1, 0, 0, 0, 0].item() == 179.0
    assert item1['y'].item() == 0
    assert item1['vortex_target'].item() == 182.5
    assert item1['vortex_change_target'].item() == 5.0
    assert torch.equal(item1['x'], item2['x'])

    model = SSW3DNetV43().eval()
    with torch.no_grad():
        out, vortex_prediction, vortex_change_prediction = model(
            torch.zeros(1, 24, 4, 20, 19, 72),
            torch.zeros(1, 24, 20),
            torch.zeros(1, 24, 19, 11),
            return_aux=True)
    assert out.shape == (1, len(CFG['lead_times']))
    assert vortex_prediction.shape == (1,)
    assert vortex_change_prediction.shape == (1,)
    print('SELF_TEST_OK', flush=True)


def run_data_check():
    """Read metadata only and verify all monthly joins before a GPU run."""
    all_files, all_steps, all_cum, winter_mask = _build_all_file_list()
    winter_files = [
        f for f, keep in zip(all_files, winter_mask) if keep]
    winter_steps = all_steps[winter_mask]
    if not all_files or len(winter_files) != int(winter_mask.sum()):
        raise ValueError('Invalid UVT file inventory')

    ep_files = sorted(glob.glob(CFG['ep_dir'] + '*_ep_flux_6h.nc'))
    ep_files = [f for f in ep_files if _is_winter_month(f)]
    ep_by_month = {
        '%04d%02d' % _year_month_from_file(f): f for f in ep_files}
    for uvt_file, n_steps in zip(winter_files, winter_steps):
        yyyymm = '%04d%02d' % _year_month_from_file(uvt_file)
        if yyyymm not in ep_by_month:
            raise FileNotFoundError('Missing EP file for %s' % yyyymm)
        with NcFile(ep_by_month[yyyymm], 'r') as ds:
            if ds.variables['epdiv'].shape[0] != int(n_steps):
                raise ValueError('EP/UVT time mismatch for %s' % yyyymm)

    with NcFile(winter_files[0], 'r') as ds:
        lat = ds.variables['latitude'][:]
        lev = ds.variables['pressure_level'][:]
        lon = ds.variables['longitude'][:]
        print('UVT grid: lat=%d [%.2f, %.2f], lon=%d [%.2f, %.2f], '
              'lev=%s %s' % (
                  len(lat), lat.min(), lat.max(), len(lon), lon.min(), lon.max(),
                  lev.tolist(), getattr(ds.variables['pressure_level'], 'units', '')),
              flush=True)
        if len(lon[::2]) != 72 or len(lev) < CFG['n_lev']:
            raise ValueError('Unexpected UVT grid')

    ssw_full = load_ssw_labels()
    if np.any(ssw_full < 0) or np.any(ssw_full >= all_cum[-1]):
        raise ValueError('SSW onset outside full ERA5 timeline')
    ssw_winter = [
        _full2winter_step(si, all_cum, all_steps, winter_mask)
        for si in ssw_full]
    if any(si < 0 for si in ssw_winter):
        raise ValueError('An SSW onset falls outside selected winter months')

    train_stop = _training_stop_step(
        winter_steps, winter_files, CFG['train_end_year'])
    print('Inventory: UVT=%d, winter=%d, EP winter=%d, '
          'winter steps=%d, train steps=%d, SSW=%d' % (
              len(all_files), len(winter_files), len(ep_files),
              winter_steps.sum(), train_stop, len(ssw_full)), flush=True)
    print('DATA_CHECK_OK', flush=True)


def main(seed):
    set_seed(seed)
    os.makedirs(CFG['out_dir'], exist_ok=True)
    if CFG['window_len'] != CFG['input_steps']:
        raise ValueError('Input window must not include forecast lead steps')
    if CFG['lead_times'][-1] > CFG['max_horizon_days']:
        raise ValueError('Forecast lead cannot exceed the label horizon')

    print('=' * 60, flush=True)
    print('SSW %s: direct precursor and vortex-change forecast' %
          CFG['version'], flush=True)
    print('=' * 60, flush=True)
    for k, v in sorted(CFG.items()):
        if isinstance(v, (int, float, str, bool, list, tuple)):
            print('  %s: %s' % (k, v), flush=True)
    print(flush=True)

    # 1. Build file metadata
    all_files, all_steps, all_cum, winter_mask = _build_all_file_list()
    print('Total files: %d, winter files: %d' % (len(all_files), winter_mask.sum()), flush=True)

    # 2. Pre-load UVT data (4-channel with U_dz)
    data, qbo, steps_pf, files, norm_stats = preload_uvt()
    cum_uvt = np.cumsum(np.concatenate([[0], steps_pf]))

    # 3. Pre-load aligned EP data
    ep, ep_steps_pf, ep_cum, ep_mu, ep_sd = preload_ep_aligned(steps_pf, files)
    norm_stats['ep_mean'] = ep_mu
    norm_stats['ep_std'] = ep_sd
    np.savez(os.path.join(
        CFG['out_dir'], 'norm_stats_%s.npz' % CFG['version']), **norm_stats)
    print('Data shapes: UVT=%s QBO=%s EP=%s' % (
        data.shape, qbo.shape, ep.shape), flush=True)

    # 4. Convert SSW onsets to winter space
    ssw_full = load_ssw_labels()
    ssw_idx = []
    for si in ssw_full:
        wi = _full2winter_step(si, all_cum, all_steps, winter_mask)
        if wi >= 0:
            ssw_idx.append(wi)
    ssw_idx = np.array(sorted(ssw_idx), dtype=np.int64)
    print('SSW in winter space: %d (%d original)' % (len(ssw_idx), len(ssw_full)), flush=True)
    if len(ssw_idx) != len(ssw_full) or len(ssw_idx) == 0:
        raise ValueError('SSW label conversion lost events')

    # 5. Build year/season mapping
    season_of_step, year_of_step, _ = assign_step_seasons(steps_pf, files)

    # 6. Train
    result = train_split(
        'main', CFG['train_end_year'], CFG['val_end_year'],
        data, qbo, steps_pf, files,
        ep, ep_steps_pf, ep_cum,
        ssw_idx, season_of_step, year_of_step, cum_uvt)

    # 7. Summary
    print('\n' + '=' * 60)
    print('Training complete.')
    print('  Best 5-day PR-AUC: %.4f @ epoch %d' % (
        result['best_primary_auc_pr'], result['best_primary_pr_epoch']))
    print('  5-day BSS at best PR: %+.4f' %
          result['best_pr5_evaluation']['horizons']['5'][
              'brier_skill_score'])
    print('  Best 5-day Brier: %.5f @ epoch %d' % (
        result['best_primary_brier'],
        result['best_primary_brier_epoch']))
    print('  Checkpoint: %s' % result['best_pr5_checkpoint'])
    print('=' * 60, flush=True)

    summary = {
        'cfg': {k: v for k, v in CFG.items()
                if isinstance(v, (int, float, str, bool, list, type(None)))},
        **result,
    }
    result_path = os.path.join(
        CFG['out_dir'], '%s_seed%d_results.json' % (
            CFG['version'], CFG['seed']))
    with open(result_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print('  Results: %s' % result_path, flush=True)


if __name__ == '__main__':
    if '--self-test' in sys.argv:
        run_self_test()
    elif '--data-check' in sys.argv:
        run_data_check()
    else:
        seed = CFG['seed']
        if '--seed' in sys.argv:
            seed_pos = sys.argv.index('--seed') + 1
            if seed_pos >= len(sys.argv):
                raise ValueError('--seed requires an integer')
            seed = int(sys.argv[seed_pos])
        main(seed)
