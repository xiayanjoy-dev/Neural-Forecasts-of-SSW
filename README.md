# Neural forecasts of sudden stratospheric warming onset

Research code and processed verification inputs supporting *Ranking Is Not Probability Skill: Lessons from Neural Forecasts of Sudden Stratospheric Warmings* (Yan Xia and Fei Xie). This repository contains historical forecasts and the reviewer-requested fixed-formulation sampling sensitivity study. It is not an operational prediction service.

## Scientific comparisons

- Chronological folds A/B/C test 2000–2004, 2005–2010 and 2011–2021, training only on preceding winters from winter 1979.
- All 1,918 eligible origins, 22 test winters and 16 SSW events remain fixed. Seeds are 42, 73 and 2026; the ensemble is the arithmetic mean of aligned probabilities.
- Historical corrected cumulative forecasts, the historical natural-frequency hazard configuration, and the NEW raw balanced/uniform cumulative-model pair are distinct systems. They must not be relabelled as the same baseline.
- The new pair holds architecture, losses, initialization, update counts and epoch rules fixed. It does NOT match retained positive-example exposure. It is a retrospective reviewer-requested sensitivity study, not public preregistration or model selection.
- Fixed thresholds and all null/uncertain comparisons are retained. Winter bootstrap intervals hold predictions fixed; they do not include retraining uncertainty.

## CPU verification replay

Use Python 3.12 with NumPy, SciPy and Matplotlib; analysis was replayed with NumPy 2.5.4, SciPy 1.18.1 and Matplotlib 3.11.2. These are the analysis environment versions, not the GPU training environment. From the repository root:

```bash
python verify_manifest.py
python manuscript/20261117_R67_revision_v19/analyze_r67.py --self-test
python manuscript/20261117_R67_revision_v19/analyze_r67.py
python manuscript/20261117_R67_revision_v19/plot_r67.py
```

The replay verifies forecast-origin/start/winter/target-class identity, 96 previously reported probability-score cells, stored event metrics, and 2,000 paired winter bootstrap intervals. It outputs pooled/fold/seed and equal-fold scores, fixed-0.20 warnings, event identities, reliability bins, actual positive exposure and the January 2021 trajectories. Plots are generated from those outputs without fitting a model or selecting a threshold. Existing output CSVs are figure source data.

## Training code and raw data

`ssw_train_v43_core_versioned.py` is the frozen 397,171-parameter network. The 1-degree/rolling wrappers and `run_sampling_control.py` implement the historical and new sampling protocols. GPU retraining requires the large canonical ERA5 monthly inputs, QG EP-flux inputs and audited label catalogue; these raw archives and GPU checkpoints are NOT redistributed here. Frozen source defaults include the original cluster paths. Set `SSW_FROZEN_CODE` to this repository root when using the sampling wrapper, and configure data/output paths explicitly before running training. Do not run the archived cluster scripts as an unreviewed batch submission.

Raw ERA5: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels and https://cloud.google.com/storage/docs/public-datasets/era5 . SSW catalogue: https://csl.noaa.gov/groups/csl8/sswcompendium/ . ECMWF S2S: https://apps.ecmwf.int/datasets/data/s2s-reforecasts-instantaneous-accum-ecmf/ . Product licences and access conditions remain those of their providers. The frozen ECMWF comparison uses cycle 48r1 and only the registered common origins, not a search across model cycles.

This release supplies source and processed verification inputs. It does not claim that full GPU training has been independently rerun outside the original cluster. Internal version codes in filenames are retained for traceability, not scientific model names.

## Licence and release status

Public reviewer access has been authorized by the corresponding research workflow. A reuse licence has not yet been selected by the authors; no MIT/BSD permission is implied. Author confirmation of the final licence and a persistent versioned archive DOI remain separate release decisions. Reviewer correspondence, manuscript drafts, credentials and raw data are excluded.
