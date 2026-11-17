# Reviewer-requested R67 sampling contrast

Recorded 2026-11-11 before new model outputs. This is a retrospective reviewer-requested sensitivity study, NOT original public preregistration and NOT model optimization/promotion.

Question R3 major 4: under a fixed cumulative-output model/loss, does changing origin sampling change ranking, probability error and warnings? Existing V65 remains a bundled head/loss/sampling intervention and cannot answer this alone.

Two NEW arms: event-balanced (12 draws per eligible event, 2 hard + 2 background per positive) and uniform without replacement from all eligible origins. Use original A/B/C, seeds42/73/2026, 21 epochs, 45-epoch LR schedule, batch16, same steps per fold, same trunk, cumulative head, horizon BCE weights1/.3/.2, auxiliary weights.1/.1, monotonicity weight.05. To prevent sampling-dependent corrections entering the loss, compute monotonicity on raw probabilities in BOTH arms, with no logit shift during training. Old event-balanced output therefore is an anchor, NOT the balanced member of this pair. This requires 18 new fits.

Primary new-arm contrast: raw probabilities, common proper-score reference and fixed0.20 policy. Save a separately labelled training-only scalar-prior-correction diagnostic for the balanced arm; it is not proof of calibration or a new selected forecast. No thresholds/checkpoints/seeds/configurations selected from outer results. Same seed gives the same initial model. Sample exposure records use actually retained batches (drop_last), not nominal draws. No positive-exposure matching claim: record this limitation and do not infer that sampling changes alone distinguish prior alignment from dose/representation effects.

Report all horizons/arms/seeds/folds, equal probability means, pooled metrics, equal-fold sensitivity, paired winter bootstrap and event identities. Correct the original causal language irrespective of the new result. No success threshold or permission to search for a better model is introduced.

Resource plan: serial one-H100, six CPUs,128GB RAM; 18 fits. Historical V65 train times plus loading are indicative only; provisional total120–180 GPU-hours excluding queueing, to be refreshed from first R67 fold. Hard wall limit24h/task. Require >=70GiB free on the assigned GPU; fail rather than kill another user's process. Results isolated under /share/home/xiayan/ssw_phase1/revision_R67_20261111. Input files and source modules stay read-only/unmodified. No additional ACL is needed on xiayan's own data.
