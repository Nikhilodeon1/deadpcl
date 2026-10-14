# DRAFT author responses, v2 (not for posting until [PENDING] and [OWNER] items are resolved)

Written against the five reviews on the submission page. Posting format there: one "Rebuttal" per review (R1 xM3J,
R2 JHNu, R3 xCda, R4 7MQE, R5 fDtQ) plus an optional paper-level "Official Comment" that I use for the shared evidence, so each
rebuttal can stay short and point to it. Source of every number: `rebuttal/pack/RESULTS.md` and the tables it cites.

Markers: **[PENDING-ATTR]** = waits for the attribution variants (no-Severinghaus, timestep masking, 9 variables, original
mask rate and warmup; scoring finishes within hours). **[OWNER]** = a decision or disclosure that is the author's, not mine.

Tone rules used below: concede what is right, give a number or a plain "not done", never argue with how a score was reached.

---

## Official Comment (paper level): shared evidence

We thank the reviewers. Most of the criticism of the statistics, the task definition and the missing details was correct, and
checking it exposed problems in the submission that we report here rather than defend. We ran new experiments and audits.

**1. Ten seeds, paired intervals.** We re-ran ERM (lambda = 0) and PCL (lambda = 0.5) on the corrected pipeline with ten seeds
(the original three plus seven), identical schedules, plus a fixed-hyperparameter XGBoost on the same Site A split.
Target AUROC, mean (SD over seeds):

| | Site B | MIMIC-IV | eICU-CRD |
|---|---|---|---|
| ERM | 0.689 (0.042) | 0.642 (0.035) | 0.594 (0.042) |
| PCL | 0.662 (0.052) | 0.642 (0.047) | 0.591 (0.050) |
| XGBoost | 0.731 (0.012) | 0.646 (0.010) | 0.661 (0.009) |

Paired PCL minus ERM (95% bootstrap over seeds, then stays): -0.026 [-0.063, +0.013], +0.001 [-0.021, +0.020], -0.003
[-0.031, +0.027]. Every interval contains zero. The submission's claim that PCL "falls below ERM at every site" is not
supported; what is supported is "no evidence that the constraints help, and no evidence that they hurt". XGBoost is above both
neural methods at eICU (+0.067 over ERM, [+0.039, +0.093]) and above PCL at Site B (+0.068, [+0.021, +0.115]); at MIMIC-IV the
three are indistinguishable. The seed SD of a neural arm is 3.5 to 5.2 points (range 12 to 17), as large as the effects the
original three-seed comparison argued about.

**2. Lambda.** On seeds 42-46 (all six lambda values run) choosing lambda by source-validation AUROC gives PCL minus ERM of
+0.008, +0.019, +0.016 at the three sites, intervals containing zero (two lower bounds at 0.000). Choosing lambda on target data
(an oracle, not a deployable method) gives +0.039 [+0.023, +0.057] at eICU, and that interval ignores the selection. Source
validation AUROC itself falls from 0.817 (lambda 0) to 0.769 (lambda 5).

**3. The task is event detection in the input window.** The label is "any positive hour in the first 48 hours" and the model reads
those same 48 hours. In a diagnostic of the existing models, cutting each positive stay's input to end 0, 6 or 12 hours before
its first positive label lowers Site B AUROC from 0.689 (ERM) / 0.662 (PCL) to 0.54-0.57, near chance. We will describe the task
as detection and drop any forecasting language. (We did not train a model on cut inputs.)

**4. Errors in the submission we will correct.** (a) "Exceeds 1.0 on four of five site pairs": the five values come from detector
validation scenarios, not site pairs. (b) "+4.7 smallest single-seed gap" vs "+4.2 source gain" is inconsistent. (c) The abstract's
XGBoost statement ignores Group DRO, which we did not re-run (with one source site its groups collapse to one). (d) The
constraint mask is built after a 6-hour forward fill, not on the raw record: Henderson-Hasselbalch inputs are jointly present in
3.0% of Site A hours raw vs 27.7% after fill (1.8% vs 17.0% at MIMIC-IV, 0.55% vs 7.2% at eICU). (e) The optimized loss includes a
loss-balancing factor that the paper omits. (f) PaO2 is reconstructed from saturation at the PhysioNet sites with an approximate
inverse and measured at MIMIC-IV and eICU. (g) From the surviving files we cannot reconstruct which lambda was selected how for
the submitted PCL row; the revision reports three explicit rules.

**5. Not done in this period:** reproduction of the original pipeline, label re-scoring against the old ICD labels (they no
longer exist), all-site pretraining, Group DRO, constraint-activity statistics, a decomposition of seed variance. We list them as
limitations. **[OWNER: confirm the repository update before pushing to the anonymous link.]**

---

## Rebuttal to R1 (xM3J)

Thank you for the detailed comments; several are fair and we have acted on them.

**"Four of five lessons are well known."** We agree the individual lessons have precedent and will not present them as new. What
the paper offers is a quantified case study: one concrete pipeline, ten seeds, paired intervals, and the finding that the previously
reported gain does not survive.

**Lesson 4 (imputed PaO2).** If the constraint were exactly valid, an imputed channel would match the measured one, as you say.
In our code the equality does not hold exactly: PaO2 is missing from the raw PhysioNet files and is reconstructed from saturation
with an approximate inverse that drops one term of the loss's own forward function (7.3% of Site A and 1.9% of Site B hours),
while at MIMIC-IV and eICU PaO2 is measured. So the constraint carries different information at source and target. We have not
shown it is the cause of the failure. **[PENDING-ATTR: effect of removing the Severinghaus term.]**

**Concurrent submission.** You are right that praising a method that tunes on target data conflicts with our lesson 3. We will
rewrite that passage: methods using unlabeled target data are transductive, not zero-shot, and we say so. **[OWNER: wording.]**

**Novelty of the constraint, transfer to other settings.** Not claimed / not tested. One architecture, one task, three external
sites; we state this as a limitation.

**Writing.** We will remove the flagged punchlines and edit the prose for plain statements. **[OWNER: disclose any AI assistance
in writing as the venue's generative-AI policy requires; do not state that none was used unless that is true.]**

**Constraint at masked positions (at least two masked variables).** In the implementation a constraint is applied where all its
variables are present after forward fill and at least one is masked, computed on the model's predictions for all variables
(`pcl_loss.py`, `train_utils.py`). "At least two" is not required, so your concern about double penalization when exactly one
variable is masked applies. Whether it makes the penalty redundant with the masked loss is empirical and we have not measured it;
we will say so rather than assert either answer.

**Timestep masking.** **[PENDING-ATTR: ERM and PCL with whole-timestep masking, five seeds, paired margin.]**

**How often the constraint triggers.** Share of stay-hours with all constraint inputs present (raw / after forward fill):
Henderson-Hasselbalch 1.8% / 17.0% (MIMIC-IV), 0.55% / 7.2% (eICU), 3.0% / 27.7% (Site A), 0.19% / 0.56% (Site B);
Severinghaus 5.1% / 23.7%, 2.4% / 13.2%, 7.0% / 26.1%, 1.7% / 8.8%. Activity at masked positions was not computed.

**L1 vs squared error.** We found no recorded rationale in the code or its history (L1 for MAP, squared for the other two) and
will describe it as an unexamined design choice. **Lambda.** The PCL rows use 0.5; the full sweep and three selection rules are
in the shared comment. **SOFA.** Sepsis-3 as suspected infection (antibiotic-culture pairing at MIMIC-IV; a culture or an
infection diagnosis at eICU) with single-point SOFA >= 2 at the suspicion time, "ever in the ICU stay", broadcast over 48 hours;
PhysioNet uses the challenge's hourly label inside the first 48 hours, so a definitional difference remains and we state it.
**Why confound 1 favors PCL; how to fix confound 4.** **[PENDING-ATTR.]**

We hope the ten-seed paired results, which change the paper's central claim, address the concern that the negative result is
not established.

---

## Rebuttal to R2 (JHNu)

Thank you. **One architecture, one task.** Correct; stated as a limitation. **Three seeds.** Now ten (original three plus seven);
a neural arm's seed SD is 3.5 to 5.2 AUROC points, so three seeds could not support the original claims. The paired intervals are in
the shared comment. **No new algorithm.** The paper is an evaluation; we say so.
**PINN shift and key assumptions.** We checked the assumptions that matter here: the constraint inputs must be present (they are
sparse; availability above), PaO2 must mean the same thing across sites (it does not), the penalty must add information beyond the
masked loss (not measured), and MAP = DBP + (SBP - DBP)/3 is an approximation, not an identity (its residual by cohort was not
measured). We will shorten the PINN analogy to a motivation. **"Only two sites."** There are three external sites (Site B,
MIMIC-IV, eICU-CRD) and they appear in every table; we will make that explicit. **Writing.** We will fix spelling and repetition.

---

## Rebuttal to R3 (xCda)

Thank you for a careful review; it improved the paper. Numbering follows your weaknesses and questions.

**W1 / Q1, Q2 (attribution; confound 4, 5).** You are right that the original claim "five confounds explain the gain" was not
shown. **[PENDING-ATTR: one-factor-at-a-time table for Severinghaus term off, 9 vs 17 variables, original mask rate and warmup,
timestep masking; paired margin change per site under the rule "explanatory if the change exceeds the seed SD of the margin and
its interval excludes zero".]** Re-scoring against the ICD labels is not possible (those labels no longer exist); all-site
pretraining was not rerun (the cohorts are not on the training machine). Confound 5 concerns the violation score across sites and
cannot inflate prediction AUROC; we will say so. At Site B, where ICD labels were never used, we explain the margin change using
the other factors only.

**W2 / Q3 (statistics).** Per-seed values, SDs (not ranges) and paired intervals are now reported (shared comment); under the rule
"worse only if the interval lies below zero over ten seeds", none qualifies, so the paper will say "no evidence of benefit".
The 4-7 points was a range over three seeds; the larger variance is not explained by the pipeline and we did not separate
initialization from the Site A split. The 3.3-point Henderson-Hasselbalch cost was not re-tested and will not be interpreted.

**W3 / Q4 (lambda).** Target AUROC for lambda in {0, 0.1, 0.5, 1, 2, 5} (seeds 42-46, mean (SD)): Site B 0.674 (0.039), 0.654
(0.075), 0.668 (0.066), 0.627 (0.056), 0.669 (0.052), 0.673 (0.028); MIMIC-IV 0.628, 0.639, 0.644, 0.620, 0.637, 0.636; eICU
0.570, 0.581, 0.592, 0.571, 0.605, 0.609. Lambda 0 is ERM. Lambda 0 and 0.5 also have ten seeds. With source validation the
best lambda is mostly 0 or small, as you suspected; a target-selected lambda gives a gain at eICU only (shared comment, item 2).

**W4 / Q5, Q6 (labels, circularity).** (a) Site A/B labels are the challenge's own hourly labels cut to the first 48 hours;
MIMIC-IV and eICU are our Sepsis-3 derivation, so some definitional shift remains. Kept-stay prevalence is 3.28% and 2.06%;
in the same files 8.8% / 5.7% of records are ever septic and 5.6% / 3.7% have a positive hour in the first 48 rows, and the
24-hour and hemodynamic filters remove the rest. Cohorts: 20,336 and 20,000 records, 10,381 and 14,779 kept stays. (b) Bicarbonate
circularity was not tested. (c) Raw PhysioNet files have no PaO2; the loader reconstructs it from saturation at 7.3% (Site A)
and 1.9% (Site B) of hours, an approximate inverse. We will remove the 8.6x figure unless we can reproduce it.

**W5 (task).** Detection in the input window, not prediction (shared comment, item 3). Post-onset hours are inside the window;
onset inside the window defines a positive. XGBoost: 200 trees, depth 4, learning rate 0.05, subsample 0.8, column subsample 0.8,
68 aggregate features (mean, min, max, observation rate of 17 variables), no tuning. ERM here is lambda 0 in the same code path.
Group DRO groups are the site id, one group with one source site; not re-run.

**W6 (second test).** We will remove it from the abstract and contributions; it is summarized without numbers. **W7.** Agreed on
precedent for the known lessons. Constraint availability by cohort is above; the derived-value proxy and the frequency of the
composition artifact across all cohorts were not run. We accept that MAP is an approximation and weaken the causal-mechanisms
argument for that constraint accordingly.

**Q7 (constraint randomization).** Not re-run after correction; your point that the ordering could come from target data in
pretraining and target-selected lambda is plausible, and we will say the ordering is untested under the corrected pipeline.
**Minor.** Running head, +4.7 / +4.2, the abstract's XGBoost / Group DRO sentence will be fixed; the composition-to-gap ratio will be
defined (ratio of the relative gap with only composition changed to the relative gap in the naive comparison, unstable when the
denominator is near zero).

---

## Rebuttal to R4 (7MQE)

Thank you. **Per-fix attribution; why a pipeline error would affect PCL more.** **[PENDING-ATTR.]** In the fixed pipeline there is
no detectable PCL-ERM difference at ten seeds (intervals about -6 to +3, -2 to +2, -3 to +3 points at the three sites), so an
asymmetric effect can neither be shown nor excluded beyond that width. **Seed variance.** SD 3.5-5.2 points (range 12-17) for
neural arms, about 1 point for XGBoost; paired-margin SD 3.6-5.6. We did not decompose initialization from the data split.
**Prediction horizon.** There is none: detection within the window (shared comment, item 3). **Training budgets.** Neural arms:
3.28 M parameters, 30 + 30 epochs, batch 64, about 7 minutes per arm on one V100; XGBoost: CPU, under a minute per seed, fixed
hyperparameters, no tuning. **Prior evidence.** Only the physiological argument and the original (flawed) comparison; we cannot
claim more. **Generic vs novel modes.** We cannot assert novelty and will frame the paper as an audit.
**AUPRC and calibration.** Now reported for every arm and site: e.g. ECE at Site B is 0.093 (ERM), 0.174 (PCL), 0.007 (XGBoost); at
eICU 0.113, 0.182, 0.068; at MIMIC-IV 0.146, 0.130, 0.282. Prevalence ranges from 2.1% to 30.9%, so AUPRC must be read against it.

---

## Rebuttal to R5 (fDtQ)

Thank you; the criticism is fair and we will restructure the paper.

**Hard to follow / writing.** We will rewrite around three findings (the task is detection; ten-seed paired evidence shows no benefit
of the constraint; lambda selection decides the sign) and cut the narrative. **[OWNER: disclose any AI assistance as the venue's
policy requires.]** **No figures.** The revision adds a per-seed AUROC figure, the lambda sweep (source-validation vs target
AUROC) and the horizon diagnostic. **PCL as a withdrawn strawman.** The revised abstract and introduction will state in the first
paragraph what PCL is, why it was plausible, and what is tested; the paper is an audit and we will say so. **Failure modes known;
overclaim from one dataset.** Agreed that leakage is known; the contribution is the measured size of each effect in one pipeline and
a result that changes the original claim. We will remove the claim that physiology cannot serve as a constraint and state the scope:
one architecture, one task, one source site, three external sites. **Anonymous repository.** It is being cleaned with commands to
regenerate each table and figure; **[OWNER: update the link only after review.]**
