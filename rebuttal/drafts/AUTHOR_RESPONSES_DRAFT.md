# Author responses, draft v3 (not for posting until the [OWNER] items are resolved)

Posting plan on the submission page: one paper-level Official Comment with the shared evidence (below), then one Rebuttal per
review. Every number comes from `rebuttal/pack/RESULTS.md` and the tables it cites. Two `[OWNER]` items need your decision:
(1) the generative-AI disclosure wording, (2) the wording about the concurrent submission. Nothing else is pending.
Each block is kept under about 5,000 characters.

---

## Official Comment (paper level)

Thank you to all five reviewers. Most of the criticism of the statistics, the task definition and the missing details was correct,
and checking it turned up problems in the submission that we report here instead of defending. We ran new experiments and audits.

**1. Ten seeds with paired intervals.** We re-ran ERM (lambda = 0) and PCL (lambda = 0.5) on the corrected pipeline with ten seeds
(42-44 are the submission's three; 45-51 are new), identical schedules, and a fixed-hyperparameter XGBoost on the same Site A split.
Target AUROC, mean (SD over seeds):

| | Site B | MIMIC-IV | eICU-CRD |
|---|---|---|---|
| ERM | 0.689 (0.042) | 0.642 (0.035) | 0.594 (0.042) |
| PCL | 0.662 (0.052) | 0.642 (0.047) | 0.591 (0.050) |
| XGBoost | 0.731 (0.012) | 0.646 (0.010) | 0.661 (0.009) |

Paired PCL minus ERM (95% bootstrap over seeds, then stays): -0.026 [-0.063, +0.013], +0.001 [-0.021, +0.020], -0.003
[-0.031, +0.027]. Every interval contains zero. The submission's statement that PCL falls below ERM at every site is therefore not
supported. What the data support is: no evidence that the constraints help, and no evidence that they hurt. XGBoost is higher than ERM at
eICU (+0.067, [+0.039, +0.093]) and higher than PCL at Site B (+0.068, [+0.021, +0.115]) and eICU (+0.070, [+0.042, +0.099]); at
MIMIC-IV the three are indistinguishable. A neural arm's seed SD is 3.5 to 5.2 points (min-max range 12 to 17), as large as the
effects the three-seed comparison argued about.

**2. Lambda.** With all six lambda values run on seeds 42-46, choosing lambda by source-validation AUROC (the only rule that uses
no target data) gives PCL minus ERM of +0.008, +0.019, +0.016 at the three sites; the intervals contain zero (two lower bounds are
0.000). Choosing lambda on target data, which is an oracle and not a deployable method, gives +0.039 [+0.023, +0.057] at eICU; that
interval ignores the selection. Source-validation AUROC falls from 0.817 (lambda 0) to 0.769 (lambda 5), so it does not favor
large lambda.

**3. The task is event detection inside the input window.** The label is "any positive hourly label in the first 48 hours", and the
model reads the same 48 hours. In a diagnostic of the existing models, we cut each positive stay's input so that it ends 0, 6 or 12
hours before its first positive label (negatives are cut to lengths drawn from the positives' cut lengths). Site B AUROC falls from
0.689 (ERM) and 0.662 (PCL) to 0.54-0.57, close to chance, and PCL minus ERM stays indistinguishable from zero. We will describe
the task as detection and remove forecasting language. We did not train a model on the cut inputs.

**4. One-factor-at-a-time checks (five seeds).** Flipping one factor of the corrected pipeline back (Severinghaus term removed, 9 instead of 17
variables, mask rate 0.5 with warmup 0.3, whole-timestep masking) did not restore a PCL advantage in any variant (largest margin +0.022,
[-0.015, +0.051]). We could not test the label definition or all-site pretraining, the factors most likely behind the original gain.

**5. Corrections we will make.** (a) "Exceeds 1.0 on four of five site pairs": the five values come from detector validation
scenarios, not site pairs. (b) "+4.7 smallest single-seed gap" is inconsistent with "+4.2 source gain". (c) The abstract's XGBoost
statement ignores Group DRO, which we did not re-run (with one source site it has one group). (d) The constraint mask is built after
a 6-hour forward fill, not on the raw record: Henderson-Hasselbalch inputs are jointly present in 3.0% of Site A hours raw and 27.7%
after fill (1.8% and 17.0% at MIMIC-IV; 0.55% and 7.2% at eICU). (e) The optimized loss includes a loss-balancing factor that the
paper does not describe. (f) PaO2 is reconstructed from saturation at the PhysioNet sites and measured at MIMIC-IV and eICU.
(g) From the surviving files we cannot reconstruct how the submitted PCL row's lambda was chosen; the revision reports three explicit
rules.

**6. Not done in this period:** reproducing the original pipeline, re-scoring against the old ICD labels (they no longer exist),
all-site pretraining, Group DRO, a decomposition of seed variance, the gradient-level redundancy check between the constraint and the
masked loss. These become stated limitations.

---

## Rebuttal to R1 (xM3J)

Thank you for the specific questions; several changed the paper.

**Known lessons.** We agree the individual lessons have precedent and will stop presenting them as new. The contribution is a
quantified case study: one concrete pipeline, ten seeds, paired intervals, and a previously reported gain that does not survive.

**Lesson 4 (imputed PaO2).** If the constraint were exact, an imputed channel would match the measured one, as you say. In our code
it does not: PaO2 is absent from the raw PhysioNet files, and our loader reconstructs it from SaO2 with an approximate inverse that
drops one term of the loss's own forward function (every PaO2 value at the two PhysioNet sites, 7.3% of Site A hours and 1.9% of
Site B hours), whereas at MIMIC-IV and eICU PaO2 is a measured blood-gas value. So the term carries different information at source
and target. We then removed the Severinghaus term and re-ran ERM and PCL (five seeds): PCL minus ERM was +0.007, +0.014, +0.022 at the
three external sites against -0.006, +0.017, +0.022 for the full loss on the same seeds, with intervals about +-0.06 wide. We cannot
detect an effect of that term, so we do not claim it caused the failure.

**Concurrent submission.** The passage describes a selection criterion computed on unlabeled target inputs. That uses target data, so
it is a transductive setting and does not meet the rule in our lesson 3; you are right that praising it is inconsistent. We will label
it transductive, remove it from the abstract and contributions, and not use it as supporting evidence. **[OWNER: confirm wording.]**

**Novelty and transfer.** We do not claim the constraints are novel. Transfer is untested: one architecture, one task, one source
site, three external sites. We state this as a limitation.

**Writing.** We will rewrite the prose and cut the flagged lines. **[OWNER: add the generative-AI disclosure that the venue's policy
requires; do not state that none was used unless that is true.]**

**Masked variables and double penalization.** A constraint is applied where all its variables are present after forward fill and at
least one is masked, on the model's predictions for all variables (`pcl_loss.py`, `train_utils.py`). "At least two" is not required,
so your concern applies. With masking independent at 30%, among positions where a constraint is active, exactly one of its variables
is masked at 67% of positions (MAP, Henderson-Hasselbalch) and 82% (Severinghaus), the same in every cohort we simulated. Whether the
penalty is then redundant with the masked loss is not measured; we say so.

**Timestep masking.** We tested it (30% of timesteps masked entirely, ERM and PCL, five seeds). PCL minus ERM was -0.054, -0.034,
-0.039 at Site B, MIMIC-IV and eICU (eICU [-0.072, -0.011], below zero). The hypothesis that it lets the constraint help is not
supported.

**How often it triggers.** Share of stay-hours at which a penalty is active in a masking draw (MAP, Henderson-Hasselbalch,
Severinghaus): Site A 50.6%, 15.9%, 11.6%; Site B 58.3%, 0.34%, 3.9%; MIMIC-IV 59.2%, 10.9%, 11.9%; eICU 60.1%, 4.8%, 6.7%. These
use forward-filled inputs: the Henderson-Hasselbalch inputs are jointly present in only 1.8% (MIMIC-IV) and 0.55% (eICU) of hours before
filling.

**L1 vs squared error.** We found no recorded rationale in the code or its history (L1 for MAP, squared error for the other two); we
will call it an unexamined choice. **Lambda.** The PCL rows use 0.5; the sweep and three selection rules are in the shared comment.
**SOFA.** Sepsis-3 labels are suspected infection (antibiotic-culture pairing at MIMIC-IV; a culture or an infection diagnosis at eICU)
plus a single-point SOFA of at least 2 at the suspicion time, "ever in the ICU stay", broadcast over the 48 hours. PhysioNet uses the
challenge's hourly label in the first 48 hours, so a definitional difference remains; we say so.
**Why confound 1 favors PCL; fixing confound 4.** We could not test the label factor (the ICD-derived labels no longer exist), so we
will not claim a mechanism. For confound 4, dropping the term had no detectable effect (above); the revision reports that it was tried.

---

## Rebuttal to R2 (JHNu)

Thank you. **One architecture, one task.** Correct; we state it as a limitation. **Three seeds.** Now ten; a neural arm's seed SD is
3.5 to 5.2 AUROC points, so three seeds could not support the original claims. The paired intervals are in the shared comment.
**No new algorithm.** The paper is an evaluation, and we will say so.
**PINN shift and key assumptions.** The assumptions that matter here, and our check of each: the constraint inputs must be present (they are
sparse, see the availability numbers); PaO2 must mean the same thing at every site (it does not); the penalty must add something
beyond the masked loss (not measured); and MAP = DBP + (SBP - DBP)/3 is an approximation, not an identity (its residual by cohort was
not measured). We will shorten the PINN analogy to a motivation. **Sites.** There are three external sites (Site B, MIMIC-IV, eICU-CRD)
and all three are in every table; the text will say so. **Writing.** We will fix the spelling and the repetition.

---

## Rebuttal to R3 (xCda)

Thank you; this review improved the paper. Numbering follows your weaknesses (W) and questions (Q).

**W1, Q1, Q2 (attribution).** You are right that "five confounds explain the gain" was not shown. We now flip one factor at a time
on the corrected pipeline (five seeds; PCL minus ERM per site, ERM retrained per variant): Severinghaus term off, 9 instead of 17
variables, mask rate 0.5 with warmup 0.3, whole-timestep masking. None restores a PCL advantage (largest margin +0.022, [-0.015,
+0.051]). Two of twelve cells meet the rule "change in margin exceeds the seed SD of the margin and its interval excludes zero", both
making PCL worse (9 variables at MIMIC-IV, -0.046 [-0.079, -0.012]; timestep masking at eICU, -0.061 [-0.116, -0.002]); with twelve
comparisons we read little into them. So the 9-to-17 variable expansion did not remove the gain. Re-scoring on the ICD labels is not
possible (they no longer exist) and all-site pretraining was not re-run. Confound 5 concerns the violation score and cannot inflate
prediction AUROC; we will say so. At Site B, where ICD labels were never used, we cannot isolate which other factor moved the margin.

**W2, Q3 (statistics).** Per-seed values, SDs (not ranges) and paired intervals are reported. Under the rule "worse" only if the
interval lies below zero over ten seeds, none qualifies; the paper will say "no evidence of benefit". We could not trace "4 to 7 points"
to per-seed files. On ten seeds a neural arm's SD is 3.5 to 5.2 points (range 12 to 17); XGBoost's is 0.9 to 1.2 (range 2.9 to 4.3). We
did not separate initialization from the Site A split. The 3.3-point Henderson-Hasselbalch ablation cost was not re-tested and will not be
interpreted.

**W3, Q4 (lambda).** Target AUROC for lambda in {0, 0.1, 0.5, 1, 2, 5} (seeds 42-46, mean (SD)); lambda 0 is ERM. Site B: 0.674
(0.039), 0.654 (0.075), 0.668 (0.066), 0.627 (0.056), 0.669 (0.052), 0.673 (0.028). MIMIC-IV: 0.628, 0.639, 0.644, 0.620, 0.637,
0.636. eICU: 0.570, 0.581, 0.592, 0.571, 0.605, 0.609. Source-validation selection picks lambda 0 or 0.1 in four of five seeds, as you
suspected; selecting on target data gives a gain at eICU only (shared comment, item 2).

**W4, Q5, Q6 (labels, circularity).** (a) Site A and B labels are the challenge's own hourly labels cut to the first 48 hours; MIMIC-IV
and eICU labels are our Sepsis-3 derivation, so some definitional shift remains that we cannot quantify. Cohorts: Site A, 20,336 records
minus 4,499 under 24 hours minus 5,456 with no SBP/DBP/MAP hour = 10,381 (340 positive, 3.28%); Site B, 20,000 minus 5,176 under 24
hours, 2 empty and 43 without that hour = 14,779 (304 positive, 2.06%). In the same files 8.8% and 5.7% of records are ever septic, but
only 5.6% and 3.7% have a positive hour in the first 48 rows. (b) PhysioNet has a raw `HCO3` column whose source the files do not
document, so we cannot say whether it is calculated; at MIMIC-IV it is serum chemistry (item 50882), not the blood-gas calculated value,
and at eICU the chemistry `bicarbonate`. A violation-rate split by source was not run. (c) Our loader, not the PhysioNet release,
reconstructs PaO2 (the paper's sentence saying otherwise is wrong): from SaO2, for every PaO2 value at both PhysioNet sites (7.3% of
Site A and 1.9% of Site B hours), with an approximate inverse. The Severinghaus term compares SpO2 (pulse oximetry) with a function of
that PaO2, so it is not an exact tautology but is close to one there. We will remove the 8.6x figure unless we can reproduce it.

**W5 (task and baselines).** Detection inside the window (shared comment, item 3); post-onset hours are inside the input and a stay is
positive if its onset falls inside the window. XGBoost: 200 trees, depth 4, learning rate 0.05, subsample 0.8, column subsample 0.8,
68 aggregate features (mean, min, max, observation rate of 17 variables), no tuning, same 80% Site A split. ERM here is lambda 0 in the
same code path. Positive-class weighting is applied to ERM and PCL fine-tuning too, so it is not specific to Group DRO. Group DRO
groups are the site id, so with one source site there is one group; we did not re-run it.

**W6, W7.** We will remove the second test from the abstract and contributions. We agree the known lessons have precedent and that MAP
is an approximation, and weaken the causal-mechanisms argument for it. The frequency of derived-value and composition effects across
all cohorts was not measured.

**Q7 (constraint randomization).** Not re-run after correction. Your explanation is plausible (target data in PCL pretraining and a
target-selected lambda), so we will state the ordering is untested under the corrected pipeline.
**Minor.** We will fix the running head, the +4.7 / +4.2 sentence and the abstract's XGBoost / Group DRO statement, and define the
composition-to-gap ratio: the relative gap with only composition changed divided by the relative gap in the naive comparison. It is
unstable when the denominator is near zero.

---

## Rebuttal to R4 (7MQE)

Thank you. **Per-fix attribution.** One-factor-at-a-time results are in the response to R3 (W1): none of the four factors we could test
restores a PCL advantage; label definition and all-site pretraining could not be tested. **Why a pipeline error would affect PCL more.**
In the fixed pipeline there is no detectable PCL-ERM difference at ten seeds (intervals about -6 to +1 points at Site B, -2 to +2 at
MIMIC-IV, -3 to +3 at eICU), so an asymmetric effect can be neither shown nor excluded beyond that width.
**Seed variance.** SD 3.5 to 5.2 points (range 12 to 17) for neural arms, about 1 point for XGBoost; the SD of the paired margin is 3.6 to
5.6. We did not separate initialization from the split. **Prediction horizon.** There is none: the task is detection inside the window
(shared comment, item 3). **Training budgets.** Neural arms: 3.28 M parameters, 30 pretraining plus 30 fine-tuning epochs, batch 64, about
7 minutes per arm on one V100; XGBoost: CPU, under a minute per seed, fixed hyperparameters. **Prior evidence.** Only the physiological
argument and the original, flawed comparison. **Novel versus generic modes.** We cannot assert novelty and will frame the paper as an
audit. **AUPRC and calibration.** Now reported for every arm and site. ECE (10 bins) at Site B is 0.093 (ERM), 0.174 (PCL), 0.007
(XGBoost); at eICU 0.113, 0.182, 0.068; at MIMIC-IV 0.146, 0.130, 0.282. Prevalence ranges from 2.1% to 30.9%, so AUPRC must be read
against it.

---

## Rebuttal to R5 (fDtQ)

Thank you; the criticism is fair and we will restructure the paper.

**Writing.** We will rewrite around three findings (the task is detection; ten-seed evidence shows no benefit of the constraint; lambda
selection decides the sign) and cut the narrative. **[OWNER: AI-use disclosure as above.]** **Figures.** The revision adds a per-seed AUROC
figure, the lambda sweep (source-validation versus target AUROC), the one-factor attribution figure and the horizon diagnostic.
**PCL as a withdrawn baseline.** The first paragraph will say what PCL is, why it looked plausible and what we test; the paper is an
audit, and we will say that. **Known failure modes; one dataset.** Leakage is known; what we add is the measured size of each effect in one
pipeline and a result that overturns the original claim. We will drop the sentence that physiology cannot serve as a constraint and state
the scope: one architecture, one task, one source site, three external sites. **Repository.** It is being cleaned, with commands that
regenerate each table and figure. **[OWNER: update the link only after review.]**
