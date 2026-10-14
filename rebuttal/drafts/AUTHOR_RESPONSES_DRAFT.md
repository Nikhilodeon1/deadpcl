# DRAFT author responses (not for posting until the [PENDING] items are filled and the owner has edited)

Source for every number: `rebuttal/pack/RESULTS.md` and the tables it points to. Items marked
**[PENDING-ATTR]** wait for the attribution variants (no-Severinghaus, timestep masking, 9 variables, original
hyperparameters); items marked **[OWNER]** are decisions that are not mine to make. Each block below is written to fit a
single review-system comment (about 5,000 characters or fewer). No author, institution or venue names appear here.

---

## Comment to all reviewers (general response)

We thank the reviewers. Every concern about statistics, task definition, metrics and missing details was fair, and several
exposed real problems in the submission. We ran a new set of experiments and audits; here is what changed.

**1. New main result (10 seeds, paired intervals).** We re-ran ERM (lambda = 0) and PCL (lambda = 0.5) on the corrected
pipeline with ten seeds (the original three plus seven new), identical schedules, and a fixed-hyperparameter XGBoost on
the same split. Target AUROC, mean (SD over seeds):

| | Site B | MIMIC-IV | eICU-CRD |
|---|---|---|---|
| ERM | 0.689 (0.042) | 0.642 (0.035) | 0.594 (0.042) |
| PCL | 0.662 (0.052) | 0.642 (0.047) | 0.591 (0.050) |
| XGBoost | 0.731 (0.012) | 0.646 (0.010) | 0.661 (0.009) |

Paired PCL minus ERM (95% two-level bootstrap over seeds and stays): -0.026 [-0.063, +0.013] (Site B), +0.001
[-0.021, +0.020] (MIMIC-IV), -0.003 [-0.031, +0.027] (eICU). **Every interval contains zero.** The submission's statement
that PCL falls below ERM at every site is therefore not supported; the supported statement is "no evidence that the
constraints help, and no evidence that they hurt". XGBoost is higher than both neural methods at eICU (+0.067 over ERM,
[+0.039, +0.093]) and higher than PCL at Site B (+0.068, [+0.021, +0.115]); at MIMIC-IV the three are indistinguishable.
The seed SD of a neural arm is 3.5 to 5.2 AUROC points (min-max range 12 to 17 points), which is as large as the effects
the original three-seed comparison argued about.

**2. Lambda selection decides the sign.** On seeds 42-46 (all six lambda values run), choosing lambda by source-validation
AUROC gives PCL minus ERM of +0.008 / +0.019 / +0.016 at the three sites, with intervals that include zero (lower bounds at
0.000 for MIMIC-IV and eICU). Choosing lambda on the target site (an oracle, not a deployable method) gives +0.039
[+0.023, +0.057] at eICU. We report all rows; the oracle is labeled as such, and its interval ignores the selection. Source
validation AUROC itself falls from 0.817 at lambda 0 to 0.769 at lambda 5, so source validation does not favor large lambda.

**3. The task is detection inside the input window, not prediction (R4 Q4, R3 C5).** The per-stay label is "any positive
hourly label in the first 48 hours" and the model reads the same 48 hours. In a diagnostic of the existing models, cutting
each positive stay's input so that it ends 0, 6 or 12 hours before the first positive label lowers Site B AUROC from 0.689
(ERM) / 0.662 (PCL) to 0.54 to 0.57 (close to chance), and PCL minus ERM stays indistinguishable from zero. We will state the
task as event detection and remove language that implies forecasting. (We did not train a model on the cut inputs.)

**4. Corrections to the submission that we will make.** (a) The sentence that a ratio "exceeds 1.0 on four of five site
pairs" is wrong: the five values come from detector validation scenarios, not site pairs. (b) "+4.7 smallest single-seed
gap" is inconsistent with "+4.2 source gain". (c) The abstract's claim that XGBoost beats both neural methods ignores Group
DRO; Group DRO was not re-run here (with one source site it has a single group). (d) The constraint-availability mask is
computed after a 6-hour forward fill, not on the raw record: the Henderson-Hasselbalch term is computable in 3.0% of Site A
hours raw but 27.7% after fill (1.8% vs 17.0% at MIMIC-IV, 0.55% vs 7.2% at eICU). (e) The optimized loss includes a
loss-balancing factor not described in the paper. (f) PaO2 is reconstructed from saturation at the two PhysioNet sites (an
approximate inverse) and measured at MIMIC-IV and eICU. (g) We cannot reconstruct from the surviving files which lambda was
chosen how for the submission's PCL row; the new tables report three explicit rules.

**5. What we did not do.** Reproduction of the original pipeline, the full attribution ladder, Group DRO, constraint-activity
statistics, a decomposition of seed variance, and label-free selectors based on reconstruction error were not run in this
period. They are listed as limitations. Code, per-seed tables and per-stay predictions (hashed ids) are prepared for the
anonymous repository **[OWNER: confirm before pushing]**.

---

## R1 (xM3J)

We agree the paper must be more careful about novelty and scope, and we appreciate the specific questions.

**M1 (lessons known).** We do not claim the individual lessons are new. The contribution is a quantified audit of one
concrete pipeline with ten seeds and paired intervals, and the finding that the previously reported effect does not survive it.

**M2 (confound 4).** The new audit shows the premise "if the constraint is valid, the imputed channel equals the measured one"
does not hold exactly here: PaO2 is absent from the raw PhysioNet files and is reconstructed from saturation with an
approximate inverse of the loss's own forward function (7.3% of Site A hours, 1.9% of Site B hours), while at MIMIC-IV and
eICU it is a measured blood-gas value. **[PENDING-ATTR: no-Severinghaus result and interval, margin change vs main.]**

**M3 (concurrent submission).** The reviewer is right that praising a method that uses target data conflicts with our
lesson 3. We will rewrite it: methods that use unlabeled target data are a transductive setting, not zero-shot, and we say so.
**[OWNER: wording]**

**M4, M5.** Novelty of the constraint is not claimed; transfer to other settings is untested (one architecture, one task,
three external sites). We state this as a limitation.

**M7 (at least two masked variables).** In the implementation a constraint is applied at a position when all its variables are
present after forward fill and at least one of them is masked; the penalty is computed on the model's predictions for all
variables, not observed values (`pcl_loss.py`, `train_utils.py`). The condition "at least two" is not implemented, so whether
the penalty is redundant with the masked loss when exactly one variable is masked is an empirical question that we have not
measured. We say so rather than assert either answer.

**M8 (timestep masking).** **[PENDING-ATTR: ERM and PCL with whole-timestep masking, five seeds, paired margin.]**

**M9 (how often it triggers).** Share of stay-hours in which the constraint inputs are jointly present (before / after forward
fill): Henderson-Hasselbalch 1.8% / 17.0% (MIMIC-IV), 0.55% / 7.2% (eICU), 3.0% / 27.7% (Site A), 0.19% / 0.56% (Site B);
Severinghaus 5.1% / 23.7% (MIMIC-IV), 2.4% / 13.2% (eICU), 7.0% / 26.1% (Site A), 1.7% / 8.8% (Site B). Activity at masked
positions by number of masked variables was not computed.

**M10 (L1 vs squared).** We found no recorded rationale in the code or surviving history (L1 for the MAP term, squared error
for the other two). We will say it is an unexamined design choice.

**M11 (final lambda).** The PCL rows use lambda = 0.5. The full sweep, per-seed selection and oracle are reported (see the
general response).

**M12 (SOFA).** Labels follow Sepsis-3: suspected infection (antibiotic-culture pairing at MIMIC-IV; a culture or an infection
diagnosis at eICU) with a single-point SOFA score at the suspicion time of at least 2, "ever in the ICU stay", broadcast over the
48 hours. We will spell this out; the sepsis label at PhysioNet is the challenge's hourly label within the first 48 hours, so a
residual definitional difference remains and we state it.

**M13, M14.** **[PENDING-ATTR: effect of each factor on the PCL minus ERM margin; no-Severinghaus variant.]**

---

## R2 (JHNu)

**J1 (one architecture, one task).** Correct; we state it as a limitation and have not tested other architectures or tasks.

**J2 (three seeds).** Ten seeds now. Seed SD is 3.5 to 5.2 points per neural arm, so three seeds could not support the original
claims; the paired intervals in the general response are the corrected evidence.

**J4, J5 (PINN framing, assumptions).** The assumptions that matter in practice and that we checked: the constraint inputs must be
present (they are sparse, and the availability above shows the loss acts on few raw hours), PaO2 must mean the same thing across
sites (it does not, see M2), and the penalty must not duplicate the masked loss (not measured). We reduce the PINN analogy to a
one-sentence motivation. The MAP identity is an approximation, not an exact equality; we did not measure its residual by cohort.

**J6.** There are three external sites (Site B, MIMIC-IV, eICU-CRD); all three appear in every table. We will make this explicit.

---

## R3 (xCda)

**C1 (attribution).** **[PENDING-ATTR: leave-one-in table for: Severinghaus term off, 9 vs 17 variables, original mask rate and
warmup, timestep masking; paired margin change per site, with the pre-registered rule "explanatory if the change exceeds the seed
SD of the margin and its interval excludes zero".]** Not run: label definition (the ICD-derived labels no longer exist) and
all-site pretraining (the cohorts are not on the training machine).

**C2 (statistics).** Per-seed values are in the new per-seed table; we report SD (not range); paired differences with 95%
intervals are above. Under our own rule a "worse" claim needs an interval entirely below zero over ten seeds; none of the
PCL minus ERM intervals qualifies, so we now say "no evidence of benefit". The 3.3-point Henderson-Hasselbalch ablation cost was
not re-tested and we will not interpret it.

**C3 (every lambda).** Target AUROC for lambda in {0, 0.1, 0.5, 1, 2, 5} (seeds 42-46; mean (SD)): Site B 0.674 (0.039), 0.654
(0.075), 0.668 (0.066), 0.627 (0.056), 0.669 (0.052), 0.673 (0.028); MIMIC-IV 0.628, 0.639, 0.644, 0.620, 0.637, 0.636; eICU 0.570,
0.581, 0.592, 0.571, 0.605, 0.609. ERM (lambda 0) is the first column. Lambda 0 and 0.5 also have ten seeds in the main table.

**C4.** (a) Site A/B labels are the challenge's own hourly labels cut to the first 48 hours. The prevalence in kept stays is 3.28%
and 2.06% (reproduced); in the same files 8.8% / 5.7% of records are ever septic and 5.6% / 3.7% have a positive hour in the first
48 rows; the 24-hour and hemodynamic filters remove the rest. (b) Circularity of Henderson-Hasselbalch through calculated
bicarbonate was not tested (we did not compute a near-zero-residual proxy). (c) Raw PhysioNet files have no PaO2; the repository's
loader reconstructs it from saturation at 7.3% (Site A) and 1.9% (Site B) of hours. The "8.6x" figure is carried from the
submission and was not re-derived; we will remove it unless we can reproduce it.

**C5.** Task: see general response. XGBoost: fixed 200 trees, depth 4, learning rate 0.05, subsample 0.8, column subsample 0.8,
68 aggregate features (mean, min, max, observation rate of 17 variables), no tuning, trained on the same 80% Site A split. ERM
pretraining here is lambda 0 in the same code path (the submission's ERM routine differed in gradient clipping). Group DRO groups
are the site id, so with one source site it reduces to a reweighted ERM; we did not re-run it.

**C6, C7, C8.** Second test: the sweep and a source-validation selector are now in the paper; selection by reconstruction error or
constraint violation on unlabeled target inputs was not run (and would be transductive). Per-cohort availability is given above;
the derived-value proxy and constraint randomization after correction were not run.

**Minor.** The running head, the +4.7 / +4.2 inconsistency and the abstract's XGBoost / Group DRO statement will be fixed; the
composition-to-gap ratio is defined in the appendix as the ratio of the relative gap with only composition changed to the relative
gap in the naive comparison, which is unstable when the denominator is near zero.

---

## R4 (7MQE)

**Q1, Q2.** **[PENDING-ATTR.]** Under one fixed pipeline there is no detectable PCL-ERM difference at ten seeds (intervals
about -6 to +3 points at Site B, -2 to +2 at MIMIC-IV, -3 to +3 at eICU), so an asymmetric effect can neither be shown nor excluded beyond that width.

**Q3 (seed variance).** SD 3.5 to 5.2 points and range 12 to 17 points for neural arms, about 1 point (range 3 to 4) for XGBoost;
the SD of the paired margin is 3.6 to 5.6 points. The seed sets both initialization and the Site A split; we did not separate them.
The submission's "4 to 7 points" was not traceable to per-seed files.

**Q4.** See general response, point 3.

**Q5 (budgets).** Neural arms: 3.28 M parameters, 30 pretraining + 30 fine-tuning epochs, batch size 64, about 7 minutes per arm on
one V100 (40 main arms, 4.6 GPU-hours; the attribution arms add about 4.8 more). XGBoost: CPU, under a minute per seed, no tuning.

**Q6, Q7.** We have no prior evidence beyond the physiological argument for PCL, and we cannot assert which failure modes are novel
rather than generic; we will frame the paper as an audit.

**Q8 (AUPRC, calibration).** AUPRC / Brier / ECE (10 bins) for ERM, PCL, XGBoost at each site are in the new table. Prevalence
differs widely (2.1% to 30.9%), so AUPRC must be read against it. PCL is worse calibrated than ERM at Site B (ECE 0.174 vs 0.093)
and eICU (0.182 vs 0.113). XGBoost is best calibrated at Site B (0.007) and worst at MIMIC-IV (0.282).

---

## R5 (fDtQ)

**D1 (writing).** We will restructure around three findings (task is detection; no evidence of benefit with ten seeds; lambda
selection decides the sign) and shorten the narrative. **D2 (figures).** The revision adds a per-seed AUROC figure, the lambda
sweep (source-validation vs target AUROC), and the horizon diagnostic. **D3, D4.** The submission is an audit of a pipeline, not
a defense of PCL; known failure modes (leakage, label shift, shortcut variables) are acknowledged, and our contribution is the
quantification on one case. **D5.** We now state the single-task, single-architecture scope. **D6 (repository).** The repository is
being cleaned (README with commands to regenerate every table and figure, fixed seeds, no identifying strings); we will update the
anonymous link **[OWNER: confirm before pushing]**.
