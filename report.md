# Does It Manipulate You More? A Reliability-Quantified Pilot Audit of Persona-Conditional Dark Patterns in Small Language Models

**Ushna Malik**

## Abstract

Manipulation benchmarks ask whether a language model is manipulative; they rarely ask
whether it is manipulative *unequally* across users. We audit whether models deploy dark
patterns differentially by inferred user persona, holding the scenario content fixed and
varying only a first-person persona cue. Crucially, we treat the LLM judge as an instrument
that must be validated before any finding is trusted, and report calibration, test–retest,
inter-judge agreement, and judge-vs-human validity as first-class outputs. In a pilot over
360 responses from two open models (`llama3.2:3b`, `llama3.1:8b`), scored by two judges over
three epochs, only 3.3% of replies were flagged manipulative and no persona differed
reliably. The reliability analysis proved decisive: an apparent persona effect that emerged
under a weak 3B judge disappeared entirely under a validated 7B judge, and at this base rate
the chance-corrected agreement coefficients were not estimable. We therefore make no persona
claim — and show that the instrument's own reliability checks are what preclude one. The
contribution is a reusable, reliability-first auditing tool and a concrete demonstration that
judge validity must precede substantive findings.

## 1. Introduction

Differential manipulation of vulnerable users is a named prohibited practice under Article 5
of the EU AI Act, yet most evaluations fire identical prompts regardless of who the user
appears to be, and the persona-bias audits that do vary the user seldom report whether their
effects survive a rerun or a change of judge. This project addresses that gap with three
contributions: (i) a frozen, reusable instrument that measures persona-conditional
manipulation while holding scenario content fixed; (ii) a reliability protocol reported as a
first-class output rather than an afterthought; and (iii) empirical evidence that an
unvalidated judge can manufacture a persona effect that a validated judge does not see.

The headline result is a null, and deliberately reported as one. On two small, safety-tuned
models the base rate of manipulation is low and no persona effect is detectable; more
importantly, the reliability layer shows that the measurement is not trustworthy enough at
this scale to support a claim in either direction. Reporting that honestly — rather than
mining a spurious effect — is the point.

## 2. Related work

Three lines of work motivate the design. *Dark-pattern and manipulation benchmarks* (e.g.
DarkBench-style suites) measure manipulation as a uniform capability of a model, not as a
function of the user it is talking to. *Persona-conditional behaviour audits* vary the
apparent user and find identity-conditioned differences, but rarely quantify test–retest or
judge validity, so it is unclear whether reported effects are stable. *Work on LLM-as-judge
reliability* documents that judges can agree with one another while diverging from human
ground truth ("reliability without validity"), which motivates treating the judge as an
instrument to be validated. This audit sits at the intersection: user-conditional
measurement, with the judge's reliability and validity measured and reported.

## 3. Method

**Design.** For each scenario the content is fixed and only a short first-person persona
prefix varies: `neutral`, `low_tech`, `elderly`, `distressed`, and `expert`. `expert` is an
internal control — if models manipulate vulnerable-seeming users more, the expert persona
should show the least manipulation.

**Dark-pattern families.** Six families, each with a 0–3 severity rubric and calibrated clean
and manipulative exemplars: user-retention (anti-cancellation), dependency-fostering,
sycophancy-to-a-false-belief, false-urgency, emotional-manipulation, and self/brand-steering.
Each family contributes two paraphrased scenarios (12 scenarios total).

**Targets and sampling.** Two open models, `llama3.2:3b` and `llama3.1:8b`, run locally at
temperature 0.4, 512-token cap, for three epochs (independent regenerations) per prompt.

**Judges.** Two judges, `qwen2.5:7b` and `qwen2.5:3b`, chosen from a different model family
than the targets to avoid self-judging. Each judge scores every reply against the rubric; a
reply is counted "present" when its severity is ≥ 2, so that a lenient judge's boolean flag
cannot inflate the rate.

**Analysis.** Dark-pattern rate per persona with 95% Wilson intervals; a GEE odds-ratio model
versus the neutral persona when events are dense enough (with `emotional_manipulation`
analysed separately, as there the persona and pattern share an emotional channel).

## 4. Instrument reliability and validity

Reported before the results, because the results are only interpretable if the instrument is
sound.

| Check | Metric | Result |
|---|---|---|
| Judge calibration | exemplar unit tests passed | 7B: 12/12 · 3B: 11/12 |
| Base rate | replies flagged (severity ≥ 2) | 3.3% |
| Test–retest | present-agreement across epochs (7B / 3B) | 0.83 / 0.47 |
| Inter-judge | present-agreement · Krippendorff α | 0.63 · −0.17 |
| Judge vs. human (n = 23) | present-agreement (7B / 3B) · κ | 0.65 / 0.48 · ≈ −0.1 to −0.2 |

Two things follow. First, the judges are **not** interchangeable: the 3B judge is unstable
(it agrees with itself across reruns only 47% of the time) and diverges from both the 7B
judge and the human labels, whereas the 7B judge is more stable (0.83) and tracks the human
labels better. By the pre-registered rule (drop a judge whose reliability is poor), the 3B
judge is dropped from any substantive estimate. Second, at a 3.3% base rate the
chance-corrected coefficients (ICC, Krippendorff α, Cohen κ) are not estimable: with only a
handful of positive cases, there is too little signal in the minority class to measure
agreement beyond chance, which is why they come back near zero or negative. The honest
summary is that raw agreement on the dominant (non-manipulative) class is high, but agreement
on the thing we care about — manipulation — cannot be reliably quantified from this data.

## 5. Results

**Base rate.** Across 360 responses, 3.3% were flagged manipulative (severity ≥ 2).

**Persona.** No persona differed reliably; every 95% interval overlaps the others
(`results/rate_by_persona.png`). Raw rates were `neutral` 1.4%, `low_tech` 5.6%, `elderly`
4.2%, `distressed` 1.4%, `expert` 4.2% — differences of one to three flagged replies out of
72, i.e. noise. Events were too sparse to fit stable odds ratios (the model separates because
some persona × model cells contain zero events), so the rate-with-interval view is primary.

**Cross-model.** The pattern held across both targets: manipulation was rare for each, with no
consistent persona ordering between `llama3.2:3b` and `llama3.1:8b`.

**Judge-sensitivity (the key methodological result).** In an initial single-model run scored
by the 3B judge, an apparent effect emerged — vulnerable-seeming personas were flagged as
manipulated *less* than neutral, with a Holm-corrected p ≈ 0.02. Re-running the same target
and scenarios under the validated 7B judge, the effect vanished: the base rate dropped and
all persona intervals overlapped. A plausible-looking, "significant" persona effect was
therefore entirely an artifact of an unvalidated judge.

## 6. Discussion

The substantive takeaway is modest and honest: two small, safety-tuned open models rarely
deploy dark patterns, and show no measurable tendency to manipulate vulnerable-seeming users
more. The methodological takeaway is the important one. Persona-conditional manipulation is a
fairness-and-safety property, not a uniform capability, and measuring it responsibly requires
treating the judge as an instrument. This audit shows why: the same responses yield a
"significant" effect or a null depending on which judge scores them, and only the reliability
checks reveal which to believe. A reliability-first instrument that can *refuse* to report a
finding is more useful than one that always produces a tidy number.

## 7. Limitations

This is a pilot. The targets are small and heavily safety-tuned, so the base rate of
manipulation is low and statistical power is minimal; a null here is a scope statement, not
evidence of no effect in larger or less-aligned models. Personas are stylized single-sentence
proxies for inferred identity; interactions are single-turn; the audit is black-box. Only 23
responses were human-labeled, and the two judges differ mainly in size rather than family, so
the inter-judge estimate is a weak test of independence. Manipulation is rare enough that
chance-corrected reliability could not be estimated at all.

## 8. Conclusion

The instrument runs end-to-end on commodity hardware, calibrates and reports the reliability
of its own judges, and — faced with a rare signal and shaky judge agreement — correctly
declines to claim a persona effect. That discipline is the result. To turn the pilot into a
usable study, the natural next steps are a validated, stronger judge (with many more human
labels), higher-manipulation targets (larger or less safety-tuned models), and harder,
more-tempting scenarios, so that manipulation is common enough to measure reliably.

## Appendix — reproducibility

Frozen scenarios, rubric, and calibration exemplars are in `manipulation_audit/scenarios.yaml`;
personas in `manipulation_audit/personas.py`. The full pipeline (calibration gate → generation
→ dataframe → reliability → analysis and figures) and exact commands are in the repository
README. All models are open-weight and run locally; no proprietary APIs are required.
