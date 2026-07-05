# Stage 3 — Baseline run report

**Date:** 2026-06-26
**Stage:** 3 (baseline model — the floor Stage 4 must beat)
**Code:** working tree atop `2385e04` (`baseline.py` uncommitted at run time)
**Command:** `uv run --extra baseline python -m finance_mlops.baseline`
**Reproducible:** yes — deterministic split + `lbfgs` solver, no sampling, no seed needed.

---

## Model

- **Pipeline:** `TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True)` → `LogisticRegression(max_iter=1000, class_weight="balanced")`.
- **Classes:** 3 — `bullish` / `bearish` / `neutral`. (No `irrelevant`; that class is learned later from live RSS weak-labels, per design Stage 0/2.)
- **Features:** word unigrams + bigrams, dropping terms appearing in fewer than 2 docs, sublinear term-frequency scaling.
- **Why this model:** deliberately trivial. Its only job is to be a real, reproducible floor that the Stage 4 fine-tune has to clear — bag-of-words sentiment vocabulary transfers across the PhraseBank→RSS skew well enough for that.

## Data

- **Source:** Financial PhraseBank v1.0 (Malo et al. 2014), `data/phrasebank/` (gitignored).
- **Label mapping:** `positive→bullish`, `negative→bearish`, `neutral→neutral`.
- **Unit:** single sentence (proxy for the headline grain the live model serves).
- **Split (leak-free by construction):**
  - **Train = `75Agree \ AllAgree` → 1189 sentences.** 75Agree (~3453) is a *superset* of AllAgree, so the set-difference removes every eval sentence from training (`3453 − 2264 = 1189`).
  - **Eval = `AllAgree` → 2264 sentences.** The highest-agreement labels, frozen as the eval anchor; never trained on.
- **Note:** eval (2264) is larger than train (1189) by design — train on the noisier lower-agreement remainder, test on the cleanest set. Makes the score a conservative, honest floor.
- **Class balance (eval):** neutral 1391 (61%), bullish 570 (25%), bearish 303 (13%) — heavily skewed toward neutral, which drives the per-class results below.

---

## Results

```
              precision    recall  f1-score   support

     bearish      0.453     0.624     0.525       303
     bullish      0.574     0.482     0.524       570
     neutral      0.885     0.871     0.878      1391

    accuracy                          0.740      2264
   macro avg      0.638     0.659     0.642      2264
weighted avg      0.749     0.740     0.742      2264
```

Trained on 1189, evaluated on 2264. **3-class macro-F1 on AllAgree = 0.642.**

### How to read it

- **precision** — of the headlines the model called X, how many really were X.
- **recall** — of the truly X headlines, how many it caught.
- **f1** — harmonic mean of the two (one number per class).
- **support** — how many true examples of that class are in eval.
- **macro avg (0.642)** — unweighted mean of the three F1s; each class counts equally regardless of size. **This is the gate metric.**
- **weighted avg (0.742)** — mean weighted by support, so neutral dominates it.
- **accuracy (0.740)** — fraction of all 2264 predicted correctly.

### What the values actually say

- **neutral — P .885 / R .871 / F1 .878** → strong. Majority class and linguistically distinct (flat factual statements), so it's the easy one.
- **bullish — P .574 / R .482 / F1 .524** → weak recall: it misses over half of truly-bullish headlines, mostly filing them as neutral.
- **bearish — P .453 / R .624 / F1 .525** → opposite failure mode: decent recall (catches 62%) but poor precision (only 45% of its bearish calls are right). It over-predicts bearish.

That asymmetry is `class_weight="balanced"` at work: it up-weights the rare sentiment classes to stop them being ignored, which helps bearish recall but makes the model trigger-happy on bearish (false positives → low precision). A reasonable trade for a baseline.

The 10-point macro-vs-weighted gap (.642 vs .742) is the whole story in one number: it quantifies how much the two minority sentiment classes drag the average down once you stop letting neutral dominate. This is precisely why the design picked macro-F1 as the gate (so neutral can't mask weak sentiment detection) and added a separate bearish-F1 floor (the easiest class to quietly tank).

---

## What it means for the project

- **Floor locked:** gate metric (3-class macro-F1) = **0.642**, bearish-F1 = **0.525**. These are the two numbers Stage 4 must clear.
- **Healthy headroom:** TF-IDF + logreg on PhraseBank typically lands ~0.6–0.7 macro; FinBERT on AllAgree reaches ~0.85–0.95. So a 0.642 floor leaves clear room for the Stage 4 fine-tune to demonstrably earn its place — the baseline's entire purpose.
- **Caveat:** measured on clean, single-sentence PhraseBank. Live RSS headlines are messier and out-of-distribution, so 0.642 is the floor *on the eval anchor*, not a forecast of live performance.
