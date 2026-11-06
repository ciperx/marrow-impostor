# marrow-impostor

Continuous behavioural authentication from the input events an operating
system already sees. Decide whether the person at the pointer and keyboard is
the **enrolled user** — or an impostor — using nothing but:

* **pointer movement and clicks**, and
* **text as it arrives** (a physical keyboard, an on-screen keyboard, or text
  inserted by software such as dictation, *including the pattern of
  corrections*).

No camera. No microphone. No assumption about the input device. A session is a
device-neutral JSONL recording (`FORMAT.md`); the pipeline reduces it to a
fixed feature vector, fits a per-user Gaussian novelty detector on enrolment,
and scores every later session against it.

## One command

```
python3 run.py
```

That alone regenerates a deterministic fake dataset, trains a profile per input
style, fixes the decision threshold on a hold-out split, and prints the
evaluation. **Stdlib only — nothing to install.** (Python 3.9+.)

## The five requirements, and where each is met

| # | Requirement | Where |
|---|---|---|
| 1 | Open source, runs from a fresh copy with one command | `LICENSE` (MIT), `python3 run.py`, stdlib only |
| 2 | Fake data: two input styles, plus a tired owner, a stranger and a mimic | `marrow/__init__.py` `PERSONAS` + `generate_synthetic_data` |
| 3 | Swap test: a profile trained on someone else must not open for the owner | `run.py` `evaluate_style` (`swap`) |
| 4 | Report the FRR and FAR | `run.py` — `FAR`, `FRR (normal)`, `tired-recall` |
| 5 | Zero-knowledge of the user, OS-neutral format, offline train/test | `FORMAT.md`, `extract_features`, `fit_gaussian` |

## How it works

```
recording (JSONL)  ->  extract_features()  ->  13-dim vector
                                                  |
enrolment vectors  ->  fit_gaussian()  ->  mean + per-feature variance
                                                  |
hold-out vectors   ->  fixed threshold (max owner distance)
                                                  |
new session  ->  score() = Σ (xᵢ - μᵢ)²/σᵢ²  ->  accept if ≤ threshold
```

The novelty score is the squared Mahalanobis distance under a **diagonal**-
covariance Gaussian (a sum of squared z-scores). Diagonal is deliberate: with
a training set of a few dozen sessions, a full covariance matrix is too noisy
to invert reliably, while per-feature variance is stable and keeps every
distance interpretable ("this session was N standard deviations off on
features …").

### Features (see `extract_features` for exact definitions)

| ix | feature | what it measures (persona signature) |
|---|---|---|
| 0 | mean pointer velocity | pointer speed |
| 1 | peak pointer velocity | fastest move |
| 2 | mean pointer hesitation | dwell before a move |
| 3 | mean inter-key latency | typing cadence (within a burst only) |
| 4 | std inter-key latency | typing rhythm / steadiness |
| 5 | backspace rate | error-and-correct habit |
| 6 | mean typing-burst length | how long they type before pausing |
| 7 | mean inserted-chunk length | dictation chunking |
| 8 | std inserted-chunk length | dictation chunk-size steadiness |
| 9 | mean gap between insertions | dictation rhythm |
| 10 | std gap between insertions | dictation rhythm steadiness |
| 11 | click-to-move ratio | click behaviour |
| 12 | correction rate | on-screen edits per dictation insert |

Inter-key latency is measured **within a burst only** (gaps shorter than
`BURST_GAP_MS`), so the pointer travel between bursts does not contaminate the
keystroke rhythm — this is what lets the model catch the "mimic", who copies
the owner's means but is unnaturally steady.

### Fake personas (per style)

| style | roster |
|---|---|
| `mk` — mouse + physical keyboard | `mk_owner`, `mk_tired`, `mk_stranger`, `mk_mimic` |
| `pd` — slow pointer + dictation | `pd_owner`, `pd_tired`, `pd_stranger`, `pd_mimic` |

Per style: 36 enrol sessions, 12 hold-out (threshold), 12 owner test, and 60
each of tired / stranger / mimic. The `stranger` is a clearly different
person; the `mimic` copies the owner's *means* but with near-zero *variance*;
the `tired` owner is a moderate slowdown with more corrections.

## Evaluation (fixed seed, reproducible)

```
style   FAR   FRR(norm)  tired-recall  swap
mk      0.000  0.000      0.000         0/12   OK
pd      0.008  0.083      0.000         0/12   OK
```

* **FAR** — a stranger or mimic let in. This is the core job; ~0 means
  impostors are caught.
* **FRR (normal)** — the genuine owner wrongly rejected. ~0 means the owner
  is recognized reliably.
* **tired-recall** — the same, fatigued person. A strict lock doubts them;
  that is the honest, *safe* default (it errs toward locking).
* **swap** — a profile trained on the *stranger* opening the owner; 0 means
  profiles are person-specific (requirement 3).

The threshold is fixed on the hold-out split **before** any test is scored, so
every number is out-of-sample.

## Honesty note

This evaluates the **method** on openly synthetic data, not on real users, and
makes no anthropometric claim. The simulator exists only to make the pipeline
run end-to-end and to demonstrate (a) that impostors separate from the owner,
(b) that a mimic who copies means but not variance is still caught, and (c) that
the swap test holds. A production system would enrol on real, consenting input
sessions and calibrate the threshold on much more data.

## Layout

```
run.py            one-command evaluation driver
marrow/__init__.py  format, features, model, synthetic-data generator
FORMAT.md         the OS-neutral recording format
data/             generated fake recordings (JSONL + index.json)
```