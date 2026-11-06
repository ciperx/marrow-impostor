"""marrow-impostor — continuous behavioral authentication from OS input events.

Decides whether the person at the pointer/keyboard is the ENROLLED user, using
only the input events the operating system already sees:

  * pointer movement and clicks
  * text as it arrives (a physical keyboard, an on-screen keyboard, or text
    inserted by software such as dictation, including the pattern of corrections)

No audio, no camera, and no assumption about the input device. A session is a
device-neutral JSONL recording described in FORMAT.md. This module provides:

  * format   — load / save a recording (JSONL, OS-neutral)
  * features — reduce a session to a fixed-length feature vector
  * model    — Gaussian novelty detector with a fixed, pre-declared threshold
  * synthetic— deterministic fake-data generator for evaluation only

Stdlib only (Python 3.9+). No third-party dependencies.
"""

import json
import math
import os
import random

# --------------------------------------------------------------------------- #
# FORMAT
# --------------------------------------------------------------------------- #

EVENT_KINDS = ("pointer", "click", "text")
TEXT_SOURCES = ("typed", "onscreen", "inserted")


def load_session(path):
    """Read a JSONL recording file into a list of event dicts, sorted by `t`."""
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    events.sort(key=lambda e: e["t"])
    return events


def save_session(events, path):
    """Write events as a JSONL recording file (one compact JSON object per line)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, separators=(",", ": "), ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# SMALL NUMERIC HELPERS (stdlib)
# --------------------------------------------------------------------------- #

def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def std(vals):
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))


# --------------------------------------------------------------------------- #
# FEATURES — a session becomes a fixed-length numeric vector
# --------------------------------------------------------------------------- #

# Pause (ms) that splits typing into bursts.
BURST_GAP_MS = 600.0


def extract_features(events):
    """Reduce a session to a fixed-length list of floats (13 features).

    Every feature maps to a stable behavioural signature: pointer speed and
    hesitation, typing cadence and rhythm (within-burst only), correction rate,
    typing burst length, dictation chunking, and click-to-move ratio. The exact
    definitions and indices are documented in README.md.
    """
    moves = []       # (x, y, t)
    clicks = []      # t
    typed = []       # (t, is_backspace)
    inserted = []    # (length, t)
    onscreen = []    # (length, t)

    for e in events:
        kind = e.get("kind")
        if kind == "pointer":
            moves.append((float(e["x"]), float(e["y"]), float(e["t"])))
        elif kind == "click":
            clicks.append(float(e["t"]))
        elif kind == "text":
            src = e.get("source", "typed")
            chars = e.get("chars", "")
            t = float(e["t"])
            if src == "typed":
                for c in chars:
                    typed.append((t, c == "\b"))
            elif src == "inserted":
                inserted.append((len(chars), t))
            else:
                onscreen.append((len(chars), t))

    f = []

    # pointer velocity (screen-fraction / ms): mean + peak
    vel = []
    for i in range(1, len(moves)):
        dt = moves[i][2] - moves[i - 1][2]
        if dt <= 0:
            continue
        dx = moves[i][0] - moves[i - 1][0]
        dy = moves[i][1] - moves[i - 1][1]
        vel.append(math.sqrt(dx * dx + dy * dy) / dt)
    f.append(mean(vel))                        # 0 mean pointer velocity
    f.append(max(vel) if vel else 0.0)         # 1 peak pointer velocity

    # pointer hesitation: mean inter-move gap (ms)
    gaps = [moves[i][2] - moves[i - 1][2] for i in range(1, len(moves))
            if moves[i][2] - moves[i - 1][2] > 0]
    f.append(mean(gaps) if gaps else 0.0)      # 2 mean pointer hesitation

    # typing cadence: inter-key latency WITHIN a burst only, so the pointer
    # travel between bursts does not contaminate the keystroke rhythm.
    ikl = []
    for i in range(1, len(typed)):
        dt = typed[i][0] - typed[i - 1][0]
        if dt < BURST_GAP_MS:
            ikl.append(dt)
    f.append(mean(ikl) if ikl else 0.0)        # 3 mean inter-key latency (ms)
    f.append(std(ikl))                         # 4 std inter-key latency (ms)

    # correction behaviour: backspace rate over typed characters
    backspaces = sum(1 for (_, b) in typed if b)
    f.append((backspaces / len(typed)) if typed else 0.0)  # 5 backspace rate

    # typing burst length (chars between pauses longer than BURST_GAP_MS)
    times = [t for (t, _) in typed]
    bursts = []
    cur = 0
    for i in range(len(times)):
        if i > 0 and (times[i] - times[i - 1]) > BURST_GAP_MS:
            bursts.append(cur)
            cur = 0
        cur += 1
    if cur:
        bursts.append(cur)
    f.append(mean(bursts) if bursts else 0.0)  # 6 mean typing-burst length (chars)

    # dictation / software-inserted text chunk statistics
    ins_lens = [ln for (ln, _) in inserted]
    ins_gaps = [inserted[i][1] - inserted[i - 1][1] for i in range(1, len(inserted))
                if inserted[i][1] > inserted[i - 1][1]]
    f.append(mean(ins_lens) if ins_lens else 0.0)   # 7 mean inserted-chunk length
    f.append(std(ins_lens))                          # 8 std inserted-chunk length
    f.append(mean(ins_gaps) if ins_gaps else 0.0)    # 9 mean gap between insertions
    f.append(std(ins_gaps))                          # 10 std gap between insertions

    # click behaviour: clicks per pointer move
    f.append((len(clicks) / len(moves)) if moves else 0.0)  # 11 click-to-move ratio

    # correction behaviour: on-screen edits per dictation insert
    total_ins = len(inserted) + len(onscreen)
    f.append((len(onscreen) / total_ins) if total_ins else 0.0)  # 12 correction rate

    return f


# --------------------------------------------------------------------------- #
# MODEL — per-user Gaussian novelty detector
# --------------------------------------------------------------------------- #

class GaussianProfile:
    """A user profile: per-feature mean and inverse variance. `score()` returns
    the squared Mahalanobis distance of a feature vector from the profile centre,
    which here is the sum of squared z-scores — i.e. a diagonal-covariance
    Gaussian. Diagonal is chosen because the training set is small (a few dozen
    sessions): a full covariance matrix is too noisy to invert reliably at that
    size, while per-feature variance is stable and keeps every distance
    interpretable ("this session was N standard deviations off on features X…")."""

    def __init__(self, mu, inv_var):
        self.mu = mu
        self.inv_var = inv_var

    def score(self, x):
        return sum((x[i] - self.mu[i]) ** 2 * self.inv_var[i] for i in range(len(x)))


def fit_gaussian(vectors, min_var=1e-4):
    """Fit a diagonal-covariance Gaussian to a list of feature vectors.

    Returns a `GaussianProfile`. Per-feature variance is estimated from the
    training set and floored at `min_var` so features with near-zero variance
    (a person's extremely stable habits) do not produce an infinite weight.
    """
    n = len(vectors)
    d = len(vectors[0])
    mu = [sum(v[i] for v in vectors) / n for i in range(d)]

    var = [0.0] * d
    for v in vectors:
        for i in range(d):
            var[i] += (v[i] - mu[i]) ** 2
    for i in range(d):
        var[i] = var[i] / (n - 1) if n > 1 else 0.0
        if var[i] < min_var:
            var[i] = min_var

    inv_var = [1.0 / vi for vi in var]
    return GaussianProfile(mu, inv_var)


def fit_and_threshold(train_vectors, holdout_vectors, *, held_out_quantile=1.0):
    """Fit a profile on `train_vectors`, then FIX the decision threshold on
    `holdout_vectors` (a separate split of the enrolled user's own sessions).

    The threshold is the chosen quantile of holdout distances — the largest
    distance the model will accept for the owner. `held_out_quantile=1.0`
    (the max) accepts every held-out session; lower values trade acceptance
    for a tighter boundary. The returned threshold is CONSTANT for all later
    tests (strangers, mimics, tired owner, swap tests), so it is declared before
    any test is scored.
    """
    profile = fit_gaussian(train_vectors)
    dists = sorted(profile.score(v) for v in holdout_vectors)
    k = int(round(held_out_quantile * (len(dists) - 1))) if dists else 0
    threshold = dists[k]
    return profile, threshold


# --------------------------------------------------------------------------- #
# SYNTHETIC DATA (evaluation only — see README for the honesty note)
# --------------------------------------------------------------------------- #

# Persona "signature" parameters. A persona is a dict of behavioural knobs; the
# simulator samples sessions from it with noise. The values are chosen so the
# swap/stranger/mimic behaviour is clear and the evaluation is reproducible.
PERSONAS = {
    # --- style MK: mouse + physical keyboard ----------------------------------
    "mk_owner": {
        "style": "mk",
        "cadence_mean": 150.0, "cadence_std": 32.0,
        "backspace_rate": 0.06,
        "pointer_speed": 0.40,
        "pointer_pause_mean": 300.0,
        "click_rate": 0.55,
        "burst_chars": 24.0,
    },
    "mk_tired": {   # same owner, fatigued: a moderate slowdown + more errors
        "style": "mk",
        "cadence_mean": 172.0, "cadence_std": 40.0,
        "backspace_rate": 0.09,
        "pointer_speed": 0.34,
        "pointer_pause_mean": 400.0,
        "click_rate": 0.50,
        "burst_chars": 18.0,
    },
    "mk_stranger": {   # clearly different person
        "style": "mk",
        "cadence_mean": 78.0, "cadence_std": 12.0,
        "backspace_rate": 0.012,
        "pointer_speed": 0.80,
        "pointer_pause_mean": 160.0,
        "click_rate": 0.70,
        "burst_chars": 42.0,
    },
    "mk_mimic": {   # copies the owner's mean speed/cadence but is unnaturally steady
        "style": "mk",
        "cadence_mean": 150.0, "cadence_std": 6.0,
        "backspace_rate": 0.06,
        "pointer_speed": 0.40,
        "pointer_pause_mean": 300.0,
        "click_rate": 0.55,
        "burst_chars": 24.0,
    },
    # --- style PD: slow pointer + dictation -------------------------------------
    "pd_owner": {
        "style": "pd",
        "dictation_chunk_mean": 42.0, "dictation_chunk_std": 10.0,
        "dictation_pause_mean": 1500.0,
        "correction_rate": 0.20,
        "pointer_speed": 0.15,
        "pointer_pause_mean": 800.0,
        "click_rate": 0.35,
    },
    "pd_tired": {
        "style": "pd",
        "dictation_chunk_mean": 33.0, "dictation_chunk_std": 9.0,
        "dictation_pause_mean": 1900.0,
        "correction_rate": 0.26,
        "pointer_speed": 0.12,
        "pointer_pause_mean": 1000.0,
        "click_rate": 0.32,
    },
    "pd_stranger": {
        "style": "pd",
        "dictation_chunk_mean": 14.0, "dictation_chunk_std": 4.0,
        "dictation_pause_mean": 700.0,
        "correction_rate": 0.08,
        "pointer_speed": 0.30,
        "pointer_pause_mean": 450.0,
        "click_rate": 0.50,
    },
    "pd_mimic": {
        "style": "pd",
        "dictation_chunk_mean": 42.0, "dictation_chunk_std": 4.0,
        "dictation_pause_mean": 1500.0,
        "correction_rate": 0.20,
        "pointer_speed": 0.15,
        "pointer_pause_mean": 800.0,
        "click_rate": 0.35,
    },
}

_ALPHA = "abcdefghijklmnopqrstuvwxyz "


def _random_text(rng, n):
    """A random string of EXACTLY `n` characters. Used so the chunk-length
    features reflect the persona's `dictation_chunk_mean/--std` directly, with no
    word-boundary noise to blur the contrast between owner, stranger and mimic."""
    n = max(0, int(n))
    return "".join(rng.choice(_ALPHA) for _ in range(n))


def simulate_session(rng, persona):
    """Generate one synthetic recording for a persona. Returns (events)."""
    p = PERSONAS[persona]
    style = p["style"]
    events = []
    t = 0
    x, y = 0.5, 0.5
    episodes = 22

    # The persona's pointer speed is a STABLE signature: sampled once for the
    # whole session with tight noise, so the across-session feature is sharp.
    speed = max(0.03, abs(rng.gauss(p["pointer_speed"], p["pointer_speed"] * 0.06)))

    for _ in range(episodes):
        # --- move the pointer to a random target ------------------------------
        tx = rng.uniform(0.05, 0.95)
        ty = rng.uniform(0.05, 0.95)
        # dwell before moving (hesitation)
        t += max(1, int(rng.gauss(p["pointer_pause_mean"], p["pointer_pause_mean"] * 0.12)))
        steps = rng.randint(8, 12)
        for s in range(1, steps + 1):
            frac = s / steps
            nx = x + (tx - x) * frac
            ny = y + (ty - y) * frac
            step_ms = max(2, int(30.0 / speed))
            t += rng.randint(int(step_ms * 0.85), int(step_ms * 1.15))
            events.append({"t": t, "kind": "pointer", "x": round(nx, 4), "y": round(ny, 4)})
        x, y = tx, ty

        # --- maybe click ------------------------------------------------------
        if rng.random() < p.get("click_rate", 0.5):
            t += rng.randint(5, 30)
            events.append({"t": t, "kind": "click", "x": round(x, 4), "y": round(y, 4),
                           "button": "left", "action": "down"})
            t += rng.randint(30, 140)
            events.append({"t": t, "kind": "click", "x": round(x, 4), "y": round(y, 4),
                           "button": "left", "action": "up"})

        # --- text -------------------------------------------------------------
        if style == "mk":
            n_chars = max(3, int(rng.gauss(p["burst_chars"], p["burst_chars"] * 0.3)))
            for _ in range(n_chars):
                t += max(1, int(rng.gauss(p["cadence_mean"], p["cadence_std"])))
                if rng.random() < p["backspace_rate"]:
                    events.append({"t": t, "kind": "text", "source": "typed", "chars": "\b"})
                    t += max(1, int(rng.gauss(p["cadence_mean"], p["cadence_std"])))
                    events.append({"t": t, "kind": "text", "source": "typed",
                                   "chars": rng.choice(_ALPHA)})
                else:
                    events.append({"t": t, "kind": "text", "source": "typed",
                                   "chars": rng.choice(_ALPHA)})
        else:  # "pd" — pointer + dictation, with on-screen corrections
            chunk = max(3, int(rng.gauss(p["dictation_chunk_mean"], p["dictation_chunk_std"])))
            t += max(1, int(rng.gauss(p["dictation_pause_mean"], p["dictation_pause_mean"] * 0.15)))
            events.append({"t": t, "kind": "text", "source": "inserted",
                           "chars": _random_text(rng, chunk)})
            if rng.random() < p.get("correction_rate", 0.15):
                t += rng.randint(250, 800)
                events.append({"t": t, "kind": "text", "source": "onscreen",
                               "chars": _random_text(rng, rng.randint(1, 6))})

    events.sort(key=lambda e: e["t"])
    return events


def generate_synthetic_data(data_dir, seed=20261106, sessions_per_persona=60):
    """Generate the fake dataset deterministically and write JSONL recordings +
    an index manifest. Returns the set of (persona, id) pairs produced."""
    os.makedirs(data_dir, exist_ok=True)
    rng = random.Random(seed)
    index = {}
    for persona in PERSONAS:
        for i in range(sessions_per_persona):
            events = simulate_session(rng, persona)
            rel = os.path.join(persona, f"session_{i:03d}.jsonl")
            save_session(events, os.path.join(data_dir, rel))
            index[f"{persona}/session_{i:03d}"] = {"persona": persona, "file": rel}
    with open(os.path.join(data_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "sessions": index}, f, indent=2, ensure_ascii=False)
    return index