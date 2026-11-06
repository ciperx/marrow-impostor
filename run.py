"""marrow-impostor — one-command evaluation entry point.

Run from a fresh copy with:

    python3 run.py

That single command (1) regenerates the deterministic synthetic dataset,
(2) trains a per-style user profile, (3) declares the decision threshold on a
hold-out split before any test is scored, and (4) prints the evaluation:

  * FAR          — how often a stranger or mimic is let in (the core job)
  * FRR (normal) — how often the genuine owner is wrongly rejected
  * fatigue doubt— how often a tired owner is doubted (reported separately)
  * swap test    — whether a profile trained on someone else opens the owner

Stdlib only. No third-party dependencies.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from marrow import (
    extract_features,
    fit_and_threshold,
    generate_synthetic_data,
    load_session,
)

DATA_DIR = "data"
SEED = 20261106

STYLES = ("mk", "pd")


def load_persona_features(data_dir, persona):
    base = os.path.join(data_dir, persona)
    feats = []
    for fn in sorted(os.listdir(base)):
        if fn.endswith(".jsonl"):
            feats.append(extract_features(load_session(os.path.join(base, fn))))
    return feats


def evaluate_style(data_dir, style, n_train=36, n_holdout=12):
    owner = load_persona_features(data_dir, f"{style}_owner")
    tired = load_persona_features(data_dir, f"{style}_tired")
    stranger = load_persona_features(data_dir, f"{style}_stranger")
    mimic = load_persona_features(data_dir, f"{style}_mimic")

    train = owner[:n_train]
    holdout = owner[n_train:n_train + n_holdout]
    owner_test = owner[n_train + n_holdout:]

    profile, threshold = fit_and_threshold(train, holdout, held_out_quantile=1.0)

    def accepted(vecs):
        return [profile.score(v) <= threshold for v in vecs]

    attackers = stranger + mimic
    far = sum(accepted(attackers)) / len(attackers)
    frr_normal = 1.0 - sum(accepted(owner_test)) / len(owner_test)
    tired_recall = sum(accepted(tired)) / len(tired)

    # swap test: a profile trained on *someone else* must not open for the owner
    stranger_profile, _ = fit_and_threshold(stranger[:n_train], stranger[n_train:n_train + n_holdout],
                                            held_out_quantile=1.0)
    swap_accepts = sum(stranger_profile.score(v) <= threshold for v in owner_test)

    return {
        "style": style,
        "threshold": threshold,
        "n_train": len(train),
        "n_holdout": len(holdout),
        "far": far,
        "frr_normal": frr_normal,
        "tired_recall": tired_recall,
        "swap_accepts": swap_accepts,
        "swap_total": len(owner_test),
    }


def main():
    index = generate_synthetic_data(DATA_DIR, seed=SEED)
    n_sessions = len(index)
    print(f"Synthetic dataset: {n_sessions} sessions written to {DATA_DIR}/")
    print("Per style: 00-35 train, 36-47 threshold hold-out, 48-59 owner test;")
    print("plus 60 tired-owner, 60 stranger and 60 mimic sessions.\n")

    hdr = (f"{'style':<5} {'FAR':>7} {'FRR(norm)':>9} {'tired-recall':>13} "
           f"{'swap':>7}   verdict")
    print(hdr)
    print("-" * 84)

    for style in STYLES:
        r = evaluate_style(DATA_DIR, style)
        swap = f"{r['swap_accepts']}/{r['swap_total']}"
        verdict = ("OK" if (r["far"] < 0.05 and r["swap_accepts"] == 0) else "FAIL")
        print(f"{r['style']:<5} {r['far']:>7.3f} {r['frr_normal']:>9.3f} "
              f"{r['tired_recall']:>13.3f} {swap:>7}   {verdict}")

    print("-" * 84)
    print("Reading the table:")
    print("  * FAR (stranger + mimic let in) is the core job — ~0 means impostors")
    print("    are caught. The mimic copies the owner's means but not their natural")
    print("    rhythm/variance, so it is still rejected.")
    print("  * FRR (normal) is the genuine owner in a normal session — ~0 means the")
    print("    owner is recognized reliably.")
    print("  * tired-recall is the same person, fatigued. A strict lock doubts them;")
    print("    that is the honest, safe default (it errs toward locking, not opening).")
    print("  * swap = a profile trained on the *stranger* opening the owner — 0 means")
    print("    profiles are person-specific (requirement 3 holds).")
    print()
    print("Threshold is fixed on the hold-out split BEFORE any test is scored, so")
    print("every number above is out-of-sample. Seed is fixed for reproducibility.")


if __name__ == "__main__":
    main()