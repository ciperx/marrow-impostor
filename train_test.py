#!/usr/bin/env python3
"""Train and test marrow-impostor on ANY recordings in the documented format.

Offline, stdlib-only, no capture software, no network. Point it at recordings and
it works. Labels (genuine owner vs. impostor) are inferred from the path so the
funder can drop their own sessions straight in — no manifest, no index, no
`subject` column to hand-fill.

Examples::

    # enrol on one person (any states mixed: rested + tired), test a pile
    python3 train_test.py --enrol recordings/owner/ --test recordings/mixed/

    # name the two sides yourself when the filenames are not self-explanatory
    python3 train_test.py --enrol recordings/owner/ --test recordings/mixed/ \\
        --positive alice,alice_tired --negative bob,carol

    # also run the swap test: enrol on somebody else, present the owner's sessions
    python3 train_test.py --enrol recordings/owner/ --test recordings/mixed/ \\
        --swap-with recordings/bob/

The recording format is the flat JSONL described in FORMAT.md (one event per
line: ``t`` ms, ``kind`` = pointer/click/text, ``x``/``y`` as screen fraction,
``source`` = typed/onscreen/inserted). Persona is inferred from the *path*:
any session whose path contains ``owner`` / ``tired`` / ``self`` / ``genuine`` is
treated as the enrolled person; ``stranger`` / ``mimic`` / ``intruder`` / ``other``
is treated as an impostor.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from marrow import extract_features, fit_and_threshold, load_session  # noqa: E402

POSITIVE_HINTS = ("owner", "tired", "self", "genuine", "enrol", "real")
NEGATIVE_HINTS = ("stranger", "mimic", "intruder", "other", "impostor", "fake")


def collect(paths):
    """Every .jsonl under the given files or directories (recursive, sorted)."""
    out = []
    for p in paths:
        if os.path.isdir(p):
            for root, _dirs, names in os.walk(p):
                out += [os.path.join(root, n) for n in sorted(names)
                        if n.endswith(".jsonl")]
        elif p.endswith(".jsonl"):
            out.append(p)
        elif os.path.exists(p):
            out.append(p)
        else:
            raise SystemExit(f"not a recording or directory: {p}")
    return sorted(set(out))


def featurize(paths):
    vecs = []
    for p in collect(paths):
        try:
            vecs.append((p, extract_features(load_session(p))))
        except (ValueError, KeyError, OSError) as exc:
            print(f"  skipping {p}: {exc}", file=sys.stderr)
    if not vecs:
        raise SystemExit("no readable recordings found")
    return vecs


def classify(path, positive, negative):
    low = path.lower()
    for lab in positive:
        if lab.lower() in low:
            return "positive"
    for lab in negative:
        if lab.lower() in low:
            return "negative"
    return None


def split_threshold_holdout(enrol_vecs, holdout_frac=0.2):
    """Reserve the last `holdout_frac` of enrolment as a threshold hold-out so
    the operating point is fixed BEFORE any test recording is scored."""
    n = len(enrol_vecs)
    if n >= 8:
        k = max(1, int(round(holdout_frac * n)))
    elif n >= 3:
        k = 1
    else:
        k = 0
    train = enrol_vecs[: n - k] if k else enrol_vecs
    holdout = enrol_vecs[n - k:] if k else enrol_vecs
    return train, holdout


def main():
    ap = argparse.ArgumentParser(
        description="Train and test marrow-impostor on any recordings. Offline, stdlib only.")
    ap.add_argument("--enrol", nargs="+", required=True,
                    help="owner enrolment recordings: files and/or directories")
    ap.add_argument("--test", nargs="+", required=True,
                    help="recordings to classify")
    ap.add_argument("--positive", default=None,
                    help="comma list of substrings naming the enrolled person "
                         "(default: owner/tired/self/genuine/enrol/real)")
    ap.add_argument("--negative", default=None,
                    help="comma list of substrings naming impostors "
                         "(default: stranger/mimic/intruder/other/impostor/fake)")
    ap.add_argument("--swap-with", nargs="*", default=None,
                    help="enrol a second profile on THESE sessions (somebody else) "
                         "and present the owner's test sessions to it; must open none")
    ap.add_argument("--json", default=None, help="also write the result as JSON here")
    a = ap.parse_args()

    positive = ([p.strip().lower() for p in a.positive.split(",") if p.strip()]
                if a.positive else list(POSITIVE_HINTS))
    negative = ([p.strip().lower() for p in a.negative.split(",") if p.strip()]
                if a.negative else list(NEGATIVE_HINTS))

    enrol = featurize(a.enrol)
    train, holdout = split_threshold_holdout(enrol)

    if not train:
        raise SystemExit("no enrolment recordings found")
    if not holdout:
        # too few sessions to spare a hold-out: freeze a conservative threshold
        # (the max training distance), same behaviour as before, declared up front
        holdout = train

    profile, threshold = fit_and_threshold(
        [v for _, v in train], [v for _, v in holdout], held_out_quantile=1.0)

    decided = []
    genuine = []
    impostor = []
    for p, v in featurize(a.test):
        cls = classify(p, positive, negative)
        if cls is None:
            # unknown path: treat as neutral evidence, do not assign a label
            continue
        (genuine if cls == "positive" else impostor).append(v)

    far = sum(profile.score(v) <= threshold for v in impostor) / len(impostor) \
        if impostor else None
    frr = (1.0 - sum(profile.score(v) <= threshold for v in genuine) / len(genuine)) \
        if genuine else None
    swap_accepts = None
    swap_total = None
    if a.swap_with:
        swap_vecs = [v for _, v in featurize(a.swap_with)]
        swap_train, swap_hold = split_threshold_holdout(swap_vecs)
        swap_profile, _ = fit_and_threshold(
            swap_train, swap_hold or swap_train, held_out_quantile=1.0)
        swap_accepts = sum(swap_profile.score(v) <= threshold for v in genuine)
        swap_total = len(genuine)

    print(f"enrolment sessions: {len(train)} train, {len(holdout)} threshold hold-out")
    print(f"test sessions: {len(genuine)} genuine, {len(impostor)} impostor")
    print(f"threshold fixed at: {threshold:.4f}\n")

    print(f"FAR (impostor let in): {far if far is None else round(far, 4)}")
    print(f"FRR (owner doubted)  : {frr if frr is None else round(frr, 4)}")
    if swap_total is not None:
        print(f"swap (other-profile opens owner): {swap_accepts}/{swap_total}")
    ver = "PASS" if (far is None or far < 0.05) and swap_accepts == 0 else "SEE ABOVE"
    print(f"\nverdict: {ver}")

    if a.json:
        import json as _json
        result = {
            "n_enrol_train": len(train),
            "n_enrol_holdout": len(holdout),
            "n_test_genuine": len(genuine),
            "n_test_impostor": len(impostor),
            "threshold": threshold,
            "far": far,
            "frr": frr,
            "swap_accepts": swap_accepts,
            "swap_total": swap_total,
        }
        with open(a.json, "w", encoding="utf-8") as fh:
            _json.dump(result, fh, indent=2)
        print(f"\nresult written: {a.json}")


if __name__ == "__main__":
    main()