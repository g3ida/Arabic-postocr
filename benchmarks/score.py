"""Score post-OCR correction: reproduce the README benchmark.

Two modes. Run the pipeline and score it (needs torch + transformers):

    python benchmarks/score.py --eval benchmarks/eval.jsonl
    python benchmarks/score.py --eval benchmarks/eval.jsonl --gate 0.0 --out predictions.jsonl

Or score precomputed pairs, e.g. raw engine output against truth (no models needed):

    python benchmarks/score.py --gt benchmarks/data/msa.jsonl --hyp benchmarks/data/msa.tesseract.jsonl

Prints overall CER (plus edit precision/recall/F1/damage for --eval) and a
per-source breakdown. Scoring is on undiacritised text, the pipeline's own terms:
it cannot emit diacritics inside a repaired span, so scoring them would measure
the tokenizer.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_postocr.editmetrics import score as edit_score

# The exact stripping the published numbers were scored with. Deliberately not
# arabic_postocr.strip_diacritics, whose range is one interval wider (ؕ-ؚ): scoring
# must match the eval the README reports, not the runtime normalisation.
_DIACRITIC_RE = re.compile(r"[ً-ْٰـۖ-ۭ࣓-ࣿ]")


def _strip(text: str) -> str:
    return _DIACRITIC_RE.sub("", text)


try:
    from rapidfuzz.distance import Levenshtein as _RFL

    def _lev(a: str, b: str) -> int:
        return _RFL.distance(a, b)
except ImportError:                                        # pure-python fallback
    def _lev(a: str, b: str) -> int:
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                               prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[len(b)]


def _doc_cer(ref: str, hyp: str) -> float:
    return 0.0 if not ref else _lev(hyp, ref) / len(ref)


def _agg_cer(pairs: list[tuple[str, str]]) -> float:
    """Aggregate CER: total edit distance over total reference length."""
    return (sum(_lev(h, r) for r, h in pairs)
            / max(1, sum(len(r) for r, _ in pairs)))


def _rows(path: str, limit: int = 0) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def report(items: list[tuple[str, str, str, str]], with_edits: bool,
           raw: bool = False) -> None:
    """items: (source, stripped input, stripped prediction, stripped truth)."""
    srcs = [s for _, s, _, _ in items]
    hyps = [h for _, _, h, _ in items]
    gts = [g for _, _, _, g in items]
    base, after = _agg_cer(list(zip(gts, srcs))), _agg_cer(list(zip(gts, hyps)))

    print(f"\npairs {len(items)}")
    if raw:
        # No correction applied (e.g. raw engine output): the absolute CER is the
        # number; there is no before/after to compare.
        print(f"raw CER {after*100:6.2f}%")
    else:
        print(f"CER   {base*100:6.2f}% -> {after*100:6.2f}%   "
              f"reduction {(base-after)/base*100:+.1f}%")
    if with_edits:
        m = edit_score(zip(srcs, hyps, gts))
        print(f"edit  precision {m['edit_precision']*100:5.1f}%  "
              f"recall {m['edit_recall']*100:5.1f}%  "
              f"F1 {m['edit_f1']*100:5.1f}%  "
              f"damage {m['damage_rate']*100:5.1f}%")
        print(f"      good {int(m['edits_good'])}  wrong {int(m['edits_wrong'])}  "
              f"damaging {int(m['edits_damaging'])}  missed {int(m['edits_missed'])}")

    by_src: dict[str, list] = defaultdict(list)
    for source, s, h, g in items:
        by_src[source].append((g, h, s))
    if len(by_src) < 2 and not raw:
        return
    if raw:
        print(f"\n{'source':15} {'samples':>7} {'CER':>8}")
        groups = sorted(by_src.items())
        groups.append(("OVERALL", [x for _, v in groups for x in v]))
        for label, triplets in groups:
            cp = _agg_cer([(g, h) for g, h, _ in triplets])
            print(f"{label:15} {len(triplets):7d} {cp:8.4f}")
        return
    print(f"\n{'source':15} {'samples':>7} {'CER in':>8} {'CER out':>8} "
          f"{'reduce':>8} {'better':>8} {'worse':>8} {'same':>8} {'exact':>7}")
    groups = sorted(by_src.items())
    groups.append(("OVERALL", [x for _, v in groups for x in v]))
    for label, triplets in groups:
        n = len(triplets)
        ci = _agg_cer([(g, s) for g, _, s in triplets])
        cp = _agg_cer([(g, h) for g, h, _ in triplets])
        deltas = [_doc_cer(g, s) - _doc_cer(g, h) for g, h, s in triplets]
        b = sum(1 for d in deltas if d > 1e-9) / n
        w = sum(1 for d in deltas if d < -1e-9) / n
        sm = sum(1 for d in deltas if abs(d) <= 1e-9) / n
        ex = sum(1 for g, h, _ in triplets if h == g) / n
        print(f"{label:15} {n:7d} {ci:8.4f} {cp:8.4f} {(ci-cp)/ci*100:7.1f}% "
              f"{b*100:7.1f}% {w*100:7.1f}% {sm*100:7.1f}% {ex*100:6.1f}%")


def run_eval(args) -> int:
    from arabic_postocr import Corrector

    rows = _rows(args.eval, args.limit)
    required = {"input", "target"}
    missing = required - set(rows[0])
    if missing:
        sys.exit(f"{args.eval}: rows need {sorted(required)}, "
                 f"first row has {sorted(rows[0])}")
    corrector = Corrector.from_pretrained(args.detector, args.infiller,
                                          args.device)
    gate = None if args.gate is None else float(args.gate)
    results = corrector.correct_batch(
        [r["input"] for r in rows], threshold=args.threshold, gate=gate,
        gate_max_words=args.gate_max_words, num_beams=args.num_beams,
        batch_size=args.batch_size)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            for row, res in zip(rows, results):
                fh.write(json.dumps({"input": row["input"], "prediction": res.text,
                                     "target": row["target"]},
                                    ensure_ascii=False) + "\n")
    report([(r.get("source", "unknown"), _strip(r["input"]), _strip(res.text),
             _strip(r["target"])) for r, res in zip(rows, results)],
           with_edits=True)
    return 0


def run_pairs(args) -> int:
    gt = {str(r[args.key]): r for r in _rows(args.gt)}
    items = []
    for r in _rows(args.hyp):
        g = gt.get(str(r.get(args.key, "")))
        if g is None or args.hyp_field not in r or args.gt_field not in g:
            continue
        hyp, truth = _strip(r[args.hyp_field]), _strip(g[args.gt_field])
        items.append((g.get("source", "unknown"), hyp, hyp, truth))
    if not items:
        sys.exit("no joinable rows: check --key/--gt-field/--hyp-field")
    # No original inputs here, so there are no edits to attribute -- CER tables only.
    report(items, with_edits=False, raw=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="score.py",
                                description="score post-OCR correction")
    p.add_argument("--eval", default="",
                   help="rows with {input, target[, source]}: run the pipeline and score")
    p.add_argument("--gt", default="", help="ground-truth jsonl for --hyp")
    p.add_argument("--hyp", default="", help="predictions jsonl to score against --gt")
    p.add_argument("--key", default="id", help="join field for --gt/--hyp")
    p.add_argument("--gt-field", default="gt")
    p.add_argument("--hyp-field", default="text")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default="", help="write {input, prediction, target} here")
    p.add_argument("--detector", default=None)
    p.add_argument("--infiller", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--gate", type=float, default=None)
    p.add_argument("--gate-max-words", type=int, default=1)
    p.add_argument("--num-beams", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.eval:
        if args.detector is None or args.infiller is None:
            from arabic_postocr.corrector import (DEFAULT_DETECTOR,
                                                  DEFAULT_INFILLER)
            args.detector = args.detector or DEFAULT_DETECTOR
            args.infiller = args.infiller or DEFAULT_INFILLER
        return run_eval(args)
    if args.gt and args.hyp:
        return run_pairs(args)
    build_parser().print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
