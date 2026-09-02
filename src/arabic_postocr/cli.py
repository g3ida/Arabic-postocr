"""Command line entry point: ``arabic-postocr``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .corrector import DEFAULT_DETECTOR, DEFAULT_INFILLER, Corrector


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="arabic-postocr",
        description="Correct OCR errors in Arabic text.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  arabic-postocr page.txt
  arabic-postocr page.txt --gate 0.0 --review queue.txt
  cat page.txt | arabic-postocr - > fixed.txt
  arabic-postocr pages.jsonl --jsonl --field text -o fixed.jsonl
""")
    p.add_argument("input", help="text file, .jsonl file, or - for stdin")
    p.add_argument("-o", "--output", default="-", help="output file (default: stdout)")
    p.add_argument("--jsonl", action="store_true",
                   help="treat input as JSON Lines and correct one field per line")
    p.add_argument("--field", default="text", help="field to correct with --jsonl")
    p.add_argument("--detector", default=DEFAULT_DETECTOR)
    p.add_argument("--infiller", default=DEFAULT_INFILLER)
    p.add_argument("--device", default=None, help="cuda, cpu (default: autodetect)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="detector logit cutoff; higher flags fewer words (default: 0.5)")
    p.add_argument("--gate", type=float, default=None,
                   help="re-read the output and revert repairs still flagged above this. "
                        "Raises precision, lowers CER gain, produces the review queue.")
    p.add_argument("--gate-max-words", type=int, default=1,
                   help="only veto flagged regions of at most this many words. The "
                        "default of 1 is measured: vetoing larger regions discards good "
                        "repairs alongside the bad one and loses more recall than the "
                        "precision is worth.")
    p.add_argument("--review", default="",
                   help="write the <hard> review queue here (implies --gate 0.0)")
    p.add_argument("--num-beams", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--quiet", action="store_true")
    return p


def _read(path: str, jsonl: bool, field: str):
    stream = sys.stdin if path == "-" else open(path, encoding="utf-8")
    with stream as fh:
        if jsonl:
            rows = [json.loads(line) for line in fh if line.strip()]
            return rows, [r.get(field, "") for r in rows]
        body = fh.read()
    # one document per non-empty line keeps batching useful on a whole page
    return None, [ln for ln in body.splitlines() if ln.strip()]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    gate = args.gate
    if args.review and gate is None:
        gate = 0.0

    rows, texts = _read(args.input, args.jsonl, args.field)
    if not texts:
        print("nothing to correct", file=sys.stderr)
        return 1
    if not args.quiet:
        print(f"correcting {len(texts)} document(s)...", file=sys.stderr)

    corrector = Corrector.from_pretrained(args.detector, args.infiller, args.device)
    results = corrector.correct_batch(texts, threshold=args.threshold, gate=gate,
                                      gate_max_words=args.gate_max_words,
                                      num_beams=args.num_beams,
                                      batch_size=args.batch_size)

    out = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8")
    with out as fh:
        if args.jsonl:
            for row, res in zip(rows, results):
                row[args.field] = res.text
                if gate is not None:
                    row["hard"] = res.hard
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            for res in results:
                fh.write(res.text + "\n")

    if args.review:
        Path(args.review).parent.mkdir(parents=True, exist_ok=True)
        with open(args.review, "w", encoding="utf-8") as fh:
            for i, res in enumerate(results):
                for word in res.hard:
                    fh.write(f"{i}\t{word}\n")

    if not args.quiet:
        edits = sum(len(r.edits) for r in results)
        hard = sum(len(r.hard) for r in results)
        msg = f"{edits} repair(s) applied"
        if gate is not None:
            msg += f", {hard} word(s) flagged for review"
        print(msg, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
