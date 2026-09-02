"""Run one OCR engine over a collected benchmark group.

One engine per invocation, one output file per engine. That separation is deliberate: the
VL pipeline takes ~7s per page against Tesseract's fraction of a second, and a crash in
one engine should never cost the results of another.

    python run_ocr.py --engine tesseract --group msa
    python run_ocr.py --engine ppocr     --group ara
    python run_ocr.py --engine ppocr_vl  --group ara
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Tesseract 4 parallelises internally with OpenMP and defaults to one thread per core.
# Combined with process-level parallelism that is a disaster: 36 workers x 40 threads
# oversubscribes a 40-core box ~35x and every page thrashes. Measured on this hardware,
# one page went from 137s to 8.5s with this single line. Set before any worker is forked,
# so the tesseract subprocess inherits it.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")


# ---------------------------------------------------------------- reading order
def reading_order(texts: list[str], boxes) -> str:
    """Join detected lines in Arabic reading order: top to bottom, right to left.

    Detection returns boxes in whatever order the detector emitted them, which is not
    reading order. Concatenating them raw produces text that scores terribly for reasons
    that have nothing to do with recognition quality -- the words are all correct and all
    in the wrong places. For Arabic the horizontal sort must be DESCENDING x.
    """
    if boxes is None or len(boxes) != len(texts):
        return "\n".join(texts)
    items = []
    for text, box in zip(texts, boxes):
        pts = [(float(p[0]), float(p[1])) for p in box] if hasattr(box[0], "__len__") \
            else [(float(box[0]), float(box[1])), (float(box[2]), float(box[3]))]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        items.append((min(ys), max(ys), min(xs), max(xs), text))
    if not items:
        return ""
    heights = sorted(b - a for a, b, _, _, _ in items)
    tol = max(6.0, heights[len(heights) // 2] * 0.6)

    items.sort(key=lambda it: it[0])
    lines: list[list] = [[items[0]]]
    for it in items[1:]:
        # same line if its top sits within a tolerance of the current line's top
        if abs(it[0] - lines[-1][0][0]) <= tol:
            lines[-1].append(it)
        else:
            lines.append([it])
    out = []
    for line in lines:
        line.sort(key=lambda it: it[3], reverse=True)      # right to left
        out.append(" ".join(it[4] for it in line))
    return "\n".join(out)


_MD = re.compile(r"^#{1,6}\s+|^\s*[-*+]\s+|\*\*|__|`{1,3}", re.MULTILINE)


def demarkdown(text: str) -> str:
    """VL emits markdown; the benchmark stores plain text."""
    text = _MD.sub("", text)
    text = re.sub(r"<[^>]{1,40}>", "", text)               # stray html from tables
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ---------------------------------------------------------------- engines
def run_tesseract(rows, root: Path, workers: int):
    """Tesseract is CPU-only and perfectly parallel, so use every core."""
    import multiprocessing as mp

    paths = [str(root / r["image"]) for r in rows]
    with mp.Pool(workers) as pool:
        texts = pool.map(_tess_one, paths, chunksize=4)
    return texts


def _tess_one(path: str) -> str:
    import pytesseract
    from PIL import Image

    try:
        return pytesseract.image_to_string(Image.open(path), lang="ara").strip()
    except Exception as exc:                                    # noqa: BLE001
        return f"__ERROR__ {type(exc).__name__}"


def run_ppocr(rows, root: Path, device: str):
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(lang="ar", device=device, use_doc_orientation_classify=False,
                    use_doc_unwarping=False, use_textline_orientation=False)
    out = []
    for i, row in enumerate(rows):
        try:
            res = ocr.predict(str(root / row["image"]))[0]
            out.append(reading_order(res["rec_texts"],
                                     res.get("rec_boxes", res.get("rec_polys"))))
        except Exception as exc:                                # noqa: BLE001
            out.append(f"__ERROR__ {type(exc).__name__}")
        if i and i % 50 == 0:
            print(f"  ppocr {i}/{len(rows)}", file=sys.stderr, flush=True)
    return out


def run_ppocr_vl(rows, root: Path, device: str):
    from paddleocr import PaddleOCRVL

    vl = PaddleOCRVL()
    out = []
    for i, row in enumerate(rows):
        try:
            res = vl.predict(str(root / row["image"]))[0]
            out.append(demarkdown(res.markdown.get("markdown_texts", "")))
        except Exception as exc:                                # noqa: BLE001
            out.append(f"__ERROR__ {type(exc).__name__}")
        if i and i % 25 == 0:
            print(f"  ppocr_vl {i}/{len(rows)}", file=sys.stderr, flush=True)
    return out


ENGINES = {"tesseract": run_tesseract, "ppocr": run_ppocr, "ppocr_vl": run_ppocr_vl}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--engine", required=True, choices=sorted(ENGINES))
    ap.add_argument("--group", required=True)
    ap.add_argument("--root", default="bench")
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    args = ap.parse_args(argv)

    root = Path(args.root)
    rows = [json.loads(l) for l in open(root / f"{args.group}.jsonl", encoding="utf-8")]
    print(f"{args.engine} over {len(rows)} {args.group} pages", file=sys.stderr)

    start = time.time()
    if args.engine == "tesseract":
        texts = run_tesseract(rows, root, args.workers)
    else:
        texts = ENGINES[args.engine](rows, root, args.device)
    elapsed = time.time() - start

    out = root / f"{args.group}.{args.engine}.jsonl"
    failed = 0
    with open(out, "w", encoding="utf-8") as fh:
        for row, text in zip(rows, texts):
            failed += text.startswith("__ERROR__")
            fh.write(json.dumps({"id": row["id"], "text": text},
                                ensure_ascii=False) + "\n")
    print(f"{args.engine}/{args.group}: {len(rows)} pages in {elapsed/60:.1f} min "
          f"({elapsed/max(1,len(rows)):.2f}s each), {failed} failed -> {out}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
