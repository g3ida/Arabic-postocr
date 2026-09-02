"""Assemble the MSA and ARA benchmark image sets from public HuggingFace corpora.

Two sets of 300 scanned pages each:

    MSA   modern printed Arabic   -- Hindawi book pages, contemporary prose, news pages
    ARA   classical / turath      -- Shamela and hadith book scans, isnad chains,
                                     heavy citation apparatus, archaic orthography

Both are PAGE level. That is deliberate and worth preserving if you extend this: the two
sets must differ in register and typography and in nothing else. An earlier draft used
rendered single LINES for MSA, which would have confounded "modern vs classical" with
"line vs page" and made any measured difference uninterpretable.

Downloads images and ground truth only. OCR is a separate step (`run_ocr.py`) so that
re-running one engine does not re-download 600 pages.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Source:
    name: str
    hf_id: str
    text_fields: list[str]
    quota: int
    configs: list[str] | None = None      # None = single default config
    image_field: str = "image"
    image_is_url: bool = False
    min_chars: int = 48
    meta: list[str] = field(default_factory=list)


GROUPS: dict[str, list[Source]] = {
    "msa": [
        Source("arocrbench_hindawi", "ahmedheakl/arocrbench_hindawi", ["text"], 100),
        Source("arabic_ocrr", "Omar-youssef/arabic-ocrr-dataset", ["text"], 100),
        Source("qari_news", "oddadmix/qari-0.2.2-news-dataset-large", ["text"], 100,
               image_is_url=True),
    ],
    "ara": [
        Source("freococo_books", "freococo/ocr_arabic_books",
               ["page_header", "body", "footnotes"], 150, configs="AUTO"),
        Source("synth_shamela", "freococo/synth_shamela_ocr_arabic_books",
               ["page_header", "body", "footnotes"], 150, configs="AUTO"),
    ],
}

#: How many distinct books to draw from, for the multi-config sources. Spreading the quota
#: across books matters more than it looks: a single book is one typesetter, one scanner
#: and one set of running headers, so 150 pages from one book measures that book.
BOOKS_PER_SOURCE = 10


def gather_text(row: dict, fields: list[str]) -> str:
    return "\n".join(str(row[f]).strip() for f in fields
                     if row.get(f) not in (None, "")).strip()


def iter_source(src: Source, log, want: int, skip: int = 0):
    """Yield (record_id, PIL image, ground truth), at most `want`, skipping the first
    `skip` qualifying rows so a top-up pass does not re-collect what it already has."""
    from datasets import get_dataset_config_names, load_dataset

    configs = [None]
    if src.configs == "AUTO":
        names = sorted(get_dataset_config_names(src.hf_id))
        configs = names[:BOOKS_PER_SOURCE]
        log(f"  {src.name}: {len(names)} books available, sampling {len(configs)}")

    per_config = -(-(want + skip) // len(configs))   # ceil, so the quota is reachable
    taken = seen = 0
    for cfg in configs:
        if taken >= want:
            break
        try:
            ds = (load_dataset(src.hf_id, cfg, split="train", streaming=True) if cfg
                  else load_dataset(src.hf_id, split="train", streaming=True))
        except Exception as exc:                                   # noqa: BLE001
            log(f"  {src.name}[{cfg}]: SKIPPED ({type(exc).__name__}: {exc!s:.80})")
            continue

        from_this = 0
        for i, row in enumerate(ds):
            if from_this >= per_config or taken >= want:
                break
            gt = gather_text(row, src.text_fields)
            if len(gt) < src.min_chars:
                continue
            seen += 1
            if seen <= skip:                  # already collected on an earlier pass
                continue
            image = load_image(row[src.image_field], src.image_is_url)
            if image is None:
                continue
            rid = f"{src.name}/{cfg or 'default'}/{i:06d}"
            taken += 1
            from_this += 1
            yield rid, image, gt
    log(f"  {src.name}: collected {taken}/{want}")


def load_image(value, is_url: bool):
    from PIL import Image

    if not is_url:
        return value.convert("RGB") if hasattr(value, "convert") else None
    import requests

    try:
        resp = requests.get(value, timeout=45)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:                                              # noqa: BLE001
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="bench", help="output directory")
    ap.add_argument("--groups", default="msa,ara")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply every quota, for a quick smaller run")
    args = ap.parse_args(argv)

    def log(msg):
        print(msg, file=sys.stderr, flush=True)

    root = Path(args.out)
    for group in args.groups.split(","):
        target = int(sum(s.quota for s in GROUPS[group]) * args.scale)
        images = root / "images" / group
        images.mkdir(parents=True, exist_ok=True)
        manifest = root / f"{group}.jsonl"
        taken_per_source: dict[str, int] = {s.name: 0 for s in GROUPS[group]}
        n = 0

        with open(manifest, "w", encoding="utf-8") as fh:
            def drain(src, want):
                nonlocal n
                got = 0
                for rid, image, gt in iter_source(src, log, want,
                                                  skip=taken_per_source[src.name]):
                    stem = rid.replace("/", "__")
                    path = images / f"{stem}.png"
                    image.save(path)
                    fh.write(json.dumps({
                        "id": stem, "group": group, "source": src.name,
                        "image": f"images/{group}/{path.name}",
                        "width": image.width, "height": image.height,
                        "gt": gt,
                    }, ensure_ascii=False) + "\n")
                    n += 1
                    got += 1
                taken_per_source[src.name] += got
                return got

            for src in GROUPS[group]:
                drain(src, max(1, int(src.quota * args.scale)))

            # A source can hold far less than its quota -- arabic-ocrr-dataset is 10 rows
            # in total. Top up from whichever sources still have pages, so the group
            # reaches its target instead of silently shipping short.
            #
            # Spread the shortfall EVENLY rather than draining sources in list order. The
            # order-first version handed 190 of 300 MSA pages to one source, which would
            # have made the set mostly clean digital renders rather than the mix of render
            # and photograph it is supposed to be. What a benchmark is made of decides what
            # it measures.
            exhausted: set[str] = set()
            while n < target:
                live = [s for s in GROUPS[group] if s.name not in exhausted]
                if not live:
                    log(f"  {group}: exhausted every source at {n}/{target}")
                    break
                share = -(-(target - n) // len(live))
                for src in live:
                    if n >= target:
                        break
                    if drain(src, min(share, target - n)) < min(share, target - n):
                        exhausted.add(src.name)
        log(f"{group}: {n} pages -> {manifest}")
        for name, count in taken_per_source.items():
            log(f"    {name:22} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
