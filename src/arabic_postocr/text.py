"""Text primitives shared by both stages.

The two models read the same document through slightly different lenses, and keeping
those lenses in one place is what stops the pipeline from silently desynchronising:
the detector's word `i` and the infiller's word `i` must be the same word.
"""

from __future__ import annotations

import difflib
import re

SHADDA = "ّ"
ERR_OPEN, ERR_CLOSE = "<err>", "</err>"

#: Harakat, tatweel, Quranic annotation marks. Everything the infiller's vocabulary
#: cannot represent, and therefore can never emit.
DIACRITIC_RE = re.compile(r"[ً-ْٰـؕ-ؚۖ-ۭ࣓-ࣿ]")


def strip_diacritics(text: str) -> str:
    """Remove everything the infiller cannot tokenise."""
    return DIACRITIC_RE.sub("", text)


def detector_view(text: str) -> str:
    """The detector's view: shadda removed, everything else intact.

    Deliberately *not* the same as `strip_diacritics`. The detector only emits a per-word
    flag, so it can read a lightly-normalised view while its verdicts land on the original
    words. Stripping further would be a domain shift away from its training data for no
    gain.
    """
    return text.replace(SHADDA, "")


def word_spans(text: str) -> list[tuple[int, int, str]]:
    """(start, end, word) for every whitespace-delimited word, with character offsets.

    Offsets rather than token indices, because a tokenizer splits words into subwords and
    a flag has to be attributed back to the whole word.
    """
    spans, i = [], 0
    for word in text.split(" "):
        if word:
            start = text.index(word, i)
            spans.append((start, start + len(word), word))
            i = start + len(word)
        else:
            i += 1
    return spans


def project(src: list[str], dst: list[str]) -> list[str]:
    """For each source word, the destination word(s) aligned to it.

    Anchoring on whole difflib spans is too coarse: if the model rewrites three adjacent
    words where only one needed changing, the spans differ and a genuine repair reads as
    damage. Projecting word by word keeps each site separate. An unequal-length
    replacement is folded onto its first position -- an approximation, but a local one.
    """
    out: list[str | None] = [None] * len(src)
    matcher = difflib.SequenceMatcher(None, src, dst, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                out[i1 + k] = dst[j1 + k]
        elif tag == "delete":
            for i in range(i1, i2):
                out[i] = ""
        elif tag == "insert":
            anchor = min(i1, len(src) - 1)
            if anchor >= 0:
                head = out[anchor] if out[anchor] is not None else src[anchor]
                out[anchor] = (" ".join(dst[j1:j2]) + " " + head).strip()
        else:  # replace
            if i2 - i1 == j2 - j1:
                for k in range(i2 - i1):
                    out[i1 + k] = dst[j1 + k]
            else:
                out[i1] = " ".join(dst[j1:j2])
                for i in range(i1 + 1, i2):
                    out[i] = ""
    return [src[i] if v is None else v for i, v in enumerate(out)]


def mark_spans(words: list[str], flags: list[int]) -> str:
    """Fence each flagged run of words as ``<err> ... </err>``, keeping its text.

    Keeping the corrupted characters is the single most important choice in the input
    format. Replacing them with a mask token measures far worse, because roughly 55% of
    OCR errors are *light* corruptions -- a dot, a hamza seat, a join -- where the broken
    characters carry most of the answer.

    Consecutive flagged words become one span: a two-word garble is one problem, and the
    model needs to be free to emit a different number of words than it was given.
    """
    parts: list[str] = []
    i = 0
    while i < len(words):
        if not flags[i]:
            parts.append(words[i])
            i += 1
            continue
        j = i
        while j < len(words) and flags[j]:
            j += 1
        parts.append(f"{ERR_OPEN} {' '.join(words[i:j])} {ERR_CLOSE}")
        i = j
    return " ".join(parts)
