"""Edit-level metrics: what the corrector *did*, not how the characters landed.

CER is the standard headline for this task and it is worth keeping, but it is
misaligned with correction in one specific way: it awards partial credit for being
wrong. A mangled guess that lands nearer the truth than the original still lowers CER.
Measured on 1,200 held-out pairs, a model scoring a respectable +20.7% CER reduction
was right on only 40.3% of the edits it made, and damaged already-correct text almost
as often as it fixed something.

Aligning the OCR input against both the prediction and the ground truth sorts every
edit site into four buckets, which say what CER cannot:

    good       the model changed it, and matched the ground truth
    wrong_fix  it changed it, but not to the right thing
    bad_edit   it changed text that was ALREADY CORRECT   <- the dangerous one
    missed     it left an error untouched                 <- the timid one

`bad_edit` and `missed` are not symmetric in practice. A corrector that misses an error
leaves the text no worse than it found it; one that rewrites correct text actively
destroys information the OCR got right. `damage_rate` reports that separately so it can
be held down deliberately rather than averaged away.
"""

from __future__ import annotations

import difflib
import re
from typing import Iterable

# Orthographic variation that Arabic writers and typesetters disagree about, within this
# corpus and between it and itself. Measured on 12,406 words the rollout called
# unrepairable: 21.5% differ only in hamza seat, 7.3% in ى/ي, 3.0% in ة/ه, 12.1% in
# punctuation. `والذي` vs `والذى` appears in BOTH directions in the same corpus, so there
# is no convention to learn -- only a coin flip to be punished for.
#
# Scoring these as errors misattributes a typesetting choice to the model. It is not the
# same as folding them in the DATA, which would destroy information the corrector should
# restore; this only says "do not count it as a mistake either way".
_ORTHO = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه"})
_EDGE_PUNCT = "().,،؛:؟!\"«»'’‘\u201c\u201d-–—[]{}*"


def ortho_key(word: str, punctuation: bool = True) -> str:
    """Collapse a word to the form that ignores conventional spelling differences."""
    w = word.translate(_ORTHO)
    w = re.sub(r"[\u0640\u064b-\u0652\u0670]", "", w)      # tatweel + any stray marks
    return w.strip(_EDGE_PUNCT) if punctuation else w


# Letters sharing a rasm: the routine OCR confusions. A substitution inside a group is a
# dot or a hamza seat and costs little; anything else is a different letter.
_RASM = ["بتثنيئ", "جحخ", "دذ", "رز", "سش", "صض", "طظ", "عغ", "فق",
         "هة", "ىياأإآ", "وؤ", "كل"]
_RASM_ID: dict[str, set[int]] = {}
for _i, _g in enumerate(_RASM):
    for _ch in _g:
        _RASM_ID.setdefault(_ch, set()).add(_i)


def ocr_distance(a: str, b: str) -> float:
    """Levenshtein weighted by how plausible each substitution is as an OCR confusion.

    Normalised to [0, 1] by the longer string. Used as a sanity guard on a proposed
    fill: a genuine repair of a garbled word stays near it, while a word accidentally
    borrowed from elsewhere in the document (difflib alignment drift on a whole-document
    corrector) lands far away and can be rejected on sight.
    """
    n, m = len(a), len(b)
    if not n and not m:
        return 0.0
    prev = [float(j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [float(i)]
        for j in range(1, m + 1):
            ca, cb = a[i - 1], b[j - 1]
            if ca == cb:
                sub = 0.0
            elif _RASM_ID.get(ca, set()) & _RASM_ID.get(cb, set()):
                sub = 0.25
            else:
                sub = 1.0
            cur.append(min(prev[j] + 1.0, cur[j - 1] + 1.0, prev[j - 1] + sub))
        prev = cur
    return prev[m] / max(n, m)


def _project(src: list[str], dst: list[str]) -> list[str | None]:
    """For each source word, the destination words aligned to it.

    Anchoring on whole difflib spans is too coarse: if the model rewrites three
    adjacent words and the ground truth only needed one changed, the spans differ and a
    genuine fix gets scored as damage. Projecting word by word keeps each site separate.
    An unequal-length replacement is folded onto its first position, which is an
    approximation but a local one.
    """
    out: list[str | None] = [None] * len(src)
    sm = difflib.SequenceMatcher(None, src, dst, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
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


def classify(src_text: str, pred_text: str, gt_text: str,
             ortho_equiv: bool = False) -> list[tuple[str, str, str, str]]:
    """Sort every edit site into (kind, original, prediction, ground truth).

    With `ortho_equiv`, words are compared through `ortho_key`, so a difference that is
    purely conventional (hamza seat, ى/ي, ة/ه, edge punctuation) counts as agreement --
    for deciding both whether an edit was NEEDED and whether it was RIGHT.
    """
    key = (lambda w: ortho_key(w)) if ortho_equiv else (lambda w: w)
    src = src_text.split()
    pred = _project(src, pred_text.split())
    gt = _project(src, gt_text.split())
    events: list[tuple[str, str, str, str]] = []
    for i, word in enumerate(src):
        changed, needed = key(pred[i]) != key(word), key(gt[i]) != key(word)
        if changed and needed:
            events.append(("good" if key(pred[i]) == key(gt[i]) else "wrong_fix",
                           word, pred[i], gt[i]))
        elif changed and not needed:
            events.append(("bad_edit", word, pred[i], word))
        elif needed and not changed:
            events.append(("missed", word, word, gt[i]))
    return events


def score(triples: Iterable[tuple[str, str, str]],
          ortho_equiv: bool = False) -> dict[str, float]:
    """Aggregate edit metrics over (input, prediction, ground truth) triples."""
    n = {"good": 0, "wrong_fix": 0, "bad_edit": 0, "missed": 0}
    for src, pred, gt in triples:
        for kind, *_ in classify(src, pred, gt, ortho_equiv):
            n[kind] += 1

    made = n["good"] + n["wrong_fix"] + n["bad_edit"]
    present = n["good"] + n["wrong_fix"] + n["missed"]
    precision = n["good"] / made if made else 0.0
    recall = n["good"] / present if present else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "edit_precision": precision,
        "edit_recall": recall,
        "edit_f1": f1,
        # of everything it chose to touch, how much was already correct
        "damage_rate": n["bad_edit"] / made if made else 0.0,
        "edits_good": float(n["good"]),
        "edits_wrong": float(n["wrong_fix"]),
        "edits_damaging": float(n["bad_edit"]),
        "edits_missed": float(n["missed"]),
    }
