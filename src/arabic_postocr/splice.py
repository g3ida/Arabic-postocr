"""Stage C: rebuild the document, keeping the model's output only where it was invited.

This is the load-bearing stage of the whole design. A word the detector did not flag is
copied from the *original* string, byte for byte, diacritics intact -- whatever the model
emitted at that position is discarded.

The consequence is a hard bound rather than a hope: the damage this pipeline can do is
capped by the detector's false-alarm rate (4.7%), not by a decoder's willingness to keep
talking. An unbounded corrector on the same data damaged 8-22% of the text it touched and
once emitted 748 consecutive dots. Neither failure is reachable from here.

The cost is equally concrete: an error the detector misses can never be repaired, because
nothing downstream is permitted to look at it. Recall is capped by Stage A.
"""

from __future__ import annotations

from typing import NamedTuple

from .text import project


class Span(NamedTuple):
    """One flagged region: which source words it covers, and where its repair landed."""

    start: int                 #: first source word index, inclusive
    end: int                   #: last source word index, exclusive
    output: list[int]          #: indices into the spliced output; empty means deleted


def splice(original: list[str], flags: list[int], generated: str,
           stripped: list[str]) -> tuple[list[str], list[Span]]:
    """Returns (output words, one Span per flagged region).

    Spans are grouped exactly the way `mark_spans` grouped them for the model: a run of
    consecutive flagged words is ONE region. Tracking regions rather than individual words
    is not a detail -- when the model answers a three-word span with two words, the
    alignment folds both onto the region's first position and the other two source words
    map to nothing. Anything that later wants to undo that repair has to restore the whole
    region, because the region is what was replaced.
    """
    projected = project(stripped, [w for w in generated.split(" ") if w])
    out: list[str] = []
    spans: list[Span] = []
    i = 0
    while i < len(original):
        if not flags[i]:
            out.append(original[i])       # verbatim, diacritics and all
            i += 1
            continue
        j = i
        while j < len(original) and flags[j]:
            j += 1
        start = len(out)
        for k in range(i, j):
            out.extend(projected[k].split())
        # an empty region is a deletion, which is a real and needed operation here:
        # page numbers, running headers and folio marks are errors best removed
        spans.append(Span(i, j, list(range(start, len(out)))))
        i = j
    return out, spans
