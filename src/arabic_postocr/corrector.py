"""The assembled corrector."""

from __future__ import annotations

from dataclasses import dataclass, field

from .detector import Detector
from .infiller import Infiller
from .splice import splice
from .text import detector_view, mark_spans, strip_diacritics

DEFAULT_DETECTOR = "g3ida/araelectra-arabic-ocr-detector"
DEFAULT_INFILLER = "g3ida/arabart-arabic-ocr-infiller"


@dataclass
class Edit:
    """One accepted repair."""
    original: str
    corrected: str
    index: int            # word index in the source document

    def __str__(self) -> str:
        return f"{self.original} -> {self.corrected or '(deleted)'}"


@dataclass
class Correction:
    """The result of correcting one document."""

    text: str
    edits: list[Edit] = field(default_factory=list)
    hard: list[str] = field(default_factory=list)
    """Words the detector flagged and no repair satisfied it. Left exactly as the OCR
    wrote them, and reported here as a review queue. These are not failures the pipeline
    hides -- they are the ones it declines to guess at."""

    def __str__(self) -> str:
        return self.text


class Corrector:
    """Detect -> mark -> infill -> splice -> re-read.

        >>> corrector = Corrector.from_pretrained()
        >>> result = corrector.correct("سنن این ماجه - حدثنا علي بنمحمد")
        >>> result.text
        'سنن ابن ماجه - حدثنا علي بن محمد'

    Set `gate` to enable a second reading: the detector re-examines the corrector's own
    output and vetoes repairs it still objects to, reverting them to the original text and
    listing them in `Correction.hard`. Measured on 1,000 unseen-book documents, `gate=0.0`
    moves edit precision from 59.1% to 64.8% at a small recall cost (49.4% to 48.5%).

    `gate_max_words` limits the veto to regions of at most that many words, and defaults
    to 1 for a measured reason. A veto restores the whole region, because the region is
    what the model replaced -- but the detector's objection is to a single word. Vetoing a
    multi-word region discards the good repairs beside the bad one, and the alignment
    cannot say which output word came from which source word once an unequal-length
    replacement has been folded onto the region's head:

        no gate                     precision 59.1%   recall 49.4%   F1 53.8
        gate, single-word regions   precision 64.8%   recall 48.5%   F1 55.5

    Raise the limit only if you want a larger review queue and accept losing repairs
    to get it.
    """

    def __init__(self, detector: Detector, infiller: Infiller) -> None:
        self.detector = detector
        self.infiller = infiller

    @classmethod
    def from_pretrained(cls, detector: str = DEFAULT_DETECTOR,
                        infiller: str = DEFAULT_INFILLER,
                        device: str | None = None, **kwargs):
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:                                    # pragma: no cover
                device = "cpu"
        return cls(Detector.from_pretrained(detector, device, **kwargs),
                   Infiller.from_pretrained(infiller, device, **kwargs))

    # ------------------------------------------------------------------ public API
    def correct(self, text: str, **kwargs) -> Correction:
        """Correct a single document."""
        return self.correct_batch([text], **kwargs)[0]

    def correct_batch(self, texts: list[str], threshold: float = 0.5,
                      gate: float | None = None, gate_max_words: int = 1,
                      num_beams: int = 1, batch_size: int = 16,
                      detect_batch_size: int = 32) -> list[Correction]:
        """Correct many documents. Order is preserved."""
        flags = self.detector.flags(texts, threshold, detect_batch_size)

        # Documents with nothing flagged never reach the generator: there is nothing to
        # repair, and running one would only create an opportunity for damage.
        prepared, todo = [], []
        for i, (text, flag) in enumerate(zip(texts, flags)):
            words = [w for w in detector_view(text).split(" ") if w]
            if len(words) != len(flag):                            # pragma: no cover
                prepared.append(None)
                continue
            kept = [(w, s, f) for w, s, f in
                    ((w, strip_diacritics(w), f) for w, f in zip(words, flag)) if s]
            if not kept or not any(f for _, _, f in kept):
                prepared.append(None)
                continue
            entry = ([k[0] for k in kept], [k[1] for k in kept], [k[2] for k in kept])
            prepared.append(entry)
            todo.append(i)

        marked = [mark_spans(prepared[i][1], prepared[i][2]) for i in todo]
        filled = self.infiller.generate(marked, batch_size, num_beams)
        generated = dict(zip(todo, filled))

        spliced, spans = {}, {}
        for i in todo:
            original, stripped, flag = prepared[i]
            spliced[i], spans[i] = splice(original, flag, generated[i], stripped)

        scores = {}
        if gate is not None and todo:
            texts_to_score = [" ".join(spliced[i]) for i in todo]
            scores = dict(zip(todo, self.detector.scores(texts_to_score,
                                                         detect_batch_size)))

        results: list[Correction] = []
        for i, text in enumerate(texts):
            if prepared[i] is None:
                results.append(Correction(text=text))
                continue
            results.append(self._assemble(prepared[i], spliced[i], spans[i],
                                          scores.get(i), gate, gate_max_words))
        return results

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _assemble(prepared, words, spans, scores, gate, gate_max_words=1) -> Correction:
        """Rebuild the document region by region, deciding each flagged region once.

        Rebuilt rather than patched. An earlier version spliced first and then reverted
        vetoed positions in place, which lost words: when a three-word region is answered
        with two, reverting the position that carried the fill left its siblings deleted.
        A region is the unit that was replaced, so it has to be the unit that is restored.
        """
        original, stripped, _flags = prepared
        pieces: list[str] = []
        edits: list[Edit] = []
        hard: list[str] = []
        cursor = 0

        for span in spans:
            pieces.extend(original[cursor:span.start])       # untouched, verbatim
            cursor = span.end
            source = stripped[span.start:span.end]
            fill = [words[j] for j in span.output if j < len(words)]

            if (gate is not None and scores is not None
                    and span.end - span.start <= gate_max_words):
                seen = [scores[j] for j in span.output
                        if j < len(scores) and scores[j] is not None]
                # No score at all means the region fell past the detector's context
                # window, so there is no second opinion. Keep the repair rather than
                # vetoing on absence of evidence.
                if seen and max(seen) > gate:
                    pieces.extend(source)                    # the WHOLE region back
                    hard.extend(source)
                    continue

            pieces.extend(fill)
            if fill != source:
                edits.append(Edit(" ".join(source), " ".join(fill), span.start))

        pieces.extend(original[cursor:])
        return Correction(text=" ".join(pieces), edits=edits, hard=hard)
