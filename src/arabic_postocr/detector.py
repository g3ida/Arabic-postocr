"""Stage A, and Stage D: AraELECTRA answering "is this word an OCR error?".

The same model does both jobs. Stage A reads the raw OCR and decides where to look;
Stage D reads the pipeline's own output and decides whether to believe it. They are the
same question asked about different text, which is why one model serves both and why the
second pass costs a single forward pass rather than a new model.
"""

from __future__ import annotations

from .text import detector_view, word_spans


class Detector:
    """Per-word OCR-error detection.

    `threshold` is a logit, not a probability: 0.0 is p=0.5. Higher means fewer flags,
    which trades recall for precision and for a lower false-alarm rate. The pipeline pays
    for a false alarm twice -- a wasted edit, and possible damage -- but for a miss only
    once, so its optimum sits above the detector's own F1 peak.

        threshold 0.0   precision 93.8%   recall 89.8%   false alarms 6.2%
        threshold 0.5   precision 95.3%   recall 87.7%   false alarms 4.7%   <- default
        threshold 1.0   precision 96.6%   recall 85.5%   false alarms 3.4%
    """

    def __init__(self, model, tokenizer, device: str = "cpu",
                 max_length: int = 512) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    @classmethod
    def from_pretrained(cls, name_or_path: str, device: str = "cpu",
                        half: bool | None = None, **kwargs):
        from transformers import AutoModelForPreTraining, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        model = AutoModelForPreTraining.from_pretrained(name_or_path).to(device).eval()
        if half is None:
            half = device != "cpu"
        if half:
            model = model.half()
        return cls(model, tokenizer, device, **kwargs)

    def _logits(self, texts: list[str], batch_size: int):
        """Yield (normalised text, word spans, per-token logits, offsets) per document."""
        import torch

        for start in range(0, len(texts), batch_size):
            chunk = [detector_view(t) for t in texts[start:start + batch_size]]
            enc = self.tokenizer(chunk, truncation=True, max_length=self.max_length,
                                 padding=True, return_offsets_mapping=True,
                                 return_tensors="pt")
            offsets = enc.pop("offset_mapping").tolist()
            moved = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**moved).logits.float().cpu().tolist()
            for text, offs, row in zip(chunk, offsets, logits):
                yield text, word_spans(text), row, offs

    def flags(self, texts: list[str], threshold: float = 0.5,
              batch_size: int = 32) -> list[list[int]]:
        """Per word: 1 if the detector thinks it is an OCR error.

        A word is flagged if *any* of its subword tokens fires. Tokens past the truncation
        point cannot fire, so words in a very long document default to "clean" -- which is
        the safe direction here, producing a miss rather than damage.
        """
        out: list[list[int]] = []
        for _text, spans, row, offs in self._logits(texts, batch_size):
            marks = [0] * len(spans)
            for value, (a, b) in zip(row, offs):
                if value <= threshold or a == b:
                    continue
                for i, (ws, we, _) in enumerate(spans):
                    if a < we and b > ws:
                        marks[i] = 1
                        break
            out.append(marks)
        return out

    def scores(self, texts: list[str],
               batch_size: int = 32) -> list[list[float | None]]:
        """Per word, the strongest objection to it: max logit over its subword tokens.

        `None` means *no opinion* -- the word fell past `max_length` and was never scored.
        That is emphatically not the same as a clean score, and callers must not conflate
        them. Reading an unscored word as "confidently fine" is how a truncated document
        turns into a silent auto-accept.
        """
        out: list[list[float | None]] = []
        for _text, spans, row, offs in self._logits(texts, batch_size):
            best: list[float | None] = [None] * len(spans)
            for value, (a, b) in zip(row, offs):
                if a == b:
                    continue
                for i, (ws, we, _) in enumerate(spans):
                    if a < we and b > ws:
                        best[i] = value if best[i] is None else max(best[i], value)
                        break
            out.append(best)
        return out
