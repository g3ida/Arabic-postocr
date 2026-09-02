"""Stage B: AraBART rewriting the spans the detector flagged."""

from __future__ import annotations

from .text import ERR_CLOSE, ERR_OPEN


def _wraps_sequences(tokenizer) -> bool:
    """Does this tokenizer actually wrap sequences in <s> ... </s>?

    AraBART ships only a raw sentencepiece model, and the conversion to a fast tokenizer
    in transformers 5.x drops the post-processing template -- so `add_special_tokens=True`
    silently adds nothing. That is not cosmetic. Generating from differently-wrapped
    inputs than the model was trained on degrades everything downstream, quietly.

    Detected rather than assumed, so this stays correct if the tokenizer is ever fixed
    upstream.
    """
    ids = tokenizer("سنن ابن ماجه", add_special_tokens=True)["input_ids"]
    return bool(ids) and ids[-1] == tokenizer.eos_token_id


class Infiller:
    """Sequence-to-sequence repair of marked spans."""

    def __init__(self, model, tokenizer, device: str = "cpu",
                 max_length: int = 256) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.adds_specials = not _wraps_sequences(tokenizer)

    @classmethod
    def from_pretrained(cls, name_or_path: str, device: str = "cpu",
                        half: bool | None = None, **kwargs):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(name_or_path).to(device).eval()
        if half is None:
            half = device != "cpu"
        if half:
            model = model.half()
        return cls(model, tokenizer, device, **kwargs)

    def generate(self, marked: list[str], batch_size: int = 16,
                 num_beams: int = 1) -> list[str]:
        """Repair a batch of marked documents.

        `num_beams=1` is the default because it was measured to be *better* here, not
        merely faster: greedy decoding beat beam 8 on CER (+20.8% against +18.6%).
        """
        import torch

        out: list[str] = []
        for start in range(0, len(marked), batch_size):
            chunk = marked[start:start + batch_size]
            if self.adds_specials:
                ids = self.tokenizer(chunk, add_special_tokens=False, truncation=True,
                                     max_length=self.max_length - 2)["input_ids"]
                ids = [[self.tokenizer.bos_token_id] + x + [self.tokenizer.eos_token_id]
                       for x in ids]
                enc = self.tokenizer.pad({"input_ids": ids},
                                         return_tensors="pt").to(self.device)
            else:
                enc = self.tokenizer(chunk, truncation=True, max_length=self.max_length,
                                     padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                generated = self.model.generate(**enc, max_length=self.max_length,
                                                num_beams=num_beams)
            out.extend(t.strip() for t in
                       self.tokenizer.batch_decode(generated, skip_special_tokens=True))
        return out

    @staticmethod
    def respace_markers(text: str) -> str:
        """Sentencepiece decodes markers glued to their neighbours; separate them again.

        Decoding produces `علي<err>` and `ان</err>` rather than separated tokens, so any
        code that splits the model's *input* back into words has to undo that first.
        """
        return text.replace(ERR_OPEN, f" {ERR_OPEN} ").replace(ERR_CLOSE, f" {ERR_CLOSE} ")
