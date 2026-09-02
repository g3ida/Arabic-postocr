"""Arabic post-OCR correction: a detector that finds errors and an infiller that repairs
only what it found.

    from arabic_postocr import Corrector

    corrector = Corrector.from_pretrained()
    result = corrector.correct(ocr_text, gate=0.0)

    print(result.text)      # corrected
    print(result.hard)      # words needing a human
    print(result.edits)     # what changed
"""

from .corrector import Correction, Corrector, Edit
from .detector import Detector
from .infiller import Infiller
from .splice import splice
from .text import detector_view, mark_spans, strip_diacritics

__version__ = "0.1.0"
__all__ = ["Corrector", "Correction", "Edit", "Detector", "Infiller", "splice",
           "strip_diacritics", "detector_view", "mark_spans"]
