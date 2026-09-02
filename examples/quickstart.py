"""Minimal end-to-end example.

Run:  python examples/quickstart.py
"""

from arabic_postocr import Corrector

# Real OCR output: a wrong dot (این/ابن), a broken word join (بنمحمد), a garbled
# name (وكيفي،) and a missing letter (سن/سنن).
OCR = ("سن این ماجه - ت عبد الباقي ٢٥٩٣ - حدثنا علي بنمحمد "
       "قال حدثنا وكيفي، عن سفيان")

corrector = Corrector.from_pretrained()

print("OCR      ", OCR)
print()

plain = corrector.correct(OCR)
print("corrected", plain.text)
for edit in plain.edits:
    print("   ", edit)
print()

# With the gate, repairs the detector still objects to are reverted and reported
# instead of being applied.
reviewed = corrector.correct(OCR, gate=0.0)
print("gated    ", reviewed.text)
print("review   ", reviewed.hard or "(nothing needs a human)")
