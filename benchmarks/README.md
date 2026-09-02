# Benchmarks

Two self-contained sets, one JSON document per line:

- `eval.jsonl` — the 1,000 documents behind the README numbers (947 real scans
  from *Musnad Ahmad* / *al-Matalib al-Aliyah*, 53 synthetic Shamela pages).
  Fields: `input` (OCR), `target` (truth), `source`, `doc_id`.
- `data/{ara,msa}.jsonl` — 300 ground-truth pages each, classical vs modern
  (`gt` + image reference).
- `data/{ara,msa}.{ppocr,ppocr_vl,tesseract}.jsonl` — the same pages as read by
  three OCR engines (`id` + `text`), for cross-engine scoring.

`collect.py` and `run_ocr.py` are the scripts that built them (need Hugging Face
access and the OCR engines); the `.jsonl` files are the result, loadable with
`datasets.load_dataset("json", data_files=...)`.

`score.py` reproduces the README numbers (needs torch + transformers):

```bash
python benchmarks/score.py --eval benchmarks/eval.jsonl             # headline row
python benchmarks/score.py --eval benchmarks/eval.jsonl --gate 0.0  # gated row
python benchmarks/score.py --gt data/msa.jsonl --hyp data/msa.tesseract.jsonl  # raw CER
```
