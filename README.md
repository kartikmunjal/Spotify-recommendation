# Spotify Data + ML Project

This repository contains:
- `spotify-recommender/` (ML pipeline code)

Privacy note:
- Raw Spotify exports and generated personal outputs are intentionally excluded from this public repository.
- Keep data locally in sibling folders:
  - `../Spotify Account Data`
  - `../Spotify Extended Streaming History`
  - `../Spotify Technical Log Information`

Run pipeline:
```bash
cd spotify-recommender
python3 src/pipeline.py \
  --account-dir "../Spotify Account Data" \
  --extended-dir "../Spotify Extended Streaming History" \
  --tech-dir "../Spotify Technical Log Information" \
  --output-dir "./outputs"
```
