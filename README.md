# Spotify Recommender

This repository contains a Spotify listening project that:
- predicts play completion vs skip from historical listening events
- ranks tracks using the prediction output and behavior priors
- evaluates results with chronological holdout and rolling backtests

## What To Read First
- [Project README](./spotify-recommender/README.md)
- [Technical Brief](./spotify-recommender/docs/TECHNICAL_BRIEF.md)
- [Reproducibility Guide](./spotify-recommender/docs/REPRODUCIBILITY.md)

## Output Artifacts
`outputs/` is not committed to this public repo (privacy-safe by design).

Generate locally, then review:
- `outputs/RESULTS.md`
- `outputs/model_comparison.csv`
- `outputs/rolling_backtests.csv`
- `outputs/top_resume_playlist.csv`
- `outputs/top_favorites_playlist.csv`

## Current Metrics
- ROC-AUC: `0.8657`
- PR-AUC: `0.8214`
- F1: `0.7381`
- NDCG@10: `0.8654`
- Attribution filtering keeps `89.82%` of rows and improves AUC from `0.8553` to `0.8657`

## Repository Layout
- `spotify-recommender/`: pipeline code and project docs
- `spotify-recommender/src/`: modeling and visualization scripts
- `spotify-recommender/outputs/`: generated metrics and recommendation artifacts (local)

## Data Layout (Local)
Expected sibling folders:
- `../Spotify Account Data`
- `../Spotify Extended Streaming History`
- `../Spotify Technical Log Information`

## Run
```bash
cd spotify-recommender
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-optional.txt
python src/pipeline.py \
  --account-dir "../Spotify Account Data" \
  --extended-dir "../Spotify Extended Streaming History" \
  --tech-dir "../Spotify Technical Log Information" \
  --output-dir "./outputs" \
  --filter-primary-user true \
  --primary-user-config "./primary_user_config.json" \
  --favorite-artists-config "./favorite_artists.json"

python src/visualize.py \
  --scored-samples "./outputs/scored_samples.csv" \
  --out-dir "./outputs/figures"

python3 -m streamlit run app.py
```

## Privacy
Ignored by git:
- raw account/streaming/log exports
- generated personal recommendation outputs and model metrics
