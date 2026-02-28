# Spotify Recommender

This repository contains a production-style ML project that predicts playback completion probability and ranks tracks for personalized recommendation.

## Project Highlights
This project is structured to demonstrate:
- End-to-end ownership: ingestion, feature engineering, model training, evaluation, and recommendation outputs.
- Product framing: optimize completion probability, not just offline model metrics.
- Multi-model evaluation with ranking metrics (`NDCG`, `MAP`, `Recall@K`).
- Primary-user attribution filtering to reduce family-plan multi-user noise.
- Visualization of listening behavior and model score distributions.
- Responsible data handling: no personal export data or generated personal outputs are published.

## Repository Contents
- `spotify-recommender/`: model code, pipeline script, and project documentation.

## Snapshot Metrics
- ROC-AUC: `0.8657`
- PR-AUC: `0.8214`
- F1: `0.7381`
- NDCG@10: `0.8654`

## Local Data Contract
Run the project with Spotify export folders present locally as siblings to this repo:
- `../Spotify Account Data`
- `../Spotify Extended Streaming History`
- `../Spotify Technical Log Information`

## Quick Start
```bash
cd spotify-recommender
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/pipeline.py \
  --account-dir "../Spotify Account Data" \
  --extended-dir "../Spotify Extended Streaming History" \
  --tech-dir "../Spotify Technical Log Information" \
  --output-dir "./outputs"
```

## Privacy
Ignored by git:
- raw account/streaming/log exports
- generated recommendation outputs and model metrics
