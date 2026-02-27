# Spotify Recommender

This project builds a ML pipeline from my Spotify data export.

## Use Case
Predict whether a track play will complete (not be skipped), then rank tracks by predicted completion probability to generate high-confidence playlist recommendations.

Spotify applications:
- Uses real Spotify export modalities: account data, extended streaming history, and technical client logs.
- Demonstrates end-to-end ML lifecycle: ingestion, feature engineering, modeling, evaluation, and product-style output.
- Produces measurable metrics and an artifact (`top_resume_playlist.csv`) that maps model output to user value.

## Data Inputs
Expected sibling folders (same parent directory):
- `Spotify Account Data`
- `Spotify Extended Streaming History`
- `Spotify Technical Log Information`

## Project Structure
- `src/pipeline.py`: full training + recommendation pipeline
- `outputs/`: local generated artifacts (ignored in git for privacy)

## Setup
```bash
cd spotify-recommender
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
From `spotify-recommender` directory:
```bash
python src/pipeline.py \
  --account-dir "../Spotify Account Data" \
  --extended-dir "../Spotify Extended Streaming History" \
  --tech-dir "../Spotify Technical Log Information" \
  --output-dir "./outputs"
```
