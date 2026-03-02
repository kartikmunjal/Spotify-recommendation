# Reproducibility Guide

## 1) Environment
```bash
cd spotify-recommender
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-optional.txt
```

## 2) Required Local Data Folders
Expected as siblings of this repository:
- `../Spotify Account Data`
- `../Spotify Extended Streaming History`
- `../Spotify Technical Log Information`

## 3) Train + Evaluate + Generate Recommendations
```bash
python3 src/pipeline.py \
  --account-dir "../Spotify Account Data" \
  --extended-dir "../Spotify Extended Streaming History" \
  --tech-dir "../Spotify Technical Log Information" \
  --output-dir "./outputs" \
  --filter-primary-user true \
  --primary-user-config "./primary_user_config.json" \
  --favorite-artists-config "./favorite_artists.json"
```

## 4) Generate Visuals
```bash
python3 src/visualize.py \
  --scored-samples "./outputs/scored_samples.csv" \
  --out-dir "./outputs/figures"
```

## 5) Launch Dashboard
```bash
python3 -m streamlit run app.py
```

## 6) Key Artifacts to Inspect
- `outputs/RESULTS.md`
- `outputs/model_comparison.csv`
- `outputs/dataset_comparison.csv`
- `outputs/rolling_backtests.csv`
- `outputs/top_resume_playlist.csv`
- `outputs/top_favorites_playlist.csv`
- `outputs/cold_start_recommendations.csv`
