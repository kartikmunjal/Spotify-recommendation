# Spotify Recommender

A session-aware recommendation pipeline built from Spotify export data.

## Key Results
- Primary-user filtered model: ROC-AUC `0.8657`, PR-AUC `0.8214`, F1 `0.7381`
- Session ranking quality: NDCG@10 `0.8654`, MAP@10 `0.7777`, Recall@10 `0.7451`
- Family-plan attribution filter improved full-model AUC from `0.8553` to `0.8657`

## Problem
Given user listening events, estimate the probability that a track play will complete (not be skipped), then rank candidate tracks by expected completion to create recommendation-ready outputs.

## Why This Project
- Uses multiple Spotify export modalities together: account profile, long-horizon streaming history, and technical logs.
- Applies chronological evaluation instead of random split to better reflect real deployment behavior.
- Produces interpretable artifacts for product and modeling discussions.

## System Design
1. Ingest data from three export sources.
2. Build temporal/session features (hour, day, session position, recent skip rate).
3. Attribute likely primary-user sessions to handle family-plan multi-user noise.
4. Add behavioral priors (track/artist skip rates from train window only).
5. Merge daily technical reliability signals (connection/playback errors).
6. Train logistic regression (NumPy implementation) on chronological train split.
7. Score plays and generate top-ranked recommendation candidates.

## Session Features (Explicit)
- `hour`, `day_of_week`, `month`, `is_weekend`: temporal context of listening behavior
- `session_position`: position of track inside current session (30-minute gap based)
- `recent_skip_rate_10`: skip rate over preceding 10 events, capturing local intent/friction
- `reason_start`, `reason_end`: playback transitions (e.g., trackdone, fwdbtn)
- `platform`, `conn_country`: environment and geo context
- `track_train_skip_rate`, `artist_train_skip_rate`: historical priors from train window only

## Key Features
- Time: `hour`, `day_of_week`, `month`, `is_weekend`
- Session: `session_position`, `recent_skip_rate_10`
- Behavior priors: `track_train_skip_rate`, `artist_train_skip_rate`, play counts
- Context: `platform`, `conn_country`, `reason_start`, `reason_end`
- Reliability: `connection_error_count`, `playback_error_count`

## Outputs (Local Only)
The pipeline writes these to `outputs/` (ignored in git):
- `model_metrics.json`
- `feature_importance.csv`
- `top_resume_playlist.csv`
- `top_favorites_playlist.csv` (favorite-artist boosted rerank)
- `scored_samples.csv`
- `resume_project_summary.md`
- `model_comparison.csv` and `model_comparison.json`
- `dataset_comparison.csv` and `dataset_comparison.json`
- `attribution_summary.json` and `user_attribution_report.csv`
- `RESULTS.md` (summary of classification + ranking metrics)
- `figures/*.svg`, `figures/dashboard.html`, and `figures/visualization_summary.md`

## Recommendation Bridge
Recommendations are generated from skip/completion predictions using an explicit ranking function:

`score = (0.45 * predicted_completion + 0.35 * bayesian_track_completion + 0.20 * artist_completion_prior) * confidence(plays)`

- `top_resume_playlist.csv`: pure global ranking output
- `top_favorites_playlist.csv`: favorite-artist aware rerank (configured in `favorite_artists.json`)

## Visual Diagnostics

### 1) Completion Rate by Hour
<img width="2838" height="1450" alt="image" src="https://github.com/user-attachments/assets/b4dab451-c609-4355-ae06-a5904b4d94f8" />

**Takeaway:** Completion probability varies by time-of-day, indicating that temporal context is a meaningful recommendation feature.

### 2) Prediction Score Distribution (Completed vs Skipped)
<img width="2864" height="1450" alt="image" src="https://github.com/user-attachments/assets/479a5e97-8d80-4f9a-a76e-aa37bc9a08bf" />

**Takeaway:** The model assigns higher completion probabilities to completed plays than skipped plays, showing good class separation.

### 3) Completion Rate for Most Played Artists
<img width="2868" height="1638" alt="image" src="https://github.com/user-attachments/assets/2bb8915d-ef14-4064-9e61-237a7dc58781" />

**Takeaway:** Artist-level behavior is strongly differentiated, supporting the use of artist priors in ranking.

### 4) Platform Mix
<img width="2864" height="1452" alt="image" src="https://github.com/user-attachments/assets/562b472a-60b1-4b1b-bb3d-18147b56345b" />

**Takeaway:** Listening context is platform-dependent, which justifies platform features in completion prediction. Not_applic depicts to data listened to while driving

## Run
```bash
cd spotify-recommender
python3 src/pipeline.py \
  --account-dir "../Spotify Account Data" \
  --extended-dir "../Spotify Extended Streaming History" \
  --tech-dir "../Spotify Technical Log Information" \
  --output-dir "./outputs" \
  --filter-primary-user true \
  --primary-user-config "./primary_user_config.json" \
  --favorite-artists-config "./favorite_artists.json"
```

Primary-user attribution is configured in:
- `primary_user_config.json`
Favorite-artist reranking is configured in:
- `favorite_artists.json`

## Model Benchmarking
- Baselines: `full_logistic`, `baseline_logistic`, `track_artist_heuristic`
- Additional benchmark: `hist_gradient_boosting` (runs automatically when `scikit-learn` is available)
- If unavailable, it is reported as `status=not_available` in `model_comparison.csv`
- Install optional benchmark dependency with:
  - `pip install -r requirements-optional.txt`

## Visualize
```bash
cd spotify-recommender
python3 src/visualize.py \
  --scored-samples "./outputs/scored_samples.csv" \
  --out-dir "./outputs/figures"
```

## Repo Structure
- `src/pipeline.py`: full training + scoring + recommendation pipeline
- `run_pipeline.sh`: convenience runner
- `requirements.txt`: dependencies

## Privacy and Security
This public repository excludes all personal Spotify export files and generated personal model outputs.

## Limitations and Next Steps
- Cold-start users/tracks are not fully solved yet; current model relies on behavioral priors and session context.
- Future work: content/audio embeddings, candidate retrieval, and dedicated cold-start rankers.
