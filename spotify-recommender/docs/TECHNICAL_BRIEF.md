# Technical Brief

## Objective
Predict play-level completion probability (`not skipped`) and convert scores into ranked recommendation lists.

## Data Modalities
- Spotify account export: identity, saved library
- Extended streaming history: play events across multiple years
- Technical logs: daily connection/playback error counts

## Core Challenges
- Family-plan multi-user contamination in exported events
- Session-dependent user intent
- Sparse track histories and noisy one-off interactions
- Recommendation ranking must be connected to skip prediction outputs

## Key Design Decisions
1. Session-aware features
- Session boundaries: new session if gap > 30 minutes
- Sequential context: `session_position`, `recent_skip_rate_10`
- Temporal context: hour/day/month/weekend

2. Attribution filter for primary user
- Session-level scoring with country, platform, time window, artist affinity, and driving hints
- Configurable thresholds in `primary_user_config.json`
- Prevents blended family-profile behavior from degrading personalization quality

3. Leakage controls
- Chronological train/test split only
- Track/artist priors computed from train slice and merged forward
- Backtests run on forward windows, never random shuffle

4. Ranking bridge
- Recommendation score:
  - `0.45 * predicted_completion`
  - `0.35 * bayesian_track_completion`
  - `0.20 * artist_completion_prior`
  - multiplied by confidence from play evidence

5. Dual recommendation outputs
- `top_resume_playlist.csv`: pure global rank
- `top_favorites_playlist.csv`: preference-aware rerank with per-favorite coverage guarantees

6. Cold-start fallback
- `cold_start_recommendations.csv`
- Heuristic fallback for sparse/new contexts using artist completion prior + popularity

## Evaluation
- Classification: ROC-AUC, PR-AUC, Accuracy, F1, Logloss
- Ranking: NDCG@K, MAP@K, Recall@K (session-based)
- Stability: rolling chronological backtests
- Model comparison: logistic baseline, reduced baseline, heuristic, optional gradient boosting

## Current Snapshot (Primary-User Filtered)
- ROC-AUC: 0.8657
- PR-AUC: 0.8214
- F1: 0.7381
- NDCG@10: 0.8654
- Coverage retained after attribution: 89.82%

## Known Limitations
- No content/audio embedding model yet
- Cold-start fallback is heuristic, not learned
- Candidate retrieval stage is not separated from reranking stage

## Next Improvements
- Add embedding-driven candidate generation
- Add calibrated probability diagnostics and threshold policy by product objective
- Add stronger tree/sequence models under strict leakage controls
