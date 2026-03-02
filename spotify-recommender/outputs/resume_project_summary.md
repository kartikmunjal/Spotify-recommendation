# Spotify Recommender Project

**Profile analyzed:** Kartik Munjal

## Model Performance (Holdout Test)
- Rows (train/test): 172455 / 43114
- ROC-AUC: 0.8657
- PR-AUC: 0.8214
- Accuracy: 0.7881
- F1 Score: 0.7381
- NDCG@10: 0.8654
- Primary-user coverage kept: 215569/240001 (89.82%)

## Resume Bullets
- Built an end-to-end Spotify listening intelligence pipeline over multi-year streaming and client telemetry data.
- Added a primary-user session attribution layer to reduce family-plan multi-user noise before model training.
- Trained and compared recommendation models with chronological holdout validation and ranking metrics (NDCG/MAP/Recall@K).