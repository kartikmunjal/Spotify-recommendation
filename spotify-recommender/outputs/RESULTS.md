# Results

- Profile: Kartik Munjal
- Train rows: 172455
- Test rows: 43114

## Dataset Attribution
- Filter enabled: True
- Rows kept as primary-user: 215569/240001 (89.82%)
- Sessions kept: 8740/11004

## Primary Model (Full Feature Set)
- ROC-AUC: 0.8657
- PR-AUC: 0.8214
- Accuracy: 0.7881
- F1: 0.7381
- Recommendation score bridge: score = (0.45 * predicted_completion + 0.35 * bayesian_track_completion + 0.20 * artist_prior) * confidence(plays).

## Ranking Metrics (Session-Based)
- Sessions evaluated: 1480
- NDCG@5: 0.8529
- MAP@5: 0.7864
- Recall@5: 0.5325
- NDCG@10: 0.8654
- MAP@10: 0.7777
- Recall@10: 0.7451

## Dataset Comparison (Full Model)
- all_rows: rows=240001, AUC=0.8553, PR-AUC=0.8053, Accuracy=0.7795, F1=0.7224
- primary_user_filtered: rows=215569, AUC=0.8657, PR-AUC=0.8214, Accuracy=0.7881, F1=0.7381

## Model Comparison
- full_logistic: AUC=0.8657, PR-AUC=0.8214, Accuracy=0.7881, F1=0.7381
- baseline_logistic: AUC=0.8016, PR-AUC=0.7236, Accuracy=0.7241, F1=0.6615
- track_artist_heuristic: AUC=0.5030, PR-AUC=0.4019, Accuracy=0.4950, F1=0.4157
- hist_gradient_boosting: AUC=0.7031, PR-AUC=0.6107, Accuracy=0.6905, F1=0.5770

## Top Features (Absolute Coefficient)
- recent_skip_rate_10: -1.42854823
- reason_end=trackdone: 0.61747081
- track_train_skip_rate: -0.60486393
- ms_played: 0.54482927
- platform=ios: -0.53461047
- conn_country=US: 0.39426721
- reason_end=fwdbtn: -0.29635663
- artist_train_skip_rate: -0.22453172
- platform=iOS 14.6 (iPhone13,3): 0.18189145
- reason_start=trackdone: 0.18093323

## Rolling Backtests
- Windows: 3
- Mean AUC: 0.8693
- Min/Max AUC: 0.8534 / 0.8936