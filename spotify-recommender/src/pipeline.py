import argparse
import csv
import json
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import numpy as np


RANDOM_STATE = 42
EPS = 1e-12


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def load_extended_streaming(ext_dir: Path):
    rows = []
    for path in sorted(ext_dir.glob("Streaming_History_Audio*.json")):
        data = load_json(path)
        for r in data:
            ts = r.get("ts")
            uri = r.get("spotify_track_uri")
            track = r.get("master_metadata_track_name")
            artist = r.get("master_metadata_album_artist_name")
            skipped = r.get("skipped")
            if ts is None or uri is None or track is None or artist is None or skipped is None:
                continue
            rows.append(
                {
                    "ts": ts,
                    "dt": parse_ts(ts),
                    "date": ts[:10],
                    "uri": uri,
                    "track_name": track,
                    "artist": artist,
                    "platform": str(r.get("platform", "unknown") or "unknown"),
                    "conn_country": str(r.get("conn_country", "unknown") or "unknown"),
                    "reason_start": str(r.get("reason_start", "unknown") or "unknown"),
                    "reason_end": str(r.get("reason_end", "unknown") or "unknown"),
                    "ms_played": float(r.get("ms_played", 0) or 0),
                    "skipped": int(bool(skipped)),
                }
            )

    if not rows:
        raise ValueError(f"No valid rows in {ext_dir}")

    rows.sort(key=lambda x: x["ts"])
    return rows


def load_identity_and_library(account_dir: Path):
    identity = {}
    id_path = account_dir / "Identity.json"
    if id_path.exists():
        identity = load_json(id_path)

    library_tracks = []
    lib_path = account_dir / "YourLibrary.json"
    if lib_path.exists():
        data = load_json(lib_path)
        for t in data.get("tracks", []):
            uri = t.get("uri")
            if uri:
                library_tracks.append(
                    {
                        "uri": uri,
                        "track_name": t.get("track", "unknown"),
                        "artist": t.get("artist", "unknown"),
                    }
                )
    return identity, library_tracks


def load_technical_daily_counts(tech_dir: Path):
    out = defaultdict(lambda: {"connection_error_count": 0, "playback_error_count": 0})

    conn_path = tech_dir / "ConnectionError.json"
    if conn_path.exists():
        for r in load_json(conn_path):
            ts = r.get("timestamp_utc")
            if ts:
                out[str(ts)[:10]]["connection_error_count"] += 1

    play_path = tech_dir / "PlaybackError.json"
    if play_path.exists():
        for r in load_json(play_path):
            ts = r.get("timestamp_utc")
            if ts:
                out[str(ts)[:10]]["playback_error_count"] += 1

    return out


def enrich_rows(rows, tech_daily):
    session_position = 0
    session_id = 0
    prev_dt = None
    recent_skips = deque(maxlen=10)

    for r in rows:
        dt = r["dt"]
        if prev_dt is None:
            session_position = 0
            session_id = 0
        else:
            gap = (dt - prev_dt).total_seconds() / 60.0
            if gap > 30:
                session_id += 1
                session_position = 0
            else:
                session_position += 1

        r["session_id"] = session_id
        r["hour"] = dt.hour
        r["day_of_week"] = dt.weekday()
        r["month"] = dt.month
        r["is_weekend"] = 1 if dt.weekday() >= 5 else 0
        r["session_position"] = session_position
        if recent_skips:
            r["recent_skip_rate_10"] = sum(recent_skips) / len(recent_skips)
        else:
            r["recent_skip_rate_10"] = 0.5

        daily = tech_daily.get(r["date"], {})
        r["connection_error_count"] = float(daily.get("connection_error_count", 0))
        r["playback_error_count"] = float(daily.get("playback_error_count", 0))

        r["target_completed"] = 1 - r["skipped"]

        recent_skips.append(r["skipped"])
        prev_dt = dt


def split_rows(rows, train_ratio=0.8):
    cut = int(len(rows) * train_ratio)
    train = rows[:cut]
    test = rows[cut:]
    return train, test


def attach_train_priors(rows_all, rows_train):
    track_plays = defaultdict(int)
    track_skip = defaultdict(int)
    artist_plays = defaultdict(int)
    artist_skip = defaultdict(int)

    for r in rows_train:
        track_plays[r["uri"]] += 1
        track_skip[r["uri"]] += r["skipped"]
        artist_plays[r["artist"]] += 1
        artist_skip[r["artist"]] += r["skipped"]

    global_skip = sum(r["skipped"] for r in rows_train) / max(len(rows_train), 1)

    for r in rows_all:
        tp = track_plays.get(r["uri"], 0)
        ap = artist_plays.get(r["artist"], 0)
        r["track_train_plays"] = float(tp)
        r["artist_train_plays"] = float(ap)
        r["track_train_skip_rate"] = (track_skip[r["uri"]] / tp) if tp else global_skip
        r["artist_train_skip_rate"] = (artist_skip[r["artist"]] / ap) if ap else global_skip


def build_vocab(rows_train, cat_fields):
    vocab = {}
    for field in cat_fields:
        values = sorted({str(r.get(field, "unknown") or "unknown") for r in rows_train})
        vocab[field] = {v: i for i, v in enumerate(values)}
    return vocab


def prepare_encoder(rows_train, num_fields, cat_fields):
    train_num = np.array([[float(r.get(f, 0.0)) for f in num_fields] for r in rows_train], dtype=float)
    mu = train_num.mean(axis=0)
    sigma = train_num.std(axis=0)
    sigma[sigma < EPS] = 1.0

    vocab = build_vocab(rows_train, cat_fields)
    offsets = {}
    cur = len(num_fields)
    for field in cat_fields:
        offsets[field] = cur
        cur += len(vocab[field])

    feature_names = list(num_fields)
    for f in cat_fields:
        inv = sorted(vocab[f].items(), key=lambda x: x[1])
        feature_names.extend([f"{f}={val}" for val, _ in inv])

    return {
        "num_fields": num_fields,
        "cat_fields": cat_fields,
        "mu": mu,
        "sigma": sigma,
        "vocab": vocab,
        "offsets": offsets,
        "dim": cur,
        "feature_names": feature_names,
    }


def encode_rows(rows, encoder):
    n = len(rows)
    x = np.zeros((n, encoder["dim"]), dtype=float)
    for i, r in enumerate(rows):
        for j, f in enumerate(encoder["num_fields"]):
            x[i, j] = (float(r.get(f, 0.0)) - encoder["mu"][j]) / encoder["sigma"][j]
        for f in encoder["cat_fields"]:
            val = str(r.get(f, "unknown") or "unknown")
            idx = encoder["vocab"][f].get(val)
            if idx is not None:
                x[i, encoder["offsets"][f] + idx] = 1.0
    y = np.array([int(r["target_completed"]) for r in rows], dtype=float)
    return x, y


def sigmoid(z):
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def train_logistic_regression(x, y, lr=0.05, epochs=260, reg=0.0005):
    np.random.seed(RANDOM_STATE)
    n, d = x.shape
    w = np.zeros(d, dtype=float)
    b = 0.0

    for _ in range(epochs):
        z = x @ w + b
        p = sigmoid(z)
        err = p - y
        grad_w = (x.T @ err) / n + reg * w
        grad_b = float(np.mean(err))
        w -= lr * grad_w
        b -= lr * grad_b

    return w, b


def predict_proba(x, w, b):
    return sigmoid(x @ w + b)


def roc_auc_score_manual(y_true, y_score):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    sum_ranks_pos = ranks[pos].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def pr_auc_manual(y_true, y_score):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(-y_score)
    y = y_true[order]

    tp = 0.0
    fp = 0.0
    total_pos = float(y_true.sum())
    if total_pos == 0:
        return 0.0

    prev_recall = 0.0
    prev_precision = 1.0
    auc = 0.0

    for label in y:
        if label == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / total_pos
        precision = tp / max(tp + fp, EPS)
        auc += (recall - prev_recall) * ((precision + prev_precision) / 2.0)
        prev_recall = recall
        prev_precision = precision

    return float(auc)


def evaluate(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, EPS)

    return {
        "auc": roc_auc_score_manual(y_true, y_prob),
        "pr_auc": pr_auc_manual(y_true, y_prob),
        "accuracy": float(accuracy),
        "f1": float(f1),
    }


def _dcg_at_k(labels, k):
    val = 0.0
    top = labels[:k]
    for i, rel in enumerate(top, start=1):
        gain = float(rel)
        val += gain / np.log2(i + 1)
    return val


def _ndcg_at_k(labels, k):
    ideal = sorted(labels, reverse=True)
    denom = _dcg_at_k(ideal, k)
    if denom <= EPS:
        return 0.0
    return _dcg_at_k(labels, k) / denom


def _map_at_k(labels, k):
    hits = 0.0
    acc = 0.0
    total_relevant = max(sum(labels), 1)
    for i, rel in enumerate(labels[:k], start=1):
        if rel == 1:
            hits += 1
            acc += hits / i
    return acc / min(total_relevant, k)


def _recall_at_k(labels, k):
    total_relevant = sum(labels)
    if total_relevant == 0:
        return 0.0
    return sum(labels[:k]) / total_relevant


def ranking_metrics_by_session(rows, probs, ks=(5, 10), min_session_len=5):
    buckets = defaultdict(list)
    for r, p in zip(rows, probs):
        buckets[r["session_id"]].append((float(p), int(r["target_completed"])))

    metrics = {}
    valid_sessions = 0
    for k in ks:
        metrics[f"ndcg@{k}"] = []
        metrics[f"map@{k}"] = []
        metrics[f"recall@{k}"] = []

    for events in buckets.values():
        if len(events) < min_session_len:
            continue
        events = sorted(events, key=lambda x: x[0], reverse=True)
        labels = [rel for _, rel in events]
        if sum(labels) == 0:
            continue

        valid_sessions += 1
        for k in ks:
            metrics[f"ndcg@{k}"].append(_ndcg_at_k(labels, k))
            metrics[f"map@{k}"].append(_map_at_k(labels, k))
            metrics[f"recall@{k}"].append(_recall_at_k(labels, k))

    out = {"ranking_sessions_evaluated": valid_sessions}
    for key, vals in metrics.items():
        out[key] = float(np.mean(vals)) if vals else 0.0
    return out


def build_recommendations(rows, probs, library_tracks, top_n=100):
    track_stats = {}
    for r, p in zip(rows, probs):
        uri = r["uri"]
        if uri not in track_stats:
            track_stats[uri] = {
                "uri": uri,
                "track_name": r["track_name"],
                "artist": r["artist"],
                "plays": 0,
                "actual_completed_sum": 0.0,
                "pred_completed_sum": 0.0,
                "last_ts": r["ts"],
            }
        s = track_stats[uri]
        s["plays"] += 1
        s["actual_completed_sum"] += r["target_completed"]
        s["pred_completed_sum"] += float(p)
        s["last_ts"] = max(s["last_ts"], r["ts"])

    artist_scores = defaultdict(list)
    for s in track_stats.values():
        artist_scores[s["artist"]].append(s["actual_completed_sum"] / s["plays"])
    artist_prior = {k: float(sum(v) / len(v)) for k, v in artist_scores.items()}
    global_prior = float(sum((s["actual_completed_sum"] / s["plays"]) for s in track_stats.values()) / max(len(track_stats), 1))

    recs = []
    lib_by_uri = {t["uri"]: t for t in library_tracks}
    candidate_uris = set(track_stats.keys()) | set(lib_by_uri.keys())

    for uri in candidate_uris:
        s = track_stats.get(uri)
        if s is None:
            t = lib_by_uri[uri]
            prior = artist_prior.get(t.get("artist", "unknown"), global_prior)
            plays = 0
            actual = prior
            pred = prior
            track_name = t.get("track_name", "unknown")
            artist = t.get("artist", "unknown")
        else:
            plays = s["plays"]
            actual = s["actual_completed_sum"] / plays
            pred = s["pred_completed_sum"] / plays
            track_name = s["track_name"]
            artist = s["artist"]

        score = 0.65 * pred + 0.35 * actual
        reason = (
            "High predicted completion and repeated positive listening history"
            if plays >= 3
            else "Strong artist-level preference with high completion probability"
        )
        recs.append(
            {
                "uri": uri,
                "track_name": track_name,
                "artist": artist,
                "plays": plays,
                "actual_completion_rate": round(actual, 6),
                "predicted_completion_rate": round(pred, 6),
                "score": round(score, 6),
                "reason": reason,
            }
        )

    recs.sort(key=lambda x: (x["score"], x["plays"]), reverse=True)
    return recs[:top_n]


def build_heuristic_probs(rows):
    probs = []
    for r in rows:
        p = 1.0 - (0.65 * r["track_train_skip_rate"] + 0.35 * r["artist_train_skip_rate"])
        probs.append(float(np.clip(p, 0.0, 1.0)))
    return np.array(probs, dtype=float)


def train_and_score_model(rows_train, rows_test, rows_all, num_fields, cat_fields):
    encoder = prepare_encoder(rows_train, num_fields, cat_fields)
    x_train, y_train = encode_rows(rows_train, encoder)
    x_test, y_test = encode_rows(rows_test, encoder)
    x_all, _y_all = encode_rows(rows_all, encoder)

    weights, bias = train_logistic_regression(x_train, y_train)
    train_prob = predict_proba(x_train, weights, bias)
    test_prob = predict_proba(x_test, weights, bias)
    all_prob = predict_proba(x_all, weights, bias)

    return {
        "weights": weights,
        "bias": bias,
        "feature_names": encoder["feature_names"],
        "train_prob": train_prob,
        "test_prob": test_prob,
        "all_prob": all_prob,
        "y_train": y_train,
        "y_test": y_test,
    }


def write_csv(path: Path, rows, headers):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_results_md(output_dir, identity, metrics, ranking_metrics, model_comparison, top_features):
    name = identity.get("displayName", "Spotify User") if isinstance(identity, dict) else "Spotify User"
    lines = [
        "# Results",
        "",
        f"- Profile: {name}",
        f"- Train rows: {metrics['train_rows']}",
        f"- Test rows: {metrics['test_rows']}",
        "",
        "## Primary Model (Full Feature Set)",
        f"- ROC-AUC: {metrics['test_auc']:.4f}",
        f"- PR-AUC: {metrics['test_pr_auc']:.4f}",
        f"- Accuracy: {metrics['test_accuracy']:.4f}",
        f"- F1: {metrics['test_f1']:.4f}",
        "",
        "## Ranking Metrics (Session-Based)",
        f"- Sessions evaluated: {ranking_metrics['ranking_sessions_evaluated']}",
        f"- NDCG@5: {ranking_metrics['ndcg@5']:.4f}",
        f"- MAP@5: {ranking_metrics['map@5']:.4f}",
        f"- Recall@5: {ranking_metrics['recall@5']:.4f}",
        f"- NDCG@10: {ranking_metrics['ndcg@10']:.4f}",
        f"- MAP@10: {ranking_metrics['map@10']:.4f}",
        f"- Recall@10: {ranking_metrics['recall@10']:.4f}",
        "",
        "## Model Comparison",
    ]

    for row in model_comparison:
        lines.append(
            f"- {row['model']}: AUC={row['test_auc']:.4f}, PR-AUC={row['test_pr_auc']:.4f}, "
            f"Accuracy={row['test_accuracy']:.4f}, F1={row['test_f1']:.4f}"
        )

    lines.extend(["", "## Top Features (Absolute Coefficient)"])
    for feat in top_features[:10]:
        lines.append(f"- {feat['feature']}: {feat['coefficient']}")

    (output_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def write_outputs(output_dir, identity, rows, probs, metrics, feature_names, weights, recs, ranking_metrics, model_comparison):
    output_dir.mkdir(parents=True, exist_ok=True)

    top_features = []
    for name, weight in sorted(zip(feature_names, weights), key=lambda x: abs(x[1]), reverse=True)[:100]:
        top_features.append(
            {"feature": name, "coefficient": round(float(weight), 8), "abs_coefficient": round(abs(float(weight)), 8)}
        )
    write_csv(output_dir / "feature_importance.csv", top_features, ["feature", "coefficient", "abs_coefficient"])

    write_csv(
        output_dir / "top_resume_playlist.csv",
        recs,
        ["uri", "track_name", "artist", "plays", "actual_completion_rate", "predicted_completion_rate", "score", "reason"],
    )

    scored_samples = []
    for r, p in list(zip(rows, probs))[-10000:]:
        scored_samples.append(
            {
                "ts": r["ts"],
                "uri": r["uri"],
                "track_name": r["track_name"],
                "artist": r["artist"],
                "platform": r["platform"],
                "target_completed": r["target_completed"],
                "predicted_completion_prob": round(float(p), 6),
            }
        )
    write_csv(
        output_dir / "scored_samples.csv",
        scored_samples,
        ["ts", "uri", "track_name", "artist", "platform", "target_completed", "predicted_completion_prob"],
    )

    full_metrics = metrics.copy()
    full_metrics.update(ranking_metrics)
    with (output_dir / "model_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2)

    with (output_dir / "model_comparison.json").open("w", encoding="utf-8") as f:
        json.dump(model_comparison, f, indent=2)
    write_csv(
        output_dir / "model_comparison.csv",
        model_comparison,
        ["model", "test_auc", "test_pr_auc", "test_accuracy", "test_f1", "test_logloss"],
    )

    display_name = identity.get("displayName", "Spotify User") if isinstance(identity, dict) else "Spotify User"
    top10 = top_features[:10]
    summary_lines = [
        "# Spotify Recommender Project",
        "",
        f"**Profile analyzed:** {display_name}",
        "",
        "## Model Performance (Holdout Test)",
        f"- Rows (train/test): {metrics['train_rows']} / {metrics['test_rows']}",
        f"- ROC-AUC: {metrics['test_auc']:.4f}",
        f"- PR-AUC: {metrics['test_pr_auc']:.4f}",
        f"- Accuracy: {metrics['test_accuracy']:.4f}",
        f"- F1 Score: {metrics['test_f1']:.4f}",
        f"- NDCG@10: {ranking_metrics['ndcg@10']:.4f}",
        "",
        "## Top Drivers",
    ]
    for feat in top10:
        summary_lines.append(f"- {feat['feature']}: {feat['coefficient']}")

    summary_lines.extend(
        [
            "",
            "## Resume Bullets",
            "- Built an end-to-end Spotify listening intelligence pipeline over multi-year streaming and client telemetry data.",
            "- Trained and compared recommendation models with chronological holdout validation and ranking metrics (NDCG/MAP/Recall@K).",
            "- Delivered a recommendation layer that ranks tracks by completion probability with interpretable feature drivers.",
        ]
    )
    (output_dir / "resume_project_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    write_results_md(output_dir, identity, metrics, ranking_metrics, model_comparison, top_features)


def main():
    parser = argparse.ArgumentParser(description="Spotify playback intelligence pipeline (NumPy implementation).")
    parser.add_argument("--account-dir", default="../Spotify Account Data")
    parser.add_argument("--extended-dir", default="../Spotify Extended Streaming History")
    parser.add_argument("--tech-dir", default="../Spotify Technical Log Information")
    parser.add_argument("--output-dir", default="./outputs")
    args = parser.parse_args()

    account_dir = Path(args.account_dir).resolve()
    extended_dir = Path(args.extended_dir).resolve()
    tech_dir = Path(args.tech_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    rows = load_extended_streaming(extended_dir)
    identity, library_tracks = load_identity_and_library(account_dir)
    tech_daily = load_technical_daily_counts(tech_dir)

    enrich_rows(rows, tech_daily)
    train_rows, test_rows = split_rows(rows, train_ratio=0.8)
    attach_train_priors(rows, train_rows)

    full_num_fields = [
        "ms_played",
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "session_position",
        "recent_skip_rate_10",
        "track_train_plays",
        "track_train_skip_rate",
        "artist_train_plays",
        "artist_train_skip_rate",
        "connection_error_count",
        "playback_error_count",
    ]
    full_cat_fields = ["platform", "conn_country", "reason_start", "reason_end"]

    baseline_num_fields = [
        "ms_played",
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "session_position",
        "recent_skip_rate_10",
    ]
    baseline_cat_fields = ["platform"]

    full_model = train_and_score_model(train_rows, test_rows, rows, full_num_fields, full_cat_fields)
    base_model = train_and_score_model(train_rows, test_rows, rows, baseline_num_fields, baseline_cat_fields)

    y_train = full_model["y_train"]
    y_test = full_model["y_test"]

    test_eval = evaluate(y_test, full_model["test_prob"])
    ranking_metrics = ranking_metrics_by_session(test_rows, full_model["test_prob"], ks=(5, 10), min_session_len=5)

    def logloss(y, p):
        p = np.clip(p, EPS, 1 - EPS)
        return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))

    heuristic_test_probs = build_heuristic_probs(test_rows)

    model_comparison = []
    for model_name, probs in [
        ("full_logistic", full_model["test_prob"]),
        ("baseline_logistic", base_model["test_prob"]),
        ("track_artist_heuristic", heuristic_test_probs),
    ]:
        ev = evaluate(y_test, probs)
        model_comparison.append(
            {
                "model": model_name,
                "test_auc": float(ev["auc"]),
                "test_pr_auc": float(ev["pr_auc"]),
                "test_accuracy": float(ev["accuracy"]),
                "test_f1": float(ev["f1"]),
                "test_logloss": logloss(y_test, probs),
            }
        )

    metrics = {
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "test_auc": test_eval["auc"],
        "test_pr_auc": test_eval["pr_auc"],
        "test_accuracy": test_eval["accuracy"],
        "test_f1": test_eval["f1"],
        "base_completion_rate_test": float(np.mean(y_test)),
        "base_completion_rate_train": float(np.mean(y_train)),
        "train_logloss": logloss(y_train, full_model["train_prob"]),
        "test_logloss": logloss(y_test, full_model["test_prob"]),
    }

    recs = build_recommendations(rows, full_model["all_prob"], library_tracks)
    write_outputs(
        output_dir,
        identity,
        rows,
        full_model["all_prob"],
        metrics,
        full_model["feature_names"],
        full_model["weights"],
        recs,
        ranking_metrics,
        model_comparison,
    )

    print("Pipeline completed.")
    print(f"Rows used: {len(rows)}")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
