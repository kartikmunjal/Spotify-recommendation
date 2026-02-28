import argparse
import csv
import json
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import numpy as np


RANDOM_STATE = 42
EPS = 1e-12


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


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
        r["recent_skip_rate_10"] = (sum(recent_skips) / len(recent_skips)) if recent_skips else 0.5

        daily = tech_daily.get(r["date"], {})
        r["connection_error_count"] = float(daily.get("connection_error_count", 0))
        r["playback_error_count"] = float(daily.get("playback_error_count", 0))
        r["target_completed"] = 1 - r["skipped"]

        recent_skips.append(r["skipped"])
        prev_dt = dt


def _dominant_key(counter_dict):
    if not counter_dict:
        return "unknown"
    return max(counter_dict.items(), key=lambda x: x[1])[0]


def _hour_in_window(hour, start_hour, end_hour):
    if start_hour <= end_hour:
        return start_hour <= hour <= end_hour
    return hour >= start_hour or hour <= end_hour


def load_primary_user_config(path: Path):
    defaults = {
        "home_country": "US",
        "primary_platforms": ["ios", "osx"],
        "driving_platform_hints": ["not_applicable", "tizen", "cast"],
        "preferred_hour_window": {"start": 6, "end": 1},
        "artist_affinity_top_n": 60,
        "artist_affinity_min_ratio": 0.20,
        "min_session_score": 2.6,
        "require_country_match": True,
        "weights": {
            "country": 1.2,
            "platform": 1.2,
            "hour": 0.7,
            "artist_affinity": 1.1,
            "driving_hint": 0.6,
        },
    }
    if not path.exists():
        return defaults

    user_cfg = load_json(path)
    if not isinstance(user_cfg, dict):
        return defaults

    out = defaults.copy()
    out.update({k: v for k, v in user_cfg.items() if k != "weights"})
    w = defaults["weights"].copy()
    if isinstance(user_cfg.get("weights"), dict):
        w.update(user_cfg["weights"])
    out["weights"] = w
    return out


def build_primary_user_filter(rows, cfg):
    home_country = str(cfg["home_country"])
    primary_platforms = {str(x) for x in cfg["primary_platforms"]}
    driving_hints = {str(x) for x in cfg["driving_platform_hints"]}
    top_n = int(cfg["artist_affinity_top_n"])
    min_ratio = float(cfg["artist_affinity_min_ratio"])
    min_score = float(cfg["min_session_score"])
    require_country = bool(cfg["require_country_match"])
    start_hour = int(cfg["preferred_hour_window"]["start"])
    end_hour = int(cfg["preferred_hour_window"]["end"])
    w = cfg["weights"]

    seed_rows = [r for r in rows if r["conn_country"] == home_country and r["platform"] in primary_platforms]
    affinity_source = seed_rows if len(seed_rows) >= 500 else rows
    artist_counts = defaultdict(int)
    for r in affinity_source:
        artist_counts[r["artist"]] += 1
    top_artists = {a for a, _ in sorted(artist_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]}

    session = {}
    for r in rows:
        sid = r["session_id"]
        if sid not in session:
            session[sid] = {
                "session_id": sid,
                "rows": 0,
                "country_counts": defaultdict(int),
                "platform_counts": defaultdict(int),
                "start_hour": r["hour"],
                "affinity_hits": 0,
                "completion_sum": 0.0,
            }
        s = session[sid]
        s["rows"] += 1
        s["country_counts"][r["conn_country"]] += 1
        s["platform_counts"][r["platform"]] += 1
        s["affinity_hits"] += 1 if r["artist"] in top_artists else 0
        s["completion_sum"] += r["target_completed"]

    report_rows = []
    keep_session_ids = set()
    session_keep_map = {}

    for sid, s in session.items():
        dom_country = _dominant_key(s["country_counts"])
        dom_platform = _dominant_key(s["platform_counts"])
        affinity_ratio = s["affinity_hits"] / max(s["rows"], 1)
        completion_rate = s["completion_sum"] / max(s["rows"], 1)

        country_match = dom_country == home_country
        platform_match = dom_platform in primary_platforms
        driving_hint = dom_platform in driving_hints
        hour_match = _hour_in_window(s["start_hour"], start_hour, end_hour)
        affinity_match = affinity_ratio >= min_ratio

        score = 0.0
        score += w["country"] if country_match else 0.0
        score += w["platform"] if platform_match else 0.0
        score += w["hour"] if hour_match else 0.0
        score += w["artist_affinity"] if affinity_match else 0.0
        score += w["driving_hint"] if driving_hint else 0.0

        keep = (score >= min_score) and ((not require_country) or country_match)
        session_keep_map[sid] = keep
        if keep:
            keep_session_ids.add(sid)

        report_rows.append(
            {
                "session_id": sid,
                "rows": s["rows"],
                "dominant_country": dom_country,
                "dominant_platform": dom_platform,
                "start_hour": s["start_hour"],
                "completion_rate": round(float(completion_rate), 6),
                "artist_affinity_ratio": round(float(affinity_ratio), 6),
                "country_match": int(country_match),
                "platform_match": int(platform_match),
                "hour_match": int(hour_match),
                "artist_affinity_match": int(affinity_match),
                "driving_hint": int(driving_hint),
                "attribution_score": round(float(score), 6),
                "keep_as_primary_user": int(keep),
            }
        )

    filtered_rows = []
    for r in rows:
        r["is_primary_user_session"] = int(session_keep_map.get(r["session_id"], False))
        if r["session_id"] in keep_session_ids:
            filtered_rows.append(r)

    summary = {
        "sessions_total": len(session),
        "sessions_kept": len(keep_session_ids),
        "rows_total": len(rows),
        "rows_kept": len(filtered_rows),
        "row_coverage": float(len(filtered_rows) / max(len(rows), 1)),
        "home_country": home_country,
        "primary_platforms": sorted(primary_platforms),
        "driving_platform_hints": sorted(driving_hints),
    }
    report_rows.sort(key=lambda x: (x["attribution_score"], x["rows"]), reverse=True)
    return filtered_rows, report_rows, summary


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
    for i, rel in enumerate(labels[:k], start=1):
        val += float(rel) / np.log2(i + 1)
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

    prior_strength = 8.0  # Bayesian smoothing prior pseudo-count.
    min_evidence_plays = 2

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

        artist_aff = artist_prior.get(artist, global_prior)
        bayes_actual = ((plays * actual) + (prior_strength * artist_aff)) / (plays + prior_strength)
        evidence = plays / (plays + prior_strength)
        score = 0.45 * pred + 0.35 * bayes_actual + 0.20 * artist_aff
        score = score * (0.65 + 0.35 * evidence)

        reason = (
            "High confidence from repeated positive listening history"
            if plays >= 5
            else "Strong artist preference with moderate play confidence"
        )
        recs.append(
            {
                "uri": uri,
                "track_name": track_name,
                "artist": artist,
                "plays": plays,
                "actual_completion_rate": round(actual, 6),
                "bayesian_completion_rate": round(bayes_actual, 6),
                "artist_completion_prior": round(artist_aff, 6),
                "predicted_completion_rate": round(pred, 6),
                "score": round(score, 6),
                "reason": reason,
            }
        )

    # Prefer recommendations with at least modest play evidence.
    evidence_recs = [r for r in recs if r["plays"] >= min_evidence_plays]
    evidence_recs.sort(key=lambda x: (x["score"], x["plays"]), reverse=True)
    if len(evidence_recs) >= top_n:
        return evidence_recs[:top_n]

    recs.sort(key=lambda x: (x["score"], x["plays"]), reverse=True)
    seen = {r["uri"] for r in evidence_recs}
    for r in recs:
        if len(evidence_recs) >= top_n:
            break
        if r["uri"] not in seen:
            evidence_recs.append(r)
            seen.add(r["uri"])
    return evidence_recs[:top_n]


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
    x_all, _ = encode_rows(rows_all, encoder)

    weights, bias = train_logistic_regression(x_train, y_train)
    return {
        "weights": weights,
        "bias": bias,
        "feature_names": encoder["feature_names"],
        "train_prob": predict_proba(x_train, weights, bias),
        "test_prob": predict_proba(x_test, weights, bias),
        "all_prob": predict_proba(x_all, weights, bias),
        "y_train": y_train,
        "y_test": y_test,
    }


def write_csv(path: Path, rows, headers):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_results_md(output_dir, identity, metrics, ranking_metrics, model_comparison, top_features, attribution_summary, dataset_comparison):
    name = identity.get("displayName", "Spotify User") if isinstance(identity, dict) else "Spotify User"
    lines = [
        "# Results",
        "",
        f"- Profile: {name}",
        f"- Train rows: {metrics['train_rows']}",
        f"- Test rows: {metrics['test_rows']}",
        "",
        "## Dataset Attribution",
        f"- Filter enabled: {attribution_summary['filter_enabled']}",
        f"- Rows kept as primary-user: {attribution_summary['rows_kept']}/{attribution_summary['rows_total']} ({100.0 * attribution_summary['row_coverage']:.2f}%)",
        f"- Sessions kept: {attribution_summary['sessions_kept']}/{attribution_summary['sessions_total']}",
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
        "## Dataset Comparison (Full Model)",
    ]
    for row in dataset_comparison:
        lines.append(
            f"- {row['dataset']}: rows={row['rows']}, AUC={row['test_auc']:.4f}, PR-AUC={row['test_pr_auc']:.4f}, "
            f"Accuracy={row['test_accuracy']:.4f}, F1={row['test_f1']:.4f}"
        )

    lines.extend(["", "## Model Comparison"])
    for row in model_comparison:
        lines.append(
            f"- {row['model']}: AUC={row['test_auc']:.4f}, PR-AUC={row['test_pr_auc']:.4f}, "
            f"Accuracy={row['test_accuracy']:.4f}, F1={row['test_f1']:.4f}"
        )

    lines.extend(["", "## Top Features (Absolute Coefficient)"])
    for feat in top_features[:10]:
        lines.append(f"- {feat['feature']}: {feat['coefficient']}")

    (output_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def write_outputs(
    output_dir,
    identity,
    rows,
    probs,
    metrics,
    feature_names,
    weights,
    recs,
    ranking_metrics,
    model_comparison,
    attribution_summary,
    dataset_comparison,
    attribution_report_rows,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    top_features = []
    for name, weight in sorted(zip(feature_names, weights), key=lambda x: abs(x[1]), reverse=True)[:100]:
        top_features.append({"feature": name, "coefficient": round(float(weight), 8), "abs_coefficient": round(abs(float(weight)), 8)})
    write_csv(output_dir / "feature_importance.csv", top_features, ["feature", "coefficient", "abs_coefficient"])

    write_csv(
        output_dir / "top_resume_playlist.csv",
        recs,
        [
            "uri",
            "track_name",
            "artist",
            "plays",
            "actual_completion_rate",
            "bayesian_completion_rate",
            "artist_completion_prior",
            "predicted_completion_rate",
            "score",
            "reason",
        ],
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
                "is_primary_user_session": r.get("is_primary_user_session", 1),
                "predicted_completion_prob": round(float(p), 6),
            }
        )
    write_csv(
        output_dir / "scored_samples.csv",
        scored_samples,
        ["ts", "uri", "track_name", "artist", "platform", "target_completed", "is_primary_user_session", "predicted_completion_prob"],
    )

    full_metrics = metrics.copy()
    full_metrics.update(ranking_metrics)
    full_metrics.update({
        "attribution_rows_kept": attribution_summary["rows_kept"],
        "attribution_rows_total": attribution_summary["rows_total"],
        "attribution_row_coverage": attribution_summary["row_coverage"],
    })
    with (output_dir / "model_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2)

    with (output_dir / "model_comparison.json").open("w", encoding="utf-8") as f:
        json.dump(model_comparison, f, indent=2)
    write_csv(output_dir / "model_comparison.csv", model_comparison, ["model", "test_auc", "test_pr_auc", "test_accuracy", "test_f1", "test_logloss"])

    with (output_dir / "dataset_comparison.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_comparison, f, indent=2)
    write_csv(output_dir / "dataset_comparison.csv", dataset_comparison, ["dataset", "rows", "test_auc", "test_pr_auc", "test_accuracy", "test_f1", "test_logloss"])

    with (output_dir / "attribution_summary.json").open("w", encoding="utf-8") as f:
        json.dump(attribution_summary, f, indent=2)
    write_csv(
        output_dir / "user_attribution_report.csv",
        attribution_report_rows,
        [
            "session_id",
            "rows",
            "dominant_country",
            "dominant_platform",
            "start_hour",
            "completion_rate",
            "artist_affinity_ratio",
            "country_match",
            "platform_match",
            "hour_match",
            "artist_affinity_match",
            "driving_hint",
            "attribution_score",
            "keep_as_primary_user",
        ],
    )

    display_name = identity.get("displayName", "Spotify User") if isinstance(identity, dict) else "Spotify User"
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
        f"- Primary-user coverage kept: {attribution_summary['rows_kept']}/{attribution_summary['rows_total']} ({100.0 * attribution_summary['row_coverage']:.2f}%)",
        "",
        "## Resume Bullets",
        "- Built an end-to-end Spotify listening intelligence pipeline over multi-year streaming and client telemetry data.",
        "- Added a primary-user session attribution layer to reduce family-plan multi-user noise before model training.",
        "- Trained and compared recommendation models with chronological holdout validation and ranking metrics (NDCG/MAP/Recall@K).",
    ]
    (output_dir / "resume_project_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    write_results_md(output_dir, identity, metrics, ranking_metrics, model_comparison, top_features, attribution_summary, dataset_comparison)


def clone_rows(rows):
    return [dict(r) for r in rows]


def run_dataset_experiment(rows, library_tracks):
    work_rows = clone_rows(rows)
    train_rows, test_rows = split_rows(work_rows, train_ratio=0.8)
    attach_train_priors(work_rows, train_rows)

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

    full_model = train_and_score_model(train_rows, test_rows, work_rows, full_num_fields, full_cat_fields)
    base_model = train_and_score_model(train_rows, test_rows, work_rows, baseline_num_fields, baseline_cat_fields)

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

    recs = build_recommendations(work_rows, full_model["all_prob"], library_tracks)
    return {
        "rows": work_rows,
        "metrics": metrics,
        "ranking_metrics": ranking_metrics,
        "model_comparison": model_comparison,
        "full_model": full_model,
        "recommendations": recs,
    }


def main():
    parser = argparse.ArgumentParser(description="Spotify playback intelligence pipeline (NumPy implementation).")
    parser.add_argument("--account-dir", default="../Spotify Account Data")
    parser.add_argument("--extended-dir", default="../Spotify Extended Streaming History")
    parser.add_argument("--tech-dir", default="../Spotify Technical Log Information")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--filter-primary-user", default="true", help="Whether to apply primary-user session filter.")
    parser.add_argument("--primary-user-config", default="./primary_user_config.json", help="Path to primary user filter config JSON.")
    args = parser.parse_args()

    account_dir = Path(args.account_dir).resolve()
    extended_dir = Path(args.extended_dir).resolve()
    tech_dir = Path(args.tech_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    use_filter = parse_bool(args.filter_primary_user)
    filter_cfg_path = Path(args.primary_user_config).resolve()

    rows = load_extended_streaming(extended_dir)
    identity, library_tracks = load_identity_and_library(account_dir)
    tech_daily = load_technical_daily_counts(tech_dir)
    enrich_rows(rows, tech_daily)

    all_result = run_dataset_experiment(rows, library_tracks)

    attribution_summary = {
        "filter_enabled": bool(use_filter),
        "sessions_total": 0,
        "sessions_kept": 0,
        "rows_total": len(rows),
        "rows_kept": len(rows),
        "row_coverage": 1.0,
    }
    attribution_report_rows = []
    selected_rows = rows

    if use_filter:
        cfg = load_primary_user_config(filter_cfg_path)
        selected_rows, attribution_report_rows, filter_summary = build_primary_user_filter(rows, cfg)
        attribution_summary.update(filter_summary)
        if len(selected_rows) < 5000:
            selected_rows = rows
            attribution_summary["filter_enabled"] = False
            attribution_summary["rows_kept"] = len(rows)
            attribution_summary["row_coverage"] = 1.0

    selected_result = run_dataset_experiment(selected_rows, library_tracks)
    metrics = selected_result["metrics"]
    ranking_metrics = selected_result["ranking_metrics"]
    model_comparison = selected_result["model_comparison"]
    full_model = selected_result["full_model"]
    recs = selected_result["recommendations"]

    dataset_comparison = [
        {
            "dataset": "all_rows",
            "rows": len(all_result["rows"]),
            "test_auc": all_result["metrics"]["test_auc"],
            "test_pr_auc": all_result["metrics"]["test_pr_auc"],
            "test_accuracy": all_result["metrics"]["test_accuracy"],
            "test_f1": all_result["metrics"]["test_f1"],
            "test_logloss": all_result["metrics"]["test_logloss"],
        },
        {
            "dataset": "primary_user_filtered" if attribution_summary["filter_enabled"] else "primary_user_filter_disabled",
            "rows": len(selected_result["rows"]),
            "test_auc": metrics["test_auc"],
            "test_pr_auc": metrics["test_pr_auc"],
            "test_accuracy": metrics["test_accuracy"],
            "test_f1": metrics["test_f1"],
            "test_logloss": metrics["test_logloss"],
        },
    ]

    write_outputs(
        output_dir,
        identity,
        selected_result["rows"],
        full_model["all_prob"],
        metrics,
        full_model["feature_names"],
        full_model["weights"],
        recs,
        ranking_metrics,
        model_comparison,
        attribution_summary,
        dataset_comparison,
        attribution_report_rows,
    )

    print("Pipeline completed.")
    print(f"Rows used (selected dataset): {len(selected_result['rows'])}")
    print(f"Primary-user filter enabled: {attribution_summary['filter_enabled']}")
    print(
        f"Coverage kept: {attribution_summary['rows_kept']}/{attribution_summary['rows_total']} "
        f"({100.0 * attribution_summary['row_coverage']:.2f}%)"
    )
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
