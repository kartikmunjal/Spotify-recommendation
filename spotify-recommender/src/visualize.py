import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


def read_scored_samples(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ts = r.get("ts")
            if not ts:
                continue
            rows.append(
                {
                    "hour": datetime.fromisoformat(ts.replace("Z", "+00:00")).hour,
                    "artist": r.get("artist", "unknown"),
                    "platform": r.get("platform", "unknown"),
                    "target_completed": int(float(r.get("target_completed", 0))),
                    "predicted_completion_prob": float(r.get("predicted_completion_prob", 0.0)),
                }
            )
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _svg_wrap(width, height, body, title=""):
    title_tag = f"<title>{title}</title>" if title else ""
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>"
        f"{title_tag}<rect x='0' y='0' width='{width}' height='{height}' fill='white'/>"
        f"{body}</svg>"
    )


def _axes(width, height, margin):
    left, top, right, bottom = margin
    x0, y0 = left, height - bottom
    x1, y1 = width - right, top
    return x0, y0, x1, y1


def _save(path: Path, content: str):
    path.write_text(content, encoding="utf-8")


def chart_completion_rate_by_hour(rows, out_path: Path):
    width, height = 920, 420
    margin = (60, 40, 30, 50)
    x0, y0, x1, y1 = _axes(width, height, margin)

    sums = defaultdict(float)
    counts = defaultdict(int)
    for r in rows:
        h = r["hour"]
        sums[h] += r["target_completed"]
        counts[h] += 1
    ys = [sums[h] / max(counts[h], 1) for h in range(24)]

    path_pts = []
    for h, val in enumerate(ys):
        x = x0 + (x1 - x0) * h / 23
        y = y0 - (y0 - y1) * val
        path_pts.append(f"{x:.2f},{y:.2f}")

    grid = []
    for i in range(6):
        yy = y0 - (y0 - y1) * i / 5
        grid.append(f"<line x1='{x0}' y1='{yy:.2f}' x2='{x1}' y2='{yy:.2f}' stroke='#e6e6e6'/>")
        val = i / 5
        grid.append(f"<text x='{x0 - 8}' y='{yy + 4:.2f}' font-size='11' text-anchor='end' fill='#666'>{val:.1f}</text>")

    x_ticks = []
    for h in range(0, 24, 3):
        x = x0 + (x1 - x0) * h / 23
        x_ticks.append(f"<text x='{x:.2f}' y='{y0 + 18}' font-size='11' text-anchor='middle' fill='#666'>{h}</text>")

    body = "".join(grid)
    body += f"<line x1='{x0}' y1='{y0}' x2='{x1}' y2='{y0}' stroke='#444'/>"
    body += f"<line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y1}' stroke='#444'/>"
    body += "".join(x_ticks)
    body += f"<polyline fill='none' stroke='#1db954' stroke-width='3' points='{' '.join(path_pts)}'/>"
    body += "<text x='460' y='22' font-size='16' text-anchor='middle' fill='#111'>Completion Rate by Hour</text>"
    body += "<text x='460' y='410' font-size='12' text-anchor='middle' fill='#555'>Hour of Day</text>"
    body += "<text transform='translate(18,220) rotate(-90)' font-size='12' text-anchor='middle' fill='#555'>Completion Rate</text>"

    _save(out_path, _svg_wrap(width, height, body, "Completion Rate by Hour"))


def chart_prediction_distribution(rows, out_path: Path):
    width, height = 920, 420
    margin = (60, 40, 30, 50)
    x0, y0, x1, y1 = _axes(width, height, margin)

    pos = np.array([r["predicted_completion_prob"] for r in rows if r["target_completed"] == 1], dtype=float)
    neg = np.array([r["predicted_completion_prob"] for r in rows if r["target_completed"] == 0], dtype=float)

    bins = np.linspace(0, 1, 21)
    pos_hist, _ = np.histogram(pos, bins=bins)
    neg_hist, _ = np.histogram(neg, bins=bins)
    total = max((pos_hist + neg_hist).max(), 1)

    body = ""
    body += f"<line x1='{x0}' y1='{y0}' x2='{x1}' y2='{y0}' stroke='#444'/>"
    body += f"<line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y1}' stroke='#444'/>"

    n = len(pos_hist)
    bar_w = (x1 - x0) / n
    for i in range(n):
        x = x0 + i * bar_w
        h_pos = (y0 - y1) * (pos_hist[i] / total)
        h_neg = (y0 - y1) * (neg_hist[i] / total)

        body += f"<rect x='{x + 1:.2f}' y='{y0 - h_neg:.2f}' width='{bar_w / 2 - 2:.2f}' height='{h_neg:.2f}' fill='#ef4444' opacity='0.75'/>"
        body += f"<rect x='{x + bar_w / 2 + 1:.2f}' y='{y0 - h_pos:.2f}' width='{bar_w / 2 - 2:.2f}' height='{h_pos:.2f}' fill='#1db954' opacity='0.75'/>"

    for i in range(6):
        xv = i / 5
        x = x0 + (x1 - x0) * xv
        body += f"<text x='{x:.2f}' y='{y0 + 18}' font-size='11' text-anchor='middle' fill='#666'>{xv:.1f}</text>"

    body += "<text x='460' y='22' font-size='16' text-anchor='middle' fill='#111'>Predicted Completion Probability Distribution</text>"
    body += "<text x='460' y='410' font-size='12' text-anchor='middle' fill='#555'>Predicted Probability</text>"
    body += "<text transform='translate(18,220) rotate(-90)' font-size='12' text-anchor='middle' fill='#555'>Relative Count</text>"
    body += "<rect x='670' y='48' width='14' height='14' fill='#1db954' opacity='0.75'/><text x='690' y='59' font-size='12' fill='#333'>Completed</text>"
    body += "<rect x='770' y='48' width='14' height='14' fill='#ef4444' opacity='0.75'/><text x='790' y='59' font-size='12' fill='#333'>Skipped</text>"

    _save(out_path, _svg_wrap(width, height, body, "Prediction Distribution"))


def chart_top_artists(rows, out_path: Path, top_n=12):
    width, height = 980, 520
    margin = (240, 40, 30, 50)
    x0, y0, x1, y1 = _axes(width, height, margin)

    counts = Counter(r["artist"] for r in rows)
    top_artists = [a for a, _ in counts.most_common(top_n)][::-1]

    comp_sum = defaultdict(float)
    comp_cnt = defaultdict(int)
    for r in rows:
        a = r["artist"]
        if a in top_artists:
            comp_sum[a] += r["target_completed"]
            comp_cnt[a] += 1

    bar_h = (y0 - y1) / max(len(top_artists), 1)
    body = ""
    body += f"<line x1='{x0}' y1='{y0}' x2='{x1}' y2='{y0}' stroke='#444'/>"

    for i, artist in enumerate(top_artists):
        rate = comp_sum[artist] / max(comp_cnt[artist], 1)
        y = y1 + i * bar_h
        w = (x1 - x0) * rate
        body += f"<rect x='{x0}' y='{y + 3:.2f}' width='{w:.2f}' height='{bar_h - 6:.2f}' fill='#1db954'/>"
        body += f"<text x='{x0 - 8}' y='{y + bar_h / 2 + 4:.2f}' font-size='11' text-anchor='end' fill='#333'>{artist[:36]}</text>"
        body += f"<text x='{x0 + w + 6:.2f}' y='{y + bar_h / 2 + 4:.2f}' font-size='11' fill='#333'>{rate:.2f}</text>"

    for i in range(6):
        xv = i / 5
        x = x0 + (x1 - x0) * xv
        body += f"<line x1='{x:.2f}' y1='{y0}' x2='{x:.2f}' y2='{y1}' stroke='#efefef'/>"
        body += f"<text x='{x:.2f}' y='{y0 + 18}' font-size='11' text-anchor='middle' fill='#666'>{xv:.1f}</text>"

    body += "<text x='490' y='22' font-size='16' text-anchor='middle' fill='#111'>Completion Rate for Most Played Artists</text>"
    body += "<text x='620' y='510' font-size='12' text-anchor='middle' fill='#555'>Completion Rate</text>"

    _save(out_path, _svg_wrap(width, height, body, "Top Artists Completion"))


def chart_platform_mix(rows, out_path: Path):
    width, height = 920, 420
    margin = (60, 40, 30, 60)
    x0, y0, x1, y1 = _axes(width, height, margin)

    plat_counts = Counter(r["platform"] for r in rows)
    top = plat_counts.most_common(8)
    labels = [k for k, _ in top]
    vals = [v for _, v in top]
    total = max(sum(vals), 1)
    shares = [100.0 * v / total for v in vals]

    n = max(len(labels), 1)
    bar_w = (x1 - x0) / n

    body = ""
    body += f"<line x1='{x0}' y1='{y0}' x2='{x1}' y2='{y0}' stroke='#444'/>"
    body += f"<line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y1}' stroke='#444'/>"

    max_share = max(shares) if shares else 1.0
    for i, (label, share) in enumerate(zip(labels, shares)):
        x = x0 + i * bar_w
        h = (y0 - y1) * share / max(max_share, 1)
        body += f"<rect x='{x + 8:.2f}' y='{y0 - h:.2f}' width='{bar_w - 16:.2f}' height='{h:.2f}' fill='#2563eb'/>"
        body += f"<text x='{x + bar_w / 2:.2f}' y='{y0 + 16}' font-size='10' text-anchor='middle' fill='#333'>{label[:10]}</text>"
        body += f"<text x='{x + bar_w / 2:.2f}' y='{y0 - h - 6:.2f}' font-size='10' text-anchor='middle' fill='#333'>{share:.1f}%</text>"

    body += "<text x='460' y='22' font-size='16' text-anchor='middle' fill='#111'>Platform Mix in Scored Samples</text>"
    body += "<text x='460' y='410' font-size='12' text-anchor='middle' fill='#555'>Platform</text>"

    _save(out_path, _svg_wrap(width, height, body, "Platform Mix"))


def write_visual_summary(rows, out_dir: Path):
    comp = np.array([r["target_completed"] for r in rows], dtype=float)
    pred = np.array([r["predicted_completion_prob"] for r in rows], dtype=float)

    corr = float(np.corrcoef(pred, comp)[0, 1]) if len(rows) > 1 else 0.0
    lines = [
        "# Visualization Summary",
        "",
        f"- Rows visualized: {len(rows)}",
        f"- Mean actual completion: {comp.mean():.4f}",
        f"- Mean predicted completion: {pred.mean():.4f}",
        f"- Correlation(predicted, actual): {corr:.4f}",
        "",
        "Generated figures:",
        "- completion_rate_by_hour.svg",
        "- prediction_distribution.svg",
        "- top_artists_completion_rate.svg",
        "- platform_mix.svg",
    ]
    (out_dir / "visualization_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_dashboard_html(out_dir: Path):
    html = """<!doctype html>
<html>
<head>
  <meta charset='utf-8' />
  <title>Spotify Recommender Visuals</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #fafafa; }
    h1 { margin-bottom: 6px; }
    .grid { display: grid; grid-template-columns: 1fr; gap: 18px; }
    .card { background: white; border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px; }
    img { width: 100%; height: auto; }
  </style>
</head>
<body>
  <h1>Spotify Recommender: Visual Analytics</h1>
  <p>Generated from <code>outputs/scored_samples.csv</code>.</p>
  <div class='grid'>
    <div class='card'><h3>Completion Rate by Hour</h3><img src='completion_rate_by_hour.svg' /></div>
    <div class='card'><h3>Prediction Distribution</h3><img src='prediction_distribution.svg' /></div>
    <div class='card'><h3>Top Artists Completion</h3><img src='top_artists_completion_rate.svg' /></div>
    <div class='card'><h3>Platform Mix</h3><img src='platform_mix.svg' /></div>
  </div>
</body>
</html>
"""
    (out_dir / "dashboard.html").write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate visual analytics for Spotify recommender outputs.")
    parser.add_argument("--scored-samples", default="./outputs/scored_samples.csv")
    parser.add_argument("--out-dir", default="./outputs/figures")
    args = parser.parse_args()

    samples_path = Path(args.scored_samples).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_scored_samples(samples_path)
    chart_completion_rate_by_hour(rows, out_dir / "completion_rate_by_hour.svg")
    chart_prediction_distribution(rows, out_dir / "prediction_distribution.svg")
    chart_top_artists(rows, out_dir / "top_artists_completion_rate.svg")
    chart_platform_mix(rows, out_dir / "platform_mix.svg")
    write_visual_summary(rows, out_dir)
    write_dashboard_html(out_dir)

    print(f"Visualizations written to: {out_dir}")


if __name__ == "__main__":
    main()
