import csv
import json
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"


def read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    st.set_page_config(page_title="Spotify Recommender Dashboard", layout="wide")
    st.title("Spotify Recommender Dashboard")
    st.caption("Interactive view of model outputs and recommendation artifacts.")

    metrics = read_json(OUTPUTS / "model_metrics.json", {})
    model_comp = read_csv(OUTPUTS / "model_comparison.csv")
    dataset_comp = read_csv(OUTPUTS / "dataset_comparison.csv")
    backtests = read_csv(OUTPUTS / "rolling_backtests.csv")
    top_global = read_csv(OUTPUTS / "top_resume_playlist.csv")
    top_favs = read_csv(OUTPUTS / "top_favorites_playlist.csv")
    cold_start = read_csv(OUTPUTS / "cold_start_recommendations.csv")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", f"{float(metrics.get('test_auc', 0.0)):.4f}")
    c2.metric("PR-AUC", f"{float(metrics.get('test_pr_auc', 0.0)):.4f}")
    c3.metric("F1", f"{float(metrics.get('test_f1', 0.0)):.4f}")
    c4.metric("NDCG@10", f"{float(metrics.get('ndcg@10', 0.0)):.4f}")

    st.subheader("Model Comparison")
    st.dataframe(model_comp, use_container_width=True)

    st.subheader("Dataset Comparison")
    st.dataframe(dataset_comp, use_container_width=True)

    st.subheader("Rolling Backtests")
    if backtests:
        st.dataframe(backtests, use_container_width=True)
    else:
        st.info("No rolling backtest file found yet. Run pipeline to generate `outputs/rolling_backtests.csv`.")

    st.subheader("Recommendations")
    tab1, tab2, tab3 = st.tabs(["Global Top 100", "Favorites Boosted", "Cold Start Fallback"])
    with tab1:
        st.dataframe(top_global, use_container_width=True)
    with tab2:
        st.dataframe(top_favs, use_container_width=True)
    with tab3:
        st.dataframe(cold_start, use_container_width=True)

    st.subheader("Visual Diagnostics")
    fig_dir = OUTPUTS / "figures"
    for name in [
        "completion_rate_by_hour.svg",
        "prediction_distribution.svg",
        "top_artists_completion_rate.svg",
        "platform_mix.svg",
    ]:
        p = fig_dir / name
        if p.exists():
            st.markdown(f"**{name}**")
            st.image(str(p))


if __name__ == "__main__":
    main()
