from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MAPPED_PATH = PROCESSED_DIR / "papers_mapped.csv"
SUMMARY_PATH = PROCESSED_DIR / "cluster_summary.csv"
REPORT_PATH = PROCESSED_DIR / "exploration_report.json"


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    papers = pd.read_csv(MAPPED_PATH)
    summary = pd.read_csv(SUMMARY_PATH)
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    return papers, summary, report


def highlight_matches(df: pd.DataFrame, keyword: str) -> pd.DataFrame:
    keyword = keyword.strip().lower()
    if not keyword:
        df["match_status"] = "All papers"
        return df

    title_matches = df["title"].fillna("").str.lower().str.contains(keyword, regex=False)
    abstract_matches = df["abstract"].fillna("").str.lower().str.contains(keyword, regex=False)
    df["match_status"] = (title_matches | abstract_matches).map(
        {True: f'Matches "{keyword}"', False: "Other papers"}
    )
    return df


def main() -> None:
    st.set_page_config(page_title="Literature Mapper", layout="wide")
    st.title("Literature Mapper")
    st.write("Explore clusters of papers at the intersection of neural networks and cognitive science.")

    if not (MAPPED_PATH.exists() and SUMMARY_PATH.exists() and REPORT_PATH.exists()):
        st.warning("Run the pipeline first: `python3 scripts/run_pipeline.py`")
        st.stop()

    papers, summary, report = load_data()

    keyword = st.sidebar.text_input("Search keyword")
    selected_clusters = st.sidebar.multiselect(
        "Visible clusters",
        options=sorted(papers["cluster"].unique().tolist()),
        default=sorted(papers["cluster"].unique().tolist()),
    )

    filtered = papers[papers["cluster"].isin(selected_clusters)].copy()
    filtered = highlight_matches(filtered, keyword)
    filtered["abstract_snippet"] = filtered["abstract"].fillna("").str.slice(0, 220) + "..."

    st.subheader("Mini Findings")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Papers", report["paper_count"])
    metric_cols[1].metric("Clusters", int(papers["cluster"].nunique()))
    metric_cols[2].metric("Years Covered", len(report["year_distribution"]))
    metric_cols[3].metric("Avg Abstract Words", report["mean_abstract_length_words"])

    fig = px.scatter(
        filtered,
        x="x",
        y="y",
        color="cluster",
        symbol="match_status",
        hover_name="title",
        hover_data={
            "journal": True,
            "publication_year": True,
            "authors_display": True,
            "cited_by_count": True,
            "abstract_snippet": True,
            "x": False,
            "y": False,
        },
        title="2D Map of the Literature",
        height=700,
    )
    fig.update_traces(marker=dict(size=10, opacity=0.8))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster Summaries")
    st.dataframe(summary, use_container_width=True)

    st.subheader("Top Journals")
    journal_df = pd.DataFrame(
        list(report["top_journals"].items()),
        columns=["journal", "count"],
    )
    st.dataframe(journal_df, use_container_width=True)

    st.subheader("Top Authors")
    authors_df = pd.DataFrame(report["top_authors"], columns=["author", "count"])
    st.dataframe(authors_df, use_container_width=True)

    st.subheader("Work Types")
    work_types_df = pd.DataFrame(
        list(report["work_type_distribution"].items()),
        columns=["work_type", "count"],
    )
    st.dataframe(work_types_df, use_container_width=True)


if __name__ == "__main__":
    main()
