from __future__ import annotations

from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def build_exploration_report(df: pd.DataFrame) -> dict:
    years = (
        df["publication_year"]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
    )
    author_counter = Counter()
    for authors in df["authors"]:
        author_counter.update(authors)

    journal_counts = (
        df["journal"]
        .replace("", pd.NA)
        .dropna()
        .value_counts()
        .head(10)
        .to_dict()
    )
    work_type_counts = (
        df["work_type"]
        .replace("", pd.NA)
        .dropna()
        .value_counts()
        .to_dict()
    )
    year_values = df["publication_year"].dropna().astype(int)

    return {
        "paper_count": int(len(df)),
        "year_distribution": years,
        "year_span": {
            "min": int(year_values.min()) if not year_values.empty else None,
            "max": int(year_values.max()) if not year_values.empty else None,
        },
        "top_authors": author_counter.most_common(10),
        "top_journals": journal_counts,
        "work_type_distribution": work_type_counts,
        "mean_abstract_length_words": round(df["abstract"].str.split().map(len).mean(), 2),
        "median_citations": round(float(df["cited_by_count"].median()), 2),
    }


def summarize_clusters(df: pd.DataFrame, top_n_terms: int = 5) -> pd.DataFrame:
    summaries: list[dict] = []

    for cluster_id, cluster_df in df.groupby("cluster"):
        vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
        try:
            matrix = vectorizer.fit_transform(cluster_df["abstract"])
            term_scores = matrix.mean(axis=0).A1
            terms = vectorizer.get_feature_names_out()
            ranked_terms = [terms[idx] for idx in term_scores.argsort()[::-1][:top_n_terms]]
        except ValueError:
            ranked_terms = ["unclear-theme"]

        central_papers = (
            cluster_df.sort_values("distance_to_cluster_center")
            .head(3)["title"]
            .tolist()
        )
        summaries.append(
            {
                "cluster": int(cluster_id),
                "size": int(len(cluster_df)),
                "label": ", ".join(ranked_terms[:3]),
                "top_terms": ", ".join(ranked_terms),
                "central_papers": " | ".join(central_papers),
            }
        )

    return pd.DataFrame(summaries).sort_values("cluster").reset_index(drop=True)


def render_portfolio_summary(report: dict, cluster_summary: pd.DataFrame) -> str:
    top_cluster = cluster_summary.sort_values("size", ascending=False).iloc[0].to_dict()
    lines = [
        "# Portfolio Summary",
        "",
        "## Dataset",
        f"- Papers analyzed: {report['paper_count']}",
        f"- Year span: {report['year_span']['min']} to {report['year_span']['max']}",
        f"- Average abstract length: {report['mean_abstract_length_words']} words",
        f"- Median citation count: {report['median_citations']}",
        "",
        "## Most common venues",
    ]

    for journal, count in list(report["top_journals"].items())[:5]:
        lines.append(f"- {journal}: {count}")

    lines.extend(
        [
            "",
            "## Largest cluster",
            f"- Cluster {int(top_cluster['cluster'])} contains {int(top_cluster['size'])} papers.",
            f"- Auto-label: {top_cluster['label']}",
            f"- Representative papers: {top_cluster['central_papers']}",
            "",
            "## Notes",
            "- Cluster labels are generated automatically from TF-IDF terms and should be refined by reading representative papers.",
            "- The pipeline prioritizes English-language records with meaningful abstracts and removes many book-like or duplicate entries.",
        ]
    )
    return "\n".join(lines)
