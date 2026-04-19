from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from literature_mapper.analysis import (
    build_exploration_report,
    render_portfolio_summary,
    summarize_clusters,
)
from literature_mapper.modeling import (
    attach_model_outputs,
    cluster_embeddings,
    generate_embeddings,
    reduce_embeddings,
)
from literature_mapper.openalex import fetch_openalex_papers
from literature_mapper.preprocessing import build_dataframe


def ensure_data_dirs(project_root: Path) -> dict[str, Path]:
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    outputs = {
        "data": data_dir,
        "raw": raw_dir,
        "processed": processed_dir,
    }
    for path in outputs.values():
        path.mkdir(parents=True, exist_ok=True)
    return outputs


def run_pipeline(
    project_root: Path,
    search_query: str = "neural networks cognitive science",
    total_results: int = 1000,
    model_name: str = "all-MiniLM-L6-v2",
    n_clusters: int = 12,
) -> dict[str, str]:
    paths = ensure_data_dirs(project_root)
    mailto = os.getenv("OPENALEX_MAILTO")

    raw_papers = fetch_openalex_papers(
        search_query=search_query,
        total_results=total_results,
        mailto=mailto,
    )
    raw_payload = [asdict(paper) for paper in raw_papers]
    raw_json_path = paths["raw"] / "openalex_papers.json"
    raw_json_path.write_text(json.dumps(raw_payload, indent=2), encoding="utf-8")

    clean_df = build_dataframe(raw_payload)
    if clean_df.empty:
        raise ValueError("No usable papers were returned after cleaning. Try a broader query.")
    clean_csv_path = paths["processed"] / "papers_clean.csv"
    clean_df.to_csv(clean_csv_path, index=False)

    exploration_report = build_exploration_report(clean_df)
    report_path = paths["processed"] / "exploration_report.json"
    report_path.write_text(json.dumps(exploration_report, indent=2), encoding="utf-8")

    embeddings = generate_embeddings(clean_df["abstract"].tolist(), model_name=model_name)
    np.save(paths["processed"] / "embeddings.npy", embeddings)

    embedding_2d = reduce_embeddings(embeddings)
    labels, distances = cluster_embeddings(embeddings, n_clusters=n_clusters)
    mapped_df = attach_model_outputs(clean_df, embedding_2d, labels, distances)
    mapped_csv_path = paths["processed"] / "papers_mapped.csv"
    mapped_df.to_csv(mapped_csv_path, index=False)

    cluster_summary = summarize_clusters(mapped_df)
    cluster_summary_path = paths["processed"] / "cluster_summary.csv"
    cluster_summary.to_csv(cluster_summary_path, index=False)
    portfolio_summary_path = paths["processed"] / "portfolio_summary.md"
    portfolio_summary_path.write_text(
        render_portfolio_summary(exploration_report, cluster_summary),
        encoding="utf-8",
    )

    return {
        "raw_json": str(raw_json_path),
        "clean_csv": str(clean_csv_path),
        "report_json": str(report_path),
        "mapped_csv": str(mapped_csv_path),
        "cluster_summary_csv": str(cluster_summary_path),
        "portfolio_summary_md": str(portfolio_summary_path),
    }
