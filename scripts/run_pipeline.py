from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from literature_mapper.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Literature Mapper pipeline.")
    parser.add_argument(
        "--query",
        default="neural networks cognitive science",
        help="OpenAlex search query.",
    )
    parser.add_argument(
        "--results",
        type=int,
        default=1000,
        help="Number of papers to fetch from OpenAlex.",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=12,
        help="Number of K-means clusters.",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name.",
    )
    args = parser.parse_args()

    outputs = run_pipeline(
        project_root=PROJECT_ROOT,
        search_query=args.query,
        total_results=args.results,
        model_name=args.model,
        n_clusters=args.clusters,
    )
    print("Pipeline complete.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
