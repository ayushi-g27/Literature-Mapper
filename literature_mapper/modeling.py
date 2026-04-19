from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def generate_embeddings(texts: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.encode(texts, show_progress_bar=True)


def reduce_embeddings(embeddings: np.ndarray, random_state: int = 42) -> np.ndarray:
    cache_dir = Path(".numba_cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir))

    import umap

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)


def cluster_embeddings(
    embeddings: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.cluster import KMeans

    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = model.fit_predict(embeddings)
    distances = np.linalg.norm(embeddings - model.cluster_centers_[labels], axis=1)
    return labels, distances


def attach_model_outputs(
    df: pd.DataFrame,
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    distances: np.ndarray,
) -> pd.DataFrame:
    modeled = df.copy()
    modeled["x"] = embeddings_2d[:, 0]
    modeled["y"] = embeddings_2d[:, 1]
    modeled["cluster"] = labels.astype(int)
    modeled["distance_to_cluster_center"] = distances
    return modeled
