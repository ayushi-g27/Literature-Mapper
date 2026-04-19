# Literature Mapper

Literature Mapper is a portfolio project that maps research papers at the intersection of neural networks and cognitive science into a shared semantic space. It pulls metadata and abstracts from OpenAlex, embeds each abstract with a transformer model, projects the results into two dimensions with UMAP, clusters nearby papers with K-means, and serves the final map in a Streamlit app.

This is the kind of project that works well for Symbolic Systems, cognitive science, HCI, computational social science, or research engineering applications because it combines:

- API-based data collection
- messy real-world text cleaning
- modern NLP embeddings
- unsupervised structure discovery
- interactive visualization for sensemaking

## What The Project Does

Given a search query such as `"neural networks cognitive science"`, the pipeline:

1. retrieves papers from OpenAlex
2. reconstructs and cleans abstracts
3. filters out weak records such as book-like entries, duplicates, and papers without meaningful abstracts
4. converts abstracts into dense sentence embeddings with `all-MiniLM-L6-v2`
5. reduces the embedding space to 2D with UMAP
6. clusters papers with K-means
7. generates a Plotly-based literature map inside Streamlit

## Why This Is Interesting

Most paper discovery tools are keyword-first. This project is meaning-first. Papers end up near each other because their abstracts are semantically similar, not just because they reuse the same surface words. That makes it easier to spot:

- subfields that share language
- subfields that look conceptually close but publish in different venues
- clusters that deserve manual labeling and further reading

## Repository Tour

```text
Literature Mapper/
├── app.py
├── requirements.txt
├── scripts/
│   └── run_pipeline.py
├── data/
│   ├── raw/
│   └── processed/
└── literature_mapper/
    ├── analysis.py
    ├── modeling.py
    ├── openalex.py
    ├── pipeline.py
    └── preprocessing.py
```

Core files:

- `literature_mapper/openalex.py`: OpenAlex retrieval and abstract reconstruction
- `literature_mapper/preprocessing.py`: cleaning, filtering, deduplication
- `literature_mapper/modeling.py`: embeddings, UMAP, clustering
- `literature_mapper/analysis.py`: exploratory stats, cluster summaries, portfolio summary generation
- `app.py`: Streamlit app for interactive exploration

## Methods

### Data collection

- Source: OpenAlex Works API
- Query: configurable full-text search
- Language filter: English
- Abstract requirement: only records with abstracts

### Cleaning choices

The cleaning step intentionally removes some records to improve interpretability:

- papers with missing or very short abstracts
- duplicate titles or duplicate abstracts
- many book or ebook style records
- front-matter or chapter-like entries that clutter clusters

This means the final paper count is usually lower than the requested fetch count, but the resulting map is more coherent.

### Embeddings and clustering

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Dimensionality reduction: UMAP with cosine distance
- Clustering: K-means
- Cluster labels: automatically generated from top TF-IDF terms inside each cluster

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Optional:

```bash
export OPENALEX_MAILTO="your_email@example.com"
```

## Run The Pipeline

Default run:

```bash
python3 scripts/run_pipeline.py
```

Larger run:

```bash
python3 scripts/run_pipeline.py \
  --query "neural networks cognitive science" \
  --results 1000 \
  --clusters 12
```

Outputs:

- `data/raw/openalex_papers.json`
- `data/processed/papers_clean.csv`
- `data/processed/exploration_report.json`
- `data/processed/embeddings.npy`
- `data/processed/papers_mapped.csv`
- `data/processed/cluster_summary.csv`
- `data/processed/portfolio_summary.md`

## Run The App

```bash
python3 -m streamlit run app.py
```

If your shell cannot find `streamlit`, using `python3 -m streamlit` is the most reliable option.

Features:

- interactive 2D literature map
- hover cards with title, venue, year, citations, and abstract snippet
- cluster filtering
- keyword highlighting
- summary tables for clusters, journals, authors, and work types

## Limitations

- Cluster labels are heuristic and should be refined by reading representative papers.
- Search quality depends on the OpenAlex query you choose.
- UMAP is useful for visualization, but 2D distance should be interpreted qualitatively rather than as a precise metric.

## Next Improvements

- add saved human-written labels for clusters
- add date and citation filters in the app
- support multiple search presets for different research questions
- export annotated reading lists from selected clusters
