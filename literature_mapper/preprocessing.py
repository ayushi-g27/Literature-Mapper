from __future__ import annotations

import html
import re

import pandas as pd


TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
NON_WORD_RE = re.compile(r"[^a-z0-9]+")
UNHELPFUL_TITLE_PATTERNS = (
    "table of contents",
    "index",
    "front matter",
    "back matter",
)
UNHELPFUL_ABSTRACT_PATTERNS = (
    "from the publisher",
    "this chapter",
    "chapter 1",
    "previous article next article",
)
ALLOWED_WORK_TYPES = {"article", "preprint", "proceedings-article", "peer-review"}
BLOCKLIST_TERMS = (
    "6g",
    "5g",
    "wireless",
    "internet of things",
    "cognitive radio",
    "satellite communication",
    "reconfigurable intelligent surface",
    "mobile-edge",
    "asset pricing",
)
TOPIC_TERMS = (
    "cognitive science",
    "cognitive neuroscience",
    "cognition",
    "cognitive",
    "brain",
    "neuroscience",
    "neural",
    "attention",
    "memory",
    "language",
    "semantic",
    "perception",
    "decision making",
    "human behavior",
    "cortex",
    "fmri",
    "eeg",
    "connectome",
    "representation learning",
    "deep learning",
    "neural network",
    "transformer",
)


def clean_abstract(text: str) -> str:
    text = html.unescape(text or "")
    text = TAG_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def normalize_text_key(text: str) -> str:
    return NON_WORD_RE.sub(" ", (text or "").lower()).strip()


def is_probably_useful_record(row: pd.Series) -> bool:
    title = (row.get("title") or "").strip().lower()
    abstract = (row.get("abstract") or "").strip().lower()
    journal = (row.get("journal") or "").strip().lower()
    work_type = (row.get("work_type") or "").strip().lower()
    concepts = " ".join(row.get("concepts") or []).lower()
    combined = " ".join([title, abstract[:1200], concepts])

    if any(pattern in title for pattern in UNHELPFUL_TITLE_PATTERNS):
        return False
    if any(pattern in abstract[:300] for pattern in UNHELPFUL_ABSTRACT_PATTERNS):
        return False
    if any(term in combined for term in BLOCKLIST_TERMS):
        return False
    if "ebook" in journal:
        return False
    if work_type and work_type not in ALLOWED_WORK_TYPES:
        return False
    if len(abstract.split()) < 40:
        return False
    topic_hits = sum(term in combined for term in TOPIC_TERMS)
    if topic_hits < 2:
        return False
    return True


def build_dataframe(raw_papers: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(raw_papers)
    if df.empty:
        return df

    df["abstract"] = df["abstract"].fillna("").map(clean_abstract)
    df = df[df["abstract"].str.len() > 0].copy()
    df["author_count"] = df["authors"].map(len)
    df["authors_display"] = df["authors"].map(lambda authors: ", ".join(authors[:5]))
    df["concepts_display"] = df["concepts"].map(lambda concepts: ", ".join(concepts[:5]))
    df["title"] = df["title"].fillna("").str.strip()
    df = df[df["title"].str.len() > 0].copy()
    df = df[df.apply(is_probably_useful_record, axis=1)].copy()
    df["title_key"] = df["title"].map(normalize_text_key)
    df["abstract_key"] = df["abstract"].map(normalize_text_key)
    df = df.sort_values(["cited_by_count", "publication_year"], ascending=[False, False])
    df = df.drop_duplicates(subset=["openalex_id"], keep="first")
    with_doi = df[df["doi"].fillna("").str.len() > 0].drop_duplicates(subset=["doi"], keep="first")
    without_doi = df[df["doi"].fillna("").str.len() == 0]
    df = pd.concat([with_doi, without_doi], ignore_index=True)
    df = df.drop_duplicates(subset=["title_key"], keep="first")
    df = df.drop_duplicates(subset=["abstract_key"], keep="first")
    df = df.drop(columns=["title_key", "abstract_key"])
    df.reset_index(drop=True, inplace=True)
    return df
