from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import requests


OPENALEX_WORKS_URL = "https://api.openalex.org/works"


@dataclass
class OpenAlexPaper:
    openalex_id: str
    title: str
    abstract: str
    publication_year: int | None
    journal: str
    source_type: str
    authors: list[str]
    cited_by_count: int
    concepts: list[str]
    doi: str
    work_type: str
    language: str


def decode_abstract(inverted_index: dict | None) -> str:
    if not inverted_index:
        return ""

    positions: list[tuple[int, str]] = []
    for token, indexes in inverted_index.items():
        for index in indexes:
            positions.append((index, token))
    positions.sort(key=lambda item: item[0])
    return " ".join(token for _, token in positions)


def _extract_authors(authorships: Iterable[dict] | None) -> list[str]:
    if not authorships:
        return []
    authors: list[str] = []
    for authorship in authorships:
        author_name = (authorship.get("author") or {}).get("display_name")
        if author_name:
            authors.append(author_name)
    return authors


def fetch_openalex_papers(
    search_query: str,
    total_results: int = 500,
    per_page: int = 200,
    mailto: str | None = None,
) -> list[OpenAlexPaper]:
    """Fetch papers from OpenAlex using the works search endpoint."""
    session = requests.Session()
    headers = {"User-Agent": f"literature-mapper/0.1 ({mailto or 'local-project'})"}
    papers: list[OpenAlexPaper] = []
    cursor = "*"

    while len(papers) < total_results:
        params = {
            "search": search_query,
            "per-page": min(per_page, total_results - len(papers)),
            "cursor": cursor,
            "filter": "has_abstract:true,language:en",
        }
        if mailto:
            params["mailto"] = mailto

        response = session.get(OPENALEX_WORKS_URL, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        payload = response.json()

        for result in payload.get("results", []):
            source = ((result.get("primary_location") or {}).get("source") or {})
            concept_names = [concept.get("display_name", "") for concept in result.get("concepts", [])]
            papers.append(
                OpenAlexPaper(
                    openalex_id=result.get("id", ""),
                    title=result.get("display_name", "").strip(),
                    abstract=decode_abstract(result.get("abstract_inverted_index")),
                    publication_year=result.get("publication_year"),
                    journal=source.get("display_name", ""),
                    source_type=source.get("type", "") or "",
                    authors=_extract_authors(result.get("authorships")),
                    cited_by_count=result.get("cited_by_count", 0),
                    concepts=[name for name in concept_names if name],
                    doi=result.get("doi", "") or "",
                    work_type=result.get("type", "") or "",
                    language=result.get("language", "") or "",
                )
            )
            if len(papers) >= total_results:
                break

        cursor = (payload.get("meta") or {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(0.2)

    return papers
