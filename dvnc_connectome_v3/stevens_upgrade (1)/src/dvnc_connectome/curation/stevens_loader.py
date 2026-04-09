"""
Stevens Group Corpus Loader + Dynamic Paper Ingestion

Three data sources for the Stevens Group connectome:
  1. stevens_corpus.json — Pre-built corpus of 11 core Stevens papers
  2. Google Scholar / Semantic Scholar — Fetch new papers by DOI, URL, or search
  3. Researcher Upload — Oxford researchers add papers directly (JSON, BibTeX, or plain text)

This module integrates all three into the DVNC curation pipeline.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Iterator

import requests


# ── Path Configuration ──────────────────────────────────────────────────────

# Resolve data dir: .../src/dvnc_connectome/curation/ → go up to repo root (3 levels to src, 1 more to root)
_MODULE_DIR = Path(__file__).resolve().parent  # .../src/dvnc_connectome/curation/
_REPO_ROOT = _MODULE_DIR.parent.parent.parent  # .../src/ → repo root
_DATA_DIR = _REPO_ROOT / "data" / "stevens"
_UPLOAD_DIR = _DATA_DIR / "uploads"
_CACHE_DIR = _DATA_DIR / "cache"


def _ensure_dirs():
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Core Stevens Corpus ──────────────────────────────────────────────────

def iter_stevens_corpus() -> Iterator[dict]:
    """
    Load the pre-built Stevens Group corpus (11 core papers, ~33 documents).
    This is always loaded as the foundation of the Stevens knowledge graph.
    """
    corpus_path = _DATA_DIR / "stevens_corpus.json"
    if not corpus_path.exists():
        print("[stevens] Warning: stevens_corpus.json not found — run with network to build")
        return

    with open(corpus_path) as f:
        docs = json.load(f)

    for doc in docs:
        yield {
            "doc_id": doc.get("doc_id", ""),
            "title": doc.get("title", ""),
            "text": doc.get("text", ""),
            "source": "stevens_group",
            "domain": doc.get("domain", "general"),
            "year": doc.get("year"),
            "url": doc.get("url", ""),
        }

    print(f"[stevens] Loaded {len(docs)} documents from core corpus")


# ── 2. Semantic Scholar / OpenAlex Fetch ────────────────────────────────────

def fetch_paper_by_doi(doi: str) -> dict | None:
    """
    Fetch paper metadata + abstract from Semantic Scholar API (free, no key needed).
    Falls back to OpenAlex if Semantic Scholar fails.
    """
    _ensure_dirs()
    cache_key = re.sub(r"[^a-z0-9]", "_", doi.lower())[:60] + ".json"
    cache_path = _CACHE_DIR / cache_key

    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    paper = _fetch_semantic_scholar(doi)
    if not paper:
        paper = _fetch_openalex_doi(doi)

    if paper:
        with open(cache_path, "w") as f:
            json.dump(paper, f)

    return paper


def _fetch_semantic_scholar(doi: str) -> dict | None:
    """Fetch from Semantic Scholar API."""
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
    params = {
        "fields": "title,abstract,year,authors,venue,externalIds,tldr,fieldsOfStudy"
    }
    try:
        r = requests.get(url, params=params, timeout=15,
                         headers={"User-Agent": "DVNC.AI/3.0 research"})
        if r.status_code != 200:
            return None
        data = r.json()
        title = data.get("title", "")
        abstract = data.get("abstract", "")
        tldr = data.get("tldr", {}).get("text", "")
        authors = ", ".join(a.get("name", "") for a in data.get("authors", []))

        text = f"{title}. {abstract}"
        if tldr:
            text += f" {tldr}"

        if len(text) < 50:
            return None

        return {
            "doc_id": f"scholar_{re.sub(r'[^a-z0-9]', '_', doi.lower())[:40]}",
            "title": title,
            "text": text,
            "source": "semantic_scholar",
            "domain": _classify_domain_from_fields(data.get("fieldsOfStudy", [])),
            "year": data.get("year"),
            "url": f"https://doi.org/{doi}",
            "authors": authors,
            "journal": data.get("venue", ""),
            "doi": doi,
        }
    except Exception as e:
        print(f"[stevens] Semantic Scholar error for {doi}: {e}")
        return None


def _fetch_openalex_doi(doi: str) -> dict | None:
    """Fetch from OpenAlex API as fallback."""
    url = f"https://api.openalex.org/works/doi:{doi}"
    try:
        r = requests.get(url, timeout=15,
                         headers={"User-Agent": "DVNC.AI/3.0"})
        if r.status_code != 200:
            return None
        data = r.json()
        title = data.get("title", "")
        abstract = _reconstruct_openalex_abstract(
            data.get("abstract_inverted_index", {})
        )
        text = f"{title}. {abstract}".strip()
        if len(text) < 50:
            return None

        return {
            "doc_id": f"openalex_{data.get('id', '').split('/')[-1]}",
            "title": title,
            "text": text,
            "source": "openalex",
            "domain": "general",
            "year": data.get("publication_year"),
            "url": f"https://doi.org/{doi}",
            "doi": doi,
        }
    except Exception as e:
        print(f"[stevens] OpenAlex error for {doi}: {e}")
        return None


def search_papers(query: str, max_results: int = 10) -> list[dict]:
    """
    Search for papers via Semantic Scholar.
    Returns list of document dicts ready for pipeline ingestion.
    """
    _ensure_dirs()
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": min(max_results, 20),
        "fields": "title,abstract,year,authors,venue,externalIds,fieldsOfStudy",
    }
    try:
        r = requests.get(url, params=params, timeout=20,
                         headers={"User-Agent": "DVNC.AI/3.0"})
        if r.status_code != 200:
            print(f"[stevens] Search failed: {r.status_code}")
            return []
        results = r.json().get("data", [])
    except Exception as e:
        print(f"[stevens] Search error: {e}")
        return []

    docs = []
    for item in results:
        title = item.get("title", "")
        abstract = item.get("abstract", "")
        if not abstract or len(abstract) < 50:
            continue

        text = f"{title}. {abstract}"
        authors = ", ".join(a.get("name", "") for a in item.get("authors", []))
        doi = item.get("externalIds", {}).get("DOI", "")

        docs.append({
            "doc_id": f"search_{item.get('paperId', '')[:20]}",
            "title": title,
            "text": text,
            "source": "semantic_scholar_search",
            "domain": _classify_domain_from_fields(item.get("fieldsOfStudy", [])),
            "year": item.get("year"),
            "url": f"https://doi.org/{doi}" if doi else "",
            "authors": authors,
            "doi": doi,
        })
        time.sleep(0.3)  # polite rate limit

    print(f"[stevens] Found {len(docs)} papers for query: '{query}'")
    return docs


# ── 3. Researcher Upload ────────────────────────────────────────────────────

def save_uploaded_paper(title: str, text: str, domain: str = "general",
                        authors: str = "", year: int | None = None,
                        url: str = "", doi: str = "") -> dict:
    """
    Save a researcher-uploaded paper to the uploads directory.
    Returns the document dict for immediate pipeline ingestion.
    """
    _ensure_dirs()

    doc_id = f"upload_{re.sub(r'[^a-z0-9]', '_', title.lower())[:40]}_{int(time.time())}"
    doc = {
        "doc_id": doc_id,
        "title": title,
        "text": text,
        "source": "researcher_upload",
        "domain": domain,
        "year": year,
        "url": url,
        "authors": authors,
        "doi": doi,
        "uploaded_at": time.time(),
    }

    # Save to uploads directory
    save_path = _UPLOAD_DIR / f"{doc_id}.json"
    with open(save_path, "w") as f:
        json.dump(doc, f, indent=2)

    print(f"[stevens] Saved uploaded paper: {title}")
    return doc


def iter_uploaded_papers() -> Iterator[dict]:
    """
    Iterate over all researcher-uploaded papers.
    """
    _ensure_dirs()
    upload_files = sorted(_UPLOAD_DIR.glob("*.json"))
    count = 0
    for fp in upload_files:
        try:
            with open(fp) as f:
                doc = json.load(f)
            yield {
                "doc_id": doc.get("doc_id", fp.stem),
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
                "source": doc.get("source", "researcher_upload"),
                "domain": doc.get("domain", "general"),
                "year": doc.get("year"),
                "url": doc.get("url", ""),
            }
            count += 1
        except Exception as e:
            print(f"[stevens] Error reading {fp}: {e}")

    if count:
        print(f"[stevens] Loaded {count} uploaded papers")


def parse_bibtex_entry(bibtex: str) -> dict:
    """
    Parse a single BibTeX entry into a document dict.
    Handles @article, @inproceedings, etc.
    """
    fields = {}
    # Extract entry type and key
    m = re.match(r"@(\w+)\{([^,]+),", bibtex.strip())
    if m:
        fields["entry_type"] = m.group(1)
        fields["key"] = m.group(2)

    # Extract field values
    for match in re.finditer(r"(\w+)\s*=\s*\{([^}]*)\}", bibtex):
        fields[match.group(1).lower()] = match.group(2).strip()

    title = fields.get("title", "Unknown")
    abstract = fields.get("abstract", "")
    text = f"{title}. {abstract}".strip() if abstract else title

    return {
        "doc_id": f"bibtex_{fields.get('key', 'unknown')[:30]}",
        "title": title,
        "text": text,
        "source": "bibtex_upload",
        "domain": "general",
        "year": int(fields["year"]) if "year" in fields and fields["year"].isdigit() else None,
        "url": fields.get("url", fields.get("doi", "")),
        "authors": fields.get("author", ""),
        "doi": fields.get("doi", ""),
    }


# ── 4. Master Iterator ──────────────────────────────────────────────────────

def iter_all_stevens_docs(include_uploads: bool = True,
                           extra_dois: list[str] | None = None,
                           search_queries: list[str] | None = None) -> Iterator[dict]:
    """
    Master iterator: yields all Stevens Group documents from all sources.

    Sources:
      1. Core Stevens corpus (always)
      2. Researcher uploads (if include_uploads=True)
      3. Papers fetched by DOI (if extra_dois provided)
      4. Papers found via search (if search_queries provided)
    """
    # 1. Core corpus
    yield from iter_stevens_corpus()

    # 2. Uploaded papers
    if include_uploads:
        yield from iter_uploaded_papers()

    # 3. DOI-based fetches
    if extra_dois:
        for doi in extra_dois:
            paper = fetch_paper_by_doi(doi)
            if paper:
                yield paper
            time.sleep(0.5)

    # 4. Search-based fetches
    if search_queries:
        for query in search_queries:
            papers = search_papers(query, max_results=10)
            yield from papers
            time.sleep(1.0)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _classify_domain_from_fields(fields: list[str] | None) -> str:
    if not fields:
        return "general"
    fields_lower = [f.lower() for f in fields]
    if any("raman" in f or "spectro" in f or "optic" in f for f in fields_lower):
        return "raman_spectroscopy"
    if any("cardiac" in f or "heart" in f or "myocard" in f for f in fields_lower):
        return "cardiac_biomaterials"
    if any("polymer" in f or "conduct" in f or "electroact" in f for f in fields_lower):
        return "conducting_polymers"
    if any("bone" in f or "mineral" in f or "scaffold" in f for f in fields_lower):
        return "biomineralization"
    if any("material" in f or "nano" in f for f in fields_lower):
        return "materials"
    if any("bio" in f or "cell" in f or "tissue" in f for f in fields_lower):
        return "biomechanics"
    return "general"


def _reconstruct_openalex_abstract(inv_index: dict) -> str:
    if not inv_index:
        return ""
    positions: dict[int, str] = {}
    for word, pos_list in inv_index.items():
        for pos in pos_list:
            positions[pos] = word
    return " ".join(positions[k] for k in sorted(positions))
