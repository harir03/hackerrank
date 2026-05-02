"""Retrieval layer — BM25 search over local corpus markdown files.

Public functions:
- retrieve_chunks(search_query, domain, top_k=3) -> list[dict]
- compute_coverage_score(chunks) -> float

Corpus files are loaded once from data/{domain}/**/*.md and cached
in memory. This module makes NO API calls.
"""

import re
from pathlib import Path

from rank_bm25 import BM25Okapi


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_DOMAINS = ["hackerrank", "claude", "visa"]

# data/ is at repo root: code/pipeline/../../data
DEFAULT_DATA_DIR = str(Path(__file__).parent.parent.parent / "data")

# Coverage threshold: below this, corpus does not cover the topic
COVERAGE_THRESHOLD = 0.25

# Normalisation constant for BM25 scores
NORM_K = 10.0


# ---------------------------------------------------------------------------
# Section 1: Loader
# ---------------------------------------------------------------------------

def _extract_section_slug(filename: str) -> str:
    """Extract slug from filename, stripping leading hash prefix if present.

    Pattern: "{digits}-{slug}.md" -> slug
    Or just: "{slug}.md" -> slug
    """
    stem = Path(filename).stem  # remove .md
    # Check for "{digits}-{slug}" pattern
    match = re.match(r"^\d+-(.+)$", stem)
    if match:
        return match.group(1)
    return stem


def _load_markdown_corpus(data_dir: str) -> dict[str, list[dict]]:
    """Walk data_dir/{domain}/ and load all .md files into chunks."""
    data_path = Path(data_dir)
    result: dict[str, list[dict]] = {}

    for domain in VALID_DOMAINS:
        domain_dir = data_path / domain
        chunks: list[dict] = []

        if not domain_dir.exists():
            result[domain] = chunks
            continue

        md_files = sorted(domain_dir.rglob("*.md"))

        for filepath in md_files:
            try:
                content = filepath.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            if not content.strip():
                continue

            section = _extract_section_slug(filepath.name)
            file_stem = filepath.stem
            rel_path = str(filepath.relative_to(data_path))

            # Split into paragraphs
            paragraphs = content.split("\n\n")

            chunk_index = 0
            for para in paragraphs:
                chunk_text = para.strip()
                if not chunk_text:
                    continue

                word_count = len(chunk_text.split())

                # Skip if fewer than 20 words
                if word_count < 20:
                    continue

                # Skip if no sentence punctuation
                if not any(p in chunk_text for p in (".", "?", "!", ":")):
                    continue

                # Skip if chunk is a pure header line (starts with #)
                if chunk_text.startswith("#"):
                    # But only skip if the ENTIRE chunk is header lines
                    lines = chunk_text.split("\n")
                    if all(line.strip().startswith("#") or not line.strip()
                           for line in lines):
                        continue

                chunks.append({
                    "chunk_id": f"{domain}_{file_stem}_{chunk_index}",
                    "domain": domain,
                    "section": section,
                    "text": chunk_text,
                    "filepath": rel_path,
                    "word_count": word_count,
                })
                chunk_index += 1

        result[domain] = chunks

    return result


# ---------------------------------------------------------------------------
# Section 2: Cache and BM25 index
# ---------------------------------------------------------------------------

_corpus: dict[str, list[dict]] = {}
_bm25_indices: dict[str, BM25Okapi] = {}
_corpus_loaded: bool = False


def _tokenise(text: str) -> list[str]:
    """Tokenise by whitespace and lowercase."""
    return text.lower().split()


def _ensure_loaded(data_dir: str = DEFAULT_DATA_DIR) -> None:
    """Load corpus and build BM25 indices if not already done."""
    global _corpus, _bm25_indices, _corpus_loaded

    if _corpus_loaded:
        return

    _corpus = _load_markdown_corpus(data_dir)

    for domain in VALID_DOMAINS:
        chunks = _corpus.get(domain, [])
        if not chunks:
            continue

        tokenised_docs = [_tokenise(c["text"]) for c in chunks]
        _bm25_indices[domain] = BM25Okapi(tokenised_docs)

        # Count unique files
        files = set(c["filepath"] for c in chunks)
        print(f"[CORPUS] {domain}: {len(files)} files -> {len(chunks)} chunks")

    total = sum(len(v) for v in _corpus.values())
    print(f"[CORPUS] Total: {total} chunks across 3 domains")

    _corpus_loaded = True


# ---------------------------------------------------------------------------
# Section 3: Public functions
# ---------------------------------------------------------------------------

def retrieve_chunks(
    search_query: str,
    domain: str,
    top_k: int = 3,
    data_dir: str = DEFAULT_DATA_DIR,
) -> list[dict]:
    """Retrieve top_k corpus chunks matching the search query.

    Args:
        search_query: The query string to search for.
        domain: Domain to search in. If "unknown", searches all domains.
        top_k: Number of top results to return.
        data_dir: Path to the data directory (default: "data").

    Returns:
        List of chunk dicts with keys: chunk_id, domain, section,
        text, filepath, word_count, score. Empty list if no results.
    """
    _ensure_loaded(data_dir)

    if not search_query or not search_query.strip():
        return []

    query_tokens = _tokenise(search_query)
    if not query_tokens:
        return []

    # Determine which domains to search
    if domain == "unknown" or domain not in VALID_DOMAINS:
        domains_to_search = VALID_DOMAINS
    else:
        domains_to_search = [domain]

    # Collect scored results from all domains
    all_results: list[dict] = []

    for d in domains_to_search:
        if d not in _bm25_indices:
            continue

        index = _bm25_indices[d]
        chunks = _corpus[d]
        scores = index.get_scores(query_tokens)

        for i, score in enumerate(scores):
            if score > 0:
                chunk = chunks[i].copy()
                chunk["score"] = float(score)
                all_results.append(chunk)

    # Sort by score descending and take top_k
    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:top_k]


def compute_coverage_score(chunks: list[dict]) -> float:
    """Compute normalised coverage score from retrieved chunks.

    Uses score / (score + K) normalisation to map BM25 scores to 0.0-1.0.

    Args:
        chunks: List of chunk dicts with "score" key.

    Returns:
        Float 0.0-1.0. 0.0 if chunks is empty.
    """
    if not chunks:
        return 0.0

    top_score = chunks[0].get("score", 0.0)
    if top_score <= 0:
        return 0.0

    # Normalise: score / (score + K)
    return top_score / (top_score + NORM_K)
