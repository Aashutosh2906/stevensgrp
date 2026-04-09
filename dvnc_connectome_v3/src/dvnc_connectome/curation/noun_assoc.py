"""
Concept extraction and co-occurrence edge builder.

Strategy: regex tokenisation + stopword removal + windowed co-occurrence.
No heavy NLP dependency — runs everywhere.
"""

import re
from collections import Counter
from itertools import combinations

# Comprehensive stopword list covering general + domain-specific noise
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "through", "during",
    "is", "was", "are", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "this", "that", "these", "those", "it", "its", "they", "them", "their",
    "we", "our", "you", "your", "he", "she", "his", "her", "i", "my",
    "as", "so", "not", "no", "nor", "yet", "both", "either", "each", "all",
    "more", "most", "than", "very", "too", "also", "just", "such", "how",
    "what", "which", "who", "when", "where", "why", "if", "then", "because",
    "while", "although", "though", "however", "therefore", "thus", "hence",
    "can", "any", "other", "some", "only", "used", "use", "using", "based",
    "data", "figure", "table", "paper", "study", "method", "result", "model",
    "analysis", "value", "values", "number", "total", "mean", "standard",
    "ref", "et", "al", "ibid", "doi", "http", "www", "com", "org", "pdf",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "new", "different", "large", "small", "high", "low", "given", "following",
    "across", "between", "within", "along", "well", "without", "under",
}

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z\-]{2,}")


def concepts(text: str, max_terms: int = 100) -> list[str]:
    """
    Extract top `max_terms` candidate concept tokens from text.
    Returns a list ordered by frequency (most common first).
    """
    if not text:
        return []
    tokens = _TOKEN_RE.findall(text.lower())
    tokens = [t for t in tokens if t not in _STOPWORDS and len(t) >= 3]
    freq = Counter(tokens)
    return [word for word, _ in freq.most_common(max_terms)]


def cooccurrence_edges(terms: list[str], window: int = 8) -> list[tuple[str, str, float]]:
    """
    Slide a window over `terms` and emit (a, b, weight) pairs for every
    unique pair within the window. Weight = 1 / distance (closer = stronger).
    """
    edges: dict[tuple[str, str], float] = {}
    for i, term_a in enumerate(terms):
        for j in range(i + 1, min(i + window, len(terms))):
            term_b = terms[j]
            if term_a == term_b:
                continue
            key = (min(term_a, term_b), max(term_a, term_b))
            distance = j - i
            weight = 1.0 / distance
            edges[key] = edges.get(key, 0.0) + weight
    return [(a, b, w) for (a, b), w in edges.items()]
