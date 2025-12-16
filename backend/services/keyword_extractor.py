import re
import nltk
from collections import Counter
from itertools import combinations

# Download once (safe to call multiple times)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


STOP_WORDS = set(stopwords.words("english"))

# ----------------------------------
# Clean text
# ----------------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ----------------------------------
# Extract keywords + co-occurrence
# ----------------------------------
def extract_keywords_from_text(text: str, top_k: int = 8):
    """
    Returns:
        keywords: list[str]
        keyword_score: float
        co_occurrences: list[tuple]
        co_occurrence_score: float
    """

    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)

    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    if not tokens:
        return [], 0.0, [], 0.0

    tagged = nltk.pos_tag(tokens)

    allowed_tags = {"NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN"}
    keywords = [word for word, tag in tagged if tag in allowed_tags]

    if not keywords:
        return [], 0.0, [], 0.0

    freq = Counter(keywords)
    most_common = [w for w, _ in freq.most_common(top_k)]

    # Keyword density score
    keyword_score = len(keywords) / len(tokens)

    # ----------------------------------
    # Co-occurrence detection
    # ----------------------------------
    pairs = list(combinations(set(most_common), 2))
    pair_freq = Counter(pairs)

    top_pairs = [pair for pair, _ in pair_freq.most_common(5)]

    co_occurrence_score = len(top_pairs) / max(len(pairs), 1)

    return (
        most_common,
        round(keyword_score, 3),
        top_pairs,
        round(co_occurrence_score, 3),
    )


# ----------------------------------
# Process all segments
# ----------------------------------
def extract_keywords_for_segments(segments):
    """
    Input:
        segments = [
            { "start": , "end": , "text": }
        ]

    Output:
        segments enriched with keywords & co-occurrence info
    """

    enriched_segments = []

    for seg in segments:
        keywords, kw_score, pairs, pair_score = extract_keywords_from_text(seg["text"])

        enriched_segments.append({
            **seg,
            "keywords": keywords,
            "keyword_score": kw_score,
            "co_occurrences": pairs,
            "co_occurrence_score": pair_score
        })

    return enriched_segments
# ----------------------------------
# Extract GLOBAL keywords (Title + Description)
# ----------------------------------
def extract_global_keywords(title: str, description: str, top_k: int = 15):
    """
    Extracts important global keywords from
    video title + description for context-aware scoring.
    """

    combined_text = f"{title} {description}".strip()

    if not combined_text:
        return set()

    cleaned = clean_text(combined_text)
    tokens = word_tokenize(cleaned)

    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    if not tokens:
        return set()

    tagged = nltk.pos_tag(tokens)

    allowed_tags = {"NN", "NNS", "NNP", "NNPS"}
    keywords = [word for word, tag in tagged if tag in allowed_tags]

    freq = Counter(keywords)
    most_common = [w for w, _ in freq.most_common(top_k)]

    return set(most_common)
