def select_top_segments(segments, top_k=5, min_words=20):
    """
    Selects the most important segments based on score
    while preserving original video order.

    Args:
        segments: list of scored segments
        top_k: maximum segments to include in summary
        min_words: ignore very small segments

    Returns:
        selected_segments
    """

    # 1. Filter out very small or empty segments
    valid_segments = [
        s for s in segments
        if len(s.get("text", "").split()) >= min_words
    ]

    if not valid_segments:
        return []

    # 2. Rank by score (descending)
    ranked = sorted(
        valid_segments,
        key=lambda x: x.get("score", 0),
        reverse=True
    )

    # 3. Select top-K important segments
    selected = ranked[:top_k]

    # 4. Restore original chronological order
    selected = sorted(selected, key=lambda x: x["start"])

    return selected
