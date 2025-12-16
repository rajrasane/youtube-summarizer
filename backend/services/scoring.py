def score_segments(
    segments,
    global_keywords=None,
    video_type="general"
):
    """
    Scores each segment using multiple signals and adapts scoring
    based on video type.

    Signals:
    - keyword_score           (local importance)
    - co_occurrence_score     (topic richness)
    - context_match_score     (title + description relevance)
    - length_score            (readability)

    Adds:
        seg["score"]
        seg["context_score"]
    """

    if global_keywords is None:
        global_keywords = set()

    for seg in segments:
        # --------------------------------
        # 1. Keyword importance
        # --------------------------------
        keyword_score = seg.get("keyword_score", 0)

        # --------------------------------
        # 2. Co-occurrence importance
        # --------------------------------
        co_occurrence_score = seg.get("co_occurrence_score", 0)

        # --------------------------------
        # 3. Context relevance (Title + Description)
        # --------------------------------
        segment_keywords = set(seg.get("keywords", []))
        if global_keywords:
            overlap = segment_keywords.intersection(global_keywords)
            context_score = len(overlap) / len(global_keywords)
        else:
            context_score = 0.0

        # --------------------------------
        # 4. Length normalization
        # --------------------------------
        text = seg.get("text", "")
        word_count = len(text.split())

        if word_count < 20:
            length_score = 0.3
        elif word_count > 200:
            length_score = 0.6
        else:
            length_score = 1.0

        # --------------------------------
        # 5. VIDEO-TYPE AWARE WEIGHTS
        # --------------------------------
        if video_type == "interview":
            score = (
                0.30 * keyword_score +
                0.30 * co_occurrence_score +
                0.25 * context_score +
                0.15 * length_score
            )

        elif video_type == "podcast":
            score = (
                0.25 * keyword_score +
                0.35 * co_occurrence_score +
                0.25 * context_score +
                0.15 * length_score
            )

        elif video_type == "news":
            score = (
                0.45 * keyword_score +
                0.25 * co_occurrence_score +
                0.20 * context_score +
                0.10 * length_score
            )

        elif video_type == "song":
            score = (
                0.15 * keyword_score +
                0.15 * co_occurrence_score +
                0.20 * context_score +
                0.50 * length_score
            )

        elif video_type == "lecture":
            score = (
                0.40 * keyword_score +
                0.25 * co_occurrence_score +
                0.20 * context_score +
                0.15 * length_score
            )

        else:  # general
            score = (
                0.35 * keyword_score +
                0.25 * co_occurrence_score +
                0.25 * context_score +
                0.15 * length_score
            )

        # --------------------------------
        # Store results
        # --------------------------------
        seg["score"] = round(score, 3)
        seg["context_score"] = round(context_score, 3)

    return segments
