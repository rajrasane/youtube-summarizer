def extract_keywords_for_segments(segments):
    for seg in segments:
        text = seg["text"].lower()
        words = text.split()

        # simple keyword logic (baseline)
        keywords = [w for w in words if len(w) > 4]

        seg["keywords"] = list(set(keywords))

    return segments
