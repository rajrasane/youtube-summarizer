import re
from collections import defaultdict


VIDEO_TYPE_KEYWORDS = {
    "song": [
        "song", "lyrics", "music", "official video",
        "singer", "composer", "album", "audio"
    ],
    "interview": [
        "interview", "in conversation", "talks with",
        "exclusive", "host", "guest"
    ],
    "podcast": [
        "podcast", "episode", "ep", "discussion",
        "talk show", "long conversation"
    ],
    "news": [
        "news", "breaking", "report", "debate",
        "analysis", "minister", "government"
    ],
    "lecture": [
        "lecture", "class", "lesson", "explained",
        "course", "today we will"
    ],
    "tutorial": [
        "how to", "step by step", "tutorial",
        "guide", "demo"
    ],
    "vlog": [
        "vlog", "my day", "travel", "journey"
    ],
    "documentary": [
        "documentary", "story of", "history of",
        "behind the scenes"
    ]
}


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def detect_video_type(title="", description="", transcript_text=""):
    """
    Detects video type using weighted signals:
    - Title (high weight)
    - Description (medium weight)
    - Transcript (low weight)

    Returns:
        video_type (str)
    """

    title = normalize(title)
    description = normalize(description)
    transcript = normalize(transcript_text[:3000])  # limit noise

    scores = defaultdict(float)

    for vtype, keywords in VIDEO_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in title:
                scores[vtype] += 3.0   # strong signal
            if kw in description:
                scores[vtype] += 2.0   # medium signal
            if kw in transcript:
                scores[vtype] += 0.5   # weak signal

    if not scores:
        return "general"

    detected = max(scores, key=scores.get)

    # Fallback safety
    if scores[detected] < 1.5:
        return "general"

    return detected
