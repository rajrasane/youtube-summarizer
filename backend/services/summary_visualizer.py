from typing import List, Dict


def generate_summary(selected_segments: List[Dict]) -> str:
    """
    Converts selected segments into a clean, readable summary
    suitable for final visualization.

    Input:
        selected_segments = [
            {
                "start": float,
                "end": float,
                "text": str,
                "keywords": list,
                "score": float
            }
        ]

    Output:
        summary_text (str)
    """

    if not selected_segments:
        return "No meaningful content could be summarized."

    summary_blocks = []

    for idx, seg in enumerate(selected_segments, start=1):
        text = seg.get("text", "").strip()
        keywords = seg.get("keywords", [])

        if not text:
            continue

        # Clean text formatting
        text = text.replace("  ", " ").strip()

        block = f"{idx}. {text}"

        # Add keywords if available (optional but useful)
        if keywords:
            kw_line = ", ".join(keywords[:6])
            block += f"\n   Key topics: {kw_line}"

        summary_blocks.append(block)

    summary_text = "\n\n".join(summary_blocks)

    return summary_text
