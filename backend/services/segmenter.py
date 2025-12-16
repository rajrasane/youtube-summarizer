def segment_by_pause(whisper_segments, pause_threshold=1.5):
    """
    Groups Whisper segments into logical segments
    based on pause duration.
    """

    final_segments = []

    current_segment = {
        "start": whisper_segments[0]["start"],
        "end": whisper_segments[0]["end"],
        "text": whisper_segments[0]["text"]
    }

    segment_id = 1

    for prev, curr in zip(whisper_segments, whisper_segments[1:]):
        gap = curr["start"] - prev["end"]

        if gap > pause_threshold:
            # close current segment
            final_segments.append({
                "segment_id": segment_id,
                **current_segment
            })
            segment_id += 1

            # start new segment
            current_segment = {
                "start": curr["start"],
                "end": curr["end"],
                "text": curr["text"]
            }
        else:
            # continue same segment
            current_segment["end"] = curr["end"]
            current_segment["text"] += " " + curr["text"]

    # add last segment
    final_segments.append({
        "segment_id": segment_id,
        **current_segment
    })

    return final_segments
