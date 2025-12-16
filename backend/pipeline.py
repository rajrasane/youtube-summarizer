import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging

from paths import (
    TEMP_DOWNLOAD_DIR,
    TEMP_PROCESSED_DIR,
    T5_MODEL_DIR
)

from services.downloader import download_audio
from services.audio_extractor import convert_to_wav
from services.asr_whisper import transcribe_with_whisper
from services.segmenter import segment_by_pause
from services.keyword_extractor import (
    extract_keywords_for_segments,
    extract_global_keywords
)
from services.scoring import score_segments
from services.segment_selector import select_top_segments
from services.video_type_detector import detect_video_type
from services.segment_summarizer import summarize_segments
from services.summary_visualizer import generate_summary


# -------------------------------------------------
# Logger configuration
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pipeline")


def pipeline(youtube_url: str) -> str:
    logger.info("=== Starting YouTube Video Summarization Pipeline ===\n")

    # -------------------------------------------------
    # [1] DOWNLOAD AUDIO + METADATA
    # -------------------------------------------------
    logger.info("[1] Downloading audio from YouTube...")
    dl = download_audio(youtube_url, download_dir=TEMP_DOWNLOAD_DIR)

    m4a_path = dl["m4a_path"]
    title = dl.get("title", "") or ""
    description = dl.get("raw_info", {}).get("description", "") or ""

    logger.info(f"Downloaded audio: {m4a_path}")
    logger.info(f"Video title: {title}")

    # -------------------------------------------------
    # [1.1] GLOBAL CONTEXT KEYWORDS (TITLE + DESCRIPTION)
    # -------------------------------------------------
    global_keywords = extract_global_keywords(title, description)
    logger.info(f"Global context keywords: {list(global_keywords)[:8]}")

    # -------------------------------------------------
    # [2] CONVERT AUDIO TO WAV
    # -------------------------------------------------
    logger.info("\n[2] Converting audio to WAV...")
    wav_path = convert_to_wav(m4a_path, output_dir=TEMP_PROCESSED_DIR)
    logger.info(f"WAV created: {wav_path}")

    # -------------------------------------------------
    # [3] AUTOMATIC SPEECH RECOGNITION (ASR)
    # -------------------------------------------------
    logger.info("\n[3] Transcribing audio using Whisper...")
    asr_output = transcribe_with_whisper(wav_path)

    raw_segments = asr_output["segments"]
    transcript_path = asr_output["transcript_path"]

    logger.info(f"Transcript saved at: {transcript_path}")
    logger.info(f"Total ASR segments: {len(raw_segments)}")

    full_transcript = " ".join(seg["text"] for seg in raw_segments)

    # -------------------------------------------------
    # [3.1] VIDEO TYPE DETECTION
    # -------------------------------------------------
    video_type = detect_video_type(
        title=title,
        description=description,
        transcript_text=full_transcript
    )
    logger.info(f"Detected video type: {video_type}")

    # -------------------------------------------------
    # [3.5] SEGMENTATION BY AUDIO PAUSE
    # -------------------------------------------------
    logger.info("\n[3.5] Segmenting transcript using pause detection...")
    segments = segment_by_pause(raw_segments)
    logger.info(f"Logical segments formed: {len(segments)}")

    # -------------------------------------------------
    # [3.6] KEYWORD & CO-OCCURRENCE EXTRACTION
    # -------------------------------------------------
    logger.info("\n[3.6] Extracting keywords for each segment...")
    segments = extract_keywords_for_segments(segments)

    # -------------------------------------------------
    # [3.7] SEGMENT SCORING (CONTEXT-AWARE)
    # -------------------------------------------------
    logger.info("\n[3.7] Scoring segments using global context...")
    segments = score_segments(segments, global_keywords)

    # -------------------------------------------------
    # [3.8] SEGMENT SELECTION
    # -------------------------------------------------
    logger.info("\n[3.8] Selecting top important segments...")
    selected_segments = select_top_segments(segments, top_k=5)
    logger.info(f"Selected {len(selected_segments)} segments")

    # -------------------------------------------------
    # [4] ABSTRACTIVE SEGMENT SUMMARIZATION (T5)
    # -------------------------------------------------
    logger.info("\n[4] Summarizing selected segments using T5...")
    
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"T5 Summarizer running on: {device}")

    summarized_segments = summarize_segments(
        selected_segments,
        model_dir=T5_MODEL_DIR,
        device=device
    )

    # -------------------------------------------------
    # [5] FINAL SUMMARY VISUALIZATION
    # -------------------------------------------------
    logger.info("\n[5] Generating final summary...")
    final_summary = generate_summary(summarized_segments)

    logger.info("\n=== SUMMARY COMPLETE ===")
    return final_summary


if __name__ == "__main__":
    url = input("Enter YouTube URL: ").strip()
    summary = pipeline(url)

    print("\n===== FINAL SUMMARY =====\n")
    print(summary)
    print("\n=========================")
