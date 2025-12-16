"""
asr_whisper.py
Offline transcription using faster-whisper with local model folder.
"""

import os
import math
import uuid
from pydub import AudioSegment
from faster_whisper import WhisperModel
from paths import TEMP_ASR_DIR

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def split_into_chunks(wav_path: str, chunk_length_s: int, tmp_dir: str):
    """Split long WAV audio into smaller files for stable transcription."""
    ensure_dir(tmp_dir)
    audio = AudioSegment.from_file(wav_path, format="wav")

    chunk_paths = []
    total_ms = len(audio)
    chunk_ms = chunk_length_s * 1000
    num_chunks = math.ceil(total_ms / chunk_ms)

    base = os.path.splitext(os.path.basename(wav_path))[0]

    for i in range(num_chunks):
        start = i * chunk_ms
        end = min((i + 1) * chunk_ms, total_ms)
        chunk = audio[start:end]

        chunk_name = f"{base}_chunk_{i:03d}.wav"
        out_path = os.path.join(tmp_dir, chunk_name)
        chunk.export(out_path, format="wav")
        chunk_paths.append(out_path)

    return chunk_paths


def transcribe_with_whisper(
    wav_path: str,
    model_dir: str = None,
    chunk_length_s: int = 300,
):
    """Fully offline Whisper transcription using faster-whisper."""
    if not os.path.exists(wav_path):
        raise FileNotFoundError("WAV not found")

    # Import paths
    from paths import TEMP_ASR_DIR, WHISPER_MODEL_DIR

    # FIX: Use correct model directory if not passed
    if model_dir is None:
        model_dir = WHISPER_MODEL_DIR

    base = os.path.splitext(os.path.basename(wav_path))[0]
    tmp_dir = os.path.join(TEMP_ASR_DIR, f"{base}_{uuid.uuid4().hex[:6]}")
    ensure_dir(tmp_dir)

    # 1. Split WAV into chunks
    chunks = split_into_chunks(wav_path, chunk_length_s, tmp_dir)

    # 2. Load offline Whisper model
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Whisper running on: {device}")

    model = WhisperModel(
        model_dir,
        device=device,
        compute_type="int8",
    )


    all_segments = []
    transcript_text = []

    for idx, chunk_file in enumerate(chunks):
        offset = idx * chunk_length_s

        segments, info = model.transcribe(
            chunk_file,
            language="en",  # force English only
            beam_size=5,
        )

        for seg in segments:
            start = float(seg.start) + offset
            end = float(seg.end) + offset
            text = seg.text.strip()

            all_segments.append({
                "start": start,
                "end": end,
                "text": text,
            })
            transcript_text.append(text)

    # Save transcript correctly
    transcript_path = os.path.join(tmp_dir, f"{base}_transcript.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(" ".join(transcript_text).strip())

    print("\nTranscript saved at:", transcript_path)

    return {
        "video_id": base,
        "transcript": " ".join(transcript_text).strip(),
        "segments": all_segments,
        "transcript_path": transcript_path
    }




if __name__ == "__main__":
    wav = input("Enter path to WAV file: ").strip()
    if wav:
        out = transcribe_with_whisper(wav)
        print("Transcript length:", len(out["transcript"]))
        print("First 3 segments:")
        for s in out["segments"][:3]:
            print(f"[{s['start']:.2f}-{s['end']:.2f}] {s['text']}")
