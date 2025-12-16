"""
audio_extractor.py

Converts downloaded .m4a audio into normalized, 16 kHz mono WAV format.
Deletes the original .m4a file after conversion.
"""

import os
import sys

# Add backend to Python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from paths import TEMP_PROCESSED_DIR
from pydub import AudioSegment


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def convert_to_wav(m4a_path: str, output_dir: str = TEMP_PROCESSED_DIR) -> str:
    """
    Convert .m4a audio file to normalized 16 kHz mono .wav format.

    Args:
        m4a_path: absolute path to .m4a file
        output_dir: directory to store the processed .wav file

    Returns:
        wav_path: absolute path to processed WAV audio file
    """
    ensure_dir(output_dir)

    if not os.path.exists(m4a_path):
        raise FileNotFoundError(f"Input file not found: {m4a_path}")

    # Extract video ID
    base_name = os.path.basename(m4a_path)
    video_id = os.path.splitext(base_name)[0]

    wav_path = os.path.join(output_dir, f"{video_id}.wav")

    print(f"Converting to WAV...\nOutput: {wav_path}")

    # Load m4a
    audio = AudioSegment.from_file(m4a_path, format="m4a")

    # Normalize audio
    normalized = audio.normalize()

    # Convert to 16kHz mono
    normalized = normalized.set_frame_rate(16000).set_channels(1)

    # Export WAV
    normalized.export(wav_path, format="wav")

    # Delete original m4a
    try:
        os.remove(m4a_path)
    except Exception:
        pass

    return os.path.abspath(wav_path)


if __name__ == "__main__":
    test_file = input("Enter path to .m4a file: ").strip()
    if test_file:
        try:
            result = convert_to_wav(test_file)
            print("Converted WAV saved at:", result)
        except Exception as e:
            print("Error:", e)
