import os

# Base backend directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Temp folders
TEMP_DIR = os.path.join(BASE_DIR, "temp")
TEMP_DOWNLOAD_DIR = os.path.join(TEMP_DIR, "downloads", "audio")
TEMP_PROCESSED_DIR = os.path.join(TEMP_DIR, "processed_audio")
TEMP_ASR_DIR = os.path.join(TEMP_DIR, "asr_chunks")

# Models folder
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Model paths
T5_MODEL_DIR = os.path.join(MODELS_DIR, "t5_summarizer")
WHISPER_MODEL_DIR = os.path.join(MODELS_DIR, "whisper_tiny")

# Ensure folders exist
for folder in [
    TEMP_DIR,
    TEMP_DOWNLOAD_DIR,
    TEMP_PROCESSED_DIR,
    TEMP_ASR_DIR
]:
    os.makedirs(folder, exist_ok=True)
