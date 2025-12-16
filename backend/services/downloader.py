"""
downloader.py

Downloads audio-only from a YouTube URL using yt-dlp and saves the audio file as:
    backend/temp/downloads/audio/<VIDEO_ID>.m4a

Functions:
- ensure_dir(path): create directories if missing
- download_audio(youtube_url, download_dir, prefer_format='m4a', timeout=120)
    -> returns dict { 'video_id', 'title', 'm4a_path', 'duration', 'raw_info' }
"""

import os
import sys
# Add backend directory to Python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import glob
import logging
from typing import Optional, Dict, Any
from yt_dlp import YoutubeDL
from paths import TEMP_DOWNLOAD_DIR

# Configure simple logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(ch)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _find_downloaded_file(download_dir: str, video_id: str, ext: str = "m4a") -> Optional[str]:
    """Look for file <video_id>.* (common extensions) and return full path if found."""
    patterns = [
        os.path.join(download_dir, f"{video_id}.{ext}"),
        os.path.join(download_dir, f"{video_id}.*"),
        os.path.join(download_dir, f"{video_id}_*.*"),
    ]
    for pat in patterns:
        matches = glob.glob(pat)
        if matches:
            # prefer exact ext match first
            for m in matches:
                if m.lower().endswith(f".{ext}"):
                    return os.path.abspath(m)
            # otherwise return first match
            return os.path.abspath(matches[0])
    return None


def download_audio(youtube_url: str,
                   download_dir: str = TEMP_DOWNLOAD_DIR,
                   prefer_format: str = "m4a",
                   timeout: int = 300) -> Dict[str, Any]:
    """
    Download audio-only from YouTube and save to download_dir using video ID as filename.

    Args:
        youtube_url: the YouTube video URL or id
        download_dir: directory where audio will be stored
        prefer_format: 'm4a' or 'wav' (we will download as m4a and convert later if needed)
        timeout: seconds before yt-dlp times out

    Returns:
        dict with keys:
          - video_id: str
          - title: str
          - duration: float (seconds) or None
          - m4a_path: str (absolute path to downloaded file)
          - raw_info: dict (yt-dlp info dict for further metadata)
    Raises:
        RuntimeError on download failure
    """

    ensure_dir(download_dir)

    # Output filename template: use video id (clean and stable)
    outtmpl = os.path.join(download_dir, "%(id)s.%(ext)s")

    # yt-dlp options to extract audio and convert to m4a
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "noplaylist": True,
        # Postprocessor to extract audio with ffmpeg (requires ffmpeg installed)
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": prefer_format,
            "preferredquality": "192",
        }],
        "quiet": True,
        "no_warnings": True,
        "skip_download": False,
        # network/timeouts
        "socket_timeout": timeout,
    }

    info = None
    try:
        with YoutubeDL(ydl_opts) as ydl:
            logger.info("Starting download for URL: %s", youtube_url)
            info = ydl.extract_info(youtube_url, download=True)
    except Exception as e:
        logger.exception("yt-dlp download failed: %s", e)
        raise RuntimeError(f"Failed to download audio: {e}") from e

    # Info dict may differ depending on yt-dlp; handle missing keys safely
    video_id = info.get("id") if info else None
    title = info.get("title") if info else None
    duration = info.get("duration") if info else None
    description = info.get("description") if info else ""
    uploader = info.get("uploader") if info else ""
    channel = info.get("channel") if info else uploader


    if not video_id:
        # try to infer from webpage_url or id-like fields
        video_id = info.get("webpage_url_basename") if info and info.get("webpage_url_basename") else None

    if not video_id:
        # as last resort, try to parse video id from url (simple heuristic)
        # This is best-effort — prefer info['id'] from yt-dlp
        try:
            # handle youtu.be short links and watch?v= links
            if "youtu.be/" in youtube_url:
                video_id = youtube_url.split("youtu.be/")[-1].split("?")[0]
            elif "v=" in youtube_url:
                video_id = youtube_url.split("v=")[-1].split("&")[0]
        except Exception:
            video_id = None

    if not video_id:
        raise RuntimeError("Could not determine video id after download.")

    # Determine file path
    downloaded_path = _find_downloaded_file(download_dir, video_id, ext=prefer_format)
    if not downloaded_path:
        # If still not found, raise error and include info dict for debugging
        logger.error("Downloaded file not found for video id: %s in %s", video_id, download_dir)
        raise RuntimeError(f"Downloaded audio not found for id={video_id}. yt-dlp info: {info}")

    logger.info("Downloaded audio saved to: %s", downloaded_path)

    return {
        "video_id": video_id,
        "title": title,
        "description": description,
        "channel": channel,
        "duration": duration,
        "m4a_path": downloaded_path,
        "raw_info": info,
    }



if __name__ == "__main__":
    # Simple manual test
    test_url = input("Enter YouTube URL for a quick test (or leave blank to skip): ").strip()
    if test_url:
        try:
            result = download_audio(test_url)
            print("Download completed:")
            print("Video ID:", result["video_id"])
            print("Title:", result["title"])
            print("Duration (s):", result["duration"])
            print("Audio path:", result["m4a_path"])
        except Exception as ex:
            print("Download failed:", ex)
