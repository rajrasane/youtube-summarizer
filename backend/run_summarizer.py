import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from services.summarizer_t5 import summarize_text


def main():

    # NEW: If the pipeline passes a file path, use it
    if len(sys.argv) > 1:
        text_file = sys.argv[1]
    else:
        print("=== Offline T5 Summarizer Test ===")
        text_file = input("Enter path to text file OR paste text directly: ").strip()

    # Load model directory
    model_dir = os.path.join(BASE_DIR, "models", "t5_summarizer")

    # Load text
    if os.path.exists(text_file):
        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = text_file

    # Summarize
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    summary = summarize_text(
        text,
        model_dir=model_dir,
        device=device,
        max_input_tokens=512,
        overlap=64
    )

    # IMPORTANT: Print only summary so pipeline can capture it
    print(summary)


if __name__ == "__main__":
    main()
