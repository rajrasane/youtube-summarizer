from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load once (important for performance)
_tokenizer = None
_model = None


def load_t5(model_dir, device="cpu"):
    global _tokenizer, _model

    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True
        )
        _model = AutoModelForSeq2SeqLM.from_pretrained(
            model_dir,
            local_files_only=True
        ).to(device)

        _model.eval()  # inference mode
    else:
        # Ensure model is on the correct device
        if _model.device.type != device:
             _model.to(device)

    return _tokenizer, _model


def summarize_segments(
    segments,
    model_dir,
    device="cpu",
    max_input_tokens=256,
    max_output_tokens=60
):
    """
    Applies FLAN-T5 summarization ONLY on selected segments.
    Designed to be SAFE and FAST on CPU.
    """

    tokenizer, model = load_t5(model_dir, device)
    summarized = []

    for idx, seg in enumerate(segments, 1):
        text = seg.get("text", "").strip()
        if not text:
            continue

        # 🔒 HARD LIMIT INPUT SIZE (VERY IMPORTANT)
        words = text.split()
        if len(words) > 120:
            text = " ".join(words[:120])

        print(f"T5 summarizing segment {idx}/{len(segments)} "
              f"({len(text.split())} words)")

        # FLAN-style instruction prompt
        prompt = (
            "Summarize the following content clearly and concisely:\n\n"
            f"{text}"
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=max_output_tokens,
                min_length=20,
                num_beams=2,                # 🔥 faster
                no_repeat_ngram_size=3,
                early_stopping=True
            )

        summary_text = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        ).strip()

        summarized.append({
            **seg,
            "summary": summary_text
        })

    return summarized
