"""
summarizer_t5.py

Summarization module using an offline T5/FLAN-T5 model stored locally in:
    /path/to/t5_summarizer   (the folder you already have)

Features:
- Loads tokenizer and seq2seq model from local folder (no internet).
- Splits long input text into token-limited chunks using tokenizer.
- Summarizes chunks, then optionally does a final summarization pass over chunk-summaries.
- Safe for CPU execution; uses batching for generation.
"""

import os
import math
import logging
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)


def load_model_and_tokenizer(model_dir: str,
                             device: str = "cpu"):
    """
    Load tokenizer and T5-like seq2seq model from a local directory.

    Args:
        model_dir: path to your offline t5_summarizer folder
        device: "cpu" or "cuda"

    Returns:
        tokenizer, model
    """
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    logger.info("Loading tokenizer from %s", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    logger.info("Loading model from %s (device=%s)", model_dir, device)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True)
    model.to(device)
    model.eval()
    return tokenizer, model


def chunk_text_by_tokens(text: str,
                         tokenizer,
                         max_tokens: int = 512,
                         stride: int = 64) -> List[str]:
    """
    Split text into chunks roughly <= max_tokens using tokenizer.
    Uses a sliding window with optional stride overlap for context.

    Args:
        text: input string
        tokenizer: HuggingFace tokenizer
        max_tokens: maximum input token length for the model
        stride: overlap tokens between chunks (helps coherence)

    Returns:
        list of text chunks
    """
    # encode (no truncation) to get token ids
    enc = tokenizer.encode(text, add_special_tokens=False)
    total = len(enc)
    if total == 0:
        return []

    chunks = []
    start = 0
    while start < total:
        end = min(start + max_tokens, total)
        chunk_ids = enc[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        chunks.append(chunk_text.strip())
        if end == total:
            break
        # move window with overlap
        start = end - stride if (end - stride) > start else end
    return chunks


def summarize_chunks(chunks: List[str],
                     tokenizer,
                     model,
                     device: str = "cpu",
                     gen_kwargs: Optional[Dict[str, Any]] = None,
                     batch_size: int = 1) -> List[str]:
    """
    Summarize each chunk and return list of chunk summaries.

    Args:
        chunks: list of text chunks
        tokenizer, model: loaded objects
        device: 'cpu' or 'cuda'
        gen_kwargs: generation parameters for model.generate()
        batch_size: number of chunks to summarize per pass

    Returns:
        list of chunk summary strings
    """
    if gen_kwargs is None:
        gen_kwargs = {
            "num_beams": 4,
            "max_length": 200,
            "min_length": 50,
            "length_penalty": 0.9,
            "early_stopping": True,
            "no_repeat_ngram_size": 3,
        }

    summaries = []
    model_device = torch.device(device)

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model_device)

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        # decode each summary
        for out in outputs:
            s = tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            summaries.append(s.strip())

    return summaries


def finalize_summary(intermediate_summaries: List[str],
                     tokenizer,
                     model,
                     device: str = "cpu",
                     final_gen_kwargs: Optional[Dict[str, Any]] = None) -> str:
    """
    Take multiple chunk summaries and produce a final consolidated summary.
    If there is only one intermediate summary, return it directly.

    Args:
        intermediate_summaries: list of chunk summaries
        tokenizer, model: loaded objects
        device: device string
        final_gen_kwargs: generation params for final pass
    """
    if not intermediate_summaries:
        return ""

    if len(intermediate_summaries) == 1:
        return intermediate_summaries[0]

    concat = "\n\n".join(intermediate_summaries)
    # chunk if too long for model
    chunks = chunk_text_by_tokens(concat, tokenizer, max_tokens=1024, stride=128)
    # summarize intermediate chunks into brief bullets then merge
    smaller_summaries = summarize_chunks(chunks, tokenizer, model, device=device, gen_kwargs=final_gen_kwargs, batch_size=1)

    # If we got multiple small summaries, run one final pass combining them
    if len(smaller_summaries) == 1:
        return smaller_summaries[0]
    final_input = "\n\n".join(smaller_summaries)
    # final pass
    final_kwargs = final_gen_kwargs or {
        "num_beams": 4,
        "max_length": 220,
        "min_length": 60,
        "length_penalty": 0.9,
        "early_stopping": True,
        "no_repeat_ngram_size": 3,
    }
    inputs = tokenizer(final_input, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, **final_kwargs)
    final_summary = tokenizer.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return final_summary.strip()


def summarize_text(text: str,
                   model_dir: str,
                   device: str = "cpu",
                   max_input_tokens: int = 512,
                   overlap: int = 64,
                   gen_kwargs: Optional[Dict[str, Any]] = None,
                   final_gen_kwargs: Optional[Dict[str, Any]] = None,
                   batch_size: int = 1) -> str:
    """
    High-level function: load model (if needed), chunk input text, summarize chunks,
    then consolidate into a final summary.

    Args:
        text: full input text to summarize
        model_dir: path to local t5 model directory
        device: 'cpu' or 'cuda'
        max_input_tokens: token limit for each chunk (model-specific)
        overlap: token overlap between chunks
        gen_kwargs: generation args for per-chunk summaries
        final_gen_kwargs: generation args for final pass
        batch_size: summarization batch size (1 on CPU recommended)

    Returns:
        final_summary string
    """
    tokenizer, model = load_model_and_tokenizer(model_dir, device=device)

    # 1) Break input into chunks
    logger.info("Chunking input text into token-size pieces (max_tokens=%s, overlap=%s)", max_input_tokens, overlap)
    chunks = chunk_text_by_tokens(text, tokenizer, max_tokens=max_input_tokens, stride=overlap)
    logger.info("Created %d chunks", len(chunks))

    if not chunks:
        return ""

    # 2) Summarize chunks (batching supported)
    logger.info("Summarizing each chunk (batch_size=%s)", batch_size)
    intermediate = summarize_chunks(chunks, tokenizer, model, device=device, gen_kwargs=gen_kwargs, batch_size=batch_size)

    # 3) Finalize
    logger.info("Finalizing summary from %d intermediate summaries", len(intermediate))
    final = finalize_summary(intermediate, tokenizer, model, device=device, final_gen_kwargs=final_gen_kwargs)
    return final


# Simple CLI for quick tests
if __name__ == "__main__":
    import sys
    model_path = input("Enter local t5 model folder path (e.g. /full/path/to/t5_summarizer): ").strip()
    if not model_path:
        print("Model path required.")
        sys.exit(1)
    text_file = input("Enter path to .txt file to summarize (or paste text): ").strip()
    # If path exists, read from it; else treat as direct text
    if os.path.exists(text_file):
        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = text_file

    summary = summarize_text(text, model_path, device="cpu", max_input_tokens=512, overlap=64, batch_size=1)
    print("\nFINAL SUMMARY:\n")
    print(summary)
