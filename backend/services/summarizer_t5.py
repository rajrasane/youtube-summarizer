import os
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------
# Load T5 model + tokenizer
# -----------------------------
def load_model_and_tokenizer(model_dir, device="cpu"):
    logger.info(f"Loading tokenizer and model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True).to(device)
    return tokenizer, model


# -----------------------------
# Summarize short text directly
# -----------------------------
def summarize_short(text, tokenizer, model, device):
    input_text = "summarize: " + text.strip()

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    output_ids = model.generate(
        **inputs,
        max_length=80,
        min_length=15,
        num_beams=4,
        length_penalty=1.0,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


# -----------------------------
# Chunk long text
# -----------------------------
def chunk_text(text, tokenizer, max_tokens=512, overlap=64):
    encoding = tokenizer(
        text,
        return_overflowing_tokens=True,
        max_length=max_tokens,
        truncation=True,
        stride=overlap
    )
    return encoding["input_ids"]


# -----------------------------
# Summarize large transcript chunks
# -----------------------------
def summarize_chunk(chunk_ids, tokenizer, model, device):

    input_ids = torch.tensor([chunk_ids]).to(device)

    output_ids = model.generate(
        input_ids=input_ids,
        max_length=120,
        min_length=20,
        num_beams=4,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


# -----------------------------
# Master summarization
# -----------------------------
def summarize_text(
    text,
    model_dir,
    device="cpu",
    max_input_tokens=512,
    overlap=64
):
    tokenizer, model = load_model_and_tokenizer(model_dir, device)

    text = text.strip()
    input_text = "summarize: " + text

    # Short text path
    if len(text.split()) < 40:
        return summarize_short(text, tokenizer, model, device)

    # Long text path
    chunks = chunk_text(input_text, tokenizer, max_input_tokens, overlap)

    summaries = []
    for chunk_ids in chunks:
        part = summarize_chunk(chunk_ids, tokenizer, model, device)
        summaries.append(part)

    final_summary = " ".join(summaries).strip()
    return final_summary
