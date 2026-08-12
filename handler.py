"""
RunPod Serverless handler for NLLB-200 multilingual translation.

Supports 10 languages: English, French, Arabic, German, Spanish, Hausa,
Yoruba, Igbo, Swahili, Portuguese — in any direction (any of the 10 as
source, any of the 10 as target).

NLLB is trained mainly on single sentences, not long paragraphs — if you
feed it a big multi-sentence block, it will often translate the first
portion and then stop early instead of covering the whole text. To handle
long input reliably, this handler splits text into sentences, translates
each sentence separately, then joins the results back together.

Expected input (event["input"]):
{
    "text": "Hello, how are you? I hope you are well.",
    "source_lang": "en",         # simple 2-letter code, see LANG_CODES below
    "target_lang": "yo",         # simple 2-letter code, see LANG_CODES below
    "max_length": 400,           # optional, max tokens PER SENTENCE
    "chunk": true                # optional, default true. Set false to
                                  # force single-shot translation instead
                                  # of sentence splitting.
}

Full NLLB FLORES-200 codes (e.g. "eng_Latn") are also accepted directly in
source_lang/target_lang if you prefer, for compatibility with earlier
requests.

Output:
{
    "translation": "...",
    "source_lang": "en",
    "target_lang": "yo",
    "num_chunks": 12   # how many sentence chunks were translated
}
"""

import os
import re
import runpod
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = os.environ.get("NLLB_MODEL", "facebook/nllb-200-distilled-1.3B")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {MODEL_NAME} on {DEVICE} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
).to(DEVICE)
model.eval()
print("Model loaded.")

# Simple 2-letter codes (used in the prompt/request) -> NLLB FLORES-200 codes
# (used internally by the model/tokenizer). Add more pairs here to support
# additional languages later — NLLB supports 200 total.
LANG_CODES = {
    "en": "eng_Latn",   # English
    "fr": "fra_Latn",   # French
    "ar": "arb_Arab",   # Arabic (Modern Standard)
    "de": "deu_Latn",   # German
    "es": "spa_Latn",   # Spanish
    "ha": "hau_Latn",   # Hausa
    "yo": "yor_Latn",   # Yoruba
    "ig": "ibo_Latn",   # Igbo
    "sw": "swh_Latn",   # Swahili
    "pt": "por_Latn",   # Portuguese
}
# Reverse lookup, so full NLLB codes in a request also resolve back to the
# short code for the response.
NLLB_TO_SHORT = {v: k for k, v in LANG_CODES.items()}


def resolve_lang(code: str) -> str | None:
    """Accepts either a short code ('en') or a full NLLB code
    ('eng_Latn') and returns the NLLB code, or None if unrecognized."""
    if code in LANG_CODES:
        return LANG_CODES[code]
    if code in NLLB_TO_SHORT:
        return code
    return None


# Simple sentence splitter: splits on ., !, ? followed by whitespace and a
# capital letter or end of string. Not perfect (e.g. abbreviations like
# "Dr." can trip it up) but works well for general prose.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-Ýa-zà-ÿ0-9])")


def split_into_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def translate_batch(sentences: list[str], source_lang_nllb: str, target_lang_nllb: str,
                     max_length: int = 400, batch_size: int = 8) -> list[str]:
    """Translate a list of sentences, batching several through the model
    at once for efficiency rather than one-by-one."""
    tokenizer.src_lang = source_lang_nllb
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang_nllb)

    results = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True, padding=True
        ).to(DEVICE)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=max_length,
                num_beams=5,
            )

        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        results.extend(decoded)

    return results


def handler(event):
    job_input = event.get("input", {})

    text = job_input.get("text")
    source_lang_raw = job_input.get("source_lang", "en")
    target_lang_raw = job_input.get("target_lang", "yo")
    max_length = job_input.get("max_length", 400)
    chunk = job_input.get("chunk", True)

    if not text:
        return {"error": "Missing 'text' field in input."}

    source_lang_nllb = resolve_lang(source_lang_raw)
    target_lang_nllb = resolve_lang(target_lang_raw)

    if source_lang_nllb is None or target_lang_nllb is None:
        return {
            "error": (
                "source_lang/target_lang must be one of "
                f"{sorted(LANG_CODES.keys())} (or their full NLLB codes: "
                f"{sorted(LANG_CODES.values())})"
            )
        }

    try:
        if chunk:
            sentences = split_into_sentences(text)
            if not sentences:
                sentences = [text]
            translated_sentences = translate_batch(
                sentences, source_lang_nllb, target_lang_nllb, max_length
            )
            translation = " ".join(translated_sentences)
            num_chunks = len(sentences)
        else:
            translation = translate_batch(
                [text], source_lang_nllb, target_lang_nllb, max_length
            )[0]
            num_chunks = 1
    except Exception as e:
        return {"error": str(e)}

    return {
        "translation": translation,
        "source_lang": NLLB_TO_SHORT.get(source_lang_nllb, source_lang_raw),
        "target_lang": NLLB_TO_SHORT.get(target_lang_nllb, target_lang_raw),
        "num_chunks": num_chunks,
    }


runpod.serverless.start({"handler": handler})
