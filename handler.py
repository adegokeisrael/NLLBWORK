"""
RunPod Serverless handler for NLLB-200 English<->Yoruba translation.

NLLB is trained mainly on single sentences, not long paragraphs — if you
feed it a big multi-sentence block, it will often translate the first
portion and then stop early instead of covering the whole text. To handle
long input reliably, this handler splits text into sentences, translates
each sentence separately, then joins the results back together.

Expected input (event["input"]):
{
    "text": "Hello, how are you? I hope you are well.",
    "source_lang": "eng_Latn",   # or "yor_Latn"
    "target_lang": "yor_Latn",   # or "eng_Latn"
    "max_length": 400,           # optional, max tokens PER SENTENCE
    "chunk": true                # optional, default true. Set false to
                                  # force single-shot translation (old
                                  # behavior) instead of sentence splitting.
}

Output:
{
    "translation": "...",
    "source_lang": "eng_Latn",
    "target_lang": "yor_Latn",
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

VALID_LANGS = {"eng_Latn", "yor_Latn"}

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


def translate_batch(sentences: list[str], source_lang: str, target_lang: str,
                     max_length: int = 400, batch_size: int = 8) -> list[str]:
    """Translate a list of sentences, batching several through the model
    at once for efficiency rather than one-by-one."""
    tokenizer.src_lang = source_lang
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)

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
    source_lang = job_input.get("source_lang", "eng_Latn")
    target_lang = job_input.get("target_lang", "yor_Latn")
    max_length = job_input.get("max_length", 400)
    chunk = job_input.get("chunk", True)

    if not text:
        return {"error": "Missing 'text' field in input."}

    if source_lang not in VALID_LANGS or target_lang not in VALID_LANGS:
        return {
            "error": f"source_lang/target_lang must be one of {sorted(VALID_LANGS)}"
        }

    try:
        if chunk:
            sentences = split_into_sentences(text)
            if not sentences:
                sentences = [text]
            translated_sentences = translate_batch(
                sentences, source_lang, target_lang, max_length
            )
            translation = " ".join(translated_sentences)
            num_chunks = len(sentences)
        else:
            translation = translate_batch(
                [text], source_lang, target_lang, max_length
            )[0]
            num_chunks = 1
    except Exception as e:
        return {"error": str(e)}

    return {
        "translation": translation,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "num_chunks": num_chunks,
    }


runpod.serverless.start({"handler": handler})
