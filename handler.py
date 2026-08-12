"""
RunPod Serverless handler for NLLB-200 English<->Yoruba translation.

Expected input (event["input"]):
{
    "text": "Hello, how are you?",
    "source_lang": "eng_Latn",   # or "yor_Latn"
    "target_lang": "yor_Latn",   # or "eng_Latn"
    "max_length": 400             # optional
}

Output:
{
    "translation": "...",
    "source_lang": "eng_Latn",
    "target_lang": "yor_Latn"
}
"""

import os
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


def translate(text: str, source_lang: str, target_lang: str, max_length: int = 400) -> str:
    tokenizer.src_lang = source_lang
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(DEVICE)

    # Newer transformers: convert_tokens_to_ids works for NLLB's lang codes.
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=max_length,
            num_beams=5,
        )

    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


def handler(event):
    job_input = event.get("input", {})

    text = job_input.get("text")
    source_lang = job_input.get("source_lang", "eng_Latn")
    target_lang = job_input.get("target_lang", "yor_Latn")
    max_length = job_input.get("max_length", 400)

    if not text:
        return {"error": "Missing 'text' field in input."}

    if source_lang not in VALID_LANGS or target_lang not in VALID_LANGS:
        return {
            "error": f"source_lang/target_lang must be one of {sorted(VALID_LANGS)}"
        }

    try:
        translation = translate(text, source_lang, target_lang, max_length)
    except Exception as e:
        return {"error": str(e)}

    return {
        "translation": translation,
        "source_lang": source_lang,
        "target_lang": target_lang,
    }


runpod.serverless.start({"handler": handler})
