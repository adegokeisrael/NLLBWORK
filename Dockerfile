FROM runpod/base:0.6.2-cuda12.1.0

WORKDIR /app

COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# Pre-download the model into the image so cold starts don't hit the network.
# Override NLLB_MODEL at build time if you want a different checkpoint size.
ARG NLLB_MODEL=facebook/nllb-200-distilled-1.3B
ENV NLLB_MODEL=${NLLB_MODEL}
RUN python3.11 -c "\
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer; \
import os; \
m = os.environ['NLLB_MODEL']; \
AutoTokenizer.from_pretrained(m); \
AutoModelForSeq2SeqLM.from_pretrained(m)"

COPY handler.py .

CMD ["python3.11", "-u", "handler.py"]
