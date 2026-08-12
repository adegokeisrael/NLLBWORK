# NLLB English↔Yoruba Translation on RunPod Serverless

## Files
- `handler.py` — RunPod serverless handler that loads NLLB and translates
- `requirements.txt` — Python deps
- `Dockerfile` — builds the container, bakes the model in at build time
- `test_input.json` — sample payload for local testing

## 1. Test locally (optional but recommended)

```bash
pip install -r requirements.txt
python handler.py --rp_serve_api    # spins up a local test server
# or run one-shot:
python handler.py --test_input test_input.json
```

If you have a local GPU this validates the handler before you pay for a build.
If not, skip straight to building — RunPod will build on their infra.

## 2. Build and push the Docker image

### Option A — Build locally with Docker

```bash
docker login
docker build -t <your-dockerhub-username>/nllb-yoruba:latest .
docker push <your-dockerhub-username>/nllb-yoruba:latest
```

To use a different model size, pass a build arg:
```bash
docker build --build-arg NLLB_MODEL=facebook/nllb-200-distilled-600M \
  -t <your-dockerhub-username>/nllb-yoruba:latest .
```

### Option B — Let GitHub Actions build and push it for you (no local Docker needed)

This repo includes `.github/workflows/build-and-push.yml`, which builds the
image and pushes it to GitHub Container Registry (GHCR) automatically.

1. Create a new GitHub repo and push these files to it (including the
   `.github/workflows/` folder):
   ```bash
   git init
   git add .
   git commit -m "NLLB RunPod handler"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```
2. Go to your repo → **Settings → Actions → General → Workflow permissions**
   → set to **"Read and write permissions"** (needed so the workflow can
   push to GHCR using the built-in `GITHUB_TOKEN` — no extra secrets to add).
3. The workflow runs automatically on push, or trigger it manually from the
   **Actions** tab → "Build and Push NLLB RunPod Image" → **Run workflow**.
4. Once it finishes (~10-20 min, mostly downloading the model), your image
   is live at:
   ```
   ghcr.io/<your-username>/nllb-yoruba:latest
   ```
5. By default GHCR packages are **private**. Either make the package public
   (repo → Packages → your package → Package settings → Change visibility),
   or give RunPod a GitHub Personal Access Token with `read:packages` scope
   when you set up the endpoint so it can pull a private image.

Note: baking the model into the image makes the image large (3-8GB+) but
gives you fast cold starts, since RunPod doesn't need to download weights
from Hugging Face on every worker spin-up. This is the recommended pattern
for serverless.

## 3. Create the Serverless endpoint on RunPod

1. Go to runpod.io → Serverless → New Endpoint
2. Choose "Custom Source" → Docker Image → paste your image path
   - Docker Hub: `docker.io/<your-username>/nllb-yoruba:latest`
   - GHCR: `ghcr.io/<your-username>/nllb-yoruba:latest`
3. GPU selection:
   - distilled-600M → 16GB GPU (RTX 4000/A4000) is plenty
   - distilled-1.3B → 16-24GB GPU (A4000/A5000)
   - 3.3B → 24GB+ GPU (A5000/A6000)
4. Set Container Disk to at least 15-20GB (model weights + CUDA libs)
5. Set min workers to 0 if you want scale-to-zero (cheapest, but cold starts
   ~10-30s), or 1 if you want an always-warm worker for low latency
6. Deploy

## 4. Call the endpoint

```bash
curl -X POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "text": "Good morning, how are you today?",
      "source_lang": "eng_Latn",
      "target_lang": "yor_Latn"
    }
  }'
```

Response:
```json
{
  "output": {
    "translation": "E kaaro, se alafia ni?",
    "source_lang": "eng_Latn",
    "target_lang": "yor_Latn"
  }
}
```

Use `/run` instead of `/runsync` for async jobs on longer batches, then poll
`/status/<job_id>`.

## Notes / gotchas

- **Long text handling**: NLLB is trained mostly on single sentences, not
  paragraphs. If you feed it a long multi-sentence block directly, it will
  often translate only the first portion and stop early — this is a model
  limitation, not a bug. `handler.py` now splits input text into sentences
  and translates each one, then joins the results, so long documents are
  translated fully and reliably. This is the default (`"chunk": true`).
  Set `"chunk": false` in the input if you want the old single-shot
  behavior for short single-sentence inputs (marginally faster, no
  splitting overhead).
- **Language codes**: NLLB uses FLORES-200 codes, not ISO 639-1. English is
  `eng_Latn`, Yoruba is `yor_Latn` — not `en`/`yo`.
- **Cost control**: Serverless bills per-second of GPU time while a request
  runs. Set min workers to 0 for a low-traffic app; set idle timeout
  (e.g. 5s) so workers spin down fast between requests.
- **Batching**: if you expect high volume, extend the handler to accept a
  list of strings in `text` and batch them through `model.generate()` in
  one call — much more GPU-efficient than one request per sentence.
- **Quality tuning**: `num_beams=5` in the handler is a reasonable default.
  Increase for slightly better quality at the cost of latency, or drop to
  1-2 for faster/cheaper greedy-ish decoding.
- **Alternative — RunPod Pods (not Serverless)**: if you want a persistent
  server instead of pay-per-request (e.g. for a always-on API), deploy the
  same image as a Pod and run a small FastAPI wrapper around the same
  `translate()` function instead of the RunPod handler loop.
