# PLANS.md — Hosting / deployment options

Planning notes for turning this CLI into a hosted web service: upload a
recording, get back a transcript (optionally speaker-labeled). Preference is
for Vercel-like DX — git-push / single-command deploy, managed, minimal ops.

## Why not Vercel (alone)

Whisper + pyannote is the opposite of what serverless functions are built for:

- **No GPU** on Vercel functions.
- **Execution time limits** (tens of seconds to a few minutes) — a long meeting
  exceeds them, especially on CPU.
- **Bundle size limits** can't hold the ~3 GB `large-v3` weights (plus
  torch/torchcodec/ffmpeg), and the ephemeral filesystem means a cold start
  re-downloads everything.

Vercel is still great for a **thin frontend / API**; the transcription itself
has to run on a GPU (or at least a fat, long-running container).

## Options surveyed

| Service | GPU | DX | Fit |
| --- | --- | --- | --- |
| **Modal** | ✅ serverless, scale-to-zero | Python decorators, `modal deploy` | **Best overall** for this workload |
| **Replicate** | ✅ per-second | Push a model with Cog → API | Good if you think "model API"; whisper/diarization models already exist |
| **Beam** (beam.cloud) | ✅ serverless | Modal-like decorators | Solid Modal alternative |
| **Fly.io** | ✅ (L40S/A100) | `fly deploy`, Dockerfile | Closest "always-ish container" with GPU |
| **HF Spaces / Inference Endpoints** | ✅ paid tiers | Dead-simple, esp. Gradio | We already use HF for pyannote; great demo path |
| **RunPod** | ✅ cheapest | Serverless endpoints, rougher edges | Budget GPU |
| **Render / Railway** | ❌ CPU only | Most Vercel-like git-push DX | Fine if CPU-slow transcription is acceptable |

## Architecture notes (apply to either plan)

- **Go async.** Transcription is too slow for a synchronous HTTP request:
  upload → store audio → enqueue job → worker transcribes → client polls or
  receives a webhook.
- **Blob storage** for audio in / transcript out: Cloudflare R2, Vercel Blob,
  or S3.
- **CPU is viable** if fast turnaround isn't required (`int8`, `medium` instead
  of `large-v3`) — unlocks the simplest hosts (Render/Railway/Fly).
- **Secrets / system deps:** `HF_TOKEN` for the gated pyannote models; `ffmpeg`
  must be in the image (diarization decodes via torchcodec; transcription uses
  bundled PyAV). See `CLAUDE.md` for the GPU/torch version matrix.
- **Model caching:** download weights once to a persistent volume; never bundle
  them or re-download per request.

## Chosen plans — one branch each

### Plan A — Modal only (branch: `deploy/modal`)

Single platform. Modal hosts both the web endpoint and the GPU transcription,
so there's one deploy and one place for secrets/volumes.

- `modal` app: an image with `ffmpeg` + `faster-whisper` + `pyannote.audio`.
- A persistent `modal.Volume` caches the model weights across invocations.
- `HF_TOKEN` as a `modal.Secret`.
- A GPU function wraps `transcribe_meeting.transcribe` / `diarize`.
- Expose a web endpoint (Modal ASGI/WSGI, or wrap the existing Flask app) with
  an upload route that `.spawn()`s the job and a status/result route that polls.
- `modal deploy` to ship; scales to zero when idle.

### Plan B — Vercel + Modal (branch: `deploy/vercel-modal`)

Split: keep the Vercel DX for the UI, offload the heavy work to Modal.

- **Vercel:** web UI + thin API (upload, job status, fetch result). No GPU, no
  models — just orchestration.
- **Blob storage** (Vercel Blob or R2): client uploads audio there; Vercel
  hands Modal a URL/key.
- **Modal:** GPU transcription worker (same image/volume/secret as Plan A),
  invoked from the Vercel API; writes the transcript back to blob storage and
  signals completion (webhook or status flag the UI polls).
- Two deploys, but each side stays in its sweet spot.

## Open questions

- GPU vs CPU default (turnaround speed vs cost/simplicity)?
- Auth / who can upload (personal use vs shared)?
- Retention: keep audio/transcripts, or delete after delivery (privacy — the
  whole point of the local tool was that recordings don't leave the machine)?
