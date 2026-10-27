#!/usr/bin/env bash
# Retry requests that failed in earlier runs (probe.py skips only successful ones).
#
# Usage: rerun_missing.sh
#
# Covers the 70B waves 0 and 1 after the OpenRouter 402 (insufficient credits)
# failures. Google Vertex is excluded: its endpoints are BYOK-only.
set -u
cd "$(dirname "$0")"

MODEL_70B=meta-llama/llama-3.3-70b-instruct
PROVIDERS_70B=(deepinfra/turbo novita/bf16 akashml/fp8 parasail/fp8 cloudflare/fp8
  sambanova-turbo groq coreweave/fp16 together)
OUT=results/probes.jsonl
MAX_SPEND=2.0

log() { echo "$(date '+%F %T') $*"; }

log "70B wave 0 rerun start"
uv run python probe.py --model "$MODEL_70B" --providers "${PROVIDERS_70B[@]}" \
  --n-questions 150 --wave 0 --max-spend "$MAX_SPEND" --out "$OUT"
log "70B wave 1 rerun start"
uv run python probe.py --model "$MODEL_70B" --providers "${PROVIDERS_70B[@]}" \
  --n-questions 50 --wave 1 --max-spend "$MAX_SPEND" --out "$OUT"
log "rerun finished"
