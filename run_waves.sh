#!/usr/bin/env bash
# Chain the probe runs: 70B wave 0 after the 8B wave 0 finishes, then waves 1-3
# for both models at 6-hour offsets from a base time.
#
# Usage: run_waves.sh <pid of the running 8B wave-0 probe> <base epoch seconds>
#
# Waves 1-3 use the first 50 questions so each wave finishes well inside its
# 6-hour slot. Google Vertex is excluded from the 70B runs: its endpoints are
# BYOK-only, so calls return 404 for us and would read as false unavailability.
set -u
cd "$(dirname "$0")"

MODEL_8B=meta-llama/llama-3.1-8b-instruct
MODEL_70B=meta-llama/llama-3.3-70b-instruct
PROVIDERS_70B=(deepinfra/turbo novita/bf16 akashml/fp8 parasail/fp8 cloudflare/fp8
  sambanova-turbo groq coreweave/fp16 together)
OUT=results/probes.jsonl
MAX_SPEND=2.0
WAVE_INTERVAL_SECONDS=21600
WAVE_N_QUESTIONS=50

log() { echo "$(date '+%F %T') $*"; }

while kill -0 "$1" 2>/dev/null; do sleep 30; done
log "8B wave 0 finished"

log "70B wave 0 start"
uv run python probe.py --model "$MODEL_70B" --providers "${PROVIDERS_70B[@]}" \
  --n-questions 150 --wave 0 --max-spend "$MAX_SPEND" --out "$OUT"
log "70B wave 0 finished"

for wave in 1 2 3; do
  target=$(( $2 + wave * WAVE_INTERVAL_SECONDS ))
  now=$(date +%s)
  if [ "$target" -gt "$now" ]; then sleep $(( target - now )); fi
  log "wave $wave start"
  uv run python probe.py --model "$MODEL_8B" --n-questions "$WAVE_N_QUESTIONS" \
    --wave "$wave" --max-spend "$MAX_SPEND" --out "$OUT"
  uv run python probe.py --model "$MODEL_70B" --providers "${PROVIDERS_70B[@]}" \
    --n-questions "$WAVE_N_QUESTIONS" --wave "$wave" --max-spend "$MAX_SPEND" --out "$OUT"
  log "wave $wave finished"
done
log "all waves finished"
