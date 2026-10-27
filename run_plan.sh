#!/usr/bin/env bash
# Run the remaining probe plan after the 70B reruns and smoke tests.
#
# Usage: run_plan.sh <wave 2 epoch seconds> <wave 3 epoch seconds>
#
# Phase A: wave 0 for the two new tasks, both models: mmlu with 150 questions and
# humaneval with 100 (the paper's sample sizes).
# Phase B: waves 2 and 3 for all three tasks, 50 questions each (the first 50 of
# the wave-0 sample), each wave starting at its epoch or as soon as the previous
# phase finishes, whichever is later. Models run one after another so we never
# probe the same provider from two streams at once, which would add rate limits
# that are our doing. Wave start times are logged; records carry their own
# timestamps. Google Vertex is excluded from 70B (BYOK-only).
set -u
cd "$(dirname "$0")"

MODEL_8B=meta-llama/llama-3.1-8b-instruct
MODEL_70B=meta-llama/llama-3.3-70b-instruct
PROVIDERS_70B=(deepinfra/turbo novita/bf16 akashml/fp8 parasail/fp8 cloudflare/fp8
  sambanova-turbo groq coreweave/fp16 together)
OUT=results/probes.jsonl
MAX_SPEND=3.0

log() { echo "$(date '+%F %T') $*"; }

probe_both_models() {  # task wave n_questions
  uv run python probe.py --model "$MODEL_8B" --task "$1" --wave "$2" \
    --n-questions "$3" --max-spend "$MAX_SPEND" --out "$OUT"
  uv run python probe.py --model "$MODEL_70B" --providers "${PROVIDERS_70B[@]}" \
    --task "$1" --wave "$2" --n-questions "$3" --max-spend "$MAX_SPEND" --out "$OUT"
}

log "phase A start"
log "wave 0 mmlu start"
probe_both_models mmlu 0 150
log "wave 0 mmlu finished"
log "wave 0 humaneval start"
probe_both_models humaneval 0 100
log "wave 0 humaneval finished"

epochs=("$1" "$2")
waves=(2 3)
for i in 0 1; do
  now=$(date +%s)
  if [ "${epochs[$i]}" -gt "$now" ]; then sleep $(( ${epochs[$i]} - now )); fi
  for task in gsm8k mmlu humaneval; do
    log "wave ${waves[$i]} $task start"
    probe_both_models "$task" "${waves[$i]}" 50
    log "wave ${waves[$i]} $task finished"
  done
done
log "plan finished"
