#!/usr/bin/env bash
# compare_benchmarks.sh — run ssd-llm and llama.cpp benchmarks side by side
# Usage: ./scripts/compare_benchmarks.sh <model.gguf> [memory_budget]
set -euo pipefail

MODEL="${1:?Usage: $0 <model.gguf> [memory_budget]}"
BUDGET="${2:-8G}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSD_LLM="${SSD_LLM_BIN:-$SCRIPT_DIR/../target/release/ssd-llm}"

echo "╔══════════════════════════════════════════════════════╗"
echo "║        ssd-llm vs llama.cpp — benchmark compare      ║"
echo "╚══════════════════════════════════════════════════════╝"
echo
echo "Model:  $MODEL"
echo "Budget: $BUDGET"
echo "Chip:   $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
echo

echo "━━━ ssd-llm ━━━"
# bench prints log lines before JSON; extract from first '{' to end of output
"$SSD_LLM" bench "$MODEL" --memory-budget "$BUDGET" --json 2>/dev/null \
  | sed -n '/^{/,$p' \
  | python3 -c "
import json,sys
d=json.load(sys.stdin)
s=d['summary']
print(f\"  prefill:  {s['est_prefill_tok_per_sec']:.1f} tok/s\")
print(f\"  decode:   {s['est_decode_tok_per_sec']:.1f} tok/s\")
print(f\"  SSD BW:   {s['ssd_bandwidth_mb_per_sec']:.0f} MB/s\")
print(f\"  cache:    {s['cache_efficiency_pct']:.0f}% hit rate\")
" || echo "  (ssd-llm bench failed — build first: cargo build --release)"

echo
echo "━━━ llama.cpp ━━━"
if command -v llama-bench >/dev/null 2>&1; then
  llama-bench -m "$MODEL" 2>/dev/null \
    | grep -E '\|\s*(pp512|tg[0-9]+)\s*\|' \
    | sed -E 's/^\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[[:space:]]*([a-z0-9]+)[[:space:]]*\|[[:space:]]*([0-9.]+).*/  \1  \2 t\/s/'
else
  echo "  llama-bench not found (brew install llama.cpp)"
fi
echo
echo "Done. Note: llama.cpp numbers = real inference; ssd-llm = I/O-bound estimates."
echo "ssd-llm's advantage appears with models LARGER than your RAM."
