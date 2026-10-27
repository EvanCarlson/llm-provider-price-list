# You Can't Pick an LLM Provider From the Price List

Code for a replication of a slice of [He et al., "You Cannot Pick a Provider From the Price List: Market-Aware Routing for Open-Weight LLM Inference"](https://arxiv.org/abs/2609.37902) (arXiv 2609.37902, 29 Sep 2026). The paper releases no code, so everything here is written from its description. Where the paper is specific, the code follows it. Where it is silent, the choice is ours and is listed below.

## Get the data first

This repo has the code and the seeded question samples (`data/`), not the measurements. `results/` is gitignored, so the analysis scripts and the dashboard have nothing to read until you collect it. Collecting the post's data cost about $1.70 in OpenRouter credits.

```bash
echo 'OPENROUTER_API_KEY="sk-or-..."' > .env
uv sync
uv run python prepare_data.py && uv run python prepare_humaneval.py   # only to resample the questions
uv run python price_monitor.py --models meta-llama/llama-3.3-70b-instruct meta-llama/llama-3.1-8b-instruct
uv run python probe.py --model meta-llama/llama-3.1-8b-instruct --task gsm8k --wave 0
```

`run_waves.sh` and `run_plan.sh` chain the remaining probe waves (see the table below), and `rerun_missing.sh` retries requests that failed. Then run `analyze.py`, `paired_test.py`, `facet.py` and `live_facet.py`, and open the dashboard with `uv run streamlit run dashboard_app.py`. It shows the provider market, the FACET replay (pick a model, a scenario and a policy), the robustness tests, the live runs and the analysis tables.

Together, the paper's anchor, was removed from OpenRouter's Llama-3.3-70B listing on or before 2026-10-26, so a new 70B run needs a different `--anchor` and its savings will not compare with the post's numbers.

## What runs where

| Step | Command | Needs the API |
|---|---|---|
| Sample questions | `prepare_data.py` (GSM8K, MMLU), `prepare_humaneval.py` | no |
| Poll prices for 24h | `price_monitor.py --models <slug> ...` | no (public endpoint listing) |
| Probe every provider | `probe.py --model <slug> --task <task> --wave <n>` | yes, `OPENROUTER_API_KEY` in the environment or `.env` |
| Run a probe plan | `run_plan.sh`, `rerun_missing.sh` | yes |
| Provider-axis analysis | `analyze.py` | no |
| Do providers differ? (paired permutation test) | `paired_test.py` | no |
| FACET replay | `facet.py --model <slug> --tasks gsm8k mmlu humaneval [--robustness]` | no |
| Live FACET | `live_facet.py` | yes |
| Dashboard (Streamlit) | `uv run streamlit run dashboard_app.py` | no |
| PNG figures for the post | `export_figures.py` | no |

`.env` is gitignored. `results/` is gitignored. Run `uv run pytest`, `uv run ruff check`, `uv run mypy` before trusting a change.

## What follows the paper, and what is ours

**P** = specified by the paper. **P\*** = in the paper, but the numbers or details are ours. **M** = ours.

| Piece | Where | Status |
|---|---|---|
| Paired permutation test of the accuracy range per cell, Holm-adjusted | `paired_test.py` | **M** (ours, not in the paper) |
| Pinned requests, fallbacks off, record accuracy, cost, latency, availability | `probe.py` | **P** (Sec. 2, App. D) |
| Retry with backoff, early abort on repeated failures | `probe.py` | **P\*** (3 attempts, 2^n backoff, abort after 5) |
| Generous token budget | `probe.py` | **P\*** (2048) |
| Prompts, temperature 0, answer parsing | `prompts/`, `probe.py` | **M** |
| GSM8K and MMLU, n = 150; HumanEval, n = 100 | `prepare_*.py` | **P\*** (sizes from the paper, seeded sampling ours) |
| Availability excludes client errors (billing, BYOK, bad request) | `probe.provider_availability` | **M** |
| 24h price polling at 15 minutes, re-election definition | `price_monitor.py` | **P** (App. T) |
| Price scalar = mean of input and output $/M | `price_monitor.py` | **P\*** inferred from the paper's quoted prices, never defined |
| Measured-map router, baselines, out-of-sample, confidence-supported variant | `router.py` | **P** (Sec. 5, Table 18, App. H, K, U). Fallback when no provider is eligible is ours |
| Dispersion, price correlations, margin x health sweep, route drift | `analyze.py` | **P** (Sec. 3 to 5, Tables 1, 2, 16, 18, 33). We have 2 models, not 6 |
| FACET: window 200, n_min 12, cohort margin 8, floor slack 3, floor = best - 5, CUSUM h = ln 100, alternative = min(mu0 - 0.10, floor - 0.02), anchor never quarantined | `facet.py` | **P** (Sec. 6, App. S) |
| Confidence-aware rule: n_min 20, quarantine on the Wilson upper bound vs best - 8 | `facet.py` (`CONFIDENCE_RULE`) | **P\*** (App. P gives n_min and the quarantine rule; the Wilson confidence level, one-sided 95%, is ours; certification keeps the point test) |
| Scenarios S1, S2, S3; context-blind ablation | `facet.py` | **P** (Table 3) |
| Label noise 0 to 33% and OOD traffic | `facet.py --robustness` | **P\*** (Table 5; how OOD queries' true quality is defined is ours) |
| Probe-target rule, quarantine clears the window, anchor and mine choice, rate clamp, one-task mine, mine accuracy 0.30 | `facet.py` | **M** |
| Replay by bootstrapping measured outcomes, cost against a clairvoyant oracle, false-quarantine tagging | `facet.py` | **P\*** / **M** |
| Live run: 70B, Together as the $1.04/M anchor, interleaved tasks, backoff on failing endpoints | `live_facet.py` | **P\*** (2 runs, 40 and 65 minutes, not 36 hours, at probe rates 0.24 and 0.9; probe floors come from the wave 0 samples; the 0.9 rate was chosen after the 0.24 run never certified anyone; `--duration-hours` spaces queries over a longer run, but Together, the anchor, left OpenRouter's 70B listing before a long run, so we did not do one) |
| HumanEval grader | `humaneval_grader.py` | **M** |
| Streamlit dashboard, Wilson intervals, shared charts, PNG figure export | `dashboard_app.py`, `charts.py`, `export_figures.py` | **M** |

Not implemented: bandit baselines, periodic re-probing, TOST equivalence tests, latency-SLA analysis, the evaluator-bias sweep.

## Caveats that apply to every number

- **One aggregator, one client location.** The paper never names its aggregator; we use OpenRouter. Latency includes OpenRouter and our network. Prices are OpenRouter's posted rates, which the paper found can hide a provider's time-of-day tariff.
- **Small samples.** 150 questions per provider and task gives a 95% interval of roughly 6 points. Gaps of a few points are not separable.
- **Waves hours apart.** The paper's drift waves are 13 and 43 days apart.
- **Injected mines are synthetic**, as in the paper, because our measured market may have none.
- **Google Vertex is excluded** from the 70B runs: its endpoints are BYOK-only and return 404 without your own Google credentials.
- **Billing errors are not provider failures.** An early run hit OpenRouter's 402 (insufficient credits) on several providers. Those records are tagged `client_error` and excluded from availability.

## Safety: HumanEval executes model-written code

`humaneval_grader.py` runs each completion in a separate `python -I` process in a fresh temp directory, wrapped in macOS `sandbox-exec` (network denied, writes only inside the temp directory, `.env` files and credential directories unreadable), with a CPU limit, a memory limit where the OS honours it, a 10 second timeout, and process-group kill. It refuses to run without `sandbox-exec`. This is process isolation, not a hardened sandbox. A sentinel printed after the tests prevents an early `exit(0)` from counting as a pass.
