"""Sample a fixed, seeded subset of GSM8K or MMLU test questions into data/<task>.json.

Usage:
    hf download openai/gsm8k --repo-type dataset --include "main/test-*" --local-dir <dir>
    python prepare_data.py --task gsm8k --parquet <dir>/main/test-00000-of-00001.parquet

    hf download cais/mmlu --repo-type dataset --include "all/test-*" --local-dir <dir>
    python prepare_data.py --task mmlu --parquet <dir>/all/test-00000-of-00001.parquet

    # a fresh set that shares no question with an existing sample (live run)
    python prepare_data.py --task gsm8k --parquet <p> --seed 1 --n 250 \
        --exclude data/gsm8k.json --out data/live_gsm8k.json

Writes a JSON list of {id, question, answer}. For GSM8K the answer is the final
number after the "####" marker. For MMLU the question text includes the lettered
options and the answer is the correct letter. The sample is seeded so reruns pick
the same questions and results stay comparable across providers.
"""

import argparse
import json
import logging
import random
from pathlib import Path

import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

DEFAULT_N = 150
DEFAULT_SEED = 0
ANSWER_MARKER = "####"
LETTERS = "ABCD"


def final_answer(*, solution: str) -> str:
    return solution.split(ANSWER_MARKER)[-1].strip().replace(",", "")


def format_gsm8k(*, row: dict) -> dict[str, str]:
    return {
        "question": row["question"],
        "answer": final_answer(solution=row["answer"]),
    }


def format_mmlu(*, row: dict) -> dict[str, str]:
    subject = row["subject"].replace("_", " ")
    options = "\n".join(
        f"{letter}. {choice}" for letter, choice in zip(LETTERS, row["choices"])
    )

    return {
        "question": f"Subject: {subject}\n\n{row['question']}\n\n{options}",
        "answer": LETTERS[row["answer"]],
    }


FORMATTERS = {"gsm8k": format_gsm8k, "mmlu": format_mmlu}


def sample_questions(
    *, rows: list[dict], task: str, n: int, seed: int, exclude_ids: set[int]
) -> list[dict[str, str | int]]:
    pool = [i for i in range(len(rows)) if i not in exclude_ids]
    chosen = random.Random(seed).sample(pool, n)

    return [{"id": i, **FORMATTERS[task](row=rows[i])} for i in sorted(chosen)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample benchmark test questions")
    parser.add_argument("--task", choices=sorted(FORMATTERS), required=True)
    parser.add_argument(
        "--parquet", type=Path, required=True, help="Test split parquet file"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Default: data/<task>.json"
    )
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Questions to sample")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--exclude",
        type=Path,
        default=None,
        help="Existing sample JSON whose ids to leave out (for a fresh, disjoint set)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        dest="log_level",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(message)s")
    out = args.out or Path(f"data/{args.task}.json")
    rows = pq.read_table(args.parquet).to_pylist()
    exclude_ids = (
        {q["id"] for q in json.loads(args.exclude.read_text())}
        if args.exclude
        else set()
    )
    sampled = sample_questions(
        rows=rows, task=args.task, n=args.n, seed=args.seed, exclude_ids=exclude_ids
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sampled, indent=1))
    logger.info("Wrote %d questions to %s", len(sampled), out)


if __name__ == "__main__":
    main()
