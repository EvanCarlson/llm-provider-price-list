"""Sample a fixed, seeded subset of HumanEval problems into data/humaneval.json.

Usage:
    hf download openai/openai_humaneval --repo-type dataset --local-dir <dir>
    python prepare_humaneval.py --parquet <dir>/openai_humaneval/test-00000-of-00001.parquet

Writes a JSON list of {id, question, answer, test, entry_point}. The question is
the function prompt (signature and docstring), answer is unused (grading runs
the tests), and id is the problem's index in the dataset, so a later sample can
exclude ids already used. The paper's benchmark-scale runs use n=100 for
HumanEval; the dataset has 164 problems.
"""

import argparse
import json
import logging
import random
from pathlib import Path

import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

DEFAULT_OUT = Path("data/humaneval.json")
DEFAULT_N = 100
DEFAULT_SEED = 0


def sample_problems(
    *, rows: list[dict], n: int, seed: int, exclude_ids: set[int]
) -> list[dict[str, str | int]]:
    pool = [i for i in range(len(rows)) if i not in exclude_ids]
    chosen = random.Random(seed).sample(pool, n)

    return [
        {
            "id": i,
            "question": rows[i]["prompt"],
            "answer": "",
            "test": rows[i]["test"],
            "entry_point": rows[i]["entry_point"],
        }
        for i in sorted(chosen)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample HumanEval problems")
    parser.add_argument("--parquet", type=Path, required=True, help="HumanEval parquet")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Problems to sample")
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
    rows = pq.read_table(args.parquet).to_pylist()
    exclude_ids = (
        {q["id"] for q in json.loads(args.exclude.read_text())}
        if args.exclude
        else set()
    )
    sampled = sample_problems(
        rows=rows, n=args.n, seed=args.seed, exclude_ids=exclude_ids
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(sampled, indent=1))
    logger.info("Wrote %d problems to %s", len(sampled), args.out)


if __name__ == "__main__":
    main()
